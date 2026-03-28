"""
Семантическое уточнение ролей «Оператор / Заявитель» по тексту сегментов.

Локально, бесплатно: RuBERT-Tiny2 (≈29M параметров) + косинусная близость к якорным
фразам типичного тона ЕКЦ и заявителя. Скачивание весов — один раз в кэш Hugging Face.

Отключить: CALLQA_SEMANTIC_ROLE_REFINE=0
Устройство: CALLQA_SEMANTIC_ROLE_DEVICE=auto|cpu|mps|cuda
"""

from __future__ import annotations

import os
import threading
from typing import Callable

# Якоря — не дословные цитаты из звонков, а обобщённый стиль речи (RU).
_OPERATOR_ANCHORS = (
    "Здравствуйте, меня зовут Егор, оператор единого контактного центра ДВФУ, чем могу вам помочь?",
    "Слушаю вас, подскажите, как к вам обращаться по имени?",
    "Уточню информацию по вашему вопросу, оставайтесь на линии, пожалуйста, одну минуту.",
    "Вам необходимо обратиться в деканат с паспортом и документами в рабочие часы.",
    "По данному вопросу подайте заявление через личный кабинет абитуриента на сайте ДВФУ.",
    "Я могу перезвонить вам в течение пятнадцати минут, когда уточню детали.",
    "Спасибо за обращение, всего доброго, хорошего дня, обращайтесь.",
    "Запишите, пожалуйста, номер заявления и дату подачи документов.",
    "Для заселения в общежитие направьте сканы справок на электронную почту приёмной комиссии.",
)

_APPLICANT_ANCHORS = (
    "Здравствуйте, подскажите пожалуйста, мне нужна справка об обучении для военкомата.",
    "Я хотел узнать про общежитие и какие документы нужно принести лично.",
    "Скажите, когда можно забрать справку и куда обратиться?",
    "У меня вопрос по поступлению, могу ли я подать документы онлайн?",
    "Добрый день, как записаться на личный приём к специалисту?",
    "Интересуют льготы и список документов для поступления.",
    "Спасибо большое, всё понял, до свидания.",
    "А подскажите, а если я не успеваю до конца недели?",
)

_MODEL_ID = "cointegrated/rubert-tiny2"
_MIN_CHARS = 12
# Косинусная разница прототипов; чем выше — тем реже лезем в уже согласованные роли.
_MARGIN_DEFAULT = 0.14
_MARGIN_HIGH_VOICE = 0.20

_lock = threading.Lock()
_model = None
_tokenizer = None
_device: str | None = None


def _semantic_refine_enabled() -> bool:
    v = (os.environ.get("CALLQA_SEMANTIC_ROLE_REFINE") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _pick_torch_device() -> str:
    raw = (os.environ.get("CALLQA_SEMANTIC_ROLE_DEVICE") or "auto").strip().lower()
    try:
        import torch
    except Exception:
        return "cpu"
    if raw == "cpu":
        return "cpu"
    if raw == "cuda" and torch.cuda.is_available():
        return "cuda"
    if raw == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if raw == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def _load_model(log: Callable[[str], None] | None):
    global _model, _tokenizer, _device
    import torch
    from transformers import AutoModel, AutoTokenizer

    if _model is not None and _tokenizer is not None and _device is not None:
        return _tokenizer, _model, _device
    if log:
        log(f"[semantic_roles] Загрузка {_MODEL_ID} для семантического уточнения ролей (локально, без API)…")
    _device = _pick_torch_device()
    tok = AutoTokenizer.from_pretrained(_MODEL_ID)
    mod = AutoModel.from_pretrained(_MODEL_ID)
    mod.eval()
    mod.to(_device)
    _tokenizer = tok
    _model = mod
    if log:
        log(f"[semantic_roles] Модель готова, устройство={_device}.")
    return tok, mod, _device


def _mean_pool(last_hidden: object, attention_mask: object) -> object:
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


def _encode_texts(tokenizer, model, device: str, texts: list[str]) -> object:
    import torch

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=160,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model(**enc).last_hidden_state
        pooled = _mean_pool(out, enc["attention_mask"])
        return torch.nn.functional.normalize(pooled, p=2, dim=1)


def _compute_prototypes(tokenizer, model, device: str) -> tuple[object, object]:
    import torch

    op_emb = _encode_texts(tokenizer, model, device, list(_OPERATOR_ANCHORS)).mean(dim=0, keepdim=True)
    ap_emb = _encode_texts(tokenizer, model, device, list(_APPLICANT_ANCHORS)).mean(dim=0, keepdim=True)
    op_emb = torch.nn.functional.normalize(op_emb, p=2, dim=1)
    ap_emb = torch.nn.functional.normalize(ap_emb, p=2, dim=1)
    return op_emb, ap_emb


def try_semantic_role_refine(
    chunks: list[str],
    roles: list[str],
    *,
    protect_fn: Callable[[str], bool],
    log: Callable[[str], None] | None = None,
    voice_confidence: str = "low",
) -> list[str]:
    """
    Возвращает новый список ролей той же длины. При ошибке или отключении — исходный ``roles``.
    """
    if not _semantic_refine_enabled():
        return roles
    if not chunks or not roles or len(chunks) != len(roles):
        return roles

    margin = _MARGIN_HIGH_VOICE if voice_confidence in {"high", "medium"} else _MARGIN_DEFAULT

    try:
        with _lock:
            tok, mod, dev = _load_model(log)
            op_p, ap_p = _compute_prototypes(tok, mod, dev)
    except Exception as exc:
        if log:
            log(f"[semantic_roles] Пропуск семантики: {type(exc).__name__}: {exc}")
        return roles

    texts = [(t or "").strip() for t in chunks]
    need = [i for i, t in enumerate(texts) if len(t) >= _MIN_CHARS and not protect_fn(t)]
    if not need:
        return roles

    flipped = 0
    out = list(roles)

    # Батчи по сегментам, требующим скоринга
    batch_size = 32
    for start in range(0, len(need), batch_size):
        idx_slice = need[start : start + batch_size]
        sub = [texts[i] for i in idx_slice]
        emb = _encode_texts(tok, mod, dev, sub)
        sim_op = (emb @ op_p.T).squeeze(-1)
        sim_ap = (emb @ ap_p.T).squeeze(-1)
        delta = sim_op - sim_ap
        for j, i in enumerate(idx_slice):
            d = float(delta[j].item())
            cur = out[i]
            if d > margin and cur == "Заявитель":
                out[i] = "Оператор"
                flipped += 1
            elif d < -margin and cur == "Оператор":
                out[i] = "Заявитель"
                flipped += 1

    if log and flipped:
        log(
            f"[semantic_roles] Семантика RuBERT: переназначено ролей у {flipped} из {len(need)} "
            f"проверенных сегментов (порог Δcos≈{margin:.2f}, voice={voice_confidence})."
        )
    return out
