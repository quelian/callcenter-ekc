"""
Встроенный LLM-постредактор без Ollama: модель с Hugging Face в процессе Python.

По умолчанию: ``deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`` (лёгкая, ~3.5 ГБ скачивания).

Переменные окружения:
  LLM_EMBEDDED_MODEL_ID  — другой репозиторий HF при необходимости
"""

from __future__ import annotations

import os
import re
import threading
from typing import Callable

from llm_post_edit import _SYSTEM_PROMPT, _parse_llm_blocks, _user_prompt, _validate_output

# DeepSeek (дистиллят Qwen 1.5B) — баланс качества и размера для CPU/GPU ноутбука
DEFAULT_HF_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

_load_lock = threading.Lock()
_cached: dict[str, tuple[object, object]] = {}


def _log_default(msg: str) -> None:
    print(f"[transcriber] {msg}", flush=True)


def _truncate_user_text(text: str, max_chars: int = 12_000) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = max_chars // 2 - 80
    tail = max_chars // 2 - 80
    return (
        text[:head]
        + "\n\n[... фрагмент опущен из-за лимита контекста модели ...]\n\n"
        + text[-tail:]
    )


def _strip_reasoning_tags(text: str) -> str:
    """У DeepSeek-R1 в ответе может быть служебный блок рассуждения."""
    if not text:
        return text
    out = re.sub(
        r"`\s*think\s*`.*?`\s*/think\s*`",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    out = re.sub(
        r"`\s*think\s*`.*?`</think>`",
        "",
        out,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return out.strip()


def _resolved_embedded_model_id(explicit: str | None) -> str:
    return (explicit or os.environ.get("LLM_EMBEDDED_MODEL_ID") or "").strip() or DEFAULT_HF_MODEL_ID


def _pick_device_dtype():
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float32  # fp16 на MPS даёт nan/inf при сэмплинге
    return "cpu", torch.float32


def _ensure_model(model_id: str, log: Callable[[str], None]) -> tuple[object, object]:
    with _load_lock:
        if model_id in _cached:
            return _cached[model_id]
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Нужны пакеты transformers и accelerate. Установите: pip install transformers accelerate"
            ) from exc

        device, dtype = _pick_device_dtype()
        log(f"LLM: устройство={device}, dtype={dtype}, репозиторий={model_id}")
        log("LLM: загрузка токенизатора (первый раз качается с Hugging Face)...")
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if getattr(tok, "pad_token_id", None) is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id

        log("LLM: загрузка весов модели...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model = model.to(device)
        model.eval()
        _cached[model_id] = (tok, model)
        log("LLM: модель готова к генерации.")
        return tok, model


def _build_prompt_text(tok: object, user_content: str) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    tpl = getattr(tok, "chat_template", None)
    if tpl:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    # Qwen/DeepSeek-стиль (fallback)
    parts = [
        "<|im_start|>system\n",
        _SYSTEM_PROMPT,
        "<|im_end|>\n<|im_start|>user\n",
        user_content,
        "<|im_end|>\n<|im_start|>assistant\n",
    ]
    return "".join(parts)


def run_embedded_llm_post_edit(
    flat_text: str,
    role_transcript: str,
    *,
    model_id: str | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[str | None, str | None, str]:
    lg = log or _log_default
    mid = _resolved_embedded_model_id(model_id)
    try:
        tok, model = _ensure_model(mid, lg)
    except Exception as exc:
        return None, None, f"load_error:{type(exc).__name__}:{exc}"

    import torch

    user_body = _truncate_user_text(_user_prompt(flat_text, role_transcript))
    prompt = _build_prompt_text(tok, user_body)
    device = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    try:
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                pad_token_id=getattr(tok, "pad_token_id", None) or tok.eos_token_id,
            )
    except Exception as exc:
        return None, None, f"generate_error:{type(exc).__name__}:{exc}"

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out[0, prompt_len:]
    raw = tok.decode(gen_ids, skip_special_tokens=True)
    content = _strip_reasoning_tags(raw)

    full, roles = _parse_llm_blocks(content)
    if not full or not roles:
        return None, None, "parse_blocks_failed"

    ok, reason = _validate_output(flat_text, full, roles)
    if not ok:
        return None, None, f"validation:{reason}"

    return full, roles, "ok"


def run_embedded_llm_post_edit_threaded(
    flat_text: str,
    role_transcript: str,
    *,
    model_id: str | None = None,
    wall_timeout: float,
    log: Callable[[str], None] | None = None,
) -> tuple[str | None, str | None, str]:
    box: dict[str, object] = {}
    mid = _resolved_embedded_model_id(model_id)

    def worker() -> None:
        try:
            f, r, note = run_embedded_llm_post_edit(
                flat_text, role_transcript, model_id=mid, log=log
            )
            box["full"] = f
            box["roles"] = r
            box["note"] = note
        except Exception as exc:
            box["note"] = f"worker:{type(exc).__name__}:{exc}"

    wall = max(30.0, float(wall_timeout))
    th = threading.Thread(target=worker, daemon=True)
    th.start()
    th.join(timeout=wall)
    if th.is_alive():
        return None, None, "timeout"
    return (
        box.get("full") if isinstance(box.get("full"), str) else None,
        box.get("roles") if isinstance(box.get("roles"), str) else None,
        str(box.get("note", "unknown")),
    )
