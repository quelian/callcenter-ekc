"""Совместимость SpeechBrain с torchaudio 2.2+ и тихий запуск UI.

SpeechBrain при импорте вызывает ``torchaudio.list_audio_backends()``; в torchaudio 2.10+
этого API нет — без заглушки импорт падает. Раньше подставляли ``lambda: []``, из‑за чего
``check_torchaudio_backend()`` логировал «no working backend». Нужен непустой список,
если в системе есть ``soundfile`` (рекомендуемый бэкенд для macOS/Linux).
"""

from __future__ import annotations

import warnings


def apply_torchaudio_speechbrain_compat() -> None:
    """Добавляет ``list_audio_backends`` в torchaudio, если его убрали (2.2+)."""
    try:
        import torchaudio as ta
    except Exception:
        return
    if hasattr(ta, "list_audio_backends"):
        return

    def _list_audio_backends() -> list[str]:
        try:
            import soundfile  # noqa: F401
        except Exception:
            pass
        # Непустой список — иначе SpeechBrain логирует «no working backend».
        return ["soundfile"]

    ta.list_audio_backends = _list_audio_backends  # type: ignore[attr-defined, assignment]


def suppress_noisy_third_party_warnings() -> None:
    """Гасит известные предупреждения из SpeechBrain/Streamlit (не наши баги)."""
    # SpeechBrain 1.x: редирект speechbrain.pretrained → inference при обходе модулей Streamlit.
    warnings.filterwarnings(
        "ignore",
        message=r".*speechbrain\.pretrained.*",
        category=UserWarning,
    )
    # SpeechBrain + torch 2.x: deprecated torch.cuda.amp.custom_fwd внутри библиотеки.
    warnings.filterwarnings(
        "ignore",
        message=r".*torch\.cuda\.amp\.custom_fwd.*",
        category=FutureWarning,
    )
    # PyTorch 2.x: STFT с предвыделенным out-тензором — deprecation про resize [] → [freq, time].
    # Срабатывает из torch.functional.stft (SpeechBrain / torchaudio и др.).
    warnings.filterwarnings(
        "ignore",
        message=r"An output with one or more elements was resized since it had shape \[\], which does not match the required output shape.*",
        category=UserWarning,
    )


def apply_speechbrain_environment() -> None:
    """Вызвать в самом начале приложения (до ``import speechbrain``)."""
    apply_torchaudio_speechbrain_compat()
    suppress_noisy_third_party_warnings()
