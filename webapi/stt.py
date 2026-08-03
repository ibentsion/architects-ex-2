"""Backend speech-to-text — pluggable, and off unless you turn it on.

The browser's Web Speech API does the voice leg wherever it exists, which is
most of the time. This module is the fallback for the browsers that don't have
it (Firefox, Safari-on-anything-old).

It is NOT wired to Token Factory: the shared key has no ASR at all (verified
2026-08-03 — 27 models, none audio; ``/v1/audio/transcriptions`` 404s). So the
backend leg needs a local model, and that is a heavyweight dependency this
project does not otherwise have. Hence: install it yourself and say so.

    pip install faster-whisper
    export STT_MODEL=base                    # or an ivrit-ai model on the GPU node

``faster-whisper`` is deliberately absent from requirements.txt. When it is
absent the endpoint says so, loudly — a voice UI that silently returns an empty
transcript is worse than one that admits it cannot hear.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

NOT_CONFIGURED = (
    "STT backend not configured: pip install faster-whisper and set STT_MODEL "
    "(e.g. ivrit-ai/whisper-large-v3-turbo-ct2 on the GPU node, 'base' on CPU)"
)

#: Corpus, questions and answers are all Hebrew; so is the microphone.
LANGUAGE = "he"

_MIME_SUFFIX = {
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
}


class SttNotConfigured(RuntimeError):
    """No backend transcriber is installed/configured. Surfaced to the UI as
    501 with this message — never swallowed into an empty transcript."""


def _backend_installed() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


@lru_cache(maxsize=1)
def _load_model():
    from faster_whisper import WhisperModel

    model = os.environ["STT_MODEL"]
    device = os.environ.get("STT_DEVICE", "cpu")
    compute_type = os.environ.get("STT_COMPUTE_TYPE", "int8")
    logger.info("loading STT model %s (device=%s, compute_type=%s)", model, device, compute_type)
    return WhisperModel(model, device=device, compute_type=compute_type)


def transcribe(audio: bytes, mime: str) -> str:
    """Transcribe an uploaded clip, or raise :class:`SttNotConfigured`.

    Both conditions must hold — the package installed AND a model named — so
    that installing faster-whisper for something else does not silently switch
    voice input on with a model nobody chose.
    """
    if not _backend_installed() or not os.environ.get("STT_MODEL"):
        raise SttNotConfigured(NOT_CONFIGURED)

    # faster-whisper reads a path, not a buffer.
    suffix = _MIME_SUFFIX.get(mime.split(";")[0].strip(), ".bin")
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        handle.write(audio)
        handle.close()
        segments, _info = _load_model().transcribe(handle.name, language=LANGUAGE)
        return " ".join(segment.text.strip() for segment in segments).strip()
    finally:
        Path(handle.name).unlink(missing_ok=True)
