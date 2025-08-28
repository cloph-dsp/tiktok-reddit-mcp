import logging
from os import getenv

logger = logging.getLogger(__name__)

# Optional transcription support via faster-whisper (enabled with USE_WHISPER_TRANSCRIPTION=true)
USE_WHISPER = getenv("USE_WHISPER_TRANSCRIPTION", "false").lower() in ("1", "true", "yes", "on")
_whisper_models = {}

try:
    if USE_WHISPER:
        from faster_whisper import WhisperModel  # type: ignore
    else:
        WhisperModel = None  # type: ignore
except Exception as _imp_err:  # pragma: no cover
    logger.warning(f"Disabling whisper transcription (import error): {_imp_err}")
    USE_WHISPER = False
    WhisperModel = None  # type: ignore