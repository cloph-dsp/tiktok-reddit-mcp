import asyncio
import logging
from os import path
from typing import Any, Dict

# Import custom exceptions
from exceptions import TranscriptionError

# Optional transcription support via faster-whisper (enabled with USE_WHISPER_TRANSCRIPTION=true)
from whisper_config import USE_WHISPER, WhisperModel, _whisper_models

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Encapsulates video transcription logic."""

    async def transcribe_video(self, ctx: Any, video_path: str, model_size: str = "small") -> Dict[str, Any]:
        """Transcribe a downloaded video using faster-whisper if enabled.

        Args:
            ctx: Context object for reporting progress.
            video_path: Path to local video file
            model_size: Whisper model size (tiny, base, small, medium, large-v3)

        Returns:
            Dict with transcript and segments. If transcription disabled, returns message.
        """
        if not path.exists(video_path):
            raise ValueError("Video path does not exist")

        if not USE_WHISPER:
            return { 'status': 'disabled', 'message': 'Whisper transcription disabled. Set USE_WHISPER_TRANSCRIPTION=true to enable.' }

        try:
            if model_size not in ("tiny", "base", "small", "medium", "large-v3"):
                model_size = "small"
            if model_size not in _whisper_models:
                logger.info(f"Loading whisper model: {model_size}")
                _whisper_models[model_size] = WhisperModel(model_size, compute_type="auto")
            # Check if ctx has the report_progress attribute
            if hasattr(ctx, 'report_progress'):
                report_progress = ctx.report_progress
            elif hasattr(ctx, '__event_emitter__'):
                async def report_progress(data: Dict[str, Any]) -> None:
                    await ctx.__event_emitter__({
                        "type": "status",
                        "data": {"description": data.get("message", "Unknown status"), "done": False}
                    })
            else:
                logger.warning("ctx object does not have report_progress or __event_emitter__ attribute. Creating a dummy function.")
                async def report_progress(data: Dict[str, Any]) -> None:
                    logger.info(f"Dummy report_progress called with: {data}")

            model = _whisper_models[model_size]
            segments_iter, info = await asyncio.to_thread(model.transcribe, video_path, beam_size=1)
            segments = []
            full_text_parts = []
            for seg in segments_iter:
                segments.append({
                    'id': seg.id,
                    'start': seg.start,
                    'end': seg.end,
                    'text': seg.text.strip(),
                })
                full_text_parts.append(seg.text.strip())
            transcript = " ".join(full_text_parts)
            logger.info(f"Video transcription completed successfully. Language: {info.language}, Duration: {info.duration:.2f} seconds.")
            await report_progress({"status": "completed", "message": "Transcription completed"})
            return {
                'status': 'success',
                'language': info.language,
                'duration': info.duration,
                'segments': segments,
                'transcript': transcript,
            }
        except Exception as e:
            logger.error(f"Failed to transcribe video: {e}")
            raise TranscriptionError(f"Failed to transcribe video: {e}", original_exception=e) from e