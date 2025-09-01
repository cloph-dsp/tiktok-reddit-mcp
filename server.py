import asyncio
import functools
import logging
import time
import sys # Import sys for stdout flushing
from datetime import datetime
from os import getenv, makedirs, path, remove
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypeVar, cast
from urllib.parse import urlparse, urlunparse # Import for URL sanitization
import yt_dlp
import yt_dlp.utils
import requests
import praw  # type: ignore
import os
from mcp.server.fastmcp import FastMCP

# Import services
from .reddit_service import RedditService
from .video_service import VideoService
from .transcription_service import TranscriptionService

# Import utility functions
from .utils import _format_timestamp, _format_post, _extract_reddit_id, _find_submission_by_title

# Import Reddit client manager
from .reddit_client import RedditClientManager

# Import whisper configuration
from .whisper_config import USE_WHISPER, WhisperModel, _whisper_models

# Import custom exceptions
from .exceptions import RedditPostError, VideoDownloadError, TranscriptionError

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv  # type: ignore
    # Only load if variables not already in environment (override=False)
    load_dotenv(dotenv_path=os.getenv("ENV_FILE", ".env"), override=False)
except Exception as _e:  # pragma: no cover
    logging.getLogger(__name__).warning(f"Could not load .env file: {_e}")

F = TypeVar("F", bound=Callable[..., Any])
if TYPE_CHECKING:
    pass

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def require_write_access(func: F) -> F:
    """Decorator to ensure write access is available."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        reddit_manager = RedditClientManager()
        if reddit_manager.is_read_only:
            raise ValueError(
                "Write operation not allowed in read-only mode. Please provide valid credentials."
            )
        if not reddit_manager.check_user_auth():
            raise Exception(
                "Authentication required for write operations. "
                "Please provide valid REDDIT_USERNAME and REDDIT_PASSWORD environment variables."
            )
        return func(*args, **kwargs)

    return cast(F, wrapper)


mcp = FastMCP("Tiktok Reddit MCP")
reddit_manager = RedditClientManager()

# Initialize services
reddit_service = RedditService()
video_service = VideoService()
transcription_service = TranscriptionService()
@require_write_access
async def create_post(
    ctx: Any,
    subreddit: str,
    title: str,
    content: Optional[str] = None,
    is_self: bool = True,
    video_path: Optional[str] = None,
    thumbnail_path: Optional[str] = None,
    nsfw: bool = False,
    spoiler: bool = False,
    flair_id: Optional[str] = None,
    flair_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new post in a subreddit.

    Args:
        ctx: Context object for reporting progress.
        subreddit: Name of the subreddit to post in (with or without 'r/' prefix)
        title: Title of the post (max 300 characters)
        content: Content of the post (text for self posts, URL for link posts)
        is_self: Whether this is a self (text) post (True) or link post (False)
        video_path: Path to the video file to upload.
        thumbnail_path: Path to a thumbnail for the video.
        nsfw: Whether the submission should be marked NSFW (default: False).
        spoiler: Whether the submission should be marked as a spoiler (default: False).

    Returns:
        Dictionary containing information about the created post

    Raises:
        ValueError: If input validation fails
        RuntimeError: For other errors during post creation
    """
    return await reddit_service.create_post(
        ctx=ctx,
        subreddit=subreddit,
        title=title,
        content=content,
        is_self=is_self,
        video_path=video_path,
        thumbnail_path=thumbnail_path,
        nsfw=nsfw,
        spoiler=spoiler,
        flair_id=flair_id,
        flair_text=flair_text,
    )


@require_write_access
async def reply_to_post(
    ctx: Any, post_id: str, content: str, subreddit: Optional[str] = None
) -> Dict[str, Any]:
    """Post a reply to an existing Reddit post.

    Args:
        ctx: Context object for reporting progress.
        post_id: The ID of the post to reply to (can be full URL, permalink, or just ID)
        content: The content of the reply (1-10000 characters)
        subreddit: The subreddit name if known (for validation, with or without 'r/' prefix)

    Returns:
        Dictionary containing information about the created reply and parent post

    Raises:
        ValueError: If input validation fails or post is not found
        RuntimeError: For other errors during reply creation
    """
    return await reddit_service.reply_to_post(
        ctx=ctx,
        post_id=post_id,
        content=content,
        subreddit=subreddit,
    )

@mcp.tool()
async def download_tiktok_video(ctx: Any, url: str, download_folder: str = "downloaded", keep: bool = True) -> Dict[str, Any]:
    """Download a TikTok video using yt-dlp (thumbnail discarded immediately).
 
    Args:
        ctx: Context object for reporting progress.
        url: TikTok video URL (short or full)
        download_folder: Destination folder
        keep: If False, video will be deleted after returning metadata
 
    Returns:
        Dict with video_id, video_path (may be deleted later), thumbnail_deleted flag
    """
    return await video_service.download_tiktok_video(
        ctx=ctx,
        url=url,
        download_folder=download_folder,
        keep=keep
    )

@mcp.tool()
async def transcribe_video(ctx: Any, video_path: str, model_size: str = "small") -> Dict[str, Any]:
    """Transcribe a downloaded video using faster-whisper if enabled.

    Args:
        ctx: Context object for reporting progress.
        video_path: Path to local video file
        model_size: Whisper model size (tiny, base, small, medium, large-v3)

    Returns:
        Dict with transcript and segments. If transcription disabled, returns message.
    """
    return await transcription_service.transcribe_video(
        ctx=ctx,
        video_path=video_path,
        model_size=model_size
    )

@mcp.tool()
@require_write_access
def post_downloaded_video(
    ctx: Any,
    video_path: Optional[str] = None,
    subreddit: str = "",
    title: str = "",
    thumbnail_path: Optional[str] = None,
    nsfw: bool = False,
    spoiler: bool = False,
    flair_id: Optional[str] = None,
    flair_text: Optional[str] = None,
    comment: Optional[str] = None,
    original_url: Optional[str] = None,
    comment_language: str = "both",  # 'en', 'pt', 'both'
    auto_comment: bool = True,
    video_id: Optional[str] = None,
    download_folder: str = "downloaded",
) -> Dict[str, Any]:
    """Post a previously downloaded video to Reddit, then auto-delete video & thumbnail.

    You can supply either:
      - video_path (full path), OR
      - video_id (basename without extension) + optional download_folder

    If neither is given, raises ValueError.

    Auto comment generation (when comment not provided and original_url present):
      en   -> "Original link: <url>"
      pt   -> "Link original: <url>"
      both -> "Original link / link original: <url>"
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(_post_downloaded_video_async(
            ctx=ctx,
            video_path=video_path,
            subreddit=subreddit,
            title=title,
            thumbnail_path=thumbnail_path,
            nsfw=nsfw,
            spoiler=spoiler,
            flair_id=flair_id,
            flair_text=flair_text,
            comment=comment,
            original_url=original_url,
            comment_language=comment_language,
            auto_comment=auto_comment,
            video_id=video_id,
            download_folder=download_folder
        ))
    except Exception as e:
        logger.error(f"Error in post_downloaded_video: {e}")
        raise

async def _post_downloaded_video_async(
    ctx: Any,
    video_path: Optional[str],
    subreddit: str,
    title: str,
    thumbnail_path: Optional[str],
    nsfw: bool,
    spoiler: bool,
    flair_id: Optional[str],
    flair_text: Optional[str],
    comment: Optional[str],
    original_url: Optional[str],
    comment_language: str,
    auto_comment: bool,
    video_id: Optional[str],
    download_folder: str
) -> Dict[str, Any]:
    """Helper function to run the async part of post_downloaded_video."""
    # Resolve video path from id if needed
    if not video_path:
        if not video_id:
            raise ValueError("Provide either video_path or video_id")
        # search for supported extensions
        for ext in ('.mp4', '.mov', '.mkv', '.webm'):
            candidate = path.join(download_folder, f"{video_id}{ext}")
            if path.exists(candidate):
                video_path = candidate
                break
        if not video_path or not path.exists(video_path):
            raise ValueError(f"Could not locate video for id {video_id} in '{download_folder}'")

    if not path.exists(video_path):
        raise ValueError("Video path does not exist")

    lang = comment_language.lower().strip()
    if lang not in ("en", "pt", "both"):
        lang = "both"

    auto_generated = False
    if not comment and auto_comment and original_url:
        if lang == "en":
            comment = f"Original link: {original_url}"
        elif lang == "pt":
            comment = f"Link original: {original_url}"
        else:
            comment = f"Original link / link original: {original_url}"
        auto_generated = True

    # Check subreddit video posting allowance before attempting to post
    if video_path:
        subreddit_details = await reddit_service.get_subreddit_details(subreddit)
        if not subreddit_details.get("video_post_allowed", False):
            raise ValueError(f"Subreddit r/{subreddit_details['name']} does not allow video posts.")
    
    # Pre-submission video file validation
    if video_path:
        if not os.path.exists(video_path):
            raise ValueError(f"Video file does not exist: {video_path}")
        
        # Check file size (Reddit has a 1GB limit for videos)
        max_size_bytes = 1 * 1024 * 1024 * 1024  # 1 GB
        try:
            file_size = os.path.getsize(video_path)
            if file_size > max_size_bytes:
                raise ValueError(f"Video file is too large ({file_size} bytes). Maximum allowed size is {max_size_bytes} bytes (1GB).")
            logger.info(f"Video file size is valid: {file_size} bytes")
        except OSError as e:
            logger.warning(f"Could not determine file size for {video_path}: {e}")
            # We'll proceed anyway, as Reddit might give a more informative error
    
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

    await report_progress({"status": "posting_video", "message": "Attempting to create Reddit post..."})
    try:
        post_info = await create_post(
            ctx=ctx,
            subreddit=subreddit,
            title=title,
            video_path=video_path,
            thumbnail_path=thumbnail_path if thumbnail_path and path.exists(thumbnail_path) else None,
            nsfw=nsfw,
            spoiler=spoiler,
            flair_id=flair_id,
            flair_text=flair_text,
        )
        await report_progress({"status": "posting_video", "message": f"Post created: {post_info['metadata']['permalink']}"})
        result: Dict[str, Any] = { 'post': post_info, 'used_video_path': video_path }
        if comment:
            await report_progress({"status": "posting_video", "message": "Adding comment to post..."})
            post_id = post_info['metadata']['id']
            reply = await reply_to_post(ctx=ctx, post_id=post_id, content=comment)
            await report_progress({"status": "posting_video", "message": "Comment added."})
            result['comment'] = reply
            result['comment_language'] = comment_language
            result['auto_comment_generated'] = auto_comment and original_url and not comment
        else:
            result['comment_language'] = None
            result['auto_comment_generated'] = False

        # Attempt deletion after successful post/comment
        deleted = False
        try:
            if path.exists(video_path):
                remove(video_path)
                deleted = True
            if thumbnail_path and path.exists(thumbnail_path):
                try: remove(thumbnail_path)
                except Exception: pass
        except Exception as e:
            logger.warning(f"Failed to delete video or thumbnail after posting: {e}")
            pass
        
        logger.info(f"Video posted successfully to Reddit. Permalink: {post_info['metadata']['permalink']}")
        return result
    except Exception as e:
        logger.error(f"Error in post_downloaded_video: {e}")
        raise
