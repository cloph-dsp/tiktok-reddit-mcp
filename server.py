# Load environment variables from .env FIRST, before any other imports
import os
try:
    from dotenv import load_dotenv  # type: ignore
    # Load .env file with override=True to ensure variables are set
    env_file = os.getenv("ENV_FILE", ".env")
    result = load_dotenv(dotenv_path=env_file, override=True)
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Loaded .env file from {env_file}: {result}")
    # Debug: log the environment variables
    logger.info(f"REDDIT_CLIENT_ID loaded: {'***' if os.getenv('REDDIT_CLIENT_ID') else 'None'}")
    logger.info(f"REDDIT_USERNAME loaded: {'***' if os.getenv('REDDIT_USERNAME') else 'None'}")
except Exception as _e:  # pragma: no cover
    import logging
    logging.getLogger(__name__).warning(f"Could not load .env file: {_e}")

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
from mcp.server.fastmcp import FastMCP

# Import services
from reddit_service import RedditService
from video_service import VideoService
from transcription_service import TranscriptionService

# Import utility functions
from utils import _format_timestamp, _format_post, _extract_reddit_id, _find_submission_by_title

# Import Reddit client manager
from reddit_client import RedditClientManager

# Import whisper configuration
from whisper_config import USE_WHISPER, WhisperModel, _whisper_models

# Import custom exceptions
from exceptions import RedditPostError, VideoDownloadError, TranscriptionError

F = TypeVar("F", bound=Callable[..., Any])
if TYPE_CHECKING:
    pass

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def require_write_access(func: F) -> F:
    """Decorator to ensure write access is available."""

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
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
        logger.info(f"require_write_access: Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        # Call the function and handle both sync and async functions properly
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
            
        logger.info(f"require_write_access: {func.__name__} returned {result}")
        return result

    return cast(F, wrapper)


def create_progress_reporter(ctx: Any) -> Callable[[Dict[str, Any]], None]:
    """Create a standardized async-safe progress reporter."""
    async def report_progress_async(data: Dict[str, Any]) -> None:
        try:
            if hasattr(ctx, '__event_emitter__'):
                await ctx.__event_emitter__({
                    "type": "status",
                    "data": {"description": data.get("message", "Unknown status"), "done": False}
                })
            elif hasattr(ctx, 'report_progress'):
                ctx.report_progress(data)
            else:
                # Enhanced logging for better visibility
                status = data.get("status", "unknown")
                message = data.get("message", "Unknown status")
                logger.info(f"🔄 PROGRESS [{status.upper()}]: {message}")

                # Log additional details for specific statuses
                if status == "uploading":
                    if "percentage" in data:
                        logger.info(f"📊 Upload Progress: {data['percentage']}")
                    if "downloaded_mb" in data and "total_mb" in data:
                        logger.info(f"📦 Data: {data['downloaded_mb']}MB / {data['total_mb']}MB")
                    if "speed_kbps" in data:
                        logger.info(f"⚡ Speed: {data['speed_kbps']} KB/s")
                elif status == "transcoding":
                    logger.info("🎬 Processing video for Reddit compatibility...")
                elif status == "posting_video":
                    logger.info("📤 Submitting to Reddit...")
                elif status == "media_processing":
                    logger.info("⏳ Waiting for Reddit to process video...")
        except Exception as e:
            logger.warning(f"Failed to report progress: {e}")

    def report_progress(data: Dict[str, Any]) -> None:
        # Check if there's a running event loop before using asyncio.create_task
        try:
            asyncio.get_running_loop()
            task = asyncio.create_task(report_progress_async(data))
            # Don't wait for completion to avoid blocking, but log if it fails
            task.add_done_callback(lambda t: logger.warning(f"Progress reporting task failed: {t.exception()}") if t.exception() else None)
        except RuntimeError:
            # No running event loop, fall back to synchronous logging
            status = data.get("status", "unknown")
            message = data.get("message", "Unknown status")
            logger.info(f"🔄 PROGRESS [{status.upper()}]: {message}")

    return report_progress


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
async def post_downloaded_video(
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
    logger.info(f"post_downloaded_video called with video_path={video_path}, subreddit={subreddit}, title={title}")
    try:
        result = await _post_downloaded_video_async(
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
        )
        logger.info(f"post_downloaded_video result: {result}")
        return result
    except asyncio.TimeoutError:
        logger.error("post_downloaded_video operation timed out")
        raise ValueError("Video posting operation timed out. Please try again.")
    except Exception as e:
        logger.error(f"Error in post_downloaded_video: {e}", exc_info=True)
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
    logger.info(f"_post_downloaded_video_async called with video_path={video_path}, subreddit={subreddit}, title={title}")
    return await _post_downloaded_video_async_impl(
        ctx, video_path, subreddit, title, thumbnail_path, nsfw, spoiler,
        flair_id, flair_text, comment, original_url, comment_language,
        auto_comment, video_id, download_folder
    )

async def _post_downloaded_video_async_impl(
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
    """Implementation of the async part of post_downloaded_video with timeout protection."""
    logger.info(f"_post_downloaded_video_async_impl started at {time.time()}")
    start_time = time.time()

    # Resolve video path from id if needed
    logger.info("Step 1: Resolving video path...")
    if not video_path:
        if not video_id:
            raise ValueError("Provide either video_path or video_id")
        # search for supported extensions
        logger.info(f"Searching for video file with ID {video_id} in folder {download_folder}")
        for ext in ('.mp4', '.mov', '.mkv', '.webm'):
            candidate = path.join(download_folder, f"{video_id}{ext}")
            logger.info(f"Checking candidate: {candidate}")
            if path.exists(candidate):
                video_path = candidate
                logger.info(f"Found video file: {candidate}")
                break
        if not video_path or not path.exists(video_path):
            # List all files in download folder for debugging
            try:
                files_in_folder = os.listdir(download_folder)
                logger.info(f"Files in download folder {download_folder}: {files_in_folder}")
            except Exception as e:
                logger.warning(f"Could not list files in download folder: {e}")
            raise ValueError(f"Could not locate video for id {video_id} in '{download_folder}'")
    logger.info(f"Video path resolved to {video_path}")
    if not path.exists(video_path):
        raise ValueError("Video path does not exist")
    logger.info(f"Video exists at {video_path}")

    # Additional video file validation
    try:
        file_size = os.path.getsize(video_path)
        logger.info(f"Video file size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
        if file_size == 0:
            raise ValueError(f"Video file is empty (0 bytes): {video_path}")
        if file_size > 1024 * 1024 * 1024:  # 1GB
            raise ValueError(f"Video file is too large ({file_size} bytes > 1GB): {video_path}")
    except OSError as e:
        logger.warning(f"Could not get file size for {video_path}: {e}")

    logger.info("Step 2: Processing comment language and auto-comment...")
    lang = comment_language.lower().strip()
    # Handle common variations
    if lang in ("english", "eng"):
        lang = "en"
    elif lang in ("portuguese", "portugues", "pt-br"):
        lang = "pt"
    elif lang not in ("en", "pt", "both"):
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
    logger.info(f"Comment processing complete. Comment: {comment}")

    # Check subreddit video posting allowance before attempting to post
    logger.info("Step 3: Checking subreddit details...")
    if video_path:
        logger.info("Attempting to retrieve subreddit details...")
        try:
            subreddit_details = await reddit_service.get_subreddit_details(subreddit)
            if subreddit_details is None:
                raise ValueError("Could not retrieve subreddit details.")
            logger.info(f"Subreddit details retrieved: {subreddit_details}")
            if not subreddit_details.get("video_post_allowed", False):
                raise ValueError(f"Subreddit r/{subreddit_details['name']} does not allow video posts.")
        except Exception as e:
            logger.error(f"Error retrieving subreddit details: {e}")
            raise
    
    # Pre-submission video file validation
    logger.info("Step 4: Validating video file...")
    if video_path:
        if not os.path.exists(video_path):
            raise ValueError(f"Video file does not exist: {video_path}")

        # Check file size (Reddit has a 1GB limit for videos)
        max_size_bytes = 1 * 1024 * 1024 * 1024  # 1 GB
        try:
            file_size = os.path.getsize(video_path)
            if file_size > max_size_bytes:
                raise ValueError(f"Video file is too large ({file_size} bytes). Maximum allowed size is {max_size_bytes} bytes (1GB).")
            if file_size == 0:
                raise ValueError(f"Video file is empty (0 bytes): {video_path}")
            logger.info(f"Video file size is valid: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")

            # Test file readability
            try:
                with open(video_path, 'rb') as f:
                    # Read first 1KB to ensure file is accessible
                    test_data = f.read(1024)
                    if len(test_data) == 0:
                        raise ValueError(f"Video file appears to be empty or corrupted: {video_path}")
                    # Check for common file signatures
                    if test_data.startswith(b'\x00\x00\x00'):
                        logger.info("Video file appears to be in MP4 format (good)")
                    elif test_data.startswith(b'RIFF'):
                        logger.warning("Video file appears to be in AVI format - may need transcoding")
                    else:
                        logger.info(f"Video file signature: {test_data[:4].hex()}")
                logger.info("Video file is readable and appears valid")
            except Exception as read_error:
                logger.error(f"Video file read test failed: {read_error}")
                raise ValueError(f"Video file cannot be read properly: {read_error}")

        except OSError as e:
            logger.warning(f"Could not determine file size for {video_path}: {e}")
            # We'll proceed anyway, as Reddit might give a more informative error

    # Create standardized progress reporter
    logger.info("Step 5: Setting up progress reporting...")
    report_progress = create_progress_reporter(ctx)
    report_progress({"status": "posting_video", "message": "Attempting to create Reddit post..."})

    # Transcode video to Reddit-safe format
    logger.info("Step 6: Starting video transcoding...")
    transcoded_video_path = None
    transcoding_info = None
    try:
        report_progress({"status": "transcoding", "message": "Transcoding video to Reddit-safe format..."})
        logger.info("Starting video transcoding...")
        # Check if video is already in a compatible format to skip transcoding
        logger.info("Checking if transcoding is needed...")
        video_ext = path.splitext(video_path)[1].lower()
        compatible_formats = {'.mp4'}

        if video_ext in compatible_formats:
            logger.info(f"Video is already in compatible format ({video_ext}), skipping transcoding")
            transcoded_video_path = video_path
            transcoding_info = {"skipped": True, "reason": "Already in compatible format"}
            report_progress({"status": "transcoding", "message": "Video format already compatible, skipping transcoding"})
        else:
            logger.info(f"Video needs transcoding from {video_ext} to MP4")
            # Check if transcoded version already exists
            base_name = path.splitext(path.basename(video_path))[0]
            transcoded_path = path.join("transcoded", f"{base_name}_reddit_safe.mp4")
            if path.exists(transcoded_path):
                logger.info(f"Found existing transcoded video: {transcoded_path}")
                transcoded_video_path = transcoded_path
                transcoding_info = {"skipped": True, "reason": "Using existing transcoded file"}
                report_progress({"status": "transcoding", "message": "Using existing transcoded video"})
            else:
                transcoding_result = await video_service.transcode_to_reddit_safe(ctx, video_path, "transcoded")
                if transcoding_result is None:
                    raise ValueError("Could not transcode video.")
                transcoded_video_path = transcoding_result["transcoded_path"]
                transcoding_info = transcoding_result
                report_progress({"status": "transcoding", "message": "Video transcoding completed successfully"})
                logger.info(f"Video transcoding result: {transcoding_result}")
                logger.info(f"Video transcoded to Reddit-safe format: {transcoded_video_path}")
    except Exception as e:
        logger.error(f"Error during video transcoding: {e}")
        report_progress({"status": "error", "message": f"Video transcoding failed: {e}"})
        # If transcoding fails, we'll try to upload the original video
        transcoded_video_path = video_path
        report_progress({"status": "warning", "message": "Using original video file for upload (transcoding failed)"})

    # Validate thumbnail if provided
    logger.info("Step 7: Validating thumbnail...")
    validated_thumbnail_path = None
    if thumbnail_path and path.exists(thumbnail_path):
        # Check if thumbnail has a valid extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        _, ext = path.splitext(thumbnail_path.lower())
        if ext in valid_extensions:
            validated_thumbnail_path = thumbnail_path
        else:
            logger.warning(f"Invalid thumbnail extension {ext}, skipping thumbnail")
            report_progress({"status": "warning", "message": f"Invalid thumbnail extension {ext}, skipping thumbnail"})
    elif thumbnail_path:
        logger.warning(f"Thumbnail file does not exist: {thumbnail_path}")
        report_progress({"status": "warning", "message": f"Thumbnail file does not exist: {thumbnail_path}"})
    logger.info(f"Thumbnail validation complete. Validated path: {validated_thumbnail_path}")

    logger.info("Step 8: Creating Reddit post...")
    try:
        if transcoded_video_path is None:
            raise ValueError("transcoded_video_path is None, cannot create post.")
        logger.info("Attempting to create Reddit post...")
        post_info = await create_post(
            ctx=ctx,
            subreddit=subreddit,
            title=title,
            video_path=transcoded_video_path,
            thumbnail_path=validated_thumbnail_path,
            nsfw=nsfw,
            spoiler=spoiler,
            flair_id=flair_id,
            flair_text=flair_text,
        )
        report_progress({"status": "posting_video", "message": f"Post created: {post_info['metadata']['permalink']}"})
        logger.info(f"Post created successfully: {post_info['metadata']['permalink']}")
        result: Dict[str, Any] = {
            'post': post_info,
            'used_video_path': transcoded_video_path,
            'original_video_path': video_path
        }
        
        # Add transcoding info if available
        if transcoding_info:
            result['transcoding_info'] = transcoding_info
            
        if comment:
            logger.info("Step 9: Adding comment to post...")
            report_progress({"status": "posting_video", "message": "Adding comment to post..."})
            post_id = post_info['metadata']['id']
            logger.info(f"Adding comment to post {post_id}...")
            reply = await reply_to_post(ctx=ctx, post_id=post_id, content=comment)
            report_progress({"status": "posting_video", "message": "Comment added."})
            result['comment'] = reply
            result['comment_language'] = comment_language
            result['auto_comment_generated'] = auto_comment and original_url and not comment
            logger.info("Comment added successfully")
        else:
            result['comment_language'] = None
            result['auto_comment_generated'] = False

        # Attempt deletion after successful post/comment
        logger.info("Step 10: Cleaning up files...")
        deleted = False
        try:
            # Delete original video file if it exists and is different from transcoded
            if video_path and path.exists(video_path) and video_path != transcoded_video_path:
                remove(video_path)
                logger.info(f"Deleted original video file: {video_path}")
                
            # Delete transcoded video file if it exists
            if transcoded_video_path and path.exists(transcoded_video_path):
                remove(transcoded_video_path)
                logger.info(f"Deleted transcoded video file: {transcoded_video_path}")
                deleted = True
                
            # Delete thumbnail if it exists
            if thumbnail_path and path.exists(thumbnail_path):
                try:
                    remove(thumbnail_path)
                    logger.info(f"Deleted thumbnail file: {thumbnail_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete thumbnail {thumbnail_path}: {e}")
        except Exception as e:
            logger.warning(f"Failed to delete video or thumbnail after posting: {e}")
            pass
        
        logger.info(f"Video posted successfully to Reddit. Permalink: {post_info['metadata']['permalink']}")
        end_time = time.time()
        logger.info(f"_post_downloaded_video_async_impl completed successfully in {end_time - start_time:.2f} seconds")
        return result
    except Exception as e:
        logger.error(f"Error in post_downloaded_video: {e}")
        # Keep files for debugging if post fails
        logger.info(f"Keeping video files for debugging: {video_path}, {transcoded_video_path}")
        raise
    # Fallback return to prevent NoneType error in await expression
    # This should not be reached under normal circumstances due to explicit returns/raises above
    return {
        "status": "completed",
        "message": "Video post processed successfully (unexpected completion path)."
    }


# --- Server entrypoint ---
if __name__ == "__main__":
    import argparse, subprocess, sys, shutil, os as _os
 
    parser = argparse.ArgumentParser(description="Run TikTok→Reddit MCP server")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8050)), help="Proxy HTTP port (mcpo)")
    parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"), help="Proxy host (mcpo)")
    parser.add_argument("--no-proxy", action="store_true", help="Run raw MCP server (stdio) without mcpo HTTP proxy")
    args = parser.parse_args()
 
    # Child/raw mode: just run stdio MCP server
    if args.no_proxy or os.getenv("MCPO_CHILD") == "1":
        logger.info("Starting raw MCP stdio server (read_only=%s)", RedditClientManager().is_read_only)
        try:
            mcp.run()
        except TypeError:
            mcp.run()
        sys.exit(0)
 
    # Parent: launch mcpo with proper syntax: mcpo --host H --port P [--api-key K] -- python server.py --no-proxy
    api_key = os.getenv("MCPO_API_KEY")
    mcpo_path = shutil.which("mcpo")
    if not mcpo_path:
        logger.error("mcpo console script not found on PATH. Install with: pip install mcpo")
        sys.exit(1)
 
    server_script = _os.path.abspath(__file__)
    server_cmd = [sys.executable, server_script, "--no-proxy"]
 
    mcpo_cmd = [mcpo_path, "--host", args.host, "--port", str(args.port)]
    if api_key:
        mcpo_cmd += ["--api-key", api_key]
    mcpo_cmd += ["--"] + server_cmd
 
    logger.info("Launching mcpo proxy: %s", " ".join(mcpo_cmd))
    # Mark child for clarity (not strictly needed because of --no-proxy flag)
    env = os.environ.copy()
    env["MCPO_CHILD"] = "1"
    try:
        subprocess.run(mcpo_cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        logger.error("mcpo exited with code %s", e.returncode)
