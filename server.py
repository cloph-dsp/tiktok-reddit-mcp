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
def create_post(
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
    return reddit_service.create_post(
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
def reply_to_post(
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
    return reddit_service.reply_to_post(
        ctx=ctx,
        post_id=post_id,
        content=content,
        subreddit=subreddit,
    )

@mcp.tool()
def download_tiktok_video(ctx: Any, url: str, download_folder: str = "downloaded", keep: bool = True) -> Dict[str, Any]:
    """Download a TikTok video using yt-dlp (thumbnail discarded immediately).
 
    Args:
        ctx: Context object for reporting progress.
        url: TikTok video URL (short or full)
        download_folder: Destination folder
        keep: If False, video will be deleted after returning metadata
 
    Returns:
        Dict with video_id, video_path (may be deleted later), thumbnail_deleted flag
    """
    return video_service.download_tiktok_video(
        ctx=ctx,
        url=url,
        download_folder=download_folder,
        keep=keep
    )

@mcp.tool()
def transcribe_video(ctx: Any, video_path: str, model_size: str = "small") -> Dict[str, Any]:
    """Transcribe a downloaded video using faster-whisper if enabled.

    Args:
        ctx: Context object for reporting progress.
        video_path: Path to local video file
        model_size: Whisper model size (tiny, base, small, medium, large-v3)

    Returns:
        Dict with transcript and segments. If transcription disabled, returns message.
    """
    return transcription_service.transcribe_video(
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
        subreddit_details = get_subreddit_details(subreddit)
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
    
    ctx.report_progress({"status": "posting_video", "message": "Attempting to create Reddit post..."})
    post_info = create_post(
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
    ctx.report_progress({"status": "posting_video", "message": f"Post created: {post_info['metadata']['permalink']}"})
    result: Dict[str, Any] = { 'post': post_info, 'used_video_path': video_path }
    if comment:
        ctx.report_progress({"status": "posting_video", "message": "Adding comment to post..."})
        post_id = post_info['metadata']['id']
        reply = reply_to_post(ctx=ctx, post_id=post_id, content=comment)
        ctx.report_progress({"status": "posting_video", "message": "Comment added."})
        result['comment'] = reply
        result['comment_language'] = lang
        result['auto_comment_generated'] = auto_generated
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

@mcp.tool()
def suggest_subreddits(
    query: str,
    limit_subreddits: int = 5,
    posts_limit: int = 5,
    time_filters: Optional[List[str]] = None,
    post_sort: str = "top",  # 'top', 'hot', 'new'
) -> Dict[str, Any]:
    """Suggest relevant subreddits for a topic and show sample post titles.
 
    This tool finds candidate subreddits via search and ranks them by subscriber count.
    For each selected subreddit, it retrieves sample post titles for specified time ranges
    (default: week & year for 'top' sort) or a single retrieval for other sorts.
    
    For detailed information about a specific subreddit's rules, flair, and video posting
    allowance, use the `get_subreddit_details` tool.
 
    Args:
        query: Topic / keywords to search for.
        limit_subreddits: Max number of subreddits to return.
        posts_limit: Max sample posts per subreddit per time frame (will be capped at 5 for LLM context consistency).
        time_filters: List of two time filters (e.g. ['week','year']). Ignored for non-'top' sorts.
        post_sort: Post sorting method ('top','hot','new').
 
    Returns:
        Dict containing ranked subreddit suggestions and sample titles, PLUS a compact
        'llm_context' string explicitly summarizing each subreddit with subscriber counts
        and exactly 10 titles (5 per time frame) suitable to feed directly to an LLM.
    """
    manager = RedditClientManager()
    if not manager.client:
        raise RuntimeError("Reddit client not initialized")

    if not query or not isinstance(query, str):
        raise ValueError("query is required")

    post_sort = post_sort.lower().strip()
    if post_sort not in ("top", "hot", "new"):
        post_sort = "top"

    if time_filters is None or not isinstance(time_filters, list) or len(time_filters) < 2:
        time_filters = ["week", "year"]  # default two frames
    # Normalize & validate time filters
    valid_tf = {"hour", "day", "week", "month", "year", "all"}
    cleaned_time_filters: List[str] = []
    for tf in time_filters:
        tf_l = str(tf).lower().strip()
        if tf_l in valid_tf and tf_l not in cleaned_time_filters:
            cleaned_time_filters.append(tf_l)
        if len(cleaned_time_filters) == 2:
            break
    if len(cleaned_time_filters) < 2:
        # pad with defaults ensuring two
        for fallback in ["week", "year", "all"]:
            if fallback not in cleaned_time_filters:
                cleaned_time_filters.append(fallback)
            if len(cleaned_time_filters) == 2:
                break
    time_filters = cleaned_time_filters

    suggestions: List[Dict[str, Any]] = []

    try:
        # Fetch candidate subreddits via search
        subreddits_iter = manager.client.subreddits.search(query, limit=50)
        candidates: List[praw.models.Subreddit] = []
        for sub in subreddits_iter:
            try:
                _ = sub.subscribers  # may raise for invalid subs
                candidates.append(sub)
            except Exception:
                continue

        # Rank by subscriber count (descending)
        candidates.sort(key=lambda s: getattr(s, "subscribers", 0) or 0, reverse=True)
        selected = candidates[:limit_subreddits]

        for sub in selected:
            sub_info: Dict[str, Any] = {
                "name": sub.display_name,
                "title": getattr(sub, "title", ""),
                "subscribers": getattr(sub, "subscribers", None),
                "over18": getattr(sub, "over18", None),
                "public_description": getattr(sub, "public_description", "")[:300],
                "url": f"https://reddit.com{sub.url}",
                "sample_posts": {},
            }
            try:
                if post_sort == "top":
                    for tf in time_filters:  # two time frames
                        titles: List[str] = []
                        try:
                            for p in sub.top(time_filter=tf, limit=posts_limit):
                                titles.append(p.title)
                        except Exception as e:
                            titles.append(f"<error fetching posts: {e}>")
                        sub_info["sample_posts"][f"top_{tf}"] = titles
                elif post_sort == "hot":
                    titles = [p.title for p in sub.hot(limit=posts_limit)]
                    sub_info["sample_posts"]["hot"] = titles
                else:  # new
                    titles = [p.title for p in sub.new(limit=posts_limit)]
                    sub_info["sample_posts"]["new"] = titles
            except Exception as e:
                sub_info["sample_posts"]["error"] = str(e)
                logger.warning(f"Error fetching sample posts for r/{sub.display_name}: {e}")
            suggestions.append(sub_info)
 
    except Exception as e:
        logger.error(f"Failed subreddit suggestion search: {e}")
        raise RuntimeError(f"Failed subreddit suggestion search: {e}") from e
 
    # Build LLM-friendly compact context (always two time frames, 5 titles each if available)
    llm_lines: List[str] = []
    capped = 5  # enforce 5 per timeframe for LLM context
    for idx, sub in enumerate(suggestions, start=1):
        subs = sub.get("subscribers")
 
        if post_sort == "top":
            tf1, tf2 = time_filters[0], time_filters[1]
            tf1_titles = sub["sample_posts"].get(f"top_{tf1}", [])[:capped]
            tf2_titles = sub["sample_posts"].get(f"top_{tf2}", [])[:capped]
            llm_lines.append(
                f"{idx}. r/{sub['name']} (subs: {subs}) | top_{tf1}: "
                + " | ".join(tf1_titles)
                + " || top_{tf2}: "
                + " | ".join(tf2_titles)
            )
        else:
            key = post_sort
            titles = sub["sample_posts"].get(key, [])[:10]
            llm_lines.append(
                f"{idx}. r/{sub['name']} (subs: {subs}) | {key}: "
                + " | ".join(titles)
            )
 
    llm_context_string = "\n".join(llm_lines)
 
    return {
        "query": query,
        "post_sort": post_sort,
        "time_filters_used": time_filters if post_sort == "top" else None,
        "limit_subreddits": limit_subreddits,
        "posts_limit_requested": posts_limit,
        "posts_limit_used_per_timeframe": 5 if post_sort == "top" else min(10, posts_limit),
        "suggestions": suggestions,
        "llm_context": llm_context_string,
        "llm_context_list": llm_lines,
        "note": "llm_context includes subreddit, subscriber count, and 5 titles per each of two time frames (total 10). Use get_subreddit_details for flair and rules.",
    }
 
@mcp.tool()
def get_subreddit_details(subreddit_name: str) -> Dict[str, Any]:
    """Get detailed information about a single subreddit, including rules and flair.
 
    Args:
        subreddit_name: The name of the subreddit (e.g., 'workreform' or 'r/workreform').
 
    Returns:
        A dictionary containing detailed subreddit information, including:
        - name, title, subscribers, over18 status, public_description, url
        - flair_info: List of available flair templates (id, text, text_editable, type)
        - rules_info: List of subreddit rules (short_name, description)
        - video_post_allowed: Boolean indicating if video posts are explicitly allowed or disallowed.
 
    Raises:
        ValueError: If the subreddit name is invalid or not found.
        RuntimeError: For other errors during API interaction.
    """
    manager = RedditClientManager()
    if not manager.client:
        raise RuntimeError("Reddit client not initialized")
 
    if not subreddit_name or not isinstance(subreddit_name, str):
        raise ValueError("Subreddit name is required")
 
    clean_subreddit_name = subreddit_name[2:] if subreddit_name.startswith("r/") else subreddit_name
 
    try:
        logger.info(f"Fetching details for subreddit r/{clean_subreddit_name}")
        sub = manager.client.subreddit(clean_subreddit_name)
        # Access a property to ensure the subreddit exists and is accessible
        _ = sub.display_name
 
        sub_info: Dict[str, Any] = {
            "name": sub.display_name,
            "title": getattr(sub, "title", ""),
            "subscribers": getattr(sub, "subscribers", None),
            "over18": getattr(sub, "over18", None),
            "public_description": getattr(sub, "public_description", "")[:300],
            "url": f"https://reddit.com{sub.url}",
            "flair_info": [],
            "rules_info": [],
            "video_post_allowed": True,  # Assume allowed unless rule says otherwise
            "client_is_read_only": manager.is_read_only, # Report client read-only status
        }
 
        # Fetch flair templates (specifically for posts)
        flair_templates = []
        try:
            # Check if link flair is enabled for the subreddit
            if getattr(sub, 'link_flair_enabled', False):
                for template in sub.flair.link_templates:
                    flair_templates.append({
                        "id": template.id,
                        "text": template.text,
                        "text_editable": template.text_editable,
                        "type": template.type,
                    })
            sub_info["flair_info"] = flair_templates
        except Exception as e:
            logger.warning(f"Error fetching link flair templates for r/{clean_subreddit_name}: {e}")
 
        # Add a note if no flair templates are found, and flair is required
        if not sub_info["flair_info"]:
            flair_note_message = "No post flair templates found."
            if sub_info["client_is_read_only"]:
                flair_note_message += " This is likely due to the Reddit client being in read-only mode. " \
                                      "This subreddit requires flair from a template, so you MUST provide " \
                                      "REDDIT_USERNAME and REDDIT_PASSWORD environment variables for full access " \
                                      "to fetch the available flair templates."
            else:
                flair_note_message += " This subreddit might not have public post flair templates, " \
                                      "or flair is moderator-only. Since custom flair text is not allowed, " \
                                      "posting may not be possible without access to templates."
            sub_info["flair_note"] = flair_note_message
 
        # Fetch rules and check for video restrictions
        rules = []
        try:
            for rule in sub.rules:
                rules.append({
                    "short_name": rule.short_name,
                    "description": rule.description,
                })
                # Simple keyword matching for video restrictions
                if "video" in rule.short_name.lower() or "video" in rule.description.lower():
                    if "no video" in rule.short_name.lower() or "no video" in rule.description.lower() or \
                       "images only" in rule.short_name.lower() or "images only" in rule.description.lower():
                        sub_info["video_post_allowed"] = False
                        sub_info["video_post_allowed_reason"] = "Rule text indicates no video posts."
            sub_info["rules_info"] = rules
        except Exception as e:
            logger.warning(f"Error fetching rules for r/{clean_subreddit_name}: {e}")
 
        # Log all attributes of the subreddit object if video posting is disallowed
        if not sub_info["video_post_allowed"]:
            logger.info(f"Subreddit r/{clean_subreddit_name} does not allow video posts based on rules. "
                        f"Subreddit object attributes: {sub.__dict__}")

        logger.info(f"Successfully fetched details for r/{clean_subreddit_name}")
        return sub_info
 
    except Exception as e:
        logger.error(f"Failed to get subreddit details for r/{clean_subreddit_name}: {e}")
        raise ValueError(f"Subreddit '{clean_subreddit_name}' not found or inaccessible: {e}") from e
 
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
