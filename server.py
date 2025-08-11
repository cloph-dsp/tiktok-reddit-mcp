import functools
import logging
import time
from datetime import datetime
from os import getenv, makedirs, path, remove
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypeVar, cast
import yt_dlp
import requests
import praw  # type: ignore
import os
from mcp.server.fastmcp import FastMCP

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional transcription support via faster-whisper (enabled with USE_WHISPER_TRANSCRIPTION=true)
USE_WHISPER = getenv("USE_WHISPER_TRANSCRIPTION", "false").lower() in ("1", "true", "yes", "on")
try:
    if USE_WHISPER:
        from faster_whisper import WhisperModel  # type: ignore
        _whisper_models = {}
    else:
        WhisperModel = None  # type: ignore
except Exception as _imp_err:  # pragma: no cover
    logger.warning(f"Disabling whisper transcription (import error): {_imp_err}")
    USE_WHISPER = False
    WhisperModel = None  # type: ignore


class RedditClientManager:
    """Manages the Reddit client and its state."""

    _instance = None
    _client = None
    _is_read_only = True

    def __new__(cls) -> "RedditClientManager":
        if cls._instance is None:
            cls._instance = super(RedditClientManager, cls).__new__(cls)
            cls._instance._initialize_client()
        return cls._instance

    def _initialize_client(self) -> None:
        """Initialize the Reddit client with appropriate credentials."""
        client_id = getenv("REDDIT_CLIENT_ID")
        client_secret = getenv("REDDIT_CLIENT_SECRET")
        user_agent = getenv("REDDIT_USER_AGENT", "RedditMCPServer v1.0")
        username = getenv("REDDIT_USERNAME")
        password = getenv("REDDIT_PASSWORD")

        self._is_read_only = True

        try:
            # Try authenticated access first if credentials are provided
            if all([username, password, client_id, client_secret]):
                logger.info(
                    f"Attempting to initialize Reddit client with user authentication for u/{username}"
                )
                try:
                    self._client = praw.Reddit(
                        client_id=client_id,
                        client_secret=client_secret,
                        user_agent=user_agent,
                        username=username,
                        password=password,
                        check_for_updates=False,
                    )
                    # Test authentication
                    if self._client.user.me() is None:
                        raise ValueError(f"Failed to authenticate as u/{username}")

                    logger.info(f"Successfully authenticated as u/{username}")
                    self._is_read_only = False
                    return
                except Exception as auth_error:
                    logger.warning(f"Authentication failed: {auth_error}")
                    logger.info("Falling back to read-only access")

            # Fall back to read-only with client credentials
            if client_id and client_secret:
                logger.info("Initializing Reddit client with read-only access")
                self._client = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent,
                    check_for_updates=False,
                    read_only=True,
                )
                return

            # Last resort: read-only without credentials
            logger.info(
                "Initializing Reddit client in read-only mode without credentials"
            )
            self._client = praw.Reddit(
                user_agent=user_agent,
                check_for_updates=False,
                read_only=True,
            )
            # Test read-only access
            self._client.subreddit("popular").hot(limit=1)

        except Exception as e:
            logger.error(f"Error initializing Reddit client: {e}")
            self._client = None

    @property
    def client(self) -> Optional[praw.Reddit]:
        """Get the Reddit client instance."""
        return self._client

    @property
    def is_read_only(self) -> bool:
        """Check if the client is in read-only mode."""
        return self._is_read_only

    def check_user_auth(self) -> bool:
        """Check if user authentication is available for write operations."""
        if not self._client:
            logger.error("Reddit client not initialized")
            return False
        if self._is_read_only:
            logger.error("Reddit client is in read-only mode")
            return False
        return True


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

def _format_timestamp(timestamp: float) -> str:
    """Convert Unix timestamp to human readable format.

    Args:
        timestamp (float): Unix timestamp

    Returns:
        str: Formatted date string
    """
    try:
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(timestamp)

def _format_post(post: praw.models.Submission) -> str:
    """Format post information."""
    content_type = "Text Post" if post.is_self else "Link Post"
    content = post.selftext if post.is_self else post.url

    return f"""
        • Title: {post.title}
        • Type: {content_type}
        • Content: {content}
        • Author: u/{str(post.author)}
        • Subreddit: r/{str(post.subreddit)}
        • Stats:
        - Score: {post.score:,}
        - Upvote Ratio: {post.upvote_ratio * 100:.1f}%
        - Comments: {post.num_comments:,}
        • Metadata:
        - Posted: {_format_timestamp(post.created_utc)}
        • Links:
        - Full Post: https://reddit.com{post.permalink}
        - Short Link: https://redd.it/{post.id}
        """

def _extract_reddit_id(reddit_id: str) -> str:
    """Extract the base ID from a Reddit URL or ID.

    Args:
        reddit_id: Either a Reddit ID or a URL containing the ID

    Returns:
        The extracted Reddit ID
    """
    if not reddit_id:
        raise ValueError("Empty ID provided")

    if "/" in reddit_id:
        parts = [p for p in reddit_id.split("/") if p]
        reddit_id = parts[-1]
        logger.debug(f"Extracted ID {reddit_id} from URL")

    return reddit_id

@require_write_access
def create_post(
    subreddit: str,
    title: str,
    content: Optional[str] = None,
    is_self: bool = True,
    video_path: Optional[str] = None,
    thumbnail_path: Optional[str] = None,
    nsfw: bool = False,
    spoiler: bool = False,
) -> Dict[str, Any]:
    """Create a new post in a subreddit.

    Args:
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
    manager = RedditClientManager()
    if not manager.client:
        raise RuntimeError("Reddit client not initialized")

    # Input validation
    if not subreddit or not isinstance(subreddit, str):
        raise ValueError("Subreddit name is required")
    if not title or not isinstance(title, str):
        raise ValueError("Post title is required")
    if len(title) > 300:
        raise ValueError("Title must be 300 characters or less")
    if not video_path and (not content or not isinstance(content, str)):
        raise ValueError("Post content/URL is required for non-video posts")

    clean_subreddit = subreddit[2:] if subreddit.startswith("r/") else subreddit

    try:
        logger.info(f"Creating post in r/{clean_subreddit}")
        subreddit_obj = manager.client.subreddit(clean_subreddit)
        _ = subreddit_obj.display_name

        try:
            if video_path:
                submission = subreddit_obj.submit_video(
                    title=title[:300],
                    video_path=video_path,
                    thumbnail_path=thumbnail_path,
                    nsfw=nsfw,
                    spoiler=spoiler,
                    send_replies=True,
                )
            elif is_self:
                submission = subreddit_obj.submit(
                    title=title[:300],
                    selftext=content,
                    nsfw=nsfw,
                    spoiler=spoiler,
                    send_replies=True,
                )
            else:
                if content and not content.startswith(("http://", "https://")):
                    content = f"https://{content}"
                submission = subreddit_obj.submit(
                    title=title[:300],
                    url=content,
                    nsfw=nsfw,
                    spoiler=spoiler,
                    send_replies=True,
                )

            logger.info(f"Post created successfully: {submission.permalink}")

            return {
                "post": _format_post(submission),
                "metadata": {
                    "created_at": _format_timestamp(time.time()),
                    "subreddit": clean_subreddit,
                    "is_self_post": is_self,
                    "permalink": f"https://reddit.com{submission.permalink}",
                    "id": submission.id,
                },
            }

        except Exception as post_error:
            logger.error(f"Failed to create post in r/{clean_subreddit}: {post_error}")
            raise RuntimeError(f"Failed to create post: {post_error}") from post_error

    except Exception as e:
        logger.error(f"Error in create_post for r/{clean_subreddit}: {e}")
        raise RuntimeError(f"Failed to create post: {e}") from e


@require_write_access
def reply_to_post(
    post_id: str, content: str, subreddit: Optional[str] = None
) -> Dict[str, Any]:
    """Post a reply to an existing Reddit post.

    Args:
        post_id: The ID of the post to reply to (can be full URL, permalink, or just ID)
        content: The content of the reply (1-10000 characters)
        subreddit: The subreddit name if known (for validation, with or without 'r/' prefix)

    Returns:
        Dictionary containing information about the created reply and parent post

    Raises:
        ValueError: If input validation fails or post is not found
        RuntimeError: For other errors during reply creation
    """
    manager = RedditClientManager()
    if not manager.client:
        raise RuntimeError("Reddit client not initialized")

    if not post_id or not isinstance(post_id, str):
        raise ValueError("Post ID is required")
    if not content or not isinstance(content, str):
        raise ValueError("Reply content is required")
    if len(content) < 1 or len(content) > 10000:
        raise ValueError("Reply must be between 1 and 10000 characters")

    clean_subreddit = None
    if subreddit:
        if not isinstance(subreddit, str):
            raise ValueError("Subreddit name must be a string")
        clean_subreddit = subreddit[2:] if subreddit.startswith("r/") else subreddit

    try:
        clean_post_id = _extract_reddit_id(post_id)
        logger.info(f"Creating reply to post ID: {clean_post_id}")

        submission = manager.client.submission(id=clean_post_id)

        try:
            post_title = submission.title
            post_subreddit = submission.subreddit

            logger.info(
                f"Replying to post: "
                f"Title: {post_title}, "
                f"Subreddit: r/{post_subreddit.display_name}"
            )

        except Exception as e:
            logger.error(f"Failed to access post {clean_post_id}: {e}")
            raise ValueError(f"Post {clean_post_id} not found or inaccessible") from e

        if (
            clean_subreddit
            and post_subreddit.display_name.lower() != clean_subreddit.lower()
        ):
            raise ValueError(
                f"Post is in r/{post_subreddit.display_name}, not r/{clean_subreddit}"
            )

        try:
            reply = submission.reply(content)
            logger.info(f"Reply created successfully: {reply.permalink}")

            return {
                "reply_id": reply.id,
                "reply_permalink": f"https://reddit.com{reply.permalink}",
                "parent_post_id": submission.id,
                "parent_post_title": submission.title,
            }

        except Exception as reply_error:
            logger.error(f"Failed to create reply: {reply_error}")
            raise RuntimeError(f"Failed to create reply: {reply_error}") from reply_error

    except Exception as e:
        logger.error(f"Error replying to post {post_id}: {e}")
        raise RuntimeError(f"Failed to reply to post: {e}") from e

@mcp.tool()
def download_tiktok_video(url: str, download_folder: str = "downloaded", keep: bool = True) -> Dict[str, Any]:
    """Download a TikTok video using yt-dlp (thumbnail discarded immediately).

    Args:
        url: TikTok video URL (short or full)
        download_folder: Destination folder
        keep: If False, video will be deleted after returning metadata

    Returns:
        Dict with video_id, video_path (may be deleted later), thumbnail_deleted flag
    """
    if not path.exists(download_folder):
        makedirs(download_folder)

    original_url = url
    if any(host in url for host in ("vm.tiktok.com", "vt.tiktok.com")):
        try:
            head = requests.head(url, allow_redirects=True, timeout=10)
            head.raise_for_status()
            url = head.url
        except Exception as e:
            raise RuntimeError(f"Failed to resolve short TikTok link: {e}") from e

    output_template = path.join(download_folder, '%(id)s.%(ext)s')
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'format': 'best[ext=mp4]/best',
        'outtmpl': output_template,
        'writethumbnail': True,
    }

    video_path = None
    video_id = None
    thumbnail_deleted = False
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get('id')
            video_path = ydl.prepare_filename(info)
            base_filename, _ = path.splitext(video_path)
            # Locate and immediately delete thumbnail (not retained)
            for ext in ('.jpg', '.webp', '.png', '.image'):
                pth = base_filename + ext
                if path.exists(pth):
                    try:
                        remove(pth)
                        thumbnail_deleted = True
                    except Exception:
                        pass
                    break
    except Exception as e:
        raise RuntimeError(f"Failed to download TikTok video: {e}") from e

    result = {
        'original_url': original_url,
        'resolved_url': url,
        'video_id': video_id,
        'video_path': video_path,
        'thumbnail_deleted': thumbnail_deleted,
        'kept': keep,
    }

    if not keep and video_path and path.exists(video_path):
        try:
            remove(video_path)
            result['kept'] = False
        except Exception:
            pass
    return result

@mcp.tool()
def transcribe_video(video_path: str, model_size: str = "small") -> Dict[str, Any]:
    """Transcribe a downloaded video using faster-whisper if enabled.

    Args:
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
        model = _whisper_models[model_size]
        segments_iter, info = model.transcribe(video_path, beam_size=1)
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
        return {
            'status': 'success',
            'language': info.language,
            'duration': info.duration,
            'segments': segments,
            'transcript': transcript,
        }
    except Exception as e:
        raise RuntimeError(f"Failed to transcribe video: {e}") from e

@mcp.tool()
@require_write_access
def post_downloaded_video(
    video_path: Optional[str] = None,
    subreddit: str = "",
    title: str = "",
    thumbnail_path: Optional[str] = None,
    nsfw: bool = False,
    spoiler: bool = False,
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

    post_info = create_post(
        subreddit=subreddit,
        title=title,
        video_path=video_path,
        thumbnail_path=thumbnail_path if thumbnail_path and path.exists(thumbnail_path) else None,
        nsfw=nsfw,
        spoiler=spoiler,
    )
    result: Dict[str, Any] = { 'post': post_info, 'used_video_path': video_path }
    if comment:
        post_id = post_info['metadata']['id']
        reply = reply_to_post(post_id=post_id, content=comment)
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
    except Exception:
        pass
    result['video_deleted'] = deleted
    return result

@mcp.tool()
def suggest_subreddits(
    query: str,
    limit_subreddits: int = 5,
    posts_limit: int = 5,
    time_filters: Optional[List[str]] = None,
    post_sort: str = "top",  # 'top', 'hot', 'new'
) -> Dict[str, Any]:
    """Suggest relevant subreddits for a topic and show top post titles.

    Finds candidate subreddits via search, ranks by subscriber count, then
    for each selected subreddit retrieves sample post titles for TWO time ranges
    (default: week & year) when using 'top', or a single retrieval for other sorts.

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
            suggestions.append(sub_info)

    except Exception as e:
        raise RuntimeError(f"Failed subreddit suggestion search: {e}") from e

    # Build LLM-friendly compact context (always two time frames, 5 titles each if available)
    llm_lines: List[str] = []
    capped = 5  # enforce 5 per timeframe for LLM context
    if post_sort == "top":
        tf1, tf2 = time_filters[0], time_filters[1]
        for idx, sub in enumerate(suggestions, start=1):
            subs = sub.get("subscribers")
            tf1_titles = sub["sample_posts"].get(f"top_{tf1}", [])[:capped]
            tf2_titles = sub["sample_posts"].get(f"top_{tf2}", [])[:capped]
            llm_lines.append(
                f"{idx}. r/{sub['name']} (subs: {subs}) | top_{tf1}: "
                + " | ".join(tf1_titles)
                + " || top_{tf2}: "
                + " | ".join(tf2_titles)
            )
    else:
        # For hot/new only one list; still produce up to 10 by duplicating timeframe label
        key = post_sort
        for idx, sub in enumerate(suggestions, start=1):
            titles = sub["sample_posts"].get(key, [])[:10]
            llm_lines.append(f"{idx}. r/{sub['name']} (subs: {sub.get('subscribers')}) | {key}: " + " | ".join(titles))

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
        "note": "llm_context includes subreddit, subscriber count, and 5 titles per each of two time frames (total 10). Adjust time_filters to change frames.",
    }

# --- Server entrypoint ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run TikTok→Reddit MCP server")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8050)), help="Port to bind (default 8050)")
    parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"), help="Host/IP to bind (default 0.0.0.0)")
    args = parser.parse_args()

    logger.info(
        "Starting MCP server '%s' on %s:%d (read_only=%s)",
        mcp.name,
        args.host,
        args.port,
        RedditClientManager().is_read_only,
    )
    # Run FastMCP HTTP server (blocks). Some versions don't accept host; try port only.
    try:
        mcp.run(host=args.host, port=args.port)  # preferred (newer fastmcp)
    except TypeError:
        logger.warning("fastmcp.run() rejected host kwarg; retrying with port only")
        try:
            mcp.run(port=args.port)
        except TypeError:
            logger.warning("fastmcp.run() rejected port kwarg; falling back to no-arg run()")
            mcp.run()
