import functools
import logging
import time
from datetime import datetime
from os import getenv, makedirs, path, remove
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypeVar, cast
import yt_dlp
import requests
import praw  # type: ignore
import os  # added for list_downloaded_videos
from mcp.server.fastmcp import FastMCP

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
def download_tiktok_video_and_post_to_reddit(
    tiktok_url: str,
    subreddit: str,
    post_title: str,
    download_folder: str = "downloaded",
    nsfw: bool = False,
    spoiler: bool = False,
) -> Dict[str, Any]:
    """(Deprecated) One-shot: download TikTok and post to Reddit. Use modular tools instead.
    """
    dl = download_tiktok_video(tiktok_url, download_folder=download_folder, keep=True)
    try:
        post = post_downloaded_video(
            video_path=dl['video_path'],
            subreddit=subreddit,
            title=post_title,
            thumbnail_path=dl.get('thumbnail_path'),
            nsfw=nsfw,
            spoiler=spoiler,
            comment=f"Original link / link original: {dl['resolved_url']}",
        )
        return { 'download': dl, 'post': post }
    finally:
        # Do not delete here; user can manage lifecycle manually now.
        pass

@mcp.tool()
def download_tiktok_video(url: str, download_folder: str = "downloaded", keep: bool = True) -> Dict[str, Any]:
    """Download a TikTok video (and thumbnail) using yt-dlp.

    Args:
        url: TikTok video URL (short or full)
        download_folder: Destination folder
        keep: If False, video will be deleted after returning path metadata

    Returns:
        Dict with video_id, video_path, thumbnail_path (if exists)
    """
    if not path.exists(download_folder):
        makedirs(download_folder)

    # Resolve short link
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
    thumbnail_path = None
    video_id = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get('id')
            video_path = ydl.prepare_filename(info)
            base_filename, _ = path.splitext(video_path)
            for ext in ('.jpg', '.webp', '.png', '.image'):
                pth = base_filename + ext
                if path.exists(pth):
                    thumbnail_path = pth
                    break
    except Exception as e:
        raise RuntimeError(f"Failed to download TikTok video: {e}") from e

    result = {
        'original_url': original_url,
        'resolved_url': url,
        'video_id': video_id,
        'video_path': video_path,
        'thumbnail_path': thumbnail_path,
        'kept': keep,
    }

    if not keep and video_path and path.exists(video_path):
        try:
            remove(video_path)
            if thumbnail_path and path.exists(thumbnail_path):
                remove(thumbnail_path)
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
    video_path: str,
    subreddit: str,
    title: str,
    thumbnail_path: Optional[str] = None,
    nsfw: bool = False,
    spoiler: bool = False,
    comment: Optional[str] = None,
) -> Dict[str, Any]:
    """Post a previously downloaded video to Reddit and optional comment.

    Args:
        video_path: Local video file path
        subreddit: Target subreddit
        title: Post title
        thumbnail_path: Optional thumbnail path
        nsfw: Mark NSFW
        spoiler: Mark spoiler
        comment: Optional comment body

    Returns:
        Dict with post metadata (and comment if added)
    """
    if not path.exists(video_path):
        raise ValueError("Video path does not exist")

    post_info = create_post(
        subreddit=subreddit,
        title=title,
        video_path=video_path,
        thumbnail_path=thumbnail_path if thumbnail_path and path.exists(thumbnail_path) else None,
        nsfw=nsfw,
        spoiler=spoiler,
    )
    result = { 'post': post_info }
    if comment:
        post_id = post_info['metadata']['id']
        reply = reply_to_post(post_id=post_id, content=comment)
        result['comment'] = reply
    return result

@mcp.tool()
def list_downloaded_videos(folder: str = "downloaded") -> Dict[str, Any]:
    """List downloaded video files in a folder.

    Returns list of videos with id (basename without extension), file path, thumbnail path and size.
    """
    if not path.exists(folder):
        return { 'videos': [] }
    videos: List[Dict[str, Any]] = []
    try:
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(('.mp4', '.mov', '.mkv', '.webm')):
                full = path.join(folder, fname)
                base, _ = path.splitext(full)
                thumb = None
                for ext in ('.jpg', '.webp', '.png', '.image'):
                    candidate = base + ext
                    if path.exists(candidate):
                        thumb = candidate
                        break
                videos.append({
                    'id': path.basename(base),
                    'file': full,
                    'thumbnail': thumb,
                    'size_bytes': path.getsize(full) if path.exists(full) else None,
                })
    except Exception as e:
        return { 'videos': [], 'error': f'Failed to list videos: {e}' }
    return { 'videos': videos }
