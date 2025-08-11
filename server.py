import functools
import logging
import time
from datetime import datetime
from os import getenv, makedirs, path, remove
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypeVar, cast
import requests
import praw  # type: ignore
from mcp.server.fastmcp import FastMCP

F = TypeVar("F", bound=Callable[..., Any])
if TYPE_CHECKING:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

TIKTOK_API_URL = "https://tiktok-download-video1.p.rapidapi.com/getVideo"

@mcp.tool()
def download_tiktok_video_and_post_to_reddit(
    tiktok_url: str,
    subreddit: str,
    post_title: str,
    download_folder: str = "downloaded",
    nsfw: bool = False,
    spoiler: bool = False,
) -> Dict[str, Any]:
    """
    Downloads a TikTok video, posts it to Reddit, and adds a comment with the original URL.

    Args:
        tiktok_url: The URL of the TikTok video (short or long).
        subreddit: The subreddit to post the video to.
        post_title: The title for the Reddit post.
        download_folder: The folder to download the video to. Defaults to "downloaded".
        nsfw: Whether the submission should be marked NSFW (default: False).
        spoiler: Whether the submission should be marked as a spoiler (default: False).

    Returns:
        A dictionary with the Reddit post and comment information.
    """
    original_tiktok_url = tiktok_url
    logger.info(f"Starting TikTok download for URL: {original_tiktok_url}")

    # Step 1: Resolve short link if necessary
    if "vm.tiktok.com" in original_tiktok_url or "vt.tiktok.com" in original_tiktok_url:
        logger.info("Short link detected, resolving to full URL.")
        try:
            response = requests.head(original_tiktok_url, allow_redirects=True, timeout=10)
            response.raise_for_status()
            tiktok_url = response.url
            logger.info(f"Resolved to full URL: {tiktok_url}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to resolve short link: {e}")
            raise RuntimeError(f"Failed to resolve short link: {e}") from e

    # Create download folder if it doesn't exist
    if not path.exists(download_folder):
        makedirs(download_folder)

    # Step 2: Download TikTok video
    headers = {
        "x-rapidapi-key": getenv("RAPIDAPI_KEY"),
        "x-rapidapi-host": "tiktok-download-video1.p.rapidapi.com",
    }
    querystring = {"url": tiktok_url, "hd": "1"}
    video_path = None
    thumbnail_path = None

    try:
        response = requests.get(TIKTOK_API_URL, headers=headers, params=querystring)
        response.raise_for_status()
        video_data = response.json()

        if "data" not in video_data or "play" not in video_data["data"]:
            raise ValueError("Could not retrieve video download link from API.")

        video_url = video_data["data"]["play"]
        video_id = video_data["data"]["id"]
        video_path = path.join(download_folder, f"{video_id}.mp4")

        if "cover" in video_data["data"]:
            thumbnail_url = video_data["data"]["cover"]
            thumbnail_path = path.join(download_folder, f"{video_id}.jpg")
            with requests.get(thumbnail_url, stream=True) as r:
                r.raise_for_status()
                with open(thumbnail_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            logger.info(f"Thumbnail downloaded to: {thumbnail_path}")

        logger.info(f"Got video download URL: {video_url}")
        logger.info(f"Downloading video to: {video_path}")

        with requests.get(video_url, stream=True) as r:
            r.raise_for_status()
            with open(video_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info("Video downloaded successfully.")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling TikTok API: {e}")
        raise RuntimeError(f"Failed to download TikTok video: {e}") from e

    # Step 3: Post to Reddit
    try:
        logger.info(f"Posting video to r/{subreddit}")
        post_info = create_post(
            subreddit=subreddit,
            title=post_title,
            video_path=video_path,
            thumbnail_path=thumbnail_path,
            nsfw=nsfw,
            spoiler=spoiler,
        )

        post_id = post_info["metadata"]["id"]
        logger.info(f"Reddit post created with ID: {post_id}")

        # Step 4: Add a comment with the original link
        comment_content = f"Original TikTok video: {original_tiktok_url}"
        logger.info(f"Adding comment to post {post_id}")
        comment_info = reply_to_post(post_id=post_id, content=comment_content)

        return {
            "reddit_post": post_info,
            "reddit_comment": comment_info,
        }
    finally:
        # Step 5: Delete the video and thumbnail files
        if video_path and path.exists(video_path):
            remove(video_path)
            logger.info(f"Deleted video file: {video_path}")
        if thumbnail_path and path.exists(thumbnail_path):
            remove(thumbnail_path)
            logger.info(f"Deleted thumbnail file: {thumbnail_path}")
