def sanitize_tiktok_url(url: str) -> str:
    """Remove query parameters from TikTok URLs."""
    from urllib.parse import urlparse, urlunparse
    parsed_url = urlparse(url)
    return urlunparse(parsed_url._replace(query=""))
import logging
from datetime import datetime
from typing import Any, Optional
import praw  # type: ignore

logger = logging.getLogger(__name__)


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


def _find_submission_by_title(
    subreddit_obj: praw.models.Subreddit, title: str, max_age_seconds: int = 300
) -> Optional[praw.models.Submission]:
    """Attempt to find a submission by title in a given subreddit within a time window.

    Args:
        subreddit_obj: The PRAW Subreddit object to search within.
        title: The title of the post to search for.
        max_age_seconds: Maximum age of the post in seconds to consider (default: 5 minutes).

    Returns:
        The praw.models.Submission object if found, otherwise None.
    """
    import time
    logger.info(f"Attempting to find submission by title '{title}' in r/{subreddit_obj.display_name}")
    search_time = time.time() - max_age_seconds
    
    # Search in new posts first (most likely place for recent submissions)
    for submission in subreddit_obj.new(limit=20): # Check up to 20 new posts
        if submission.created_utc > search_time and submission.title == title:
            logger.info(f"Found new submission by title: {submission.permalink}")
            return submission
    
    # Search in hot posts
    for submission in subreddit_obj.hot(limit=20): # Check up to 20 hot posts
        if submission.created_utc > search_time and submission.title == title:
            logger.info(f"Found hot submission by title: {submission.permalink}")
            return submission
            
    # Search in rising posts
    try:
        for submission in subreddit_obj.rising(limit=20): # Check up to 20 rising posts
            if submission.created_utc > search_time and submission.title == title:
                logger.info(f"Found rising submission by title: {submission.permalink}")
                return submission
    except Exception as e:
        logger.warning(f"Could not search rising posts: {e}")
    
    logger.info(f"No recent submission found with title '{title}' in r/{subreddit_obj.display_name}")
    return None