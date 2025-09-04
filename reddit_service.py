import asyncio
import logging
import time
import sys
from datetime import datetime
from os import getenv, path, remove
from typing import Any, Dict, List, Optional
import praw  # type: ignore
import requests
from utils import (
    _format_timestamp,
    _format_post,
    _extract_reddit_id,
    _find_submission_by_title,
)
from reddit_client import RedditClientManager

# Import custom exceptions
from exceptions import RedditPostError

logger = logging.getLogger(__name__)


class RedditService:
    """Encapsulates Reddit API interactions."""

    def __init__(self):
        self.manager = RedditClientManager()

    async def _report_progress(self, ctx: Any, data: Dict[str, Any]) -> None:
        """Report progress via ctx.report_progress if available."""
        logger.info(f"Reporting progress: {data}")
        if hasattr(ctx, 'report_progress'):
            logger.info("Using ctx.report_progress")
            ctx.report_progress(data)
        elif hasattr(ctx, '__event_emitter__'):
            logger.info("Using ctx.__event_emitter__")
            await ctx.__event_emitter__({
                "type": "status",
                "data": {"description": data.get("message", "Unknown status"), "done": False}
            })
        else:
            logger.info("Using fallback logging")
            logger.info(f"Progress: {data}")
            
    async def _poll_for_submission(
        self,
        ctx: Any,
        subreddit_obj: praw.models.Subreddit,
        title: str,
        timeout_seconds: int = 60,  # Reduced from 180 to 60 seconds
        poll_interval: int = 2      # Reduced from 5 to 2 seconds
    ) -> Optional[praw.models.Submission]:
        """Poll for a submission by title with timeout."""
        logger.info(f"Polling for submission with title: {title}")
        await self._report_progress(ctx, {"status": "polling", "message": f"Searching for post by title: {title}"})

        start_time = time.time()
        attempt = 0

        # Clean the title for better matching (remove extra whitespace, normalize)
        clean_title = title.strip()

        while time.time() - start_time < timeout_seconds:
            attempt += 1
            logger.info(f"Polling attempt {attempt} for submission")

            try:
                # Try to find by title using our utility function
                logger.info("Calling _find_submission_by_title...")
                submission = _find_submission_by_title(subreddit_obj, clean_title)
                logger.info("Finished calling _find_submission_by_title.")
                if submission:
                    logger.info(f"Found submission by title: {submission.permalink}")
                    await self._report_progress(ctx, {"status": "polling", "message": f"Found post: {submission.permalink}"})
                    return submission

                # Also try searching the subreddit's new posts directly
                logger.info("Also checking subreddit's new posts directly...")
                try:
                    for post in subreddit_obj.new(limit=10):  # Check last 10 posts
                        if post.title.strip() == clean_title:
                            logger.info(f"Found submission in new posts: {post.permalink}")
                            await self._report_progress(ctx, {"status": "polling", "message": f"Found post in new posts: {post.permalink}"})
                            return post
                except Exception as search_error:
                    logger.warning(f"Error searching new posts: {search_error}")

            except Exception as poll_error:
                logger.warning(f"Error during polling attempt {attempt}: {poll_error}")

            # If not found, wait before next attempt
            if time.time() - start_time < timeout_seconds:
                logger.info(f"Submission not found, waiting {poll_interval} seconds before retry")
                await self._report_progress(ctx, {"status": "polling", "message": f"Post not found yet, retrying in {poll_interval}s (attempt {attempt})"})
                logger.info(f"Sleeping for {poll_interval} seconds...")
                await asyncio.sleep(poll_interval)
                logger.info(f"Finished sleeping.")

        logger.warning(f"Timeout reached while polling for submission with title: {title}")
        await self._report_progress(ctx, {"status": "polling", "message": f"Timeout searching for post by title after {timeout_seconds}s"})
        return None
        
    async def _validate_video_submission(
        self,
        ctx: Any,
        submission: praw.models.Submission,
        timeout_seconds: int = 60
    ) -> bool:
        """Validate that a video submission has valid video content."""
        logger.info(f"Validating video submission: {submission.permalink}")

        try:
            # Refresh the submission to get latest data
            logger.info("Refreshing submission...")
            submission.refresh()
            logger.info("Finished refreshing submission.")

            # Check if it's marked as a video post
            if not (hasattr(submission, 'is_video') and submission.is_video):
                logger.error(f"Submission is not marked as video: {submission.permalink}")
                return False

            # Check for media attributes
            has_media = (hasattr(submission, 'media') and submission.media is not None)
            has_secure_media = (hasattr(submission, 'secure_media') and submission.secure_media is not None)

            if not (has_media or has_secure_media):
                logger.error(f"Submission has no media attributes: {submission.permalink}")
                return False

            # Check for v.redd.it URL
            if hasattr(submission, 'url') and 'v.redd.it' in submission.url:
                logger.info(f"Submission has v.redd.it URL: {submission.url}")

                # Try to access the video URL to verify it's working
                try:
                    import requests
                    logger.info(f"Accessing video URL: {submission.url}")
                    response = requests.head(submission.url, timeout=10, allow_redirects=True)
                    logger.info(f"Video URL access returned status {response.status_code}: {submission.url}")
                    if response.status_code == 200:
                        logger.info(f"Video URL is accessible: {submission.url}")
                        return True
                    else:
                        logger.error(f"Video URL returned status {response.status_code}: {submission.url}")
                        return False
                except requests.RequestException as e:
                    logger.error(f"Could not access video URL {submission.url}: {e}")
                    return False
            else:
                logger.warning(f"Submission does not have expected v.redd.it URL: {getattr(submission, 'url', 'No URL')}")
                return False

        except Exception as e:
            logger.error(f"Error validating video submission: {e}")
            return False

    async def _poll_for_media_readiness(
        self,
        ctx: Any,
        submission: praw.models.Submission,
        timeout_seconds: int = 45,  # Reduced from 120 to 45 seconds
        poll_interval: int = 2      # Reduced from 3 to 2 seconds
    ) -> bool:
        """Poll for media readiness with timeout."""
        logger.info(f"Polling for media readiness of submission: {submission.permalink}")
        await self._report_progress(ctx, {"status": "media_processing", "message": f"Waiting for video to process: {submission.permalink}"})

        start_time = time.time()
        attempt = 0

        while time.time() - start_time < timeout_seconds:
            attempt += 1
            logger.info(f"Media readiness check attempt {attempt}")

            try:
                # First validate the video submission
                logger.info("Calling _validate_video_submission...")
                validation_result = await self._validate_video_submission(ctx, submission)
                logger.info(f"_validate_video_submission returned: {validation_result}")
                if validation_result:
                    logger.info(f"Video submission validated successfully: {submission.permalink}")
                    await self._report_progress(ctx, {"status": "media_ready", "message": f"Video processing complete: {submission.permalink}"})
                    return True

            except Exception as e:
                logger.warning(f"Error during media validation: {e}")

            # If not ready, wait before next attempt
            if time.time() - start_time < timeout_seconds:
                logger.info(f"Media not ready, waiting {poll_interval} seconds before retry")
                await self._report_progress(ctx, {"status": "media_processing", "message": f"Video still processing, retrying in {poll_interval}s (attempt {attempt})"})
                logger.info(f"Sleeping for {poll_interval} seconds...")
                await asyncio.sleep(poll_interval)
                logger.info(f"Finished sleeping.")

        logger.warning(f"Timeout reached while waiting for media readiness: {submission.permalink}")
        await self._report_progress(ctx, {"status": "media_timeout", "message": f"Video processing timeout after {timeout_seconds}s: {submission.permalink}"})
        return False

    async def create_post(
        self,
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
        if not self.manager.client:
            raise RuntimeError("Reddit client not initialized")

        if self.manager.is_read_only:
            raise RuntimeError(
                "Reddit client is in read-only mode. Video submission requires write access. "
                "Please ensure REDDIT_USERNAME, REDDIT_PASSWORD, REDDIT_CLIENT_ID, and REDDIT_CLIENT_SECRET "
                "environment variables are correctly set for authenticated access."
            )

        # Additional authentication check
        if not self.manager.check_user_auth():
            raise RuntimeError(
                "Reddit client authentication failed. Video submission requires authenticated access. "
                "Please check your REDDIT_USERNAME, REDDIT_PASSWORD, REDDIT_CLIENT_ID, and REDDIT_CLIENT_SECRET "
                "environment variables."
            )

        # Input validation
        if not subreddit or not isinstance(subreddit, str):
            raise ValueError("Subreddit name is required")
        if not title or not isinstance(title, str):
            raise ValueError("Post title is required")
        if len(title) > 300:
            raise ValueError("Title must be 300 characters or less")
        if not video_path and (not content or not isinstance(content, str)):
            raise ValueError("Post content/URL is required for non-video posts")

        # CRITICAL: If video_path is provided, is_self should be False
        if video_path and is_self:
            logger.warning(f"Video path provided but is_self=True. Setting is_self=False for video post.")
            is_self = False

        clean_subreddit = subreddit[2:] if subreddit.startswith("r/") else subreddit
        
        logger.info(f"create_post: subreddit={subreddit}, title={title}, video_path={video_path}, thumbnail_path={thumbnail_path}")
        logger.info(f"create_post: is_self={is_self}, content={content}, nsfw={nsfw}, spoiler={spoiler}, flair_id={flair_id}, flair_text={flair_text}")

        try:
            logger.info(f"Creating post in r/{clean_subreddit}")
            subreddit_obj = self.manager.client.subreddit(clean_subreddit)
            _ = subreddit_obj.display_name

            try:
                if video_path:
                    # Log video file details before submission
                    if video_path and path.exists(video_path):
                        try:
                            file_size = path.getsize(video_path)
                            logger.info(f"Attempting to submit video: {video_path}")
                            logger.info(f"Video file size: {file_size} bytes")
                            await self._report_progress(ctx, {"status": "uploading", "message": f"Starting video upload: {file_size} bytes"})
                        except Exception as log_error:
                            logger.warning(f"Error getting video file size: {log_error}")
                            await self._report_progress(ctx, {"status": "uploading", "message": f"Starting video upload (file size unknown): {log_error}"})

                    submission = None
                    max_retries = 1  # Single attempt to avoid multiple posts
                    for attempt in range(1, max_retries + 1):
                        try:
                            logger.info(f"Attempt {attempt} to submit video.")
                            
                            # Default to without_websockets for better reliability
                            use_websockets = False if attempt < max_retries else True  # Try websockets only on last attempt
                            logger.info(f"Using without_websockets: {not use_websockets}")
                            
                            logger.info("Calling subreddit_obj.submit_video...")
                            logger.info(f"Video submission parameters: title='{title[:300]}', video_path='{video_path}', thumbnail_path='{thumbnail_path}', without_websockets={not use_websockets}")

                            # Log PRAW client state
                            logger.info(f"PRAW client read-only state: {self.manager.client.read_only}")
                            logger.info(f"PRAW client authenticated: {self.manager.client.user is not None}")

                            # Validate video file before submission
                            if not path.exists(video_path):
                                raise RedditPostError(
                                    f"Video file does not exist: {video_path}",
                                    details={"error_type": "FILE_NOT_FOUND", "video_path": video_path}
                                )

                            # Check video file size (Reddit limit is 1GB)
                            file_size = path.getsize(video_path)
                            max_size = 1024 * 1024 * 1024  # 1GB
                            if file_size > max_size:
                                raise RedditPostError(
                                    f"Video file is too large: {file_size} bytes (max: {max_size} bytes)",
                                    details={"error_type": "FILE_TOO_LARGE", "file_size": file_size, "max_size": max_size}
                                )

                            # Check video file extension
                            valid_extensions = {'.mp4', '.mov', '.avi', '.wmv', '.flv', '.webm', '.m4v', '.3gp'}
                            file_ext = path.splitext(video_path)[1].lower()
                            if file_ext not in valid_extensions:
                                raise RedditPostError(
                                    f"Unsupported video format: {file_ext}. Supported formats: {', '.join(valid_extensions)}",
                                    details={"error_type": "UNSUPPORTED_FORMAT", "file_extension": file_ext, "supported_formats": list(valid_extensions)}
                                )

                            logger.info(f"Video file validation passed: {video_path} ({file_size} bytes, {file_ext})")

                            # Attempt video submission with improved error handling
                            try:
                                logger.info(f"About to call PRAW submit_video with:")
                                logger.info(f"  Title: {title[:300]}")
                                logger.info(f"  Video path: {video_path}")
                                logger.info(f"  Thumbnail path: {thumbnail_path}")
                                logger.info(f"  Without websockets: {not use_websockets}")
                                logger.info(f"  PRAW version: {praw.__version__ if hasattr(praw, '__version__') else 'unknown'}")
                                logger.info(f"  Client authenticated: {self.manager.client.user is not None}")

                                # Check if video file is readable and has valid content
                                try:
                                    with open(video_path, 'rb') as f:
                                        # Try to read first few bytes to ensure file is accessible
                                        header = f.read(1024)
                                        if len(header) < 100:
                                            raise RedditPostError(
                                                f"Video file is too small or empty: {len(header)} bytes",
                                                details={"error_type": "FILE_TOO_SMALL", "video_path": video_path, "bytes_read": len(header)}
                                            )
   
                                        # Check for basic video file signatures
                                        if not (header.startswith(b'\x00\x00\x00') or  # MP4
                                               header.startswith(b'RIFF') or         # AVI
                                               header.startswith(b'\x66\x74\x79\x70') or  # MP4 variant
                                               header.startswith(b'moov') or         # MOV
                                               header.startswith(b'mdat')):          # MP4 data
                                            logger.warning(f"Video file may not have valid signature: {header[:4].hex()}")
   
                                    logger.info("Video file is readable and has valid header")
                                except Exception as file_error:
                                    logger.error(f"Video file is not readable: {file_error}")
                                    raise RedditPostError(
                                        f"Video file cannot be read: {file_error}",
                                        details={"error_type": "FILE_NOT_READABLE", "video_path": video_path}
                                    )

                                # Log all parameters before submission
                                logger.info("About to call PRAW submit_video with parameters:")
                                logger.info(f"  subreddit: {clean_subreddit}")
                                logger.info(f"  title: {title[:300]}")
                                logger.info(f"  video_path: {video_path}")
                                logger.info(f"  video_exists: {path.exists(video_path)}")
                                logger.info(f"  video_size: {path.getsize(video_path) if path.exists(video_path) else 'N/A'}")
                                logger.info(f"  thumbnail_path: {thumbnail_path}")
                                logger.info(f"  without_websockets: {not use_websockets}")
                                logger.info(f"  nsfw: {nsfw}")
                                logger.info(f"  spoiler: {spoiler}")

                                # Double-check video file before submission
                                if not path.exists(video_path):
                                    raise RedditPostError(
                                        f"Video file disappeared before submission: {video_path}",
                                        details={"error_type": "FILE_DISAPPEARED", "video_path": video_path}
                                    )

                                file_size = path.getsize(video_path)
                                if file_size == 0:
                                    raise RedditPostError(
                                        f"Video file is empty before submission: {video_path}",
                                        details={"error_type": "FILE_EMPTY", "video_path": video_path}
                                    )

                                logger.info(f"Final video file check passed: {file_size} bytes")

                                submission = subreddit_obj.submit_video(
                                    title=title[:300],
                                    video_path=video_path,
                                    thumbnail_path=thumbnail_path,
                                    nsfw=nsfw,
                                    spoiler=spoiler,
                                    send_replies=True,
                                    flair_id=flair_id,
                                    flair_text=flair_text,
                                    without_websockets=not use_websockets,
                                )
                                logger.info("Video submission call completed.")
                                logger.info(f"Submission result type: {type(submission)}")
                                logger.info(f"Submission result: {submission}")

                                if submission is not None:
                                    logger.info(f"Submission ID: {getattr(submission, 'id', 'N/A')}")
                                    logger.info(f"Submission permalink: {getattr(submission, 'permalink', 'N/A')}")
                                    logger.info(f"Submission URL: {getattr(submission, 'url', 'N/A')}")

                                if submission is None:
                                    logger.error("PRAW submit_video returned None - this indicates a failure")
                                    raise RedditPostError(
                                        "Video submission failed: PRAW returned None. This usually indicates an issue with the video file or Reddit's servers.",
                                        details={"error_type": "SUBMISSION_RETURNED_NONE", "video_path": video_path}
                                    )

                                logger.info(f"Submission ID: {submission.id}, Permalink: {submission.permalink}")

                            except praw.exceptions.WebSocketException as ws_error:
                                logger.error(f"WebSocket error during video submission: {ws_error}")
                                # Check if this is a recoverable WebSocket error
                                error_msg = str(ws_error).lower()
                                recoverable_errors = [
                                    "websocket error",
                                    "connection closed",
                                    "timeout",
                                    "network error",
                                    "temporary failure"
                                ]

                                is_recoverable = any(recoverable in error_msg for recoverable in recoverable_errors)

                                if is_recoverable and attempt < max_retries:
                                    logger.warning(f"Recoverable WebSocket error, will retry: {ws_error}")
                                    await self._report_progress(ctx, {"status": "retry", "message": f"WebSocket error, retrying in 5s (attempt {attempt+1}/{max_retries})"})
                                    await asyncio.sleep(5)  # Wait before retry
                                    continue  # Continue to next attempt
                                else:
                                    # Non-recoverable or max retries reached
                                    logger.warning(f"Non-recoverable WebSocket error or max retries reached: {ws_error}")

                                    # IMPORTANT: Even with WebSocket errors, the post might have been created!
                                    # Try to find it before giving up
                                    logger.info("WebSocket failed but post might have been created - searching...")
                                    found_submission = await self._poll_for_submission(ctx, subreddit_obj, title, timeout_seconds=10)
                                    if found_submission:
                                        logger.info(f"Found post despite WebSocket error: {found_submission.permalink}")
                                        # Don't raise error, return the found submission
                                        submission = found_submission
                                        break  # Break out of retry loop with success
                                    else:
                                        logger.warning("Could not find post after WebSocket error")
                                        raise RedditPostError(
                                            f"Video upload failed due to WebSocket error: {ws_error}. "
                                            "This usually indicates an issue with the video file format, corruption, or Reddit's media servers. "
                                            "Try re-downloading the video or check if the video format is supported.",
                                            details={"error_type": "WEBSOCKET_ERROR", "original_error": str(ws_error), "video_path": video_path, "recoverable": is_recoverable}
                                        ) from ws_error
                            except Exception as submit_error:
                                logger.error(f"Video submission failed with error: {submit_error}")
                                # Re-raise with more context
                                raise RedditPostError(
                                    f"Video submission failed: {submit_error}",
                                    details={"error_type": "SUBMISSION_FAILED", "original_error": str(submit_error), "video_path": video_path}
                                ) from submit_error
                            
                            # If we get here without exception, break out of retry loop
                            break
                        except praw.exceptions.WebSocketException as ws_exc:
                            logger.warning(f"WebSocket error during video submission (attempt {attempt}): {ws_exc}")
                            error_msg = str(ws_exc).lower()
                            recoverable_errors = [
                                "websocket error",
                                "connection closed",
                                "timeout",
                                "network error",
                                "temporary failure"
                            ]

                            is_recoverable = any(recoverable in error_msg for recoverable in recoverable_errors)

                            if is_recoverable and attempt < max_retries:
                                logger.info("Retrying video submission after recoverable WebSocket error...")
                                await self._report_progress(ctx, {"status": "retry", "message": f"WebSocket error, retrying (attempt {attempt+1}/{max_retries})"})
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            else:
                                logger.error("Max retries reached for WebSocket error or error is not recoverable.")
                                await self._report_progress(ctx, {"status": "error", "message": f"WebSocket error after {max_retries} attempts: {ws_exc}"})

                                # IMPORTANT: Even with WebSocket errors, the post might have been created!
                                # Try to find it before giving up
                                logger.info("WebSocket failed but post might have been created - searching...")
                                found_submission = await self._poll_for_submission(ctx, subreddit_obj, title, timeout_seconds=15)
                                if found_submission:
                                    logger.info(f"Found post despite WebSocket error: {found_submission.permalink}")
                                    submission = found_submission
                                    break  # Success! Use the found submission
                                else:
                                    logger.warning("Could not find post after WebSocket error")
                                    submission = None
                                    break
                        except praw.exceptions.APIException as api_exc:
                            logger.error(f"Reddit API error during video submission (attempt {attempt}): {api_exc}", exc_info=True)
                            await self._report_progress(ctx, {"status": "error", "message": f"Reddit API error during video submission: {api_exc}"})
                            # Specific handling for common API errors
                            error_type = api_exc.error_type if hasattr(api_exc, 'error_type') else str(api_exc)
                            if "RATELIMIT" in error_type:
                                raise RedditPostError(
                                    f"Rate limit exceeded: {api_exc}. Please wait before trying again.",
                                    details={
                                        "error_type": "RATELIMIT",
                                        "attempt": attempt,
                                        "max_retries": max_retries
                                    }
                                ) from api_exc
                            elif "VIDEO_TOO_LONG" in error_type:
                                raise RedditPostError(
                                    f"Video is too long: {api_exc}. Please check the video length.",
                                    details={
                                        "error_type": "VIDEO_TOO_LONG",
                                        "attempt": attempt,
                                        "max_retries": max_retries
                                    }
                                ) from api_exc
                            elif "VIDEO_TOO_LARGE" in error_type:
                                raise RedditPostError(
                                    f"Video is too large: {api_exc}. Please check the video size.",
                                    details={
                                        "error_type": "VIDEO_TOO_LARGE",
                                        "attempt": attempt,
                                        "max_retries": max_retries
                                    }
                                ) from api_exc
                            elif "INVALID_VIDEO_FORMAT" in error_type:
                                raise RedditPostError(
                                    f"Invalid video format: {api_exc}. Please check the video format.",
                                    details={
                                        "error_type": "INVALID_VIDEO_FORMAT",
                                        "attempt": attempt,
                                        "max_retries": max_retries
                                    }
                                ) from api_exc
                            else:
                                if attempt < max_retries:
                                    logger.info("Retrying video submission after API error...")
                                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                else:
                                    raise RedditPostError(
                                        f"Failed to submit video due to Reddit API error: {api_exc}. "
                                        "This might indicate issues with video format, size, or subreddit settings.",
                                        details={
                                            "error_type": error_type,
                                            "attempt": attempt,
                                            "max_retries": max_retries,
                                            "error_message": str(api_exc)
                                        }
                                    ) from api_exc
                        except requests.exceptions.RequestException as req_exc:
                            logger.error(f"Network error during video submission (attempt {attempt}): {req_exc}", exc_info=True)
                            await self._report_progress(ctx, {"status": "error", "message": f"Network error during video submission: {req_exc}"})
                            if attempt < max_retries:
                                logger.info("Retrying video submission after network error...")
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            else:
                                raise RedditPostError(
                                    f"Failed to submit video due to network error after {max_retries} attempts: {req_exc}. "
                                    "Please check your internet connection or try again later.",
                                    details={
                                        "error_type": "NETWORK_ERROR",
                                        "attempt": attempt,
                                        "max_retries": max_retries,
                                        "error_message": str(req_exc)
                                    }
                                ) from req_exc
                        except Exception as e:
                            logger.error(f"An unexpected error occurred during video submission (attempt {attempt}): {e}", exc_info=True)
                            await self._report_progress(ctx, {"status": "error", "message": f"Unexpected error during video submission: {e}"})
                            if attempt < max_retries:
                                logger.info("Retrying video submission after unexpected error...")
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            else:
                                raise RedditPostError(
                                    f"An unexpected error occurred during video submission after {max_retries} attempts: {e}. "
                                    "Please check the video file and try again.",
                                    details={
                                        "error_type": "UNEXPECTED_ERROR",
                                        "attempt": attempt,
                                        "max_retries": max_retries,
                                        "error_message": str(e)
                                    }
                                ) from e

                    # Handle case where submission is None or we had websocket errors
                    if submission is None:
                        logger.warning(
                            "PRAW's submit_video returned None or failed. This indicates the video upload did not complete successfully."
                        )
                        await self._report_progress(ctx, {"status": "error", "message": "Video upload failed - PRAW returned None"})

                        # IMPORTANT: Even when submission is None, the post might have been created!
                        # Reddit sometimes creates the post but the WebSocket connection fails
                        # Let's try to find the post by searching for it
                        logger.info("Attempting to find post that might have been created despite WebSocket error...")
                        await self._report_progress(ctx, {"status": "searching", "message": "Searching for post that may have been created despite WebSocket error..."})

                        # Try multiple search strategies
                        found_submission = None

                        # Strategy 1: Poll for submission by title
                        logger.info("Strategy 1: Polling for submission by title...")
                        found_submission = await self._poll_for_submission(ctx, subreddit_obj, title, timeout_seconds=20)

                        if not found_submission:
                            # Strategy 2: Try with a slightly modified title (Reddit might have truncated it)
                            logger.info("Strategy 2: Trying with truncated title...")
                            truncated_title = title[:50] + "..." if len(title) > 50 else title
                            found_submission = await self._poll_for_submission(ctx, subreddit_obj, truncated_title, timeout_seconds=10)

                        if not found_submission:
                            # Strategy 3: Check user's recent posts
                            logger.info("Strategy 3: Checking user's recent posts...")
                            try:
                                user = self.manager.client.user.me()
                                if user:
                                    for post in user.submissions.new(limit=5):
                                        if post.subreddit.display_name.lower() == clean_subreddit.lower():
                                            # Check if the post was created recently (within last 5 minutes)
                                            post_age = time.time() - post.created_utc
                                            if post_age < 300:  # 5 minutes
                                                logger.info(f"Found recent post in subreddit: {post.permalink}")
                                                found_submission = post
                                                break
                            except Exception as user_search_error:
                                logger.warning(f"Error searching user's posts: {user_search_error}")

                        if found_submission:
                            logger.info(f"Found the post despite WebSocket error: {found_submission.permalink}")
                            await self._report_progress(ctx, {"status": "post_found", "message": f"Post found: {found_submission.permalink}"})

                            # Log additional details about the found post
                            logger.info(f"Found post details: ID={found_submission.id}, URL={found_submission.url}, is_video={getattr(found_submission, 'is_video', 'unknown')}")

                            # Check if the post has a video URL
                            if hasattr(found_submission, 'url') and found_submission.url:
                                if 'v.redd.it' in found_submission.url:
                                    logger.info(f"Post has v.redd.it URL: {found_submission.url}")
                                else:
                                    logger.warning(f"Post does not have expected v.redd.it URL: {found_submission.url}")

                            submission = found_submission

                            # Give Reddit a moment to fully process the post
                            logger.info("Waiting a moment for Reddit to fully process the post...")
                            await asyncio.sleep(3)
                        else:
                            logger.warning("Could not find the post - it may not have been created or WebSocket error was fatal")
                            await self._report_progress(ctx, {"status": "post_not_found", "message": "Could not locate the post after WebSocket error"})
                            raise RedditPostError(
                                "Video submission failed: WebSocket error occurred and post could not be located. "
                                "This could be due to:\n"
                                "• Video file corruption or unsupported format\n"
                                "• Reddit server connectivity issues\n"
                                "• Temporary WebSocket connection problems\n"
                                "• Post may have been created but is awaiting moderation\n"
                                "Please check the subreddit and try again if needed.",
                                details={
                                    "error_type": "WEBSOCKET_AND_POST_NOT_FOUND",
                                    "title": title[:300],
                                    "subreddit": clean_subreddit,
                                    "video_path": video_path,
                                    "file_size": path.getsize(video_path) if path.exists(video_path) else "unknown",
                                    "search_strategies_attempted": ["title_poll", "truncated_title_poll", "user_posts_search"]
                                }
                            )

                    # Validate video submission immediately after creation
                    if submission and hasattr(submission, 'is_video') and submission.is_video:
                        logger.info(f"Validating video submission: {submission.permalink}")

                        # First immediate validation
                        video_valid = await self._validate_video_submission(ctx, submission)
                        if not video_valid:
                            logger.error(f"Video submission validation failed immediately: {submission.permalink}")
                            await self._report_progress(ctx, {"status": "error", "message": f"Video upload failed - post created but video is not accessible: {submission.permalink}"})

                            # Delete the broken post
                            try:
                                submission.delete()
                                logger.info(f"Deleted broken video post: {submission.permalink}")
                                await self._report_progress(ctx, {"status": "error", "message": "Deleted broken post. Please check video file and try again."})
                            except Exception as delete_error:
                                logger.warning(f"Could not delete broken post: {delete_error}")

                            raise RedditPostError(
                                "Video upload failed. The post was created but the video is not accessible. "
                                "This usually indicates the video file is corrupted, too large, or in an unsupported format. "
                                "Please check the video file and try again.",
                                details={
                                    "error_type": "VIDEO_UPLOAD_FAILED",
                                    "permalink": submission.permalink,
                                    "title": title[:300],
                                    "subreddit": clean_subreddit
                                }
                            )

                        # Wait for media to be fully ready
                        logger.info(f"Video validated, waiting for full media readiness: {submission.permalink}")
                        media_ready = await self._poll_for_media_readiness(ctx, submission)
                        if not media_ready:
                            logger.warning(f"Media not ready within timeout for: {submission.permalink}")
                            await self._report_progress(ctx, {"status": "media_timeout", "message": f"Video still processing: {submission.permalink}"})
                        else:
                            logger.info(f"Media is ready: {submission.permalink}")
                            await self._report_progress(ctx, {"status": "media_ready", "message": f"Video ready to view: {submission.permalink}"})
                    
                    await self._report_progress(ctx, {"status": "completed", "message": f"Video submitted successfully: {submission.permalink}"})
                elif is_self:
                    # This should NEVER happen when video_path is provided
                    if video_path:
                        logger.error("CRITICAL ERROR: Attempting to create text post when video_path is provided!")
                        logger.error(f"is_self: {is_self}, video_path: {video_path}")
                        raise RedditPostError(
                            "Logic error: Attempted to create text post when video_path was provided. "
                            "This indicates a bug in the post creation logic.",
                            details={
                                "error_type": "LOGIC_ERROR_TEXT_POST_WITH_VIDEO",
                                "is_self": is_self,
                                "video_path": video_path,
                                "content": content
                            }
                        )
                    else:
                        # Only create text post if no video_path is provided
                        logger.info("Creating text post (no video provided)")
                        submission = subreddit_obj.submit(
                            title=title[:300],
                            selftext=content,
                            nsfw=nsfw,
                            spoiler=spoiler,
                            send_replies=True,
                        )
                        await self._report_progress(ctx, {"status": "completed", "message": f"Text post created: {submission.permalink}"})
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
                    await self._report_progress(ctx, {"status": "completed", "message": f"Link post created: {submission.permalink}"})

                # Apply flair if provided for any post type
                if (flair_id or flair_text):

                    try:
                        # Only attempt to flair if submission object is valid
                        if submission:
                            submission.mod.flair(flair_template_id=flair_id, text=flair_text)
                            logger.info(f"Applied flair to post {submission.id}: ID={flair_id}, Text='{flair_text}'")
                            await self._report_progress(ctx, {"status": "applying_flair", "message": f"Flair applied: ID={flair_id}, Text='{flair_text}'"})
                    except Exception as flair_error:
                        logger.warning(f"Failed to apply flair to post {submission.id}: {flair_error}")
                        await self._report_progress(ctx, {"status": "applying_flair", "message": f"Failed to apply flair: {flair_error}"})

                logger.info(f"Post created successfully: {submission.permalink}")
                await self._report_progress(ctx, {"status": "completed", "message": f"Post created successfully: {submission.permalink}"})

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
                logger.error(f"Failed to create post in r/{clean_subreddit}: {post_error}", exc_info=True) # Log full traceback
                # Check for specific error messages related to video uploads
                error_message = str(post_error)
                if "NO_VIDEOS" in error_message:
                    raise RedditPostError(
                        f"Failed to create post: This community does not allow videos. Original error: {error_message}",
                        details={
                            "error_type": "NO_VIDEOS",
                            "subreddit": clean_subreddit
                        }
                    ) from post_error
                elif "Websocket error" in error_message or "MEDIA_UPLOAD_FAILED" in error_message:
                    raise RedditPostError(
                        f"Failed to create post: {error_message}. This often indicates an issue with the video file "
                        f"(e.g., corrupted, unsupported format, too large) or a temporary Reddit media server problem. "
                        f"Your post may still have been created and be awaiting moderation. Please check the video file "
                        f"and try again if necessary.",
                        details={
                            "error_type": "MEDIA_UPLOAD_FAILED",
                            "error_message": error_message
                        }
                    ) from post_error
                elif isinstance(post_error, RedditPostError):
                    # Re-raise RedditPostError as is
                    raise
                else:
                    raise RedditPostError(
                        f"Failed to create post: {post_error}. Please check subreddit name, title, and content.",
                        details={
                            "error_type": "GENERAL_POST_ERROR",
                            "error_message": str(post_error)
                        }
                    ) from post_error

        except Exception as e:
            logger.error(f"An unexpected error occurred during post creation for r/{clean_subreddit}: {e}", exc_info=True) # Log full traceback
            if isinstance(e, RedditPostError):
                raise
            else:
                raise RedditPostError(f"An unexpected error occurred during post creation: {e}", original_exception=e) from e

    async def reply_to_post(
        self, ctx: Any, post_id: str, content: str, subreddit: Optional[str] = None
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
        if not self.manager.client:
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
            await self._report_progress(ctx, {"status": "replying_to_post", "message": f"Creating reply to post ID: {clean_post_id}"})

            submission = self.manager.client.submission(id=clean_post_id)

            try:
                post_title = submission.title
                post_subreddit = submission.subreddit

                logger.info(
                    f"Replying to post: "
                    f"Title: {post_title}, "
                    f"Subreddit: r/{post_subreddit.display_name}"
                )
                await self._report_progress(ctx, {"status": "replying_to_post", "message": f"Replying to post: Title: {post_title}, Subreddit: r/{post_subreddit.display_name}"})

            except Exception as e:
                logger.error(f"Failed to access post {clean_post_id}: {e}")
                await self._report_progress(ctx, {"status": "error", "message": f"Failed to access post {clean_post_id}: {e}"})
                raise ValueError(f"Post {clean_post_id} not found or inaccessible") from e

            if (
                clean_subreddit
                and post_subreddit.display_name.lower() != clean_subreddit.lower()
            ):
                await self._report_progress(ctx, {"status": "error", "message": f"Post is in r/{post_subreddit.display_name}, not r/{clean_subreddit}"})
                raise ValueError(
                    f"Post is in r/{post_subreddit.display_name}, not r/{clean_subreddit}"
                )

            try:
                reply = submission.reply(content)
                logger.info(f"Reply created successfully: {reply.permalink}")
                await self._report_progress(ctx, {"status": "reply_created", "message": f"Reply created successfully: {reply.permalink}"})

                return {
                    "reply_id": reply.id,
                    "reply_permalink": f"https://reddit.com{reply.permalink}",
                    "parent_post_id": submission.id,
                    "parent_post_title": submission.title,
                }

            except Exception as reply_error:
                logger.error(f"Failed to create reply to post {clean_post_id}: {reply_error}")
                raise RuntimeError(f"Failed to create reply: {reply_error}. Please check content and permissions.") from reply_error

        except Exception as e:
            logger.error(f"An unexpected error occurred while replying to post {post_id}: {e}")
            raise RuntimeError(f"An unexpected error occurred while replying to post: {e}") from e

    async def suggest_subreddits(
        self,
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
        if not self.manager.client:
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
            subreddits_iter = self.manager.client.subreddits.search(query, limit=50)
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

    async def get_subreddit_details(self, subreddit_name: str) -> Dict[str, Any]:
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
        if not self.manager.client:
            raise RuntimeError("Reddit client not initialized")

        if not subreddit_name or not isinstance(subreddit_name, str):
            raise ValueError("Subreddit name is required")

        clean_subreddit_name = subreddit_name[2:] if subreddit_name.startswith("r/") else subreddit_name

        try:
            logger.info(f"Fetching details for subreddit r/{clean_subreddit_name}")
            sub = self.manager.client.subreddit(clean_subreddit_name)
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
                "client_is_read_only": self.manager.is_read_only, # Report client read-only status
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