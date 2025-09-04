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
    async def _submit_video_direct(
        self,
        ctx: Any,
        subreddit: str,
        title: str,
        video_path: str,
        thumbnail_path: Optional[str] = None,
        nsfw: bool = False,
        spoiler: bool = False,
        flair_id: Optional[str] = None,
        flair_text: Optional[str] = None,
    ) -> praw.models.Submission:
        """Submit a video post using Reddit's API directly (bypassing PRAW WebSocket issues).

        Args:
            ctx: Context object for reporting progress.
            subreddit: Name of the subreddit to post in.
            title: Title of the post.
            video_path: Path to the video file.
            thumbnail_path: Path to the thumbnail file (optional).
            nsfw: Whether the post should be marked NSFW.
            spoiler: Whether the post should be marked as a spoiler.
            flair_id: Flair template ID.
            flair_text: Flair text.

        Returns:
            PRAW Submission object.

        Raises:
            RedditPostError: If video upload or post creation fails.
        """
        logger.info(f"Submitting video directly via API: {video_path}")

        # Get Reddit credentials
        client_id = getenv('REDDIT_CLIENT_ID')
        client_secret = getenv('REDDIT_CLIENT_SECRET')
        username = getenv('REDDIT_USERNAME')
        password = getenv('REDDIT_PASSWORD')

        if not all([client_id, client_secret, username, password]):
            raise RedditPostError(
                "Reddit API credentials not available for direct upload",
                details={"error_type": "MISSING_CREDENTIALS"}
            )

        # Get OAuth2 access token
        logger.info("Requesting OAuth2 access token...")
        auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
        data = {'grant_type': 'password', 'username': username, 'password': password}
        headers = {'User-Agent': 'RedditMCP/1.0'}

        try:
            token_response = requests.post('https://www.reddit.com/api/v1/access_token',
                                          auth=auth, data=data, headers=headers, timeout=30)
            token_response.raise_for_status()
            token_data = token_response.json()
            access_token = token_data['access_token']
            expires_in = token_data.get('expires_in', 'unknown')
            logger.info(f"Successfully obtained access token (expires in {expires_in}s)")
        except Exception as e:
            logger.error(f"Failed to obtain access token: {e}")
            raise RedditPostError(
                f"Failed to get access token: {e}",
                details={"error_type": "TOKEN_ERROR", "original_error": str(e)}
            )

        # Step 1: Enhanced video validation before upload
        logger.info(f"Starting video validation for: {video_path}")
        await self._report_progress(ctx, {"status": "validating", "message": f"Validating video file: {path.basename(video_path)}"})

        # Validate video file exists
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

        if file_size == 0:
            raise RedditPostError(
                f"Video file is empty: {video_path}",
                details={"error_type": "FILE_EMPTY", "video_path": video_path}
            )

        # Check video file extension
        valid_extensions = {'.mp4', '.mov', '.avi', '.wmv', '.flv', '.webm', '.m4v', '.3gp'}
        file_ext = path.splitext(video_path)[1].lower()
        if file_ext not in valid_extensions:
            raise RedditPostError(
                f"Unsupported video format: {file_ext}. Supported formats: {', '.join(valid_extensions)}",
                details={"error_type": "UNSUPPORTED_FORMAT", "file_extension": file_ext, "supported_formats": list(valid_extensions)}
            )

        # Validate video file content
        try:
            with open(video_path, 'rb') as f:
                # Read first 64KB for validation
                header_data = f.read(65536)
                if len(header_data) == 0:
                    raise RedditPostError(
                        f"Video file is empty or cannot be read: {video_path}",
                        details={"error_type": "FILE_EMPTY", "video_path": video_path}
                    )

                # Check for common video file signatures
                if file_ext == '.mp4':
                    if not (header_data.startswith(b'\x00\x00\x00') or
                           header_data.startswith(b'\x66\x74\x79\x70') or
                           header_data.startswith(b'moov') or
                           header_data.startswith(b'mdat')):
                        logger.warning(f"MP4 file doesn't have expected signature: {header_data[:4].hex()}")

                elif file_ext == '.mov':
                    if not header_data.startswith(b'\x00\x00\x00'):
                        logger.warning(f"MOV file doesn't have expected signature: {header_data[:4].hex()}")

                elif file_ext == '.avi':
                    if not header_data.startswith(b'RIFF'):
                        logger.warning(f"AVI file doesn't have expected signature: {header_data[:4].hex()}")

                # Try to read from different positions to ensure file is not truncated
                f.seek(max(0, file_size - 1024))  # Seek to near end
                end_data = f.read(1024)
                if len(end_data) == 0:
                    raise RedditPostError(
                        f"Video file appears truncated: {video_path}",
                        details={"error_type": "FILE_TRUNCATED", "video_path": video_path}
                    )

        except Exception as validation_error:
            raise RedditPostError(
                f"Video file validation failed: {validation_error}",
                details={"error_type": "VALIDATION_FAILED", "video_path": video_path, "original_error": str(validation_error)}
            )

        logger.info(f"Video validation passed: {video_path} ({file_size} bytes, {file_ext})")

        # Step 2: Upload video to Reddit's media server (single attempt)
        await self._report_progress(ctx, {"status": "uploading", "message": "Uploading video to Reddit..."})

        try:
            logger.info("Starting video upload (single attempt)")

            # Get upload URL for video
            upload_headers = {
                'Authorization': f'bearer {access_token}',
                'User-Agent': 'RedditMCP/1.0'
            }

            upload_data = {
                'filepath': path.basename(video_path),
                'mimetype': 'video/mp4'
            }

            upload_response = requests.post(
                'https://oauth.reddit.com/api/v1/upload/asset.json',
                headers=upload_headers,
                json=upload_data,
                timeout=30
            )
            upload_response.raise_for_status()
            upload_info = upload_response.json()

            upload_url = upload_info['args']['action']
            upload_fields = upload_info['args']['fields']
            video_asset_id = upload_info['asset']['asset_id']

            logger.info(f"Got upload URL for video asset: {video_asset_id}")

            # Upload video file
            logger.info(f"Uploading video file ({file_size} bytes)...")
            with open(video_path, 'rb') as f:
                video_data = f.read()

            files = {'file': (path.basename(video_path), video_data, 'video/mp4')}
            form_data = {field['name']: field['value'] for field in upload_fields}

            upload_resp = requests.post(upload_url, data=form_data, files=files, timeout=300)  # 5 min timeout
            upload_resp.raise_for_status()

            logger.info(f"Video uploaded successfully to asset: {video_asset_id}")

        except Exception as upload_error:
            logger.error(f"Video upload failed: {upload_error}")
            raise RedditPostError(
                f"Failed to upload video: {upload_error}",
                details={"error_type": "UPLOAD_FAILED", "error": str(upload_error)}
            )

        # Step 2: Upload thumbnail if provided
        thumbnail_asset_id = None
        if thumbnail_path and path.exists(thumbnail_path):
            try:
                await self._report_progress(ctx, {"status": "uploading", "message": "Uploading thumbnail..."})

                thumb_upload_data = {
                    'filepath': path.basename(thumbnail_path),
                    'mimetype': 'image/jpeg'
                }

                thumb_upload_response = requests.post(
                    'https://oauth.reddit.com/api/v1/upload/asset.json',
                    headers=upload_headers,
                    json=thumb_upload_data,
                    timeout=30
                )
                thumb_upload_response.raise_for_status()
                thumb_upload_info = thumb_upload_response.json()

                thumb_upload_url = thumb_upload_info['args']['action']
                thumb_upload_fields = thumb_upload_info['args']['fields']

                with open(thumbnail_path, 'rb') as f:
                    thumb_data = f.read()

                thumb_files = {'file': (path.basename(thumbnail_path), thumb_data, 'image/jpeg')}
                thumb_form_data = {field['name']: field['value'] for field in thumb_upload_fields}

                thumb_upload_resp = requests.post(thumb_upload_url, data=thumb_form_data, files=thumb_files, timeout=60)
                thumb_upload_resp.raise_for_status()

                thumbnail_asset_id = thumb_upload_info['asset']['asset_id']
                logger.info("Thumbnail uploaded successfully")

            except Exception as e:
                logger.warning(f"Failed to upload thumbnail: {e}")
                # Continue without thumbnail

        # Step 3: Create the post
        logger.info(f"Creating Reddit post in r/{subreddit} with title: {title[:50]}...")
        await self._report_progress(ctx, {"status": "posting_video", "message": f"Creating post in r/{subreddit}"})

        try:
            submit_data = {
                'sr': subreddit,
                'title': title,
                'kind': 'video',
                'video_poster_url': f'https://reddit-uploaded-media.s3-accelerate.amazonaws.com/{video_asset_id}/poster.jpg' if thumbnail_asset_id else None,
                'url': f'https://reddit-uploaded-media.s3-accelerate.amazonaws.com/{video_asset_id}/video.mp4',
                'sendreplies': True
            }

            if nsfw:
                submit_data['nsfw'] = True
                logger.info("Post marked as NSFW")
            if spoiler:
                submit_data['spoiler'] = True
                logger.info("Post marked as spoiler")
            if flair_id:
                submit_data['flair_id'] = flair_id
                logger.info(f"Post flair ID: {flair_id}")
            if flair_text:
                submit_data['flair_text'] = flair_text
                logger.info(f"Post flair text: {flair_text}")

            logger.info(f"Submitting post data: {submit_data}")

            submit_response = requests.post(
                'https://oauth.reddit.com/api/submit',
                headers=upload_headers,
                data=submit_data,
                timeout=30
            )
            submit_response.raise_for_status()
            submit_result = submit_response.json()

            if not submit_result.get('success', False):
                logger.error(f"Post creation failed: {submit_result}")
                raise RedditPostError(
                    f"Post creation failed: {submit_result}",
                    details={"error_type": "SUBMIT_FAILED", "response": submit_result}
                )

            post_data = submit_result['jquery'][10][3][0]  # Reddit's response format
            post_id = post_data['data']['id']

            logger.info(f"Post created successfully with ID: {post_id}")
            logger.info(f"Post URL: https://reddit.com/r/{subreddit}/comments/{post_id}")

            # Get the PRAW submission object
            subreddit_obj = self.manager.client.subreddit(subreddit)
            submission = self.manager.client.submission(id=post_id)

            logger.info(f"Retrieved PRAW submission object: {submission.permalink}")
            return submission

        except Exception as e:
            raise RedditPostError(
                f"Failed to create post: {e}",
                details={"error_type": "POST_CREATION_FAILED", "original_error": str(e)}
            )

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

                    # Use the new direct video upload mechanism
                    logger.info("Using direct video upload mechanism...")
                    submission = await self._submit_video_direct(
                        ctx=ctx,
                        subreddit=clean_subreddit,
                        title=title[:300],
                        video_path=video_path,
                        thumbnail_path=thumbnail_path,
                        nsfw=nsfw,
                        spoiler=spoiler,
                        flair_id=flair_id,
                        flair_text=flair_text,
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