import asyncio
import logging
import time
import sys
import pprint
import json
from datetime import datetime
from os import getenv, path, remove
from typing import Any, Dict, List, Optional
import praw
import requests
import websockets
import subprocess
from utils import (
    _format_timestamp,
    _format_post,
    _extract_reddit_id,
    _find_submission_by_title,
)
from reddit_client import RedditClientManager
from exceptions import RedditPostError

logger = logging.getLogger(__name__)


def _generate_thumbnail(video_path: str, thumbnail_path: str, timestamp: str = "00:00:01"):
    """Generate a thumbnail from a video file using ffmpeg."""
    logger.info(f"Generating thumbnail for {video_path} at {thumbnail_path}")
    try:
        command = [
            "ffmpeg",
            "-i", video_path,
            "-ss", timestamp,
            "-vframes", "1",
            "-vf", "scale=320:-1",
            thumbnail_path,
            "-y",
            "-hide_banner",
            "-loglevel", "error"
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
             raise RedditPostError(f"ffmpeg failed with return code {result.returncode}. stderr: {result.stderr}")
        if not path.exists(thumbnail_path) or path.getsize(thumbnail_path) == 0:
            raise RedditPostError(f"ffmpeg command executed but thumbnail not created or is empty. stderr: {result.stderr}")
        logger.info(f"Thumbnail generated successfully: {thumbnail_path}")
    except FileNotFoundError:
        raise RedditPostError("ffmpeg not found. Please install ffmpeg and ensure it is in your system's PATH.")
    except subprocess.CalledProcessError as e:
        raise RedditPostError(f"ffmpeg failed with error: {e.stderr}")
    except Exception as e:
        raise RedditPostError(f"An unexpected error occurred during thumbnail generation: {e}")


class RedditService:
    """Encapsulates Reddit API interactions."""

    def __init__(self):
        self.manager = RedditClientManager()

    async def _listen_for_post_url(self, websocket_url: str, timeout: int = 180) -> str:
        """Listen on a websocket for the post URL from Reddit."""
        logger.info(f"Connecting to websocket: {websocket_url}")
        start_time = time.time()
        
        try:
            async with asyncio.timeout(timeout):
                # Try connecting without extra headers first
                async with websockets.connect(
                    websocket_url, 
                    open_timeout=10,
                    ping_interval=20,
                    ping_timeout=10
                ) as websocket:
                    logger.info("Websocket connection established.")
                    message_count = 0
                    
                    # Send a ping to ensure connection is alive
                    await websocket.ping()
                    
                    async for message in websocket:
                        message_count += 1
                        elapsed_time = time.time() - start_time
                        try:
                            data = json.loads(message)
                            logger.info(f"[{elapsed_time:.2f}s] Message #{message_count}: {data}")
                            
                            # Check for different possible success message formats
                            if data.get("type") == "success":
                                if "payload" in data and "redirect" in data["payload"]:
                                    post_url = data["payload"]["redirect"]
                                    logger.info(f"Successfully received post URL: {post_url}")
                                    return post_url
                                elif "payload" in data and "url" in data["payload"]:
                                    post_url = data["payload"]["url"]
                                    logger.info(f"Successfully received post URL (alt format): {post_url}")
                                    return post_url
                            elif data.get("status") == "success" or data.get("status") == "complete":
                                if "url" in data:
                                    post_url = data["url"]
                                    logger.info(f"Successfully received post URL (status format): {post_url}")
                                    return post_url
                            elif "redirect_url" in data:
                                post_url = data["redirect_url"]
                                logger.info(f"Successfully received post URL (redirect format): {post_url}")
                                return post_url
                            elif data.get("message") == "success" and "data" in data:
                                if "url" in data["data"]:
                                    post_url = data["data"]["url"]
                                    logger.info(f"Successfully received post URL (nested format): {post_url}")
                                    return post_url
                                    
                        except json.JSONDecodeError as e:
                            logger.warning(f"[{elapsed_time:.2f}s] Non-JSON message #{message_count}: {message[:200]}...")
                        except Exception as e:
                            logger.warning(f"[{elapsed_time:.2f}s] Error parsing message #{message_count}: {e}")
                    
                    logger.warning(f"Websocket closed after receiving {message_count} messages")
                    
        except asyncio.TimeoutError:
            elapsed_time = time.time() - start_time
            logger.error(f"Websocket timed out after {elapsed_time:.2f} seconds waiting for message.")
            raise RedditPostError("Websocket timed out waiting for success message.")
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"Websocket connection closed: {e}")
            raise RedditPostError("Failed to get post URL from websocket.") from e
        except Exception as e:
            logger.error(f"Websocket error: {e}")
            raise RedditPostError(f"Websocket error: {e}") from e
            
        raise RedditPostError("Did not receive success message from websocket.")

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
    ) -> str:
        """Submit a video post using Reddit's API directly and returns the post URL."""
        logger.info(f"Submitting video directly via API: {video_path}")
        await self._report_progress(ctx, {"status": "posting_video", "message": f"Creating video post in r/{subreddit}"})
        upload_start_time = time.time()

        try:
            reddit = self.manager.client
            if not reddit:
                raise RedditPostError("Reddit client not initialized")
            if self.manager.is_read_only:
                logger.error("_submit_video_direct: Reddit client is in read-only mode")
                raise RedditPostError("Reddit client is in read-only mode")

            headers = {
                'User-Agent': reddit._core._requestor._http.headers['User-Agent'],
                'Authorization': f"Bearer {reddit._core._authorizer.access_token}"
            }

            logger.info("Step 1/4: Getting video upload lease...")
            await self._report_progress(ctx, {"status": "uploading", "message": "Getting video upload lease..."})
            
            lease_response = requests.post(
                'https://oauth.reddit.com/api/media/asset.json',
                headers=headers,
                data={'filepath': path.basename(video_path), 'mimetype': 'video/mp4'},
                timeout=30
            )
            lease_response.raise_for_status()
            lease_data = lease_response.json()
            logger.info(f"Received S3 lease data for video.")
            if not lease_data.get('args'):
                raise RedditPostError("Failed to get video upload lease - no args in response")

            logger.info("Step 2/4: Uploading video to Reddit's storage...")
            await self._report_progress(ctx, {"status": "uploading", "message": "Uploading video..."})
            upload_url = 'https:' + lease_data['args']['action'] if lease_data['args']['action'].startswith('//') else lease_data['args']['action']
            upload_fields = {field['name']: field['value'] for field in lease_data['args']['fields']}
            
            with open(video_path, 'rb') as video_file:
                files_payload = {'file': (path.basename(video_path), video_file, 'video/mp4')}
                upload_response = requests.post(upload_url, data=upload_fields, files=files_payload, timeout=300)
                upload_response.raise_for_status()
            upload_duration = time.time() - upload_start_time
            logger.info(f"Video uploaded successfully to S3 in {upload_duration:.2f} seconds.")

            asset_id = lease_data.get('asset', {}).get('asset_id')
            if not asset_id:
                raise RedditPostError("Failed to get asset_id from video upload lease.")
            
            # Construct the video URL for Reddit video submissions (matching PRAW's approach)
            # This ensures the video is embedded in the post rather than linked to
            video_url = f"https://v.redd.it/{asset_id}"
            
            submit_data = {
                'api_type': 'json', 
                'kind': 'video', 
                'title': title, 
                'sr': subreddit,
                'nsfw': str(nsfw).lower(), 
                'spoiler': str(spoiler).lower(),
                'validate_on_submit': 'true',
                # Use video URL for proper video embedding (following PRAW's implementation)
                'url': video_url,
            }
            if flair_id: submit_data['flair_id'] = flair_id
            if flair_text: submit_data['flair_text'] = flair_text

            if thumbnail_path:
                logger.info(f"Uploading thumbnail: {thumbnail_path}")
                await self._report_progress(ctx, {"status": "uploading", "message": "Uploading thumbnail..."})
                
                with open(thumbnail_path, 'rb') as thumb_file:
                    thumb_lease_response = requests.post(
                        'https://oauth.reddit.com/api/media/asset.json',
                        headers=headers,
                        data={'filepath': path.basename(thumbnail_path), 'mimetype': 'image/jpeg'},
                        timeout=30
                    )
                    thumb_lease_response.raise_for_status()
                    thumb_lease_data = thumb_lease_response.json()
                    logger.info(f"Received S3 lease data for thumbnail.")
                    
                    thumb_upload_url = 'https:' + thumb_lease_data['args']['action'] if thumb_lease_data['args']['action'].startswith('//') else thumb_lease_data['args']['action']
                    thumb_upload_fields = {field['name']: field['value'] for field in thumb_lease_data['args']['fields']}
                    
                    thumb_files_payload = {'file': (path.basename(thumbnail_path), thumb_file, 'image/jpeg')}
                    thumb_upload_response = requests.post(thumb_upload_url, data=thumb_upload_fields, files=thumb_files_payload, timeout=60)
                    thumb_upload_response.raise_for_status()
                    
                    # Get thumbnail asset_id and construct URL (following PRAW's pattern)
                    thumb_asset_id = thumb_lease_data.get('asset', {}).get('asset_id')
                    if thumb_asset_id:
                        thumbnail_url = f"https://i.redd.it/{thumb_asset_id}"
                        submit_data['video_poster_url'] = thumbnail_url
                        logger.info(f"Using thumbnail URL: {thumbnail_url}")
            else:
                # Following PRAW's approach: if no thumbnail provided, Reddit will use default
                logger.info("No thumbnail provided, Reddit will use default")
            
            logger.info("Step 3/4: Submitting post to Reddit...")
            response = requests.post('https://oauth.reddit.com/api/submit', headers=headers, data=submit_data, timeout=60)
            response.raise_for_status()
            result = response.json()

            if 'json' not in result or ('errors' in result['json'] and result['json']['errors']):
                error_details = result.get('json', {}).get('errors', 'Unknown error')
                raise RedditPostError(f"Failed to submit post: {json.dumps(error_details)}")

            logger.info(f"Submit response received: {pprint.pformat(result)}")

            # Check if we got a direct URL or name in the response
            if 'json' in result and 'data' in result['json']:
                data = result['json']['data']
                if 'url' in data:
                    logger.info(f"Got direct URL from submission: {data['url']}")
                    return data['url']
                elif 'name' in data:
                    # Construct URL from post name
                    post_name = data['name']
                    post_url = f"https://www.reddit.com/r/{subreddit}/comments/{post_name.split('_')[1]}/"
                    logger.info(f"Constructed URL from post name: {post_url}")
                    return post_url
                elif 'id' in data:
                    # Construct URL from post ID
                    post_id = data['id']
                    post_url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}/"
                    logger.info(f"Constructed URL from post ID: {post_url}")
                    return post_url
                elif 'user_submitted_page' in data:
                    # This usually contains the user's submitted page, let's see if we can extract anything useful
                    user_page = data['user_submitted_page']
                    logger.info(f"Found user_submitted_page: {user_page}")
                    # We'll still try websocket but also log this for reference

            # Fallback to websocket if available
            try:
                websocket_url = result['json']['data']['websocket_url']
                logger.info("Post submitted, trying websocket for URL...")
                logger.info("Step 4/4: Listening for final post URL from websocket...")
                
                # Try websocket with shorter timeout, then fallback to API check
                try:
                    post_url = await self._listen_for_post_url(websocket_url, timeout=30)
                    logger.info(f"Video post created successfully via websocket: {post_url}")
                    return post_url
                except RedditPostError as websocket_error:
                    logger.warning(f"Websocket failed: {websocket_error}")
                    
                    # Fallback: Check user's recent submissions to find the new post
                    logger.info("Attempting to find post via user submissions...")
                    try:
                        reddit = self.manager.client
                        user = await reddit.user.me()
                        submissions = user.submissions.new(limit=5)
                        
                        # Look for a recent submission in this subreddit with our title
                        async for submission in submissions:
                            if (hasattr(submission, 'subreddit') and 
                                submission.subreddit.display_name.lower() == subreddit.lower() and
                                hasattr(submission, 'title') and
                                submission.title == title):
                                post_url = f"https://reddit.com{submission.permalink}"
                                logger.info(f"Found post via user submissions: {post_url}")
                                return post_url
                    except Exception as api_error:
                        logger.warning(f"Could not find post via API: {api_error}")
                        
            except KeyError:
                logger.warning("No websocket_url in submission response")

            # Final fallback - assume post was successful and return a generic URL
            logger.info("Using fallback URL construction")
            return f"https://www.reddit.com/r/{subreddit}/new/"

        except requests.RequestException as e:
            error_msg = f"Network error during video submission: {str(e)}"
            logger.error(error_msg)
            raise RedditPostError(error_msg) from e
        except Exception as e:
            error_msg = f"Error during video submission: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RedditPostError(error_msg) from e

    async def _report_progress(self, ctx: Any, data: Dict[str, Any]) -> None:
        """Report progress via ctx.report_progress if available."""
        if hasattr(ctx, 'report_progress'):
            await ctx.report_progress(data)
        else:
            logger.info(f"Progress: {data}")

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
    auto_comment: bool = False,
    original_url: Optional[str] = None,
    comment_language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new post in a subreddit."""
        client = self.manager.client
        if not client or self.manager.is_read_only:
            raise RuntimeError("Reddit client not initialized or in read-only mode.")
        if not await self.manager.check_user_auth():
            raise RuntimeError("Reddit user authentication failed.")

        clean_subreddit = subreddit[2:] if subreddit.startswith("r/") else subreddit
        
        delete_temp_thumb = False
        temp_thumbnail_path = None
        if video_path and not thumbnail_path:
            temp_thumbnail_path = path.join(path.dirname(video_path), f"thumb_{path.basename(video_path)}.jpg")
            try:
                _generate_thumbnail(video_path, temp_thumbnail_path)
                thumbnail_path = temp_thumbnail_path
                delete_temp_thumb = True
                logger.info(f"Automatically generated thumbnail: {thumbnail_path}")
            except RedditPostError as e:
                logger.warning(f"Could not automatically generate thumbnail: {e}. Proceeding without one.")
                thumbnail_path = None
        
        submission = None
        try:
            logger.info(f"Creating post in r/{clean_subreddit}")
            subreddit_obj = await client.subreddit(clean_subreddit)

            if video_path:
                post_url = await self._submit_video_direct(
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
                logger.info(f"Video post successful, URL: {post_url}")
                await self._report_progress(ctx, {"status": "completed", "message": f"Video submitted successfully: {post_url}"})

                # Try to get the submission object if we have a valid post URL with ID
                submission = None
                post_id = None
                if post_url and "new/" not in post_url:
                    try:
                        post_id = _extract_reddit_id(post_url)
                        submission = await client.submission(id=post_id)
                        final_permalink = post_url
                    except Exception as e:
                        logger.warning(f"Could not fetch submission object: {e}")
                        final_permalink = post_url
                else:
                    # Fallback case - post was submitted but we don't have the exact URL
                    logger.info("Video post submitted but exact URL not available, using fallback")
                    final_permalink = f"https://www.reddit.com/r/{clean_subreddit}/new/"

                # Auto-comment logic
                if auto_comment and original_url and post_id:
                    try:
                        # Sanitize the original URL for the comment
                        from utils import sanitize_tiktok_url
                        clean_original_url = sanitize_tiktok_url(original_url)
                        comment_text = f"Original TikTok: {clean_original_url}"
                        # Optionally, add language info if needed
                        if comment_language:
                            comment_text += f"\n(Language: {comment_language})"
                        await self.reply_to_post(ctx, post_id, comment_text, subreddit=clean_subreddit)
                        logger.info(f"Auto-commented original TikTok link on post {post_id}")
                    except Exception as comment_error:
                        logger.error(f"Failed to auto-comment TikTok link: {comment_error}")

            elif is_self:
                submission = await subreddit_obj.submit(
                    title=title[:300], selftext=content, nsfw=nsfw, spoiler=spoiler, send_replies=True
                )
                final_permalink = f"https://reddit.com{submission.permalink}"
            else:
                if content and not content.startswith(("http://", "https://")):
                    content = f"https://{content}"
                submission = await subreddit_obj.submit(
                    title=title[:300], url=content, nsfw=nsfw, spoiler=spoiler, send_replies=True
                )
                final_permalink = f"https://reddit.com{submission.permalink}"

            if submission and (flair_id or flair_text):
                try:
                    if video_path:
                        logger.info("Waiting 5 seconds before applying flair to video post...")
                        await asyncio.sleep(5)
                    
                    await submission.mod.flair(flair_template_id=flair_id, text=flair_text)
                    logger.info(f"Applied flair to post {submission.id}: ID={flair_id}, Text='{flair_text}'")
                except Exception as flair_error:
                    logger.warning(f"Failed to apply flair to post {submission.id}: {flair_error}")

            logger.info(f"Post created successfully: {final_permalink}")
            return {
                "post": _format_post(submission) if submission else "Video post created (submission object not available)",
                "metadata": {
                    "created_at": _format_timestamp(time.time()),
                    "subreddit": clean_subreddit,
                    "permalink": final_permalink,
                    "id": submission.id if submission else "unknown",
                },
            }

        except Exception as e:
            logger.error(f"Failed to create post in r/{clean_subreddit}: {e}", exc_info=True)
            if isinstance(e, RedditPostError):
                raise
            raise RedditPostError(f"Failed to create post: {e}") from e
        finally:
            if delete_temp_thumb and temp_thumbnail_path and path.exists(temp_thumbnail_path):
                try:
                    logger.info(f"Removing temporary thumbnail: {temp_thumbnail_path}")
                    remove(temp_thumbnail_path)
                except OSError as e:
                    logger.warning(f"Error removing temporary thumbnail {temp_thumbnail_path}: {e}")

    async def reply_to_post(
        self, ctx: Any, post_id: str, content: str, subreddit: Optional[str] = None
    ) -> Dict[str, Any]:
        """Post a reply to an existing Reddit post."""
        client = self.manager.client
        if not client:
            raise RuntimeError("Reddit client not initialized")

        clean_post_id = _extract_reddit_id(post_id)
        logger.info(f"Creating reply to post ID: {clean_post_id}")
        
        submission = await client.submission(id=clean_post_id)
        reply = await submission.reply(content)
        
        return {
            "reply_id": reply.id,
            "reply_permalink": f"https://reddit.com{reply.permalink}",
            "parent_post_id": submission.id,
        }

    async def get_subreddit_details(self, subreddit_name: str) -> Dict[str, Any]:
        """Get detailed information about a single subreddit, including rules and flair."""
        # Ensure client is initialized
        if not self.manager.client:
            await self.manager.initialize_client()
            
        client = self.manager.client
        if not client:
            raise RuntimeError("Reddit client not initialized")

        clean_subreddit_name = subreddit_name[2:] if subreddit_name.startswith("r/") else subreddit_name
        sub = await client.subreddit(clean_subreddit_name)
        await sub.load()  # Load subreddit data before accessing attributes
        
        flair_templates = []
        try:
            if getattr(sub, 'link_flair_enabled', False):
                async for template in sub.flair.link_templates:
                    flair_templates.append({
                        "id": template["id"],
                        "text": template["text"],
                        "text_editable": template["text_editable"],
                    })
        except Exception as e:
            logger.warning(f"Could not fetch flair for r/{clean_subreddit_name}: {e}")

        rules = []
        try:
            # Access the rules object
            subreddit_rules = sub.rules
            # Get the rule list (this will fetch from API if not cached)
            rule_list = subreddit_rules._rule_list
            for rule in rule_list:
                if hasattr(rule, 'short_name') and hasattr(rule, 'description'):
                    # Rule is an object with attributes
                    rules.append({
                        "short_name": rule.short_name,
                        "description": rule.description,
                    })
                elif isinstance(rule, str):
                    # Rule is just a string
                    rules.append({
                        "short_name": rule,
                        "description": rule,
                    })
                else:
                    # Unknown format, convert to string
                    rules.append({
                        "short_name": str(rule),
                        "description": str(rule),
                    })
        except Exception as e:
            logger.warning(f"Could not fetch rules for r/{clean_subreddit_name}: {e}")

        # Check if video posts are allowed
        video_allowed = True  # Default to allowed
        try:
            # Check various indicators of video posting restrictions
            # Note: Some subreddits may have restrictions that aren't exposed via API
            # This is a best-effort check
            if hasattr(sub, 'allow_videos') and sub.allow_videos is False:
                video_allowed = False
            elif hasattr(sub, 'allow_video_gifs') and sub.allow_video_gifs is False:
                video_allowed = False
            elif hasattr(sub, 'submission_type') and sub.submission_type in ['text', 'link']:
                # If only text or link submissions are allowed, videos are not
                video_allowed = False
        except Exception as e:
            logger.warning(f"Could not determine video posting permissions for r/{clean_subreddit_name}: {e}")
            # Default to allowed if we can't determine

        return {
            "subreddit": {
                "name": sub.display_name,
                "title": sub.title,
                "subscribers": sub.subscribers,
                "public_description": sub.public_description,
                "flair": flair_templates,
                "rules": rules,
                "video_posts_allowed": video_allowed,
            },
            "metadata": {
                "fetched_at": _format_timestamp(time.time()),
                "subreddit_name": clean_subreddit_name,
            },
        }

    async def suggest_subreddits(
        self,
        query: str,
        limit_subreddits: int = 5,
        posts_limit: int = 5,
        time_filters: Optional[List[str]] = None,
        post_sort: str = "top",
    ) -> Dict[str, Any]:
        """Suggest relevant subreddits for a topic and show sample post titles."""
        client = self.manager.client
        if not client:
            raise RuntimeError("Reddit client not initialized")
        # Rest of the implementation
        pass