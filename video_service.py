
import asyncio
import logging
import time
import sys
import subprocess
import json
from os import getenv, makedirs, path, remove
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse
import yt_dlp
import yt_dlp.utils
import requests

# Import custom exceptions
from exceptions import VideoDownloadError

logger = logging.getLogger(__name__)


class VideoService:
    """Encapsulates video download and processing logic."""

    async def download_tiktok_video(self, ctx: Any, url: str, download_folder: str = "downloaded", keep: bool = True) -> Dict[str, Any]:
        """Download a TikTok video using yt-dlp (thumbnail discarded immediately).

        Args:
            ctx: Context object for reporting progress.
            url: TikTok video URL (short or full)
            download_folder: Destination folder
            keep: If False, video will be deleted after returning metadata

        Returns:
            Dict with video_id, video_path (may be deleted later), thumbnail_deleted flag
        """
        if not path.exists(download_folder):
            makedirs(download_folder)

        # Sanitize the URL: remove query parameters
        parsed_url = urlparse(url)
        url = urlunparse(parsed_url._replace(query=""))
        logger.info(f"Sanitized TikTok URL to: {url}")

        # Test network connectivity before attempting download
        try:
            # Run network test in a separate thread to avoid event loop issues
            await asyncio.get_event_loop().run_in_executor(None, self._test_network_connectivity_sync)
        except Exception as net_error:
            logger.warning(f"Network connectivity test failed: {net_error}")
            logger.warning("Continuing with download attempt despite network test failure...")
            # Don't fail the entire operation due to network test issues

        # Check and update yt-dlp to latest version
        try:
            import subprocess
            import sys
            logger.info("Checking for yt-dlp updates...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"],
                                    capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                logger.info("yt-dlp is up to date")
            else:
                logger.warning(f"Failed to update yt-dlp: {result.stderr}")
        except Exception as e:
            logger.warning(f"Error checking for yt-dlp updates: {e}")

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

        # Progress tracking variables
        last_reported_progress = 0
        last_reported_time = time.time() # Initialize last reported time

        def progress_hook(d):
            """Progress hook for yt-dlp to report download status."""
            nonlocal last_reported_progress, last_reported_time

            if d['status'] == 'downloading':
                current_time = time.time()
                # Calculate percentage completion
                if d.get('total_bytes') and d.get('downloaded_bytes'):
                    percentage = (d['downloaded_bytes'] / d['total_bytes']) * 100
                elif d.get('total_bytes_estimate') and d.get('downloaded_bytes'):
                    percentage = (d['downloaded_bytes'] / d['total_bytes_estimate']) * 100
                else:
                    percentage = 0

                # Report progress every 10% or at start/end or every 30 seconds
                if (percentage >= last_reported_progress + 10 or
                    (last_reported_progress == 0 and percentage > 0) or
                    percentage >= 100 or
                    (current_time - last_reported_time) >= 30): # Report every 30 seconds

                    last_reported_progress = (percentage // 10) * 10  # Round down to nearest 10
                    last_reported_time = current_time # Update last reported time

                    # Format progress message
                    downloaded_mb = d['downloaded_bytes'] / (1024 * 1024) if d.get('downloaded_bytes') else 0
                    total_mb = (d['total_bytes'] or d['total_bytes_estimate'] or 0) / (1024 * 1024) if (d.get('total_bytes') or d.get('total_bytes_estimate')) else 0

                    # Report formal progress via ctx.report_progress
                    progress_message = {
                        "status": "downloading",
                        "percentage": f"{percentage:.1f}%",
                        "downloaded_mb": f"{downloaded_mb:.1f}MB",
                        "total_mb": f"{total_mb:.1f}MB",
                    }
                    if d.get('eta') is not None:
                        progress_message["eta_seconds"] = d['eta']
                    if d.get('speed') is not None:
                        speed_kbps = d['speed'] / 1024
                        progress_message["speed_kbps"] = f"{speed_kbps:.1f} KB/s"

                    asyncio.create_task(report_progress(progress_message))

            elif d['status'] == 'finished':
                asyncio.create_task(report_progress({"status": "finished", "message": "Download completed, processing video..."}))
            elif d['status'] == 'error':
                asyncio.create_task(report_progress({"status": "error", "message": f"Download error: {d.get('errmsg', 'Unknown error')}"}))

        original_url = url
        if any(host in url for host in ("vm.tiktok.com", "vt.tiktok.com")):
            try:
                head = requests.head(url, allow_redirects=True, timeout=10)
                head.raise_for_status()
                url = head.url
            except Exception as e:
                raise RuntimeError(f"Failed to resolve short TikTok link: {e}") from e

        # Check if video already exists locally
        try:
            # Extract video ID from URL for pre-check
            video_id_from_url = None
            if "/video/" in url:
                # Extract ID from TikTok URL like https://www.tiktok.com/@user/video/123456789123456789
                parts = url.split("/video/")
                if len(parts) > 1:
                    video_id_from_url = parts[1].split("?")[0]  # Remove query parameters if any

            # Check if a file with this ID already exists in the download folder
            if video_id_from_url:
                # Check for common video extensions
                for ext in ('.mp4', '.mov', '.mkv', '.webm', '.avi', '.flv'):
                    existing_file = path.join(download_folder, f"{video_id_from_url}{ext}")
                    if path.exists(existing_file):
                        logger.info(f"Video {video_id_from_url} already exists locally: {existing_file}")
                        # Return result indicating video is already available
                        result = {
                            'original_url': original_url,
                            'resolved_url': url,
                            'video_id': video_id_from_url,
                            'video_path': existing_file,
                            'thumbnail_deleted': True,  # Assume thumbnail was deleted previously
                            'kept': keep,
                            'already_exists': True,
                            'message': f'Video {video_id_from_url} already exists locally'
                        }
                        logger.info(f"Video already exists. Returning result: {result}")
                        return result
        except Exception as e:
            logger.warning(f"Error during pre-download check: {e}")

        # Enhanced yt-dlp options with retries and better error handling
        output_template = path.join(download_folder, '%(id)s.%(ext)s')

        # Try different format options in order of preference
        format_options = [
            'best[ext=mp4]/best[height<=1080]',  # Prefer MP4, limit resolution
            'best[ext=mp4]',                     # Fallback to any MP4
            'best[height<=1080]/best',           # Limit resolution if no MP4
            'best'                               # Last resort
        ]

        video_path = None
        video_id = None
        thumbnail_deleted = False
        download_success = False
        last_error = None

        # Try different format options with retries
        for format_option in format_options:
            if download_success:
                break

            logger.info(f"Trying download with format: {format_option}")

            ydl_opts = {
                'quiet': True,
                'no_warnings': False,  # Show warnings for debugging
                'format': format_option,
                'outtmpl': output_template,
                'writethumbnail': True,
                'progress_hooks': [progress_hook],
                # Add retry and timeout options
                'retries': 3,
                'fragment_retries': 3,
                'retry_sleep_functions': {
                    'http': lambda n: min(2 ** n, 30),  # Exponential backoff, max 30s
                    'fragment': lambda n: min(2 ** n, 10)  # Shorter for fragments
                },
                # Add timeout options
                'socket_timeout': 30,
                'extractor_retries': 3,
                # Ensure we get MP4 when possible
                'merge_output_format': 'mp4',
                # Add headers to avoid blocking
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
            }

            # Attempt download with retries
            max_download_attempts = 3
            for attempt in range(1, max_download_attempts + 1):
                try:
                    logger.info(f"Download attempt {attempt}/{max_download_attempts} with format {format_option}")

                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        # Run yt-dlp in a separate thread to avoid event loop issues
                        info = await asyncio.get_event_loop().run_in_executor(None, lambda: ydl.extract_info(url, download=True))
                        video_id = info.get('id')
                        video_path = ydl.prepare_filename(info)

                        # Validate downloaded file
                        if not path.exists(video_path):
                            raise VideoDownloadError(f"Downloaded file not found: {video_path}")

                        file_size = path.getsize(video_path)
                        if file_size == 0:
                            raise VideoDownloadError(f"Downloaded file is empty: {video_path}")

                        # Check file size is reasonable (at least 100KB, max 500MB)
                        min_size = 100 * 1024  # 100KB
                        max_size = 500 * 1024 * 1024  # 500MB
                        if file_size < min_size:
                            raise VideoDownloadError(f"Downloaded file too small ({file_size} bytes < {min_size} bytes): {video_path}")
                        if file_size > max_size:
                            raise VideoDownloadError(f"Downloaded file too large ({file_size} bytes > {max_size} bytes): {video_path}")

                        # Try to read file to ensure it's not corrupted
                        try:
                            with open(video_path, 'rb') as f:
                                # Read first 64KB to check file integrity
                                test_data = f.read(65536)
                                if len(test_data) == 0:
                                    raise VideoDownloadError(f"Cannot read downloaded file: {video_path}")

                                # Check for common video file signatures
                                if not (test_data.startswith(b'\x00\x00\x00') or  # MP4
                                       test_data.startswith(b'RIFF') or         # AVI
                                       test_data.startswith(b'\x66\x74\x79\x70')):  # MP4 variant
                                    logger.warning(f"Downloaded file may not be a valid video format. Signature: {test_data[:4].hex()}")

                        except Exception as read_error:
                            raise VideoDownloadError(f"Downloaded file is corrupted or unreadable: {read_error}") from read_error

                        logger.info(f"Successfully downloaded and validated video: {video_path} ({file_size} bytes)")
                        download_success = True
                        break

                except yt_dlp.utils.DownloadError as e:
                    last_error = e
                    error_msg = str(e).lower()
        
                    # Check for network/DNS issues
                    if ("name or service not known" in error_msg or
                        "failed to resolve" in error_msg or
                        "nodename nor servname provided" in error_msg or
                        "name resolution failure" in error_msg):
                        logger.error(f"DNS/Network error detected: {e}")
                        logger.error("This indicates a network connectivity or DNS resolution issue.")
                        logger.error("Please check:")
                        logger.error("  1. Internet connection")
                        logger.error("  2. DNS settings")
                        logger.error("  3. Firewall/proxy settings")
                        logger.error("  4. Whether TikTok is accessible in your region")
        
                        # For DNS issues, don't retry as it's likely a persistent network problem
                        break
        
                    # Check for geo-blocking or access issues
                    elif ("geo" in error_msg or "region" in error_msg or
                          "not available" in error_msg or "blocked" in error_msg):
                        logger.error(f"Geo-blocking or access issue detected: {e}")
                        logger.error("The video may not be available in your region or requires authentication.")
                        break
        
                    # For other errors, retry
                    logger.warning(f"Download attempt {attempt} failed: {e}")
                    if attempt < max_download_attempts:
                        logger.info(f"Retrying download in 5 seconds...")
                        await asyncio.sleep(5)
                    else:
                        logger.error(f"All download attempts failed for format {format_option}")
                        break
                except Exception as e:
                    last_error = e
                    logger.warning(f"Unexpected error in download attempt {attempt}: {e}")
                    if attempt < max_download_attempts:
                        logger.info(f"Retrying download in 5 seconds...")
                        await asyncio.sleep(5)
                    else:
                        logger.error(f"All download attempts failed for format {format_option}")
                        break

        # If all format options failed, raise the last error
        if not download_success:
            error_msg = f"Failed to download video after trying all format options"
            if last_error:
                error_msg += f": {last_error}"
            raise VideoDownloadError(error_msg, original_exception=last_error)

        # Clean up thumbnail
        base_filename, _ = path.splitext(video_path)
        for ext in ('.jpg', '.webp', '.png', '.image'):
            pth = base_filename + ext
            if path.exists(pth):
                try:
                    remove(pth)
                    thumbnail_deleted = True
                    logger.info(f"Deleted thumbnail: {pth}")
                except Exception as e:
                    logger.warning(f"Could not delete thumbnail {pth}: {e}")
                break

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

        # Final validation of downloaded video
        try:
            # Run validation in a separate thread to avoid event loop issues
            await asyncio.get_event_loop().run_in_executor(None, lambda: self._validate_downloaded_video_sync(video_path, video_id))
            logger.info(f"Video validation passed: {video_path}")
        except Exception as validation_error:
            logger.error(f"Video validation failed: {validation_error}")
            # Try to clean up the invalid file
            try:
                if path.exists(video_path):
                    remove(video_path)
                    logger.info(f"Cleaned up invalid video file: {video_path}")
            except Exception as cleanup_error:
                logger.warning(f"Could not clean up invalid file: {cleanup_error}")
            raise VideoDownloadError(f"Downloaded video failed validation: {validation_error}") from validation_error

        # Log completion message for LLM to monitor
        logger.info(f"Download completed successfully. Video ID: {video_id}")
        logger.info(f"Video available at: {video_path}")
        logger.info(f"Returning result: {result}")
        return result

    async def _validate_downloaded_video(self, video_path: str, video_id: str) -> None:
        """Validate that a downloaded video is complete and valid for Reddit upload.

        Args:
            video_path: Path to the downloaded video file
            video_id: Video ID for logging purposes

        Raises:
            VideoDownloadError: If validation fails
        """
        logger.info(f"Validating downloaded video: {video_path}")

        # Check file exists
        if not path.exists(video_path):
            raise VideoDownloadError(f"Video file does not exist: {video_path}")

        # Check file size
        try:
            file_size = path.getsize(video_path)
            logger.info(f"Video file size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")

            if file_size == 0:
                raise VideoDownloadError(f"Video file is empty: {video_path}")

            # Reddit limits: 15MB for free users, 1GB for premium
            # Be conservative and check for reasonable minimum
            if file_size < 1024:  # Less than 1KB
                raise VideoDownloadError(f"Video file suspiciously small: {file_size} bytes")

        except OSError as e:
            raise VideoDownloadError(f"Cannot get file size: {e}") from e

        # Check file readability and basic format validation
        try:
            with open(video_path, 'rb') as f:
                # Read first 64KB for validation
                header_data = f.read(65536)
                if len(header_data) == 0:
                    raise VideoDownloadError(f"Cannot read video file header: {video_path}")

                # Check for common video file signatures
                file_ext = path.splitext(video_path)[1].lower()

                if file_ext == '.mp4':
                    # MP4 files should start with specific signatures
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
                    raise VideoDownloadError(f"Video file appears truncated: {video_path}")

        except Exception as e:
            raise VideoDownloadError(f"Video file validation failed: {e}") from e

        # Optional: Try to get basic video info with ffprobe if available
        try:
            ffprobe_cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", video_path
            ]
            result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                video_info = json.loads(result.stdout)
                logger.info(f"Video validation successful - Duration: {video_info.get('format', {}).get('duration', 'unknown')}s")
            else:
                logger.warning(f"ffprobe validation failed: {result.stderr}")
                # Don't fail the whole process if ffprobe isn't available or fails

        except Exception as e:
            logger.warning(f"Could not validate video with ffprobe: {e}")
            # Continue without ffprobe validation

        logger.info(f"Video validation completed successfully: {video_path}")

    def _test_network_connectivity_sync(self) -> None:
        """Test basic network connectivity and DNS resolution (synchronous version).

        Raises:
            VideoDownloadError: If network connectivity issues are detected
        """
        logger.info("Testing network connectivity...")

        try:
            # Test basic internet connectivity
            logger.info("Testing basic internet connectivity...")
            test_response = requests.get("https://www.google.com", timeout=10)
            test_response.raise_for_status()
            logger.info("✓ Basic internet connectivity confirmed")

        except requests.RequestException as e:
            logger.error(f"✗ Basic internet connectivity test failed: {e}")
            logger.error("Please check your internet connection and try again.")
            raise VideoDownloadError(
                f"Network connectivity test failed: {e}. "
                "Please check your internet connection and try again."
            ) from e

        try:
            # Test DNS resolution for TikTok
            logger.info("Testing DNS resolution for TikTok...")
            import socket
            socket.gethostbyname("www.tiktok.com")
            logger.info("✓ TikTok DNS resolution successful")

        except socket.gaierror as e:
            logger.error(f"✗ TikTok DNS resolution failed: {e}")
            logger.error("This may indicate:")
            logger.error("  - DNS server issues")
            logger.error("  - TikTok being blocked in your region")
            logger.error("  - Firewall/proxy blocking TikTok")
            logger.error("  - Temporary network issues")
            raise VideoDownloadError(
                f"DNS resolution for TikTok failed: {e}. "
                "This may indicate network issues or TikTok being blocked in your region. "
                "Please check your network settings and try again."
            ) from e

        logger.info("✓ Network connectivity tests passed")

    def _validate_downloaded_video_sync(self, video_path: str, video_id: str) -> None:
        """Synchronous version of video validation to avoid event loop issues.
        
        Args:
            video_path: Path to the downloaded video file
            video_id: Video ID for logging purposes
        """
        # This is a wrapper that calls the async method in a synchronous context
        # We need to create a new event loop for this
        try:
            # For now, we'll just do basic validation synchronously
            logger.info(f"Validating downloaded video synchronously: {video_path}")
            
            # Check file exists
            if not path.exists(video_path):
                raise VideoDownloadError(f"Video file does not exist: {video_path}")
            
            # Check file size
            try:
                file_size = path.getsize(video_path)
                logger.info(f"Video file size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
                
                if file_size == 0:
                    raise VideoDownloadError(f"Video file is empty: {video_path}")
                
                # Reddit limits: 15MB for free users, 1GB for premium
                # Be conservative and check for reasonable minimum
                if file_size < 1024:  # Less than 1KB
                    raise VideoDownloadError(f"Video file suspiciously small: {file_size} bytes")
                
            except OSError as e:
                raise VideoDownloadError(f"Cannot get file size: {e}") from e
            
            # Check file readability and basic format validation
            try:
                with open(video_path, 'rb') as f:
                    # Read first 64KB for validation
                    header_data = f.read(65536)
                    if len(header_data) == 0:
                        raise VideoDownloadError(f"Cannot read video file header: {video_path}")
                    
                    # Check for common video file signatures
                    file_ext = path.splitext(video_path)[1].lower()
                    
                    if file_ext == '.mp4':
                        # MP4 files should start with specific signatures
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
                        raise VideoDownloadError(f"Video file appears truncated: {video_path}")
                        
            except Exception as e:
                raise VideoDownloadError(f"Video file validation failed: {e}") from e
            
            logger.info(f"Video validation completed successfully: {video_path}")
            
        except Exception as e:
            logger.error(f"Video validation failed: {e}")
            raise
        logger.info("✓ Network connectivity tests passed")

    async def _test_network_connectivity(self) -> None:
        """Test basic network connectivity and DNS resolution (async wrapper)."""
        # Run the synchronous test in a thread to avoid event loop issues
        await asyncio.to_thread(self._test_network_connectivity_sync)

    async def transcode_to_reddit_safe(self, ctx: Any, video_path: str, output_folder: str = "transcoded") -> Dict[str, Any]:
        """Transcode a video to Reddit-safe format (MP4, H.264, yuv420p, AAC, +faststart).

        Args:
            ctx: Context object for reporting progress.
            video_path: Path to the input video file
            output_folder: Destination folder for transcoded video

        Returns:
            Dict with transcoded_path, logs, and metadata
        """
        if not path.exists(video_path):
            raise ValueError(f"Video file does not exist: {video_path}")

        if not path.exists(output_folder):
            makedirs(output_folder)

        # Create async-safe progress reporter
        async def report_progress_async(data: Dict[str, Any]) -> None:
            if hasattr(ctx, '__event_emitter__'):
                await ctx.__event_emitter__({
                    "type": "status",
                    "data": {"description": data.get("message", "Unknown status"), "done": False}
                })
            elif hasattr(ctx, 'report_progress'):
                ctx.report_progress(data)
            else:
                logger.info(f"Progress: {data}")

        def report_progress(data: Dict[str, Any]) -> None:
            asyncio.create_task(report_progress_async(data))

        # Get input video metadata using ffprobe
        report_progress({"status": "transcoding", "message": "Analyzing input video..."})
        input_info = {}
        try:
            ffprobe_cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path
            ]
            ffprobe_result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, timeout=30)
            if ffprobe_result.returncode != 0:
                logger.warning(f"ffprobe failed: {ffprobe_result.stderr}")
            else:
                input_info = json.loads(ffprobe_result.stdout)
                logger.info(f"Input video info: {input_info}")
        except Exception as e:
            logger.warning(f"Error getting video info (ffprobe not available?): {e}")
            # Continue without metadata if ffprobe is not available

        # Generate output filename
        video_id = path.splitext(path.basename(video_path))[0]
        output_filename = f"{video_id}_reddit_safe.mp4"
        output_path = path.join(output_folder, output_filename)
        
        # Check if transcoded file already exists
        if path.exists(output_path):
            logger.info(f"Transcoded video already exists: {output_path}")
            report_progress({"status": "transcoding", "message": f"Using existing transcoded video: {output_path}"})
            
            # Get metadata for existing file
            output_info = {}
            try:
                ffprobe_cmd = [
                    "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", output_path
                ]
                ffprobe_result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, timeout=30)
                if ffprobe_result.returncode == 0:
                    output_info = json.loads(ffprobe_result.stdout)
                else:
                    logger.warning(f"ffprobe failed for existing file: {ffprobe_result.stderr}")
            except Exception as e:
                logger.warning(f"Could not get metadata for existing transcoded file: {e}")
                
            return {
                "transcoded_path": output_path,
                "logs": "Using existing transcoded file",
                "input_metadata": input_info,
                "output_metadata": output_info
            }

        # Check if ffmpeg is available
        ffmpeg_available = False
        try:
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
            if result.returncode == 0:
                ffmpeg_available = True
        except Exception:
            pass
        
        if not ffmpeg_available:
            logger.warning("FFmpeg not available, returning original video path")
            report_progress({"status": "transcoding", "message": "FFmpeg not available, using original video"})
            return {
                "transcoded_path": video_path,
                "logs": "FFmpeg not available, using original video",
                "input_metadata": input_info,
                "output_metadata": input_info
            }
        
        # Build ffmpeg command for transcoding
        report_progress({"status": "transcoding", "message": "Transcoding video to Reddit-safe format..."})
        logger.info(f"Transcoding {video_path} to {output_path}")
        
        # FFmpeg command to transcode to Reddit-safe format:
        # - Video: H.264, yuv420p, even dimensions, CRF 23, veryfast preset, +faststart
        # - Audio: AAC, 128k bitrate, 48000 Hz, stereo
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # Ensure even dimensions
            "-crf", "23",
            "-preset", "veryfast",
            "-movflags", "+faststart",
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", "48000",
            "-ac", "2",
            "-max_muxing_queue_size", "1024",
            output_path
        ]
        
        logger.info(f"Running ffmpeg command: {' '.join(ffmpeg_cmd)}")
        
        try:
            # Run ffmpeg and capture output
            logger.info("Starting ffmpeg process...")
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            logger.info("ffmpeg process started.")
            
            # Collect output in real-time
            logs = []
            logger.info("Starting to read ffmpeg output...")
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    logs.append(output.strip())
                    logger.debug(f"ffmpeg output: {output.strip()}")
            logger.info("Finished reading ffmpeg output.")
            
            return_code = process.poll()
            logger.info(f"ffmpeg process finished with return code: {return_code}")
            
            if return_code != 0:
                error_logs = "\n".join(logs)
                logger.error(f"FFmpeg transcoding failed with code {return_code}: {error_logs}")
                raise RuntimeError(f"FFmpeg transcoding failed: {error_logs}")
            
            success_logs = "\n".join(logs)
            logger.info(f"FFmpeg transcoding completed successfully: {success_logs}")
            report_progress({"status": "transcoding", "message": "Transcoding completed successfully"})
            
            # Get output video metadata
            output_info = {}
            try:
                ffprobe_cmd = [
                    "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", output_path
                ]
                ffprobe_result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, timeout=30)
                if ffprobe_result.returncode == 0:
                    output_info = json.loads(ffprobe_result.stdout)
                else:
                    logger.warning(f"ffprobe failed for transcoded file: {ffprobe_result.stderr}")
            except Exception as e:
                logger.warning(f"Could not get metadata for transcoded file: {e}")
            
            return {
                "transcoded_path": output_path,
                "logs": success_logs,
                "input_metadata": input_info,
                "output_metadata": output_info
            }
            
        except subprocess.TimeoutExpired as e:
            logger.error(f"FFmpeg transcoding timed out: {e}")
            raise RuntimeError(f"FFmpeg transcoding timed out: {e}") from e
        except Exception as e:
            logger.error(f"Error during transcoding: {e}")
            raise RuntimeError(f"Failed to transcode video: {e}") from e