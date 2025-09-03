
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

        output_template = path.join(download_folder, '%(id)s.%(ext)s')
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'format': 'best[ext=mp4]/best',
            'outtmpl': output_template,
            'writethumbnail': True,
            'progress_hooks': [progress_hook],  # Add progress hook
        }

        video_path = None
        video_id = None
        thumbnail_deleted = False
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = await asyncio.to_thread(ydl.extract_info, url, download=True)
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
        except yt_dlp.utils.DownloadError as e:
            logger.error(f"yt-dlp download error: {e}")
            raise RuntimeError(f"Failed to download TikTok video: {e}. This might be due to an invalid URL, geo-restrictions, or changes in TikTok's website. Please check the URL and try again.") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during download: {e}")
            raise VideoDownloadError(f"An unexpected error occurred during TikTok video download: {e}", original_exception=e) from e

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

        # Log completion message for LLM to monitor
        logger.info(f"Download completed successfully. Video ID: {video_id}")
        logger.info(f"Video available at: {video_path}")
        logger.info(f"Returning result: {result}")
        return result

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