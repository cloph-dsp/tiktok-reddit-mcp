# TikTok → Reddit Poster MCP

A minimal MCP server for downloading TikTok videos, optional transcription, subreddit & title suggestions, and posting native Reddit videos with auto-comment for source links.

## Features
- **FFmpeg Auto-Detection:** Works on Windows, Linux, and macOS.
- **Robust Error Recovery:** Multi-layer fallback for WebSocket and API errors.
- **Video Validation & Transcoding:** Converts videos to Reddit-compatible MP4.
- **Auto-Commenting:** Automatically comments the original TikTok link if none is provided.

---
## Requirements
- Python ≥ 3.8
- **FFmpeg** (auto-detected; install if missing)
- (Optional) CUDA + cuDNN for GPU transcription

---
## Installation

### Python Dependencies

```bash
pip install praw requests yt-dlp websockets aiohttp aiofiles asyncpraw
# Optional transcription support
pip install faster-whisper
```

### System Dependencies

- **Windows:**
  - Chocolatey: `choco install ffmpeg`
  - Scoop: `scoop install ffmpeg`
  - Or download manually from [ffmpeg.org](https://ffmpeg.org/download.html)
- **Linux (Debian/Ubuntu):** `sudo apt update && sudo apt install ffmpeg python3-dev`
- **macOS:** `brew install ffmpeg`

### Package Installation

```bash
pip install -e .
```

### Environment Setup

Copy the example file and update your credentials:

```bash
cp .env.example .env
```

Fill in your Reddit API credentials (`REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USERNAME`, `REDDIT_PASSWORD`) in the `.env` file. Without these, the server functions in read-only mode.

---
## Usage

Start the MCP server:

```bash
python server.py          # default port 8050
python server.py --port 9001
```

Access API docs at: `http://localhost:<port>/docs` (use `Bearer <MCPO_API_KEY>` if set).

---

## Auto-Comment Logic

The `post_downloaded_video` function auto-generates a comment if none is provided and an original TikTok URL exists:
- **en:** `Original link: <url>`
- **pt:** `Link original: <url>`
- **both:** `Original link / link original: <url>`

---
## Troubleshooting

- **FFmpeg Not Found:** Run `python install_ffmpeg.py` or install manually.
- **Reddit API Errors:** Ensure your `.env` file is correctly set up.
- **WebSocket Errors:** The system auto-recovers and retries; check logs for "WebSocket failed" messages.
- **Fallback:** If PRAW fails, a direct Reddit API call is attempted automatically.

---
