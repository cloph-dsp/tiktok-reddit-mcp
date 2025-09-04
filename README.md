# TikTok → Reddit Poster MCP

**Enhanced with robust video processing and WebSocket error recovery**

Minimal MCP server to:
1. Download TikTok videos (short or full URL) via yt-dlp (no external API key)
2. (Optional) Transcribe locally with faster‑whisper
3. (Optional) Suggest target subreddits & sample titles
4. Post as native Reddit video with source comment

## 🚀 New Features (Latest Update)
- **Automatic FFmpeg Detection**: Works on Windows, Linux, and macOS
- **WebSocket Error Recovery**: Automatically finds posts created despite WebSocket failures
- **Video Corruption Detection**: Detects and repairs corrupted video files
- **Forced Transcoding**: All videos converted to Reddit-safe MP4 format
- **Enhanced Validation**: Comprehensive codec and format checking
- **Cross-Platform Support**: Seamless operation on Windows and Linux

---
## Core Tools
| Tool | Summary |
|------|---------|
| `download_tiktok_video` | Download video (thumbnail discarded) & resolve short links. |
| `transcribe_video` | Local Whisper transcript (enable via env flag). |
| `suggest_subreddits` | Rank candidate subreddits & return 5+5 top titles (two time frames) + LLM context. |
| `post_downloaded_video` | Upload local video (by path or video_id) to Reddit; auto delete after posting. |

Recommended flow: download → (transcribe) → (suggest_subreddits) → post.

---
## Requirements
- Python ≥ 3.8
- **FFmpeg** (automatically detected on Windows/Linux/macOS)
- (Optional) CUDA + cuDNN for GPU transcription (CPU works)

### FFmpeg Installation

#### 🪟 Windows
```bash
# Option 1: Using Chocolatey (recommended)
choco install ffmpeg

# Option 2: Using Scoop
scoop install ffmpeg

# Option 3: Manual installation
# Download from https://ffmpeg.org/download.html
# Extract to C:\ffmpeg\ or E:\ffmpeg\
# Add bin folder to PATH
```

#### 🐧 Linux
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# CentOS/RHEL
sudo yum install ffmpeg

# Arch Linux
sudo pacman -S ffmpeg

# Or run our installer:
python install_ffmpeg.py
```

#### 🍎 macOS
```bash
# Using Homebrew
brew install ffmpeg
```

**Test FFmpeg installation:**
```bash
python -c "from video_service import VideoService; print('FFmpeg paths:', VideoService()._get_ffmpeg_paths())"
```

---
## Install
```bash
pip install -e .
```
(Use a virtual environment if desired.)

---
## Environment Setup
Copy the example and edit values:
```bash
cp .env.example .env
```

### Reddit API Credentials Setup
To enable video posting and other write operations, you need to set up Reddit API credentials:

1. **Create a Reddit App:**
   - Go to https://www.reddit.com/prefs/apps
   - Click "Create App" or "Create Another App"
   - Choose "script" as the app type
   - Fill in the name, description, and redirect URI (can be `http://localhost:8080` for local development)

2. **Get Your Credentials:**
   - `REDDIT_CLIENT_ID`: The 14-character string under your app name
   - `REDDIT_CLIENT_SECRET`: The "secret" field
   - `REDDIT_USERNAME`: Your Reddit username
   - `REDDIT_PASSWORD`: Your Reddit password

3. **Update .env file:**
   ```bash
   REDDIT_CLIENT_ID=your_14_char_client_id
   REDDIT_CLIENT_SECRET=your_secret_here
   REDDIT_USERNAME=your_username
   REDDIT_PASSWORD=your_password
   ```

Without Reddit credentials, the server runs in read-only mode (download & suggest subreddits only).

---
## Run
```bash
python server.py          # default :8050
python server.py --port 9001
```
Docs: http://localhost:<port>/docs  (Authorize with `Bearer <MCPO_API_KEY>` if set.)


---

## Auto Comment Logic
`post_downloaded_video` generates a comment if:
- `comment` is omitted AND `original_url` provided AND `auto_comment` (default true)
- Language mapping:
  - en   → `Original link: <url>`
  - pt   → `Link original: <url>`
  - both → `Original link / link original: <url>` (default)

Provide `comment` explicitly to override.

---
## Transcription Notes
- Enable: `USE_WHISPER_TRANSCRIPTION=true`
- Model cached on first run
- CPU fallback; force CPU via `CT2_FORCE_CPU=1`

---
## Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **FFmpeg not found** | Run `python install_ffmpeg.py` for automatic installation instructions, or manually install FFmpeg for your platform. |
| **WebSocket errors** | The system now automatically recovers from WebSocket errors. Check logs for "Found post despite WebSocket error" messages. |
| **Video corruption** | System automatically detects and attempts to repair corrupted videos using FFmpeg. |
| **Write access denied** | Add all Reddit environment variables: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USERNAME`, `REDDIT_PASSWORD`. |
| **Video upload failed** | Check Reddit API credentials in .env file. Ensure all four Reddit credentials are correctly set. |
| **cuDNN / CUDA errors** | Set `CT2_FORCE_CPU=1` or install GPU stack for transcription. |
| **Slow transcription** | Use smaller Whisper model or set `CT2_FORCE_CPU=1` for CPU-only processing. |
| **Missing video file** | Confirm file exists in download folder and matches the video_id format. |
| **Reddit API errors** | Check Reddit status at https://www.redditstatus.com/ for outages. |
| **"client_id missing"** | Copy `.env.example` to `.env` and fill in your Reddit API credentials. |

### FFmpeg Detection Issues

**Test FFmpeg detection:**
```bash
python -c "from video_service import VideoService; vs = VideoService(); print('FFmpeg paths:', vs._get_ffmpeg_paths())"
```

**Expected output:**
```
FFmpeg paths: ('C:\\ffmpeg\\bin\\ffmpeg.exe', 'C:\\ffmpeg\\bin\\ffprobe.exe')  # Windows
FFmpeg paths: ('/usr/bin/ffmpeg', '/usr/bin/ffprobe')  # Linux
```

### WebSocket Recovery

The system now automatically handles WebSocket errors by:
1. Detecting WebSocket connection failures
2. Searching for posts that may have been created despite the error
3. Using multiple search strategies (title, truncated title, user posts)
4. Continuing processing with the found post

**Check logs for recovery messages:**
```
INFO: WebSocket failed but post might have been created - searching...
INFO: Found post despite WebSocket error: https://reddit.com/r/subreddit/comments/xxx/
```

### Video Processing Pipeline

**Enhanced validation includes:**
- File size limits (Reddit's 1GB limit)
- Codec compatibility (H.264, H.265, AAC, MP3, Opus)
- Resolution checking (up to 1080p)
- Corruption detection and automatic repair
- Forced transcoding to Reddit-safe format

**Monitor processing logs:**
```bash
tail -f /var/log/your-app.log | grep -E "(transcoding|validation|WebSocket|corruption)"
```

---
## License
MIT

## Disclaimer
Ensure you have rights to download & repost. Respect subreddit rules.
