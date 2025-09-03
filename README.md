# TikTok → Reddit Poster MCP

Minimal MCP server to:
1. Download TikTok videos (short or full URL) via yt-dlp (no external API key)
2. (Optional) Transcribe locally with faster‑whisper
3. (Optional) Suggest target subreddits & sample titles
4. Post as native Reddit video with source comment

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
- ffmpeg on PATH
- (Optional) CUDA + cuDNN for GPU transcription (CPU works)

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
| Issue | Fix |
|-------|-----|
| Write denied | Add all Reddit env vars. |
| "Video submission failed and post could not be found. The video upload likely failed." | Check Reddit API credentials in .env file. Ensure REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME, and REDDIT_PASSWORD are correctly set. |
| cuDNN / CUDA errors | Set `CT2_FORCE_CPU=1` or install GPU stack. |
| Slow transcript | Use smaller model. |
| Missing video via video_id | Confirm file name in download folder. |
| "Required configuration setting 'client_id' missing" | Copy .env.example to .env and fill in your Reddit API credentials. |

---
## License
MIT

## Disclaimer
Ensure you have rights to download & repost. Respect subreddit rules.
