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
## Environment
Copy the example and edit values:
```bash
cp .env.example .env
```
The file is self‑documenting. Without Reddit creds the server runs read‑only (download & list only).

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
| cuDNN / CUDA errors | Set `CT2_FORCE_CPU=1` or install GPU stack. |
| Slow transcript | Use smaller model. |
| Missing video via video_id | Confirm file name in download folder. |

---
## License
MIT

## Disclaimer
Ensure you have rights to download & repost. Respect subreddit rules.
