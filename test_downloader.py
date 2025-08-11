import os
import sys
import time
import argparse
import yt_dlp

# We avoid importing server.py by default so Reddit auth is not triggered.
# If you explicitly pass --use-server, we will lazy-load the server transcription tool.

def download_tiktok(tiktok_url: str, download_folder: str = "downloaded") -> dict:
    """Download TikTok video (+thumbnail) and return metadata dict."""
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    output_template = os.path.join(download_folder, '%(id)s.%(ext)s')
    ydl_opts = {
        'quiet': False,
        'no_warnings': True,
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_template,
        'writethumbnail': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(tiktok_url, download=True)
        video_id = info.get('id')
        resolved_url = info.get('webpage_url', tiktok_url)
        video_path = ydl.prepare_filename(info)
        base_filename, _ = os.path.splitext(video_path)
        thumb = None
        for ext in ('.jpg', '.webp', '.png', '.image'):
            candidate = base_filename + ext
            if os.path.exists(candidate):
                thumb = candidate
                break
    return {
        'video_id': video_id,
        'resolved_url': resolved_url,
        'video_path': video_path,
        'thumbnail_path': thumb,
    }


def whisper_transcribe(video_path: str, model_size: str, force_cpu: bool = False):
    """Direct faster-whisper transcription with optional GPU attempt.
    Falls back to CPU automatically on cuDNN / CUDA errors.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as e:  # pragma: no cover
        return { 'status': 'error', 'message': f'faster-whisper not available: {e}' }

    start = time.time()
    attempted_gpu = False

    def _run(device: str, compute_type: str):
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        seg_iter, info = model.transcribe(video_path, beam_size=1)
        segments = []
        texts = []
        for s in seg_iter:
            seg_text = s.text.strip()
            segments.append({'id': s.id, 'start': s.start, 'end': s.end, 'text': seg_text})
            texts.append(seg_text)
        return {
            'status': 'success',
            'engine': f'direct-faster-whisper-{device}',
            'model': model_size,
            'language': info.language,
            'duration': info.duration,
            'segments': segments,
            'transcript': ' '.join(texts),
            'elapsed_sec': round(time.time() - start, 2),
        }

    # Decide strategy
    if force_cpu or os.getenv('CT2_FORCE_CPU') == '1':
        try:
            return _run('cpu', 'int8')
        except Exception as e:
            return {'status': 'error', 'message': f'CPU transcription failed: {e}'}

    # Try auto (GPU) first
    try:
        attempted_gpu = True
        return _run('auto', 'auto')
    except Exception as e:
        err_msg = str(e)
        if 'cudnn' in err_msg.lower() or 'cuda' in err_msg.lower() or 'cublas' in err_msg.lower():
            # Fallback to CPU
            try:
                return _run('cpu', 'int8')
            except Exception as e2:
                return {'status': 'error', 'message': f'GPU failed ({err_msg}); CPU fallback failed: {e2}'}
        # Non-GPU related failure
        return {'status': 'error', 'message': f'Transcription failed (auto device): {err_msg}'}


def server_transcribe(video_path: str, model_size: str):  # type: ignore
    """Lazy import server.transcribe_video only if requested."""
    try:
        import importlib
        server_mod = importlib.import_module('server')
        if getattr(server_mod, 'USE_WHISPER', False):
            return server_mod.transcribe_video(video_path, model_size=model_size)  # type: ignore
        return { 'status': 'disabled', 'message': 'Server whisper disabled (USE_WHISPER_TRANSCRIPTION not true).' }
    except Exception as e:
        return { 'status': 'error', 'message': f'Could not use server transcription: {e}' }


def main():
    parser = argparse.ArgumentParser(description="Test TikTok download and optional transcription without needing Reddit creds")
    parser.add_argument('url', help='TikTok URL')
    parser.add_argument('--folder', default='downloaded', help='Download folder (default: downloaded)')
    parser.add_argument('--whisper', nargs='?', const='small', help='Run faster-whisper transcription (optional model size; default small)')
    parser.add_argument('--use-server', action='store_true', help='Use server.transcribe_video (if Whisper enabled there)')
    parser.add_argument('--gpu', action='store_true', help='Attempt GPU (auto) first; fallback to CPU on failure')
    parser.add_argument('--cpu', action='store_true', help='Force CPU only (overrides --gpu)')
    args = parser.parse_args()

    if args.cpu:
        os.environ['CT2_FORCE_CPU'] = '1'

    print(f"Downloading: {args.url}")
    try:
        dl = download_tiktok(args.url, args.folder)
    except yt_dlp.utils.DownloadError as e:
        print(f"Download error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected download error: {e}")
        sys.exit(1)

    print("\n--- Download Result ---")
    for k, v in dl.items():
        print(f"{k}: {v}")
    if not os.path.exists(dl['video_path']):
        print("Video file missing!")
        sys.exit(1)

    if args.use_server or args.whisper:
        model_size = (args.whisper if args.whisper not in (True, None) else 'small') if args.whisper else 'small'
        mode_label = 'server' if args.use_server else 'local'
        if args.cpu:
            mode_label += '-cpu'
        elif args.gpu:
            mode_label += '-gpu'
        print(f"\nTranscribing with model '{model_size}' ({mode_label})...")
        if args.use_server:
            result = server_transcribe(dl['video_path'], model_size=model_size)  # type: ignore
            if result.get('status') != 'success' and args.whisper:
                print("Server transcription not successful, falling back to local.")
                result = whisper_transcribe(dl['video_path'], model_size, force_cpu=args.cpu and not args.gpu)
        else:
            # force_cpu True unless gpu flag set (and cpu not forced)
            force_cpu = args.cpu or not args.gpu
            result = whisper_transcribe(dl['video_path'], model_size, force_cpu=force_cpu)
        print("\n--- Transcription Result ---")
        if result.get('status') == 'success':
            print(f"Engine: {result.get('engine','?')}  Language: {result.get('language')}  Duration: {result.get('duration')}s")
            print(f"Transcript (first 500 chars):\n{result.get('transcript','')[:500]} ")
        else:
            print(f"Failed: {result.get('message')}")

if __name__ == '__main__':
    main()
