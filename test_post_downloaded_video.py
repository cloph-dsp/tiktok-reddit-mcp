import asyncio
import sys
import os

# Add the tiktok-reddit-mcp directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from server import post_downloaded_video

class DummyContext:
    def __init__(self):
        self.messages = []
    
    def report_progress(self, data):
        self.messages.append(data)
        print(f"Progress (report_progress): {data}")

class DummyEventEmitterContext:
    def __init__(self):
        self.messages = []
    
    async def __event_emitter__(self, data):
        self.messages.append(data)
        print(f"Progress (__event_emitter__): {data}")

def test_post_downloaded_video_with_report_progress():
    ctx = DummyContext()
    
    # Test with a non-existent video file to trigger the error handling
    try:
        result = post_downloaded_video(
            ctx=ctx,
            video_path="non_existent_video.mp4",
            subreddit="test",
            title="Test Post"
        )
        print("Result:", result)
    except Exception as e:
        print(f"Error: {e}")
        print("Progress messages:", ctx.messages)

def test_post_downloaded_video_with_event_emitter():
    ctx = DummyEventEmitterContext()
    
    # Test with a non-existent video file to trigger the error handling
    try:
        result = post_downloaded_video(
            ctx=ctx,
            video_path="non_existent_video.mp4",
            subreddit="test",
            title="Test Post"
        )
        print("Result:", result)
    except Exception as e:
        print(f"Error: {e}")
        print("Progress messages:", ctx.messages)

if __name__ == "__main__":
    print("Testing with report_progress...")
    test_post_downloaded_video_with_report_progress()
    
    print("\nTesting with __event_emitter__...")
    test_post_downloaded_video_with_event_emitter()