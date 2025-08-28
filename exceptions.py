class RedditPostError(Exception):
    """Custom exception for Reddit post-related errors."""
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception

class VideoDownloadError(Exception):
    """Custom exception for video download-related errors."""
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception

class TranscriptionError(Exception):
    """Custom exception for transcription-related errors."""
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception