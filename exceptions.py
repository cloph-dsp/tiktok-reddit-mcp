class RedditPostError(Exception):
    """Custom exception for Reddit post-related errors."""
    def __init__(self, message: str, original_exception: Exception = None, details: dict = None):
        super().__init__(message)
        self.original_exception = original_exception
        self.message = message
        self.details = details or {}

    def __str__(self):
        if self.original_exception:
            return f"{self.message} (Original error: {str(self.original_exception)})"
        return self.message
        
    def __repr__(self):
        return f"RedditPostError(message='{self.message}', details={self.details})"

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