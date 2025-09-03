import logging
import time
from os import getenv
from typing import Optional
import praw  # type: ignore

logger = logging.getLogger(__name__)


class RedditClientManager:
    """Manages the Reddit client and its state."""

    _instance = None
    _client = None
    _is_read_only = True
    _last_init_time = 0
    _init_cooldown = 60  # Cooldown period in seconds

    def __new__(cls) -> "RedditClientManager":
        if cls._instance is None:
            cls._instance = super(RedditClientManager, cls).__new__(cls)
            cls._instance._initialize_client()
        return cls._instance

    def _initialize_client(self) -> None:
        """Initialize the Reddit client with appropriate credentials."""
        current_time = time.time()
        if current_time - self._last_init_time < self._init_cooldown:
            logger.info(f"Client initialization on cooldown. Waiting {self._init_cooldown} seconds between initializations.")
            return
            
        self._last_init_time = current_time
        
        client_id = getenv("REDDIT_CLIENT_ID")
        client_secret = getenv("REDDIT_CLIENT_SECRET")
        user_agent = getenv("REDDIT_USER_AGENT", "RedditMCPServer v1.0")
        username = getenv("REDDIT_USERNAME")
        password = getenv("REDDIT_PASSWORD")

        self._is_read_only = True

        try:
            # Try authenticated access first if credentials are provided
            if all([username, password, client_id, client_secret]):
                logger.info(
                    f"Attempting to initialize Reddit client with user authentication for u/{username}"
                )
                try:
                    self._client = praw.Reddit(
                        client_id=client_id,
                        client_secret=client_secret,
                        user_agent=user_agent,
                        username=username,
                        password=password,
                        check_for_updates=False,
                    )
                    # Test authentication
                    if self._client.user.me() is None:
                        raise ValueError(f"Failed to authenticate as u/{username}")

                    logger.info(f"Successfully authenticated as u/{username}")
                    self._is_read_only = False
                    return
                except Exception as auth_error:
                    logger.warning(f"Authentication failed: {auth_error}")
                    logger.info("Falling back to read-only access")

            # Fall back to read-only with client credentials
            if client_id and client_secret:
                logger.info("Initializing Reddit client with read-only access")
                self._client = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent,
                    check_for_updates=False,
                    read_only=True,
                )
                return

            # Last resort: read-only without credentials
            logger.info(
                "Initializing Reddit client in read-only mode without credentials"
            )
            self._client = praw.Reddit(
                user_agent=user_agent,
                check_for_updates=False,
                read_only=True,
            )
            # Test read-only access
            self._client.subreddit("popular").hot(limit=1)

        except Exception as e:
            logger.error(f"Error initializing Reddit client: {e}")
            self._client = None

    def refresh_client(self) -> None:
        """Force a refresh of the Reddit client."""
        logger.info("Refreshing Reddit client.")
        self._initialize_client()

    @property
    def client(self) -> Optional[praw.Reddit]:
        """Get the Reddit client instance."""
        return self._client

    @property
    def is_read_only(self) -> bool:
        """Check if the client is in read-only mode."""
        return self._is_read_only

    def check_user_auth(self) -> bool:
        """Check if user authentication is available for write operations."""
        if not self._client:
            logger.error("Reddit client not initialized")
            return False
        if self._is_read_only:
            logger.error("Reddit client is in read-only mode")
            return False

        # Additional authentication check
        try:
            user = self._client.user.me()
            if user is None:
                logger.error("Reddit client user authentication failed - user.me() returned None")
                return False
            logger.info(f"Reddit client authenticated as: {user}")
            return True
        except Exception as e:
            logger.error(f"Reddit client authentication check failed: {e}")
            return False