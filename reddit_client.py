import logging
import time
from os import getenv
from typing import Optional
import asyncpraw  # type: ignore

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    logger = logging.getLogger(__name__)
    logger.info("Loaded .env file in reddit_client.py")
except Exception as e:
    logging.getLogger(__name__).warning(f"Could not load .env file in reddit_client.py: {e}")

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
        return cls._instance

    @property
    def is_read_only(self) -> bool:
        """Check if the client is in read-only mode."""
        return self.__class__._is_read_only

    @property
    def client(self) -> Optional[asyncpraw.Reddit]:
        """Get the Reddit client instance."""
        return self.__class__._client

    async def initialize_client(self) -> None:
        """Initialize the Reddit client with appropriate credentials."""
        current_time = time.time()
        # Only skip initialization if we have a valid authenticated client
        if self.__class__._client and not self.__class__._is_read_only and (current_time - self.__class__._last_init_time < self.__class__._init_cooldown):
            logger.info(f"Client initialization on cooldown. Waiting {self.__class__._init_cooldown} seconds between initializations.")
            return

        self.__class__._last_init_time = current_time

        client_id = getenv("REDDIT_CLIENT_ID")
        client_secret = getenv("REDDIT_CLIENT_SECRET")
        user_agent = getenv("REDDIT_USER_AGENT", "RedditMCPServer v1.0")
        username = getenv("REDDIT_USERNAME")
        password = getenv("REDDIT_PASSWORD")

        # Debug logging
        logger.info("Initializing Reddit client with credentials:")
        logger.info(f"  REDDIT_CLIENT_ID: {client_id}")
        logger.info(f"  REDDIT_CLIENT_SECRET: {client_secret}")
        logger.info(f"  REDDIT_USERNAME: {username}")
        logger.info(f"  REDDIT_PASSWORD: {password}")
        logger.info(f"  REDDIT_USER_AGENT: {user_agent}")

        self.__class__._is_read_only = True

        try:
            # Try authenticated access first if credentials are provided
            if all([username, password, client_id, client_secret]):
                logger.info(
                    f"Attempting to initialize Reddit client with user authentication for u/{username}"
                )
                try:
                    self.__class__._client = asyncpraw.Reddit(
                        client_id=client_id,
                        client_secret=client_secret,
                        user_agent=user_agent,
                        username=username,
                        password=password,
                        check_for_updates=False,
                    )
                    # Test authentication
                    user = await self.__class__._client.user.me()
                    if user is None:
                        raise ValueError(f"Failed to authenticate as u/{username}")

                    logger.info(f"Successfully authenticated as u/{username}")
                    self.__class__._is_read_only = False
                    return
                except Exception as auth_error:
                    logger.error(f"Authentication failed: {auth_error}", exc_info=True)
                    logger.error(f"Credentials used: client_id={client_id}, client_secret={client_secret}, username={username}, password={'*' * len(password)}, user_agent={user_agent}")
                    logger.error(f"Exception type: {type(auth_error)}")
                    logger.error(f"Exception args: {auth_error.args}")
                    # Check for common authentication errors
                    error_str = str(auth_error).lower()
                    if "invalid_grant" in error_str:
                        logger.error("Authentication failed: Invalid credentials or 2FA enabled. Check username/password or use app password.")
                    elif "unauthorized" in error_str:
                        logger.error("Authentication failed: Unauthorized. Check client_id and client_secret.")
                    elif "forbidden" in error_str:
                        logger.error("Authentication failed: Forbidden. Check Reddit app permissions.")
                    logger.info("Falling back to read-only access")

            # Fall back to read-only with client credentials
            if client_id and client_secret:
                logger.info("Initializing Reddit client with read-only access")
                self.__class__._client = asyncpraw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent,
                    check_for_updates=False,
                    read_only=True
                )
                return

            # Last resort: read-only without credentials
            logger.info(
                "Initializing Reddit client in read-only mode without credentials"
            )
            self.__class__._client = asyncpraw.Reddit(
                user_agent=user_agent,
                check_for_updates=False,
                read_only=True
            )
            # Test read-only access
            subreddit = await self.__class__._client.subreddit("popular")
            async for _ in subreddit.hot(limit=1):
                pass

        except Exception as e:
            logger.error(f"Error initializing Reddit client: {e}")
            self.__class__._client = None

    async def check_user_auth(self) -> bool:
        """Check if user authentication is valid."""
        if self.__class__._is_read_only:
            logger.warning("check_user_auth: Client is in read-only mode")
            return False
        if not self.__class__._client:
            logger.warning("check_user_auth: No client available")
            return False
        try:
            user = await self.__class__._client.user.me()
            if user is None:
                logger.warning("check_user_auth: user.me() returned None")
                return False
            logger.info(f"check_user_auth: Successfully authenticated as {user}")
            return True
        except Exception as e:
            logger.error(f"check_user_auth: Authentication check failed: {e}")
            # If authentication check fails, mark as read-only and try to re-authenticate
            self.__class__._is_read_only = True
            try:
                logger.info("check_user_auth: Attempting to re-authenticate...")
                await self.initialize_client()
                if not self.__class__._is_read_only:
                    logger.info("check_user_auth: Re-authentication successful")
                    return True
            except Exception as reauth_error:
                logger.error(f"check_user_auth: Re-authentication failed: {reauth_error}")
            return False

    async def force_reauth(self) -> None:
        """Force re-authentication by clearing the client and cooldown."""
        logger.info("Forcing re-authentication...")
        self.__class__._client = None
        self.__class__._is_read_only = True
        self.__class__._last_init_time = 0
        await self.initialize_client()
