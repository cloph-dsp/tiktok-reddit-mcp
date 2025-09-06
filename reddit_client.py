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
            # Debug logging
            logger.info("Initializing Reddit client with credentials:")
            logger.info(f"  REDDIT_CLIENT_ID: {client_id}")
            logger.info(f"  REDDIT_CLIENT_SECRET: {client_secret}")
            logger.info(f"  REDDIT_USERNAME: {username}")
            logger.info(f"  REDDIT_PASSWORD: {password}")
            logger.info(f"  REDDIT_USER_AGENT: {user_agent}")

            self._is_read_only = True

            try:
                # Try authenticated access first if credentials are provided
                if all([username, password, client_id, client_secret]):
                    logger.info(
                        f"Attempting to initialize Reddit client with user authentication for u/{username}"
                    )
                    try:
                        self._client = asyncpraw.Reddit(
                            client_id=client_id,
                            client_secret=client_secret,
                            user_agent=user_agent,
                            username=username,
                            password=password,
                            check_for_updates=False,
                        )
                        # Test authentication
                        user = await self._client.user.me()
                        if user is None:
                            raise ValueError(f"Failed to authenticate as u/{username}")

                        logger.info(f"Successfully authenticated as u/{username}")
                        self._is_read_only = False
                        return
                    except Exception as auth_error:
                        logger.error(f"Authentication failed: {auth_error}", exc_info=True)
                        logger.error(f"Credentials used: client_id={client_id}, client_secret={client_secret}, username={username}, password={password}, user_agent={user_agent}")
                        logger.info("Falling back to read-only access")

                # Fall back to read-only with client credentials
                if client_id and client_secret:
                    logger.info("Initializing Reddit client with read-only access")
                    self._client = asyncpraw.Reddit(
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
                self._client = asyncpraw.Reddit(
                    user_agent=user_agent,
                    check_for_updates=False,
                    read_only=True,
                )
                # Test read-only access
                subreddit = await self._client.subreddit("popular")
                async for _ in subreddit.hot(limit=1):
                    pass

            except Exception as e:
                logger.error(f"Error initializing Reddit client: {e}")
                self._client = None
                    read_only=True,
                )
                return

            # Last resort: read-only without credentials
            logger.info(
                "Initializing Reddit client in read-only mode without credentials"
            )
            self._client = asyncpraw.Reddit(
                user_agent=user_agent,
                check_for_updates=False,
                read_only=True,
            )
            # Test read-only access
            subreddit = await self._client.subreddit("popular")
            async def initialize_client(self) -> None:
                """Initialize the Reddit client with appropriate credentials."""
                current_time = time.time()
                if self._client and (current_time - self._last_init_time < self._init_cooldown):
                    logger.info(f"Client initialization on cooldown. Waiting {self._init_cooldown} seconds between initializations.")
                    return

                self._last_init_time = current_time

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

                self._is_read_only = True

                try:
                    # Try authenticated access first if credentials are provided
                    if all([username, password, client_id, client_secret]):
                        logger.info(
                            f"Attempting to initialize Reddit client with user authentication for u/{username}"
                        )
                        try:
                            self._client = asyncpraw.Reddit(
                                client_id=client_id,
                                client_secret=client_secret,
                                user_agent=user_agent,
                                username=username,
                                password=password,
                                check_for_updates=False,
                            )
                            # Test authentication
                            user = await self._client.user.me()
                            if user is None:
                                raise ValueError(f"Failed to authenticate as u/{username}")

                            logger.info(f"Successfully authenticated as u/{username}")
                            self._is_read_only = False
                            return
                        except Exception as auth_error:
                            logger.error(f"Authentication failed: {auth_error}", exc_info=True)
                            logger.error(f"Credentials used: client_id={client_id}, client_secret={client_secret}, username={username}, password={password}, user_agent={user_agent}")
                            logger.info("Falling back to read-only access")

                    # Fall back to read-only with client credentials
                    if client_id and client_secret:
                        logger.info("Initializing Reddit client with read-only access")
                        self._client = asyncpraw.Reddit(
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
                    self._client = asyncpraw.Reddit(
                        user_agent=user_agent,
                        check_for_updates=False,
                        read_only=True,
                    )
                    # Test read-only access
                    subreddit = await self._client.subreddit("popular")
                    async for _ in subreddit.hot(limit=1):
                        pass

                except Exception as e:
                    logger.error(f"Error initializing Reddit client: {e}")
            async for _ in subreddit.hot(limit=1):
                pass

        except Exception as e:
            logger.error(f"Error initializing Reddit client: {e}")
