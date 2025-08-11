# TikTok Reddit MCP Server

This server downloads a TikTok video and posts it to a specified subreddit, adding a comment with the original TikTok link.

## Setup

1.  **Install dependencies:**

    Make sure you have Python 3.8+ and pip installed. Then, run the following command in the `tiktok-reddit-mcp` directory to install the necessary packages:

    ```bash
    pip install praw fastmcp requests
    ```

2.  **Set up environment variables:**

    You need to create a `.env` file in the `tiktok-reddit-mcp` directory with the following content:

    ```
    REDDIT_CLIENT_ID=your_reddit_client_id
    REDDIT_CLIENT_SECRET=your_reddit_client_secret
    REDDIT_USER_AGENT=your_reddit_user_agent
    REDDIT_USERNAME=your_reddit_username
    REDDIT_PASSWORD=your_reddit_password
    RAPIDAPI_KEY=your_rapidapi_key
    ```

    Replace the placeholder values with your actual credentials. You can get Reddit API credentials by creating a new application on the [Reddit apps page](https://www.reddit.com/prefs/apps). For the TikTok downloader, I've used the [TikTok Download Video API on RapidAPI](https://rapidapi.com/postmaker/api/tiktok-download-video1). You will need to subscribe to this API to get your `RAPIDAPI_KEY`.

## Running the Server

To run the server, execute the following command in the `tiktok-reddit-mcp` directory:

```bash
python -m mcp.server.main --path server.py
```

The server will start, and you can interact with it using an MCP client. The available tool is `download_tiktok_video_and_post_to_reddit`, which takes the `tiktok_url`, `subreddit`, and `post_title` as arguments.
