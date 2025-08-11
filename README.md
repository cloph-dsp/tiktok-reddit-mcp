# TikTok Reddit MCP Server

This server downloads a TikTok video and posts it to a specified subreddit, adding a comment with the original TikTok link.

---

## 🚀 Getting Started

### Prerequisites

-   Python 3.8+
-   pip

### ⚙️ Setup

1.  **Install Dependencies**

    It is recommended to use a virtual environment. Once your environment is set up, install the required packages from the project's root directory:

    ```bash
    pip install -e .
    ```

2.  **Configure Environment Variables**

    Create a `.env` file by copying the example file and filling in your credentials.

    ```bash
    cp .env.example .env
    ```

    You can get Reddit API credentials by creating a new application on the [Reddit apps page](https://www.reddit.com/prefs/apps). For the TikTok downloader, you'll need an API key from [RapidAPI](https://rapidapi.com/postmaker/api/tiktok-download-video1).

---

## ▶️ Running the Server

To run the server, execute the following command in the project's root directory:

```bash
python server.py
```

By default, the server will run on port `8050`. You can specify a different port using the `--port` argument:

```bash
python server.py --port 9000
```

The server will automatically use the `MCPO_API_KEY` from your `.env` file to secure the endpoint.

Once running, you can access the interactive OpenAPI documentation for the available tools at `http://localhost:<port>/docs` (e.g., `http://localhost:8050/docs`).

---

## 🧪 Manual Testing

You can test the server manually in two ways:

### 1. Using the Interactive API Documentation

1.  Start the server as described above.
2.  Open your web browser and navigate to `http://localhost:<port>/docs` (e.g., `http://localhost:8050/docs`).
3.  You will see the Swagger UI for your API. Click on the `POST /tools/download_tiktok_video_and_post_to_reddit/run` endpoint to expand it.
4.  If you've set an `MCPO_API_KEY`, click the "Authorize" button and enter your key in the format `Bearer <your-key>`.
5.  Click the "Try it out" button.
6.  Fill in the `requestBody` with the required parameters (`tiktok_url`, `subreddit`, `post_title`).
7.  Click the "Execute" button to send the request.

### 2. Using cURL

You can also send a request from your terminal using `curl`.

```bash
curl -X POST "http://localhost:8050/tools/download_tiktok_video_and_post_to_reddit/run" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer your-secret-api-key" \
-d '{
  "tiktok_url": "https://www.tiktok.com/@user/video/12345",
  "subreddit": "test",
  "post_title": "Check out this cool TikTok!",
  "nsfw": false,
  "spoiler": false
}'
```

*Replace the placeholder values with your actual data and API key.*

---

## 🛠️ Available Tool

### `download_tiktok_video_and_post_to_reddit`

Downloads a TikTok video, posts it to Reddit, and adds a comment with the original URL.

**Arguments:**

| Argument          | Type   | Default      | Description                                                              |
| ----------------- | ------ | ------------ | ------------------------------------------------------------------------ |
| `tiktok_url`      | `str`  | **Required** | The URL of the TikTok video (short or long).                             |
| `subreddit`       | `str`  | **Required** | The subreddit to post the video to.                                      |
| `post_title`      | `str`  | **Required** | The title for the Reddit post.                                           |
| `download_folder` | `str`  | `"downloaded"` | The local folder to temporarily store the video.                         |
| `nsfw`            | `bool` | `False`      | Whether the Reddit post should be marked as NSFW.                        |
| `spoiler`         | `bool` | `False`      | Whether the Reddit post should be marked as a spoiler.                   |
