import requests
from logger import logger
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry  # type: ignore

#video URL format = https://youtu.be/FHRnQvC_dwY


def get_yt_playlist(playlist_id: str, api_key: str):
    URL = 'https://www.googleapis.com/youtube/v3/playlistItems'
    params = {
        'part': ','.join(['snippet']),
        'playlistId': playlist_id,
        'maxResults': 50,
        'key': api_key,
    }

    # Set up retry mechanism
    session = requests.Session()
    retries = Retry(
        total=3,  # Total number of retries
        backoff_factor=1,  # Delay between retries (e.g., 1s, 2s, 4s, etc.)
        status_forcelist=[500, 502, 503, 504, 400, 401, 402, 403],  # Retry on these HTTP status codes
        allowed_methods=["GET"],  # Only retry on GET requests
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))

    raw_vid_items = []

    while True:
        try:
            response = session.get(URL, params=params, timeout=10)  # Add timeout
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()

            if 'error' in data:
                raise Exception(data["error"])

            raw_vid_items += data["items"]
            nextPageToken = data.get('nextPageToken')
            if not nextPageToken:
                break
            params['pageToken'] = nextPageToken

        except requests.exceptions.RequestException as e:
            logger.error(f"YT API Error", extra={"error": str(e)})
            raise Exception(f"Request failed: {e}")

    return raw_vid_items

def process_yt_playlist(raw_vid_items: list, id_start: int):
    return [{"Id": idx+id_start, "Title": item["snippet"]["title"], "Description": item["snippet"]["description"], "Video_URL": "https://youtu.be/" + item["snippet"]["resourceId"]["videoId"], "Page_URL": "https://www.youtube.com/watch?v=" + item["snippet"]["resourceId"]["videoId"]} for idx, item in enumerate(raw_vid_items)]

def get_processed_playlist(playlist_id: str, api_key: str, id_start:int):
    raw_vid_items = get_yt_playlist(playlist_id, api_key)
    if raw_vid_items:
        return process_yt_playlist(raw_vid_items, id_start)
    else:
        return False
