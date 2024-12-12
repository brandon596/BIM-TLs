import requests
from logger import logger

#video URL format = https://youtu.be/FHRnQvC_dwY


def get_yt_playlist(playlist_id: str, api_key: str):
    URL = 'https://www.googleapis.com/youtube/v3/playlistItems'
    params = {
        'part': ','.join(['snippet']),
        'playlistId': playlist_id,
        'maxResults' : 50,
        'key': api_key,
    }

    raw_vid_items = []

    while True:
        response = requests.get(URL, params).json()
        if not response.get('error'):
            raw_vid_items += response["items"]
            nextPageToken = response.get('nextPageToken')
            if not nextPageToken:
                break
            params['pageToken'] = nextPageToken
        else:
            logger.error(response["error"]["message"])
            raise Exception(response["error"]["message"])
    return raw_vid_items

def process_yt_playlist(raw_vid_items: list, id_start: int):
    return [{"Id": idx+id_start, "Title": item["snippet"]["title"], "Description": item["snippet"]["description"], "Video_URL": "https://youtu.be/" + item["snippet"]["resourceId"]["videoId"], "Page_URL": "https://www.youtube.com/watch?v=" + item["snippet"]["resourceId"]["videoId"]} for idx, item in enumerate(raw_vid_items)]

def get_processed_playlist(playlist_id: str, api_key: str, id_start: int=24):
    raw_vid_items = get_yt_playlist(playlist_id, api_key)
    if raw_vid_items:
        return process_yt_playlist(raw_vid_items, id_start)
    else:
        return False
