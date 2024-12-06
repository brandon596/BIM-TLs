import requests

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
        raw_vid_items += response["items"]
        nextPageToken = response.get('nextPageToken')
        if not nextPageToken:
            break
        params['pageToken'] = nextPageToken
    return raw_vid_items

def process_yt_playlist(raw_vid_items: list, id_start: int):
    return [{"Id": idx+id_start, "Title": item["snippet"]["title"], "Description": item["snippet"]["description"], "Video URL": "https://youtu.be/" + item["snippet"]["resourceId"]["videoId"], "Page URL": "https://www.youtube.com/watch?v=" + item["snippet"]["resourceId"]["videoId"]} for idx, item in enumerate(raw_vid_items)]

def get_processed_playlist(playlist_id: str, api_key: str, id_start: int=24):
    return process_yt_playlist(get_yt_playlist(playlist_id, api_key), id_start)
