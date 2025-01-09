# BIM-TLs
This is a tool to retrieve videos from Revit videos playlist and search them using natural language queries.

## How it works
The tool uses a ChromaDB to store the video titles and their corresponding links. It also uses the YouTube Data API to retrieve the video links from a playlist.

## How to use
You can use this tool through Docker.

### Set up environment variables
Create a file called .env and add the following:
```
YOUTUBE_DATA_API_KEY=<YT_API_KEY> # You can get this from https://console.cloud.google.com/apis/library/youtube.googleapis.com
SECRET_KEY=<FLASK_SECRET_KEY>
API_KEY=<YOUR_CUSTOM_API_KEY>
PLAYLIST_ID=<YOUR_PLAYLIST_ID>
```

### Docker

1. Pull the image from Docker Hub
```
docker pull brandon596/revitvids
```

2. Run the container
```
docker run -d -p 5000:8000 --name revitvids -e YOUTUBE_DATA_API_KEY=<YT_API_KEY> -e SECRET_KEY=<FLASK_SECRET_KEY> -e API_KEY=<YOUR_CUSTOM_API_KEY> -e PLAYLIST_ID=<YOUR_PLAYLIST_ID> brandon596/revitvids
```

3. Open your browser and go to http://localhost:5000