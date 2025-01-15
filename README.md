# BIM-TLs

This is a tool to retrieve videos from Revit videos playlist and search them using natural language queries and serves a frontend using WebChat for the chat interface. The chatbot must be developed using Microsoft's Bot Framework SDK.

## Features

The tool uses a ChromaDB to store the video titles and their corresponding links. It also uses the YouTube Data API to retrieve the video links from a playlist.

## How to use

You can use this tool through Docker or any similar software.

### Set up environment variables

Create a file called .env and add the following:
```
YOUTUBE_DATA_API_KEY=<YT_API_KEY> # You can get this from https://console.cloud.google.com/apis/library/youtube.googleapis.com
SECRET_KEY=<FLASK_SECRET_KEY>
API_KEY=<YOUR_CUSTOM_API_KEY>
PLAYLIST_ID=<YOUR_PLAYLIST_ID>
DIRECT_LINE_SECRET=<DIRECTLINE_SECRET_KEY>
```

### Docker

1. Pull the image from Docker Hub
```
docker pull brandon596/bim-chat
```

2. Run the container
```
docker run -d -p 5000:8000 --name revit-chat-tools -e YOUTUBE_DATA_API_KEY=<YT_API_KEY> -e SECRET_KEY=<FLASK_SECRET_KEY> -e API_KEY=<YOUR_CUSTOM_API_KEY> -e PLAYLIST_ID=<YOUR_PLAYLIST_ID> -e DIRECT_LINE_SECRET=<DIRECTLINE_SECRET_KEY> --mount type=volume,src=revit-chat-data,dst=/app/persistent brandon596/bim-chat
```
Or using a .env file
```
docker run -d -p 5000:8000 --env-file .env --name revit-chat-tools --mount type=volume,src=revit-chat-data,dst=/app/persistent brandon596/bim-chat
```

3. Open your browser and go to http://localhost:5000


Default Username: firstlogin
Default Password: aVeryStrongAndSecurePassword