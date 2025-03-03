﻿# BIM-Chatbot

This is a web application to retrieve videos from Revit videos playlist and search them using natural language queries and serves a frontend using [microsoft/BotFramework-WebChat](https://github.com/microsoft/BotFramework-WebChat) for the chat interface. The chatbot must be compatible with [Microsoft's Bot Framework SDK](https://github.com/microsoft/botframework-sdk) such as Mircosoft Copilot Studio.

## Features

The tool uses a ChromaDB to store the video titles and their corresponding links. It also uses the YouTube Data API to retrieve the video links from a playlist.

## How to use

You can use this tool through Docker or any similar software such as podman. Just replace docker with podman.

### Set up environment variables

Create a file called .env put it into the compose-examples folder and add the following:
```
YOUTUBE_DATA_API_KEY=<YT_API_KEY> # You can get this from https://console.cloud.google.com/apis/library/youtube.googleapis.com
SECRET_KEY=<YOUR_FLASK_SECRET_KEY> # any long key
API_KEY=<YOUR_CUSTOM_API_KEY> # For the semantic search API
PLAYLIST_ID=<YOUR_PLAYLIST_ID> # From youtube
DIRECT_LINE_SECRET=<DIRECTLINE_SECRET_KEY> # For the chatbot
APP_IMAGE_NAME=docker.io/brandon596/bim-chat

# Choose cloudflare or zrok or ignore if running locally
# zrok 
ZROK_ENABLE_TOKEN=<YOUR_ZROK_TOKEN> # For instructions, go to zrok.io
ZROK_UNIQUE_NAME=<NAME> # name is used to construct frontend domain name, e.g. "myapp" in "myapp.share.zrok.io"

# Cloudflare 
CLOUDFLARE_TUNNEL_TOKEN=<TOKEN> #From cloudflare tunnel if you have one
```

## Running on Docker

If you are on windows, make sure to use Powershell instead of Command Prompt

### Pull the image from Docker Hub or build it yourself

Pull Image
```
docker pull docker.io/brandon596/bim-chat
```

Build image
```
docker build -t <tag name of image> .
```

Notes if building the image:
- If you have `all-MiniLM-L6-v2` folder/directory, ignore the pointers below and move on to the next steps
- If you do not have `all-MiniLM-L6-v2` directory, open the `Dockerfile` and comment out `COPY /all-MiniLM-L6-v2 /home/myuser/.cache/chroma/onnx_models/all-MiniLM-L6-v2` line
- Go to the compose files and add a bind mount under the app service volumes: `- ./all-MiniLM-L6-v2:/home/myuser/.cache/chroma/onnx_models/all-MiniLM-L6-v2`


### Run the container using Docker Compose

Open the terminal in the compose-examples folder

### For local use
To start, run
```
docker compose -f compose-local.yml up -d
```
Go to localhost:8080

To stop
```
docker compose -f compose-local.yml down
```

### zrok
To start run
```
docker compose -f compose-zrok.yml up -d
```

If encounter failures, try restarting a few times
```
docker compose -f compose-zrok.yml restart
```
Otherwise, go to https://api.zrok.io to make sure that there are shares available in your account.

To get the URL, go to https://api.zrok.io/

To stop
```
docker compose -f compose-zrok.yml down
```

### Cloudflare
To try it for free with ephemeral URL use the compose-trycf.yml file. If you have an existing domain and cloudflare tunnel, use compose-cloudflare.yml file

To run
To start run
```
docker compose -f <COMPOSE_FILE> up -d
```

If using trycf, get the URL by running
```
docker compose -f compose-trycf.yml logs cloudflaretunnel
```

If using cloudflare with your account, go the dashboard in cloudflare tunnels and create a tunnel then under the service add http://app:8001

To stop
```
docker compose -f <COMPOSE_FILE> down
```

## Using the Web app

If starting the app for the first time, use the following credentials

Default Username: firstlogin<br>
Default Password: password

Then you will have to create the accounts. Please remember your passwords and usernames as there is no way to recover them once it is lost.

If you need to reset the app because you forgot the passwords, first remove the containers through the terminal:
```
docker compose -f <COMPOSE_FILE_NAME> down
```

Then remove the volume for the web app:
```
docker volume ls # To list the existing volumes

docker volume rm bim-chat-app_app-data # To remove the app volume, make sure it contains "app-data"
```

Start your containers again:
```
docker compose -f <COMPOSE_FILE_NAME> up -d
```
