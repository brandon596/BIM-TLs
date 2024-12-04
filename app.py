from flask import Flask, request, jsonify
import json
import chromadb
import numpy as np
from yt_playlist_retriever import get_processed_playlist
import os
from logger import logger

app = Flask(__name__)
AUTODESK_VID_PATH: str = "json_data/Autodesk_Videos.json"
YT_API_KEY: str = os.environ.get("YOUTUBE_DATA_API_KEY")
PLAYLIST_ID: str = "PL2KGy-TyLFXZyWk7frVvUHrFtjCl9mWLl"
temperature:float=0.0715
threshold:float=0.389

# Revit Doc id ends at 23, youtube videos starts at 24

def create_collection(chroma_client):
    collection = chroma_client.create_collection(
        name="Video_Titles_Embeddings",
        metadata={"hnsw:space": "cosine"}
    )
    with open(AUTODESK_VID_PATH, "r") as file:
        video_links = json.load(file)
    yt_id_start = len(video_links)
    video_links += get_processed_playlist(PLAYLIST_ID, YT_API_KEY, yt_id_start)
    titles = []
    metadatas = []
    for row in video_links:
        titles.append(row["Title"].strip())
        metadatas.append({"URL": row["Video URL"], "Source": row["Page URL"], "actual_id": row["Id"], "isTitle": True, "byAutodesk": True if row["Id"] < yt_id_start else False})
        if row["Description"] != "":
            alt_titles = row["Description"].split(" | ")
            for title in alt_titles:
                titles.append(title.strip())
                metadatas.append({"URL": row["Video URL"], "Source": row["Page URL"], "actual_id": row["Id"], "isTitle": False, "byAutodesk": True if row["Id"] < yt_id_start else False})
    ids = [str(i+1) for i in range(len(titles))]
    collection.upsert(
        documents=titles,
        ids=ids,
        metadatas=metadatas
    )
    return collection

def initialise_db():
    chroma_client = chromadb.PersistentClient(path="chroma")
    try:
        collection = chroma_client.get_collection("Video_Titles_Embeddings")
    except chromadb.errors.InvalidCollectionException:
        collection = create_collection(chroma_client)
        logger.info("Collection created")
    return collection

collection = initialise_db()

def invert_and_softmax(arr:list):
    arr = (1-np.array(arr))/temperature
    return list(np.exp(arr)/np.sum(np.exp(arr)))

def get_unique_actual_ids(results):
    unique_actual_ids = set()
    for i, score in enumerate(results["distances"][0]):
        actual_id = results["metadatas"][0][i]["actual_id"]
        if score > threshold:
            unique_actual_ids.add(actual_id)
    return unique_actual_ids

def get_outputs(unique_actual_ids):
    if len(unique_actual_ids) >= 2:
        results = collection.get(
            where={
                "$and": [
                    {"$or": [{"actual_id": id} for id in unique_actual_ids]},
                    {"isTitle": True}
                ]
            },
            include=["documents", "metadatas"]
        )
    else:
        results = collection.get(
            where={"$and": [
                        {"actual_id": unique_actual_ids.pop()},
                        {"isTitle": True}
                    ]
                },
            include=["documents", "metadatas"]
        )
    
    return {"Title": results["documents"], 
             "Video_Links": [vid_url["URL"] for vid_url in results["metadatas"]],
             "Subtitle": [vid_url["Source"] for vid_url in results["metadatas"]],
             "qty": len(unique_actual_ids)
        }

def query_db(queery):
    results = collection.query(
        query_texts=queery,
        n_results=collection.count()
    )
    results.get("distances")[0] = invert_and_softmax(results.get("distances")[0])
    unique_actual_ids = get_unique_actual_ids(results)
    if len(unique_actual_ids) != 0:
        pre_output = get_outputs(unique_actual_ids)
        output = {
            "Title": pre_output["Title"],
            "Video_Links": [
                {
                    "mimeType": "video/webm", 
                    "url": url
                } for url in pre_output["Video_Links"]
                ], 
            "Subtitle": pre_output["Subtitle"],
            "qty": pre_output["qty"]
        }
    else:
        output = {
                    "Title": [""],
                    "Video_Links": [
                        {
                            "mimeType": "video/webm", 
                            "url": ""
                        }
                        ], 
                    "Subtitle": [""], 
                    "qty": 0
                }
    return output

#Routes
@app.route("/")
def home():
    return "Home"

@app.route('/api/semantic_search', methods=["POST"])
def semantic_searh():
    post:dict = request.json
    query = post.get("query")
    output = query_db(query)
    to_log = {"user_query": query, "Titles": output["Title"]}
    logger.info("Query received", extra=to_log)
    return jsonify(output), 201

@app.route("/view/get_query_logs")
def get_query_logs():
    return jsonify(logger.handlers[0].baseFilename)

@app.route("/view/create_test_logs")
def create_test_logs():
    something = {"test": "test"}
    logger.info("Test log", extra=something)
    return "Test log created"

if __name__ == "__main__":
    app.run()