from flask import Flask, request, jsonify, render_template
import json
import chromadb
import numpy as np
from yt_playlist_retriever import get_processed_playlist
import os
from logger import logger

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY")

AUTODESK_VID_PATH: str = "json_data/Autodesk_Videos.json"
YT_API_KEY: str = os.environ.get("YOUTUBE_DATA_API_KEY")
PLAYLIST_ID: str = os.environ.get("PLAYLIST_ID")
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
    yt_id_start = len(video_links)+1
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
    rowed_output = None
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
        rowed_output = [
            {
                "Title": title,
                "URL": url
            } for title, url in zip(pre_output["Title"], pre_output["Subtitle"])
        ]
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
    return {"ProcessedforAdaptiveCards": output, "asRows": rowed_output}

def authenticate_api_key():
    api_key = request.headers.get("Authorization")
    if api_key != os.environ.get("API_KEY"):
        return False
    return True

def get_all_videos(form: str=None):
    results = collection.get(
        where={"isTitle": True}
    )
    if len(results["documents"]) == 0:
        return None
    if form == "dict":
        return {"Id": [i["actual_id"] for i in results["metadatas"]], "Title": results["documents"], "URL": [r["Source"] for r in results["metadatas"]]}
    else:
        return [{"Id": i["actual_id"], "Title": document, "URL": url["Source"]} for i, document, url in zip(results["metadatas"], results["documents"], results["metadatas"])]

def simple_search(query):
    results = collection.get(
        where={
            "isTitle": True
        },
        where_document={
            "$contains": query
        }
        )
    return [{"Id": i["actual_id"], "Title": document, "URL": url["Source"]} for i, document, url in zip(results["metadatas"], results["documents"], results["metadatas"])]

#Routes
@app.route('/api/semantic_search', methods=["POST"])
def semantic_searh():
    if not authenticate_api_key():
        return jsonify({"error": "Invalid API Key"}), 401
    post:dict = request.json
    try:
        query = str(post.get("query"))
        output = query_db(query).get("ProcessedforAdaptiveCards")
        to_log = {"user_query": query, "Titles": output["Title"]}
        logger.info("Query received", extra=to_log)
        return jsonify(output), 201
    except Exception as e:
        logger.error(str(e))
        return jsonify({"error": str(e)}), 400

@app.route("/")
def home():
    pass

@app.route("/get_query_logs")
def get_query_logs():
    pass


@app.route("/video_table", methods=["GET","POST"])
def all_videos():
    # if request.method == "POST":
    query = request.form.get("search")
    toEmbed = request.form.get("embed")
    if toEmbed:
        toEmbed = "checked"
    print(query)
    if query and not toEmbed:
        return render_template("table.html", video_table=simple_search(query), query=query, ticked=toEmbed)
    elif query and toEmbed:
        return render_template("table.html", video_table=query_db(query).get("asRows"), query=query, ticked=toEmbed)
    # if request.method == "GET":
    elif get_all_videos():
        return render_template("table.html", video_table=get_all_videos(), query=query, ticked=toEmbed)
    else:
        return render_template("table.html")


if __name__ == "__main__":
    app.run(debug=True)