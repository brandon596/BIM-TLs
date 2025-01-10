from flask import Flask, request, jsonify, render_template, send_file, abort
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import check_password_hash, generate_password_hash
import json
import chromadb
import numpy as np
from yt_playlist_retriever import get_processed_playlist
import os
from logger import logger, parse_log_folder_files, parse_app_log
from collections import Counter
import time
import requests
# from secrets import token_urlsafe

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY")
auth = HTTPBasicAuth()

def getUsers():
    with open("persistent/json_data/users.json", "r") as file:
        users = json.load(file)
    return users

def getRoles():
    with open("persistent/json_data/roles.json", "r") as file:
        roles = json.load(file)
    return roles

def dumpUsers(users:dict):
    with open("persistent/json_data/users.json", "w") as file:
        json.dump(users, file, indent=4)

def dumpRoles(roles:dict):
    with open("persistent/json_data/roles.json", "w") as file:
        json.dump(roles, file, indent=4)

AUTODESK_VID_PATH: str = "persistent/json_data/Autodesk_Videos.json"
YT_API_KEY: str = os.environ.get("YOUTUBE_DATA_API_KEY")
PLAYLIST_ID: str = os.environ.get("PLAYLIST_ID")
enable_chat_endpoint = False
temperature: float=0.0715
threshold: float=0.389

@auth.get_user_roles
def get_user_roles(username):
    return getRoles().get(username)

@auth.verify_password
def verify_password(username, password):
    if username in getUsers() and getUsers().get(username) == password:
        return username
    elif username in getUsers() and check_password_hash(getUsers().get(username), password):
        return username

def getStudents():
    roles = getRoles()
    users = getUsers()
    studentUsernames = [user for user, role in roles.items() if "student" in role]
    studentAccounts = {username: password for username, password in users.items() if username in studentUsernames}
    return studentAccounts

def getNumberofStudents(roles:dict):
    c = Counter(roles.values())
    return c["student"]

def additStudentUser(account: dict):
    users = getUsers()
    users.update(account)
    roles = getRoles()
    roles.update({next(iter(account)): "student"})
    if getNumberofStudents(roles) <= 10:
        dumpUsers(users)
        dumpRoles(roles)
    else:
        raise Exception("too many access accounts limit is 10")

def deleteStudentUser(username:str):
    users = getUsers()
    users.pop(username)
    roles = getRoles()
    roles.pop(username)
    dumpUsers(users)
    dumpRoles(roles)

def create_collection(chroma_client):
    created_collection = chroma_client.create_collection(
        name="Video_Titles_Embeddings",
        metadata={"hnsw:space": "cosine"}
    )
    return created_collection

def load_empty_collection(empty_collection):
    with open(AUTODESK_VID_PATH, "r") as file:
        video_links = json.load(file)
    yt_id_start = len(video_links)+1
    yt_vids_links = get_processed_playlist(PLAYLIST_ID, YT_API_KEY, yt_id_start)
    video_links += yt_vids_links
    titles = []
    metadatas = []
    for row in video_links:
        titles.append(row["Title"].strip())
        metadatas.append({"URL": row["Video_URL"], "Source": row["Page_URL"], "actual_id": row["Id"], "isTitle": True, "onYoutube": True if row["Id"] >= yt_id_start else False})
        if row["Description"] != "":
            alt_titles = row["Description"].split(" | ")
            for title in alt_titles:
                titles.append(title.strip())
                metadatas.append({"URL": row["Video_URL"], "Source": row["Page_URL"], "actual_id": row["Id"], "isTitle": False, "onYoutube": True if row["Id"] >= yt_id_start else False})
    ids = [str(i+1) for i in range(len(titles))]
    empty_collection.add(
        documents=titles,
        ids=ids,
        metadatas=metadatas
    )
    return empty_collection #its not empty here if everything above goes well

def upsert_collection(new_collection):
    old_ids = new_collection.get(
        include=["documents"]
    ).get("ids")
    old_ids = set(old_ids)

    with open(AUTODESK_VID_PATH, "r") as file:
        video_links = json.load(file)
    yt_id_start = len(video_links)+1
    yt_vids_links = get_processed_playlist(PLAYLIST_ID, YT_API_KEY, yt_id_start)
    video_links += yt_vids_links
    titles = []
    metadatas = []
    for row in video_links:
        titles.append(row["Title"].strip())
        metadatas.append({"URL": row["Video_URL"], "Source": row["Page_URL"], "actual_id": row["Id"], "isTitle": True, "onYoutube": True if row["Id"] >= yt_id_start else False})
        if row["Description"].strip():
            alt_titles = row["Description"].split(" | ")
            for title in alt_titles:
                titles.append(title.strip())
                metadatas.append({"URL": row["Video_URL"], "Source": row["Page_URL"], "actual_id": row["Id"], "isTitle": False, "onYoutube": True if row["Id"] >= yt_id_start else False})
    ids = [str(i+1) for i in range(len(titles))]
    new_collection.upsert(
        documents=titles,
        ids=ids,
        metadatas=metadatas
    )
    new_ids = set(ids)
    toBeDeletedIds = list(old_ids - new_ids)
    if toBeDeletedIds:
        new_collection.delete(ids=toBeDeletedIds)
    return new_collection

def initialise_db():
    chroma_client = chromadb.PersistentClient(path="persistent/chroma")
    # chroma_client = chromadb.Client()
    try:
        collection = chroma_client.get_collection("Video_Titles_Embeddings")
    except chromadb.errors.InvalidCollectionException:
        empty_collection = create_collection(chroma_client)
        collection = load_empty_collection(empty_collection)
        logger.info("Collection created")
    return collection

collection = initialise_db()
os.makedirs("sent", exist_ok=True)

def invert_and_softmax(arr:list):
    arr = (1-np.array(arr))/temperature
    return list(np.exp(arr)/np.sum(np.exp(arr)))

def get_unique_actual_ids(results):
    unique_actual_ids = set()
    for i, score in enumerate(results["distances"][0]):
        actual_id = results["metadatas"][0][i]["actual_id"]
        if score > threshold:
            unique_actual_ids.add(actual_id)
    return list(unique_actual_ids)

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
                        {"actual_id": unique_actual_ids[0]},
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
        n_results=min(collection.count(), 50)
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


# Load JSON data
def load_data(current_user):
    user_autodest_vid_file_path = f"persistent/json_data/{current_user}_Autodesk_Videos_temp.json"
    if os.path.exists(user_autodest_vid_file_path):
        with open(user_autodest_vid_file_path, 'r') as file:
            return json.load(file)
    else:
        with open(AUTODESK_VID_PATH, 'r') as file:
            return json.load(file)

# Save JSON data to a temporary file
def save_data(data, current_user):
    with open(f'persistent/json_data/{current_user}_Autodesk_Videos_temp.json', 'w') as file:
        json.dump(data, file, indent=4)

def replace_data_with_temp(current_user):
    with open(AUTODESK_VID_PATH, 'w') as file:
        with open(f'persistent/json_data/{current_user}_Autodesk_Videos_temp.json', 'r') as temp_file:
            new_data = json.load(temp_file)
            reindexed_date = reindex_data(new_data)
        json.dump(reindexed_date, file, indent=4)
    os.remove(f'persistent/json_data/{current_user}_Autodesk_Videos_temp.json')

def reindex_data(data):
    for i, video in enumerate(data):
        video['Id'] = i + 1
    return data

def tempDataIsDifferent(current_user):
    with open(AUTODESK_VID_PATH, 'r') as file:
        return load_data(current_user) != json.load(file)

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
@auth.login_required(role="admin")
def home():
    
    return render_template("home.html")

@app.route("/downloads/json_logs")
@auth.login_required(role="admin")
def download_json_logs():
    log_list = parse_log_folder_files()
    with open("sent/query_logs.json", "w") as file:
        json.dump(log_list, file, indent=4)
    return send_file("sent/query_logs.json", as_attachment=True, download_name="query_logs.json")

@app.route("/logs", methods=["GET"])
@auth.login_required(role="admin")
def get_query_logs():
    log_list = parse_log_folder_files()
    all_logs = parse_app_log()
    titles_list = []
    for log in log_list:
        replaced_blanks = [title or "No videos found" for title in log["Titles"]]
        titles_list += replaced_blanks
    title_counter = Counter(titles_list)
    return render_template("get_logs.html", logs=json.dumps(log_list, indent=4), title_counter=title_counter, all_logs=all_logs)


@app.route("/video_table", methods=["GET","POST"])
@auth.login_required(role="admin")
def all_videos():
    query = request.form.get("search")
    toEmbed = request.form.get("embed")
    if toEmbed:
        toEmbed = "checked"
    if query and not toEmbed:
        return render_template("table.html", video_table=simple_search(query), query=query, ticked=toEmbed)
    elif query and toEmbed:
        return render_template("table.html", video_table=query_db(query).get("asRows"), query=query, ticked=toEmbed)
    elif get_all_videos():
        return render_template("table.html", video_table=get_all_videos(), ticked=toEmbed)
    else:
        return render_template("table.html")

@app.route('/manage_videos')
@auth.login_required(role="admin")
def manage_videos():
    if os.path.exists(f'persistent/json_data/{auth.current_user()}_Autodesk_Videos_temp.json'):
        os.remove(f'persistent/json_data/{auth.current_user()}_Autodesk_Videos_temp.json')
    data = load_data(auth.current_user())
    return render_template('manage_videos.html', videos=data)

@app.route('/add', methods=['POST'])
@auth.login_required(role="admin")
def add_video():
    data = load_data(auth.current_user())
    new_video = request.json
    new_video['Id'] = max(video['Id'] for video in data) + 1 if data else 1
    data.append(new_video)
    save_data(data, auth.current_user())
    return jsonify(new_video), 201

@app.route('/edit/<int:video_id>', methods=['POST'])
@auth.login_required(role="admin")
def edit_video(video_id):
    data = load_data(auth.current_user())
    video = next((video for video in data if video['Id'] == video_id), None)
    if not video:
        return jsonify({'error': 'Video not found'}), 404
    video.update(request.json)
    save_data(data, auth.current_user())
    return jsonify(video)

@app.route('/delete/<int:video_id>', methods=['POST'])
@auth.login_required(role="admin")
def delete_video(video_id):
    data = load_data(auth.current_user())
    data = [video for video in data if video['Id'] != video_id]
    save_data(data, auth.current_user())
    return jsonify({'message': 'Video deleted'})

@app.route('/commit', methods=['POST'])
@auth.login_required(role="admin")
def commit_changes():
    if os.path.exists(f'persistent/json_data/{auth.current_user()}_Autodesk_Videos_temp.json') and tempDataIsDifferent(auth.current_user()):
        replace_data_with_temp(auth.current_user())
    upsert_collection(collection)
    time.sleep(1)
    logger.info("Collection updated")
    logger.info(f"Changes committed by {auth.current_user()}")
    return jsonify({'message': 'Changes committed'})

@app.route('/test_querying/<int:x>')
@auth.login_required(role="admin")
def test_querying(x):
    count=collection.count()
    print(count)
    results = None
    results = collection.get()
    queryed_results = None
    queryed_results = collection.query(
        query_texts=["Schedule Sheets"],
        n_results=x
    )
    return jsonify({"count": count, "results": results, "queryed_results": queryed_results})

@app.route('/chat')
@auth.login_required(role=["admin", "student"])
def chatv2():
    headers = {
        "Authorization": f"Bearer {os.environ.get('DIRECT_LINE_SECRET')}"
    }
    directline_response = requests.post(
        url="https://directline.botframework.com/v3/directline/tokens/generate",
        headers=headers
    ).json()
    return render_template('chatv2.html', 
                           DIRECTLINE_TOKEN=directline_response.get("token")
                        )

@app.route('/student_access')
@auth.login_required(role="admin")
def manage_student_access():
    return render_template("manage_student.html", ENDPOINT_ENABLED=enable_chat_endpoint)

@app.route('/test_requests', methods=["GET", "POST", "DELETE"])
@auth.login_required(role="admin")
def test_requests():
    print(type(request.json))
    return jsonify(request.json)

@app.route('/student_access/addit', methods=["POST"])
@auth.login_required(role='admin')
def addit_student_access():
    account = request.json
    try:
        additStudentUser(account)
        return jsonify({"response": "OK"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    

@app.route('/student_access/delete', methods=["DELETE"])
@auth.login_required(role='admin')
def delete_student_access():
    try:
        deleteStudentUser(request.json["username"])
        return jsonify({"response": "OK"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/student_access/get', methods=["GET"])
@auth.login_required(role='admin')
def get_student_access():
    return jsonify(getStudents())

@app.route('/student_access/user/exists', methods=["POST"])
@auth.login_required(role="admin")
def usernameExists():
    username = request.json["username"]
    try:
        if username in getUsers():
            return jsonify({"userExists": True})
        else:
            return jsonify({"userExists": False})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.before_request
def block_disabled_endpoints():
    # Check if the request is for the specific endpoint
    if request.path == '/chat':
        # Check if the endpoint is disabled
        if not enable_chat_endpoint:
            abort(503, description="This endpoint is currently disabled.")

@app.route('/api/toggle-endpoint', methods=['GET'])
@auth.login_required(role="admin")
def toggle_endpoint():
    global enable_chat_endpoint
    enable_chat_endpoint = not enable_chat_endpoint
    return jsonify({"chatEnabled": enable_chat_endpoint})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")