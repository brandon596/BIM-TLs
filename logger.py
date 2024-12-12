import logging
from pythonjsonlogger import jsonlogger
from logging.handlers import TimedRotatingFileHandler
import os

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
formatter = jsonlogger.JsonFormatter(
    "%(asctime)s %(levelname)s %(message)s",
    "%Y-%m-%d %H:%M:%S",
)
os.makedirs("logs", exist_ok=True)
handler = TimedRotatingFileHandler(f"logs/app.log", when="D", interval=7, backupCount=4)
handler.setFormatter(formatter)
logger.addHandler(handler)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

def parse_log_folder_files():
    import json
    lines = []
    for file in os.listdir("logs"):
        if file.endswith(".log"):
            with open(f"logs/{file}", "r") as file:
                lines += file.readlines()

    return [json.loads(line) for line in lines if "Query received" in line]

def parse_all_logs():
    with open("logs/app.log", "r") as file:
        return file.read()