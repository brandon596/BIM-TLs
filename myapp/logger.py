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
os.makedirs("persistent/logs", exist_ok=True)
handler = TimedRotatingFileHandler(f"persistent/logs/app.log", when="D", interval=7, backupCount=4)
handler.setFormatter(formatter)
logger.addHandler(handler)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


def parse_log_folder_files():
    try:
        import json
        lines = []
        for file in os.listdir("persistent/logs"):
            with open(f"persistent/logs/{file}", "r") as file:
                lines += file.readlines()

        return [json.loads(line) for line in lines if "Query received" in line]
    except Exception as e:
        return [f"Error: {str(e)}"]

def parse_app_log():
    try:
        with open("persistent/logs/app.log", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return "Error: The log file was not found."
    except PermissionError:
        return "Error: Permission denied to read the log file."
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

