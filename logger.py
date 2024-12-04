import logging
from pythonjsonlogger import jsonlogger
from logging.handlers import TimedRotatingFileHandler

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
formatter = jsonlogger.JsonFormatter(
    "%(asctime)s %(levelname)s %(message)s",
    "%Y-%m-%d %H:%M:%S",
)
handler = TimedRotatingFileHandler(f"logs/app.log", when="D", interval=7, backupCount=4)
handler.setFormatter(formatter)
logger.addHandler(handler)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)