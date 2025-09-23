import logging
import os
from logging import Logger
from typing import Optional

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name: str, level: int = logging.INFO) -> Logger:
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        formatter = logging.Formatter(
            '{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","msg":"%(message)s"}'
        )
        fh = logging.FileHandler(os.path.join(LOG_DIR, "app.log"), encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    logger.setLevel(level)
    return logger