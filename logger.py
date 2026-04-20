import logging
import sys
from logging.handlers import RotatingFileHandler

import config


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:         # prevent duplicate handlers on re-import
        return logger

    logger.setLevel(config.LOG_LEVEL)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(config.LOG_FILE, maxBytes=5_242_880, backupCount=3)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
