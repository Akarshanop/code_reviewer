import logging
import sys
from loguru import logger as loguru_logger

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    loguru_logger.remove()
    loguru_logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")

    