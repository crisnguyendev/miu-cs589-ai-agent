import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

def find_file(base_path: str, filename: str) -> str:
    search_dirs = [
        base_path,
        os.path.join(base_path, "output"),
        os.path.join(base_path, "../output"),
        os.path.join(base_path, "data"),
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../output"),
    ]
    logger.debug(f"Searching for {filename} in: {search_dirs}")
    for dir_path in search_dirs:
        file_path = os.path.join(dir_path, filename)
        if Path(file_path).exists():
            logger.info(f"Found {filename} at: {file_path}")
            return file_path
    raise FileNotFoundError(f"File '{filename}' not found in common directories: {search_dirs}")

