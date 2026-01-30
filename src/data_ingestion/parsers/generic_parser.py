import json
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class GenericParser:
    def parse_file(self, file_path: Path) -> dict:
        ext = file_path.suffix.lower()
        if ext == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        elif ext in ('.yaml', '.yml'):
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        logger.warning(f"Unsupported file: {file_path}")
        return {}