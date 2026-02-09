import json
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class GenericParser:
    def parse(self, file_path: str, content: str = None) -> dict:
        """Parse a config file from path and optional pre-loaded content.
        
        This method is the primary interface used by ParserCoordinator.
        If content is not provided, reads the file from disk.
        """
        ext = Path(file_path).suffix.lower()
        try:
            if content is not None:
                if ext == '.json':
                    return {"data": json.loads(content), "format": "json"}
                elif ext in ('.yaml', '.yml'):
                    return {"data": yaml.safe_load(content) or {}, "format": "yaml"}
            else:
                return self.parse_file(Path(file_path))
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
        return {}

    def parse_file(self, file_path: Path) -> dict:
        """Parse a config file from disk."""
        ext = file_path.suffix.lower()
        if ext == '.json':
            with open(file_path, 'r') as f:
                return {"data": json.load(f), "format": "json"}
        elif ext in ('.yaml', '.yml'):
            with open(file_path, 'r') as f:
                return {"data": yaml.safe_load(f) or {}, "format": "yaml"}
        logger.warning(f"Unsupported file: {file_path}")
        return {}