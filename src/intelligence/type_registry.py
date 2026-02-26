"""
Repository-scoped type registry for type-aware impact analysis.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Any, List, Set


class TypeRegistry:
    """In-memory type registry keyed by repository hash."""

    def __init__(self) -> None:
        self._types: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    def register_type(self, repository_hash: str, type_name: str, source_path: str = "") -> None:
        if not repository_hash or not type_name:
            return
        self._types[repository_hash][type_name].add(source_path or "")

    def register_from_parsed_data(self, repository_hash: str, parsed_data: Dict[str, Any]) -> None:
        if not repository_hash:
            return
        for language_items in parsed_data.values():
            if not isinstance(language_items, list):
                continue
            for file_item in language_items:
                if not isinstance(file_item, dict):
                    continue
                file_path = file_item.get("file", "")
                data = file_item.get("data", {}) if isinstance(file_item.get("data"), dict) else {}
                for cls in data.get("classes", []):
                    if isinstance(cls, dict) and cls.get("name"):
                        self.register_type(repository_hash, cls["name"], file_path)
                for fn in data.get("functions", []):
                    if isinstance(fn, dict):
                        ret = fn.get("return_type")
                        if isinstance(ret, str) and ret.strip():
                            self.register_type(repository_hash, ret.strip(), file_path)

    def repository_types(self, repository_hash: str) -> List[str]:
        return sorted(self._types.get(repository_hash, {}).keys())

    def to_dict(self) -> Dict[str, Any]:
        return {
            repo_hash: {type_name: sorted(paths) for type_name, paths in type_map.items()}
            for repo_hash, type_map in self._types.items()
        }
