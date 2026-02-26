"""
Centralized deterministic ID generation utilities.

These helpers intentionally avoid MD5 while keeping deterministic behavior
for graph/vector identity keys across ingestion passes.
"""

from __future__ import annotations

import hashlib
from typing import Any


def generate_node_id(node_type: str, *identifiers: Any, digest_len: int = 16) -> str:
    """Generate deterministic node IDs using SHA-256."""
    combined = f"{node_type}:{'|'.join(str(i) for i in identifiers)}"
    return hashlib.sha256(combined.encode()).hexdigest()[:digest_len]


def generate_repository_hash(repo_ref: str, digest_len: int = 12) -> str:
    """Generate deterministic repository hash using SHA-256."""
    if not repo_ref:
        return ""
    return hashlib.sha256(repo_ref.encode()).hexdigest()[:digest_len]
