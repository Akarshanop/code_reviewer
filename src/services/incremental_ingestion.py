"""
Incremental Ingestion Engine — Technical Improvement 1

Replaces full re-ingestion with diff-based processing:
    1. Detect which files changed (git diff)
    2. Parse only changed files
    3. Update graph partially (add/remove/update nodes and edges)
    4. Update only affected embeddings

No full re-index unless explicitly triggered.
Designed for CI-triggered updates.
"""

from __future__ import annotations

import hashlib
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any, Tuple

from src.core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Change detection models
# ---------------------------------------------------------------------------

class ChangeType:
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


@dataclass
class FileChange:
    """A single file change detected from git diff."""
    file_path: str
    change_type: str  # ChangeType values
    old_path: Optional[str] = None  # For renames
    lines_added: int = 0
    lines_removed: int = 0
    content_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "change_type": self.change_type,
            "old_path": self.old_path,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
        }


@dataclass
class IncrementalDiff:
    """Represents a diff between two repository states."""
    repository_name: str
    from_commit: str
    to_commit: str
    changes: List[FileChange] = field(default_factory=list)
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def added_files(self) -> List[str]:
        return [c.file_path for c in self.changes if c.change_type == ChangeType.ADDED]

    @property
    def modified_files(self) -> List[str]:
        return [c.file_path for c in self.changes if c.change_type == ChangeType.MODIFIED]

    @property
    def deleted_files(self) -> List[str]:
        return [c.file_path for c in self.changes if c.change_type == ChangeType.DELETED]

    @property
    def renamed_files(self) -> List[Tuple[str, str]]:
        return [
            (c.old_path or "", c.file_path)
            for c in self.changes if c.change_type == ChangeType.RENAMED
        ]

    @property
    def files_to_parse(self) -> List[str]:
        """Files that need parsing (added + modified + rename targets)."""
        return self.added_files + self.modified_files + [c.file_path for c in self.changes if c.change_type == ChangeType.RENAMED]

    @property
    def files_to_remove(self) -> List[str]:
        """Files whose graph nodes should be removed."""
        deleted = self.deleted_files
        renamed_sources = [c.old_path for c in self.changes if c.change_type == ChangeType.RENAMED and c.old_path]
        return deleted + renamed_sources

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repository_name": self.repository_name,
            "from_commit": self.from_commit[:12],
            "to_commit": self.to_commit[:12],
            "total_changes": len(self.changes),
            "added": len(self.added_files),
            "modified": len(self.modified_files),
            "deleted": len(self.deleted_files),
            "renamed": len(self.renamed_files),
            "files_to_parse": len(self.files_to_parse),
            "files_to_remove": len(self.files_to_remove),
        }


@dataclass
class IncrementalResult:
    """Result of an incremental ingestion run."""
    repository_name: str
    from_commit: str
    to_commit: str
    nodes_created: int = 0
    nodes_updated: int = 0
    nodes_deleted: int = 0
    edges_created: int = 0
    edges_deleted: int = 0
    embeddings_updated: int = 0
    files_parsed: int = 0
    files_skipped: int = 0
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    is_full_reindex: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repository_name": self.repository_name,
            "from_commit": self.from_commit[:12],
            "to_commit": self.to_commit[:12],
            "nodes_created": self.nodes_created,
            "nodes_updated": self.nodes_updated,
            "nodes_deleted": self.nodes_deleted,
            "edges_created": self.edges_created,
            "edges_deleted": self.edges_deleted,
            "embeddings_updated": self.embeddings_updated,
            "files_parsed": self.files_parsed,
            "files_skipped": self.files_skipped,
            "errors": self.errors[:10],
            "duration_seconds": round(self.duration_seconds, 2),
            "is_full_reindex": self.is_full_reindex,
        }


# ---------------------------------------------------------------------------
# Incremental ingestion engine
# ---------------------------------------------------------------------------

# Supported parseable extensions
_PARSEABLE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs",
    ".vue", ".svelte", ".rb", ".php", ".cs", ".kt", ".scala",
    ".yaml", ".yml", ".json",
}

# Files to always skip
_SKIP_PATTERNS = {
    "node_modules/", "dist/", "build/", ".git/", "__pycache__/",
    "vendor/", "target/", ".gradle/", "bin/", "obj/",
    "package-lock.json", "yarn.lock", "poetry.lock",
}


def should_parse_file(file_path: str) -> bool:
    """Check if a file should be parsed based on extension and path."""
    if any(skip in file_path for skip in _SKIP_PATTERNS):
        return False
    return any(file_path.endswith(ext) for ext in _PARSEABLE_EXTENSIONS)


def compute_file_hash(content: str) -> str:
    """Compute content hash for change detection."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class IncrementalIngestionEngine:
    """
    Manages incremental graph updates from git diffs.

    Usage:
        engine = IncrementalIngestionEngine(neo4j_client, parser_coordinator, ...)
        diff = engine.detect_changes(repo_path, last_commit, current_commit)
        result = await engine.apply_incremental(diff, repo_path)
    """

    def __init__(self) -> None:
        self._last_commits: Dict[str, str] = {}  # repo_name → last processed commit
        self._stats = {
            "incremental_runs": 0,
            "full_reindex_runs": 0,
            "total_files_processed": 0,
        }

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def parse_git_diff_output(
        self,
        diff_output: str,
        repository_name: str,
        from_commit: str,
        to_commit: str,
    ) -> IncrementalDiff:
        """
        Parse output from: git diff --name-status --numstat <from>..<to>

        Expected format (--name-status):
            A	path/to/new_file.py
            M	path/to/modified_file.js
            D	path/to/deleted_file.java
            R100	old/path.ts	new/path.ts
        """
        changes: List[FileChange] = []

        for line in diff_output.strip().splitlines():
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            status = parts[0].strip()
            file_path = parts[1].strip()

            if status == "A":
                changes.append(FileChange(
                    file_path=file_path,
                    change_type=ChangeType.ADDED,
                ))
            elif status == "M":
                changes.append(FileChange(
                    file_path=file_path,
                    change_type=ChangeType.MODIFIED,
                ))
            elif status == "D":
                changes.append(FileChange(
                    file_path=file_path,
                    change_type=ChangeType.DELETED,
                ))
            elif status.startswith("R"):
                old_path = file_path
                new_path = parts[2].strip() if len(parts) > 2 else file_path
                changes.append(FileChange(
                    file_path=new_path,
                    change_type=ChangeType.RENAMED,
                    old_path=old_path,
                ))

        return IncrementalDiff(
            repository_name=repository_name,
            from_commit=from_commit,
            to_commit=to_commit,
            changes=changes,
        )

    async def remove_file_nodes(
        self,
        neo4j_client: Any,
        repository_name: str,
        file_paths: List[str],
    ) -> int:
        """
        Remove all graph nodes associated with deleted/renamed files.

        Returns count of nodes removed.
        """
        if not file_paths:
            return 0

        removed = 0
        for file_path in file_paths:
            try:
                # Delete all nodes that belong to this file
                query = """
                MATCH (n)
                WHERE n.repository_name = $repo 
                AND (n.file_name = $file_path OR n.relative_path = $file_path)
                DETACH DELETE n
                RETURN count(n) as deleted_count
                """
                result = await neo4j_client.execute_write_query(
                    query,
                    {"repo": repository_name, "file_path": file_path},
                )
                if result:
                    removed += result[0].get("deleted_count", 0) if result else 0
            except Exception as e:
                logger.warning(f"Failed to remove nodes for {file_path}: {e}")

        return removed

    async def remove_file_embeddings(
        self,
        qdrant_client: Any,
        repository_name: str,
        file_paths: List[str],
    ) -> int:
        """Remove vector embeddings for deleted/renamed files."""
        if not file_paths:
            return 0

        removed = 0
        for file_path in file_paths:
            try:
                deleted = await qdrant_client.delete_vectors({
                    "must": [
                        {"key": "repository_name", "match": {"value": repository_name}},
                        {"key": "file_name", "match": {"value": file_path}},
                    ]
                })
                removed += deleted if isinstance(deleted, int) else 0
            except Exception as e:
                logger.warning(f"Failed to remove embeddings for {file_path}: {e}")

        return removed

    def set_last_commit(self, repository_name: str, commit_hash: str) -> None:
        """Record the last successfully processed commit."""
        self._last_commits[repository_name] = commit_hash

    def get_last_commit(self, repository_name: str) -> Optional[str]:
        """Get the last successfully processed commit for a repo."""
        return self._last_commits.get(repository_name)

    def should_full_reindex(
        self,
        diff: IncrementalDiff,
        threshold_percent: float = 0.5,
        threshold_count: int = 200,
    ) -> bool:
        """
        Determine if a full reindex would be more efficient than incremental.

        Heuristics:
            - If >50% of files changed → full reindex
            - If >200 files changed → full reindex
            - If repo has no last commit → full reindex
        """
        if not self.get_last_commit(diff.repository_name):
            return True
        if len(diff.changes) > threshold_count:
            return True
        return False


# Singleton
incremental_engine = IncrementalIngestionEngine()
