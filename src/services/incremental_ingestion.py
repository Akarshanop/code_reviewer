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
from src.core.id_generator import generate_node_id, generate_repository_hash

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

        repo_hash = await self._resolve_repository_hash(neo4j_client, repository_name)
        removed = 0
        for file_path in file_paths:
            try:
                file_node_id = generate_node_id("File", repo_hash, file_path)
                # Prefer deterministic file-node ID. Keep a path fallback for legacy rows.
                query = """
                MATCH (f:File)
                WHERE (
                    ($repo_hash <> '' AND f.id = $file_node_id)
                    OR ($repo_hash <> '' AND f.repository_hash = $repo_hash AND f.relative_path = $file_path)
                    OR ($repo_hash = '' AND f.repository_name = $repo AND f.relative_path = $file_path)
                )
                WITH collect(f) AS files
                UNWIND files AS f
                OPTIONAL MATCH (f)-[:CONTAINS]->(child)
                WITH collect(DISTINCT f) + collect(DISTINCT child) AS nodes_to_delete
                UNWIND nodes_to_delete AS n
                WITH DISTINCT n
                DETACH DELETE n
                RETURN count(n) as deleted_count
                """
                result = await neo4j_client.execute_write_query(
                    query,
                    {
                        "repo": repository_name,
                        "repo_hash": repo_hash,
                        "file_path": file_path,
                        "file_node_id": file_node_id,
                    },
                )
                if result:
                    removed += result[0].get("deleted_count", 0) if result else 0
            except Exception as e:
                logger.warning(f"Failed to remove nodes for {file_path}: {e}")

        return removed

    async def expand_relink_scope(
        self,
        neo4j_client: Any,
        repository_name: str,
        changed_files: List[str],
        max_depth: int = 3,
    ) -> List[str]:
        """
        Build dependency-aware relink scope from changed files.

        Traverses reverse IMPORTS plus CALLS_FUNCTION-induced file dependencies.
        """
        if not changed_files:
            return []

        repo_hash = await self._resolve_repository_hash(neo4j_client, repository_name)
        scope: Set[str] = set(changed_files)
        frontier: Set[str] = set(changed_files)

        for _ in range(max_depth):
            if not frontier:
                break

            dependents = await self._find_direct_dependents(
                neo4j_client=neo4j_client,
                repo_hash=repo_hash,
                repository_name=repository_name,
                target_files=list(frontier),
            )
            next_frontier = dependents - scope
            scope.update(next_frontier)
            frontier = next_frontier

        return sorted(scope)

    async def should_promote_to_full_relink(
        self,
        neo4j_client: Any,
        repository_name: str,
        changed_files: List[str],
        fanout_threshold: int = 100,
    ) -> bool:
        """Promote high fan-out incremental changes to full relink pass."""
        if not changed_files:
            return False

        repo_hash = await self._resolve_repository_hash(neo4j_client, repository_name)
        query = """
        UNWIND $files AS file_path
        MATCH (target:File)
        WHERE (
            ($repo_hash <> '' AND target.repository_hash = $repo_hash AND target.relative_path = file_path)
            OR ($repo_hash = '' AND target.repository_name = $repo AND target.relative_path = file_path)
        )
        OPTIONAL MATCH (importer:File)-[:IMPORTS]->(target)
        WHERE (
            ($repo_hash <> '' AND importer.repository_hash = $repo_hash)
            OR ($repo_hash = '' AND importer.repository_name = $repo)
        )
        WITH file_path, count(DISTINCT importer) AS import_fanout
        RETURN max(import_fanout) AS max_fanout
        """
        rows = await neo4j_client.execute_query(
            query,
            {"files": changed_files, "repo_hash": repo_hash, "repo": repository_name},
        )
        max_fanout = int((rows or [{}])[0].get("max_fanout") or 0)
        return max_fanout >= fanout_threshold

    async def _resolve_repository_hash(self, neo4j_client: Any, repository_name: str) -> str:
        """Resolve repository hash from graph metadata with deterministic fallback."""
        repo_hash = ""
        try:
            rows = await neo4j_client.execute_query(
                """
                MATCH (r:Repository)
                WHERE r.name = $repo OR r.url = $repo
                RETURN r.hash AS hash
                LIMIT 1
                """,
                {"repo": repository_name},
            )
            if rows:
                repo_hash = rows[0].get("hash") or ""
            if not repo_hash:
                rows = await neo4j_client.execute_query(
                    """
                    MATCH (f:File)
                    WHERE f.repository_name = $repo
                    RETURN f.repository_hash AS hash
                    LIMIT 1
                    """,
                    {"repo": repository_name},
                )
                if rows:
                    repo_hash = rows[0].get("hash") or ""
        except Exception:
            repo_hash = ""

        if not repo_hash:
            repo_hash = generate_repository_hash(repository_name)
        return repo_hash

    async def _find_direct_dependents(
        self,
        neo4j_client: Any,
        repo_hash: str,
        repository_name: str,
        target_files: List[str],
    ) -> Set[str]:
        """Find files that directly depend on any file in `target_files`."""
        if not target_files:
            return set()

        params = {"files": target_files, "repo_hash": repo_hash, "repo": repository_name}

        import_rows = await neo4j_client.execute_query(
            """
            UNWIND $files AS file_path
            MATCH (target:File)
            WHERE (
                ($repo_hash <> '' AND target.repository_hash = $repo_hash AND target.relative_path = file_path)
                OR ($repo_hash = '' AND target.repository_name = $repo AND target.relative_path = file_path)
            )
            MATCH (dep:File)-[:IMPORTS]->(target)
            WHERE (
                ($repo_hash <> '' AND dep.repository_hash = $repo_hash)
                OR ($repo_hash = '' AND dep.repository_name = $repo)
            )
            RETURN DISTINCT dep.relative_path AS file_path
            """,
            params,
        )

        call_rows = await neo4j_client.execute_query(
            """
            UNWIND $files AS file_path
            MATCH (target_file:File)
            WHERE (
                ($repo_hash <> '' AND target_file.repository_hash = $repo_hash AND target_file.relative_path = file_path)
                OR ($repo_hash = '' AND target_file.repository_name = $repo AND target_file.relative_path = file_path)
            )
            MATCH (dep_file:File)-[:CONTAINS]->(:Function)-[:CALLS_FUNCTION]->(:Function)<-[:CONTAINS]-(target_file)
            WHERE (
                ($repo_hash <> '' AND dep_file.repository_hash = $repo_hash)
                OR ($repo_hash = '' AND dep_file.repository_name = $repo)
            )
            RETURN DISTINCT dep_file.relative_path AS file_path
            """,
            params,
        )

        out: Set[str] = set()
        for row in (import_rows or []):
            file_path = row.get("file_path")
            if file_path:
                out.add(file_path)
        for row in (call_rows or []):
            file_path = row.get("file_path")
            if file_path:
                out.add(file_path)
        return out

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
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
        except ImportError:
            logger.warning("qdrant_client.models not available; cannot remove embeddings")
            return 0

        for file_path in file_paths:
            try:
                # Qdrant payload uses "repository_name" and "relative_path" (or "file")
                filter_conditions = Filter(
                    must=[
                        FieldCondition(key="repository_name", match=MatchValue(value=repository_name)),
                        FieldCondition(key="relative_path", match=MatchValue(value=file_path)),
                    ]
                )
                await qdrant_client.delete_vectors(filter_conditions, wait=True)
                removed += 1
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
