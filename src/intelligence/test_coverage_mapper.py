"""
Test Coverage Mapper — Feature 3
Detects test files (pytest, unittest, jest, mocha, JUnit), parses them,
and creates TESTS edges linking test functions to the production code
they exercise.

Heuristics:
- pytest: files matching test_*.py / *_test.py, functions starting with test_
- unittest: classes extending TestCase / unittest.TestCase
- jest/mocha: *.test.ts, *.spec.ts, describe/it blocks
- JUnit: *Test.java, @Test annotation
"""
import re
import json
from typing import Dict, List, Any, Optional
from collections import defaultdict

from src.core.logger import get_logger
from src.intelligence.utils import generate_node_id

logger = get_logger(__name__)


class TestCoverageMapper:
    """
    Post-ingestion pass that:
    1. Identifies test files by naming convention
    2. Creates TestFile / TestFunction nodes
    3. Links test functions → production functions via TESTS edge
    """

    # Patterns that identify test files
    TEST_FILE_PATTERNS = [
        # Python
        r'test_\w+\.py$', r'\w+_test\.py$', r'tests?/.*\.py$', r'conftest\.py$',
        # JS/TS
        r'\w+\.test\.[jt]sx?$', r'\w+\.spec\.[jt]sx?$',
        r'__tests__/.*\.[jt]sx?$', r'tests?/.*\.[jt]sx?$',
        # Java
        r'\w+Test\.java$', r'\w+Tests\.java$', r'test/.*\.java$',
    ]

    async def map_test_coverage(self, session, repo_hash: str) -> Dict[str, Any]:
        stats = {
            "test_files_found": 0,
            "test_functions_found": 0,
            "tests_edges_created": 0,
            "errors": [],
        }

        try:
            # ── 1. Find all files, classify as test or production ──
            file_result = await session.run("""
                MATCH (f:File)
                WHERE f.repository_hash = $repo_hash
                RETURN f.id AS id, f.path AS path
            """, {"repo_hash": repo_hash})

            test_files = []
            prod_files = []
            records = [r async for r in file_result]

            for rec in records:
                path = rec["path"] or ""
                if self._is_test_file(path):
                    test_files.append(rec)
                else:
                    prod_files.append(rec)

            stats["test_files_found"] = len(test_files)

            if not test_files:
                logger.info("No test files found")
                return stats

            # ── 2. Label test files ──
            test_file_ids = [tf["id"] for tf in test_files]
            for chunk_start in range(0, len(test_file_ids), 500):
                chunk = test_file_ids[chunk_start:chunk_start + 500]
                await session.run("""
                    UNWIND $ids AS fid
                    MATCH (f:File {id: fid})
                    SET f:TestFile, f.is_test_file = true
                """, {"ids": chunk})

            # ── 3. Label test functions ──
            test_fn_result = await session.run("""
                MATCH (f:TestFile)-[:CONTAINS]->(fn:Function)
                WHERE f.repository_hash = $repo_hash
                RETURN fn.id AS id, fn.name AS name, fn.calls AS calls,
                       f.path AS file_path
            """, {"repo_hash": repo_hash})

            test_functions = [r async for r in test_fn_result]
            test_fn_ids = []
            for tf in test_functions:
                name = tf["name"] or ""
                if self._is_test_function(name):
                    test_fn_ids.append(tf["id"])

            stats["test_functions_found"] = len(test_fn_ids)

            if test_fn_ids:
                for chunk_start in range(0, len(test_fn_ids), 500):
                    chunk = test_fn_ids[chunk_start:chunk_start + 500]
                    await session.run("""
                        UNWIND $ids AS fnid
                        MATCH (fn:Function {id: fnid})
                        SET fn:TestFunction, fn.is_test = true
                    """, {"ids": chunk})

            # ── 4. Build production function index ──
            prod_fn_result = await session.run("""
                MATCH (f:File)-[:CONTAINS]->(fn:Function)
                WHERE f.repository_hash = $repo_hash AND NOT f:TestFile
                RETURN fn.id AS id, fn.name AS name, f.path AS file_path
            """, {"repo_hash": repo_hash})

            prod_index: Dict[str, str] = {}  # name → id
            async for rec in prod_fn_result:
                name = rec["name"] or ""
                if name:
                    prod_index[name] = rec["id"]

            # ── 5. Link test functions → production functions via calls ──
            tests_pairs: list[dict] = []
            for tf in test_functions:
                calls = tf["calls"] or []
                tf_id = tf["id"]
                for callee in calls:
                    bare = callee.split(".")[-1] if "." in callee else callee
                    if bare in prod_index:
                        tests_pairs.append({
                            "test_fn": tf_id,
                            "prod_fn": prod_index[bare],
                            "callee_name": bare,
                        })

            if tests_pairs:
                for chunk_start in range(0, len(tests_pairs), 500):
                    chunk = tests_pairs[chunk_start:chunk_start + 500]
                    await session.run("""
                        UNWIND $pairs AS p
                        MATCH (test:Function {id: p.test_fn})
                        MATCH (prod:Function {id: p.prod_fn})
                        MERGE (test)-[r:TESTS]->(prod)
                        SET r.callee_name = p.callee_name,
                            r.created_at = datetime()
                    """, {"pairs": chunk})
                stats["tests_edges_created"] = len(tests_pairs)

            # ── 6. Also use naming heuristic: test_create_user → create_user ──
            naming_pairs: list[dict] = []
            for tf in test_functions:
                name = tf["name"] or ""
                # test_create_user → create_user
                candidate = re.sub(r'^test_?', '', name, flags=re.IGNORECASE)
                if candidate and candidate in prod_index and tf["id"] not in [p["test_fn"] for p in tests_pairs]:
                    naming_pairs.append({
                        "test_fn": tf["id"],
                        "prod_fn": prod_index[candidate],
                        "method": "naming_convention",
                    })

            if naming_pairs:
                for chunk_start in range(0, len(naming_pairs), 500):
                    chunk = naming_pairs[chunk_start:chunk_start + 500]
                    await session.run("""
                        UNWIND $pairs AS p
                        MATCH (test:Function {id: p.test_fn})
                        MATCH (prod:Function {id: p.prod_fn})
                        MERGE (test)-[r:TESTS]->(prod)
                        SET r.method = p.method,
                            r.created_at = datetime()
                    """, {"pairs": chunk})
                stats["tests_edges_created"] += len(naming_pairs)

            logger.info(
                f"Test coverage: {stats['test_files_found']} test files, "
                f"{stats['test_functions_found']} test functions, "
                f"{stats['tests_edges_created']} TESTS edges"
            )

        except Exception as e:
            logger.error(f"Test coverage mapping failed: {e}")
            stats["errors"].append(str(e))

        return stats

    def _is_test_file(self, path: str) -> bool:
        for pattern in self.TEST_FILE_PATTERNS:
            if re.search(pattern, path):
                return True
        return False

    def _is_test_function(self, name: str) -> bool:
        """Check if a function name indicates a test."""
        lower = name.lower()
        return (
            lower.startswith("test_") or lower.startswith("test")
            or lower.startswith("it_") or lower.startswith("should_")
            or lower == "setup" or lower == "teardown"
            or lower.startswith("before_") or lower.startswith("after_")
        )
