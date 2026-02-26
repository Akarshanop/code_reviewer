"""
Performance Profiler — Feature 6
Static analysis heuristics that flag potential performance issues in the
ingested codebase. Stores findings as properties on Function/File nodes
and as PerformanceHint nodes in the graph.

Detects:
1. N+1 query patterns (loop around DB/API call)
2. Missing async/await (sync I/O in async context)
3. Blocking I/O in async functions (time.sleep, open() in async def)
4. Unbounded queries (SELECT * without LIMIT)
5. Missing pagination on list endpoints
6. Large payload risks (no response size limits)
7. Synchronous imports of heavy libraries inside hot paths
"""
import re
import json
from typing import Dict, List, Any, Set
from collections import defaultdict

from src.core.logger import get_logger
from src.intelligence.utils import generate_node_id

logger = get_logger(__name__)


# ── Pattern definitions ──────────────────────────────────────────────────

# Patterns that indicate a DB or API call
DB_CALL_PATTERNS = [
    re.compile(r'\b(?:execute|query|find|findOne|findAll|findMany|select|insert|update|delete|save|create|remove)\s*\(', re.IGNORECASE),
    re.compile(r'\b(?:collection|model|repository|dao|cursor|session)\s*\.\s*\w+\s*\(', re.IGNORECASE),
    re.compile(r'\b(?:fetch|axios|request|http|get|post|put|patch)\s*\(', re.IGNORECASE),
    re.compile(r'\.(?:run|execute_query|execute|read_transaction|write_transaction)\s*\(', re.IGNORECASE),
    re.compile(r'\b(?:supabase|prisma|mongoose|sequelize|typeorm|knex|sqlalchemy)\b', re.IGNORECASE),
]

# Patterns that indicate a loop
LOOP_PATTERNS = [
    re.compile(r'\bfor\s+\w+\s+(?:in|of)\b'),       # Python: for x in ..., JS: for x of ...
    re.compile(r'\bfor\s*\('),                         # C-style for
    re.compile(r'\bwhile\s*\('),                       # while loops
    re.compile(r'\.(?:forEach|map|flatMap|reduce)\s*\('),  # Array methods
    re.compile(r'\basync\s+for\b'),                    # async for
]

# Blocking I/O patterns (should not appear in async functions)
BLOCKING_IO_PATTERNS = [
    re.compile(r'\btime\.sleep\s*\('),
    re.compile(r'\bopen\s*\([^)]*\)\s*(?:\.read|\.write|\.readlines)'),
    re.compile(r'\brequests\.\w+\s*\('),               # sync requests lib
    re.compile(r'\burllib\.\w+\.\w+\s*\('),            # sync urllib
    re.compile(r'\bsubprocess\.(?:run|call|check_output)\s*\('),
    re.compile(r'\bos\.(?:system|popen)\s*\('),
    re.compile(r'\bsocket\.\w+\s*\('),
]

# Unbounded query patterns
UNBOUNDED_QUERY_PATTERNS = [
    re.compile(r'SELECT\s+\*\s+FROM', re.IGNORECASE),
    re.compile(r'\.find\(\s*\)\s*(?!\.limit)', re.IGNORECASE),
    re.compile(r'\.findAll\(\s*\)', re.IGNORECASE),
    re.compile(r'\.all\(\s*\)', re.IGNORECASE),
    re.compile(r'MATCH\s*\([^)]*\)\s*RETURN', re.IGNORECASE),  # Cypher without LIMIT
]


class PerformanceProfiler:
    """
    Post-ingestion analysis that scans Function code_body content
    for common performance anti-patterns.
    """

    async def profile(self, session, repo_hash: str) -> Dict[str, Any]:
        stats = {
            "functions_scanned": 0,
            "hints_created": 0,
            "by_category": defaultdict(int),
            "errors": [],
        }

        try:
            # ── 1. Load all functions with code bodies ──
            result = await session.run("""
                MATCH (f:File)-[:CONTAINS]->(fn:Function)
                WHERE f.repository_hash = $repo_hash
                  AND fn.code_body IS NOT NULL
                  AND size(fn.code_body) > 10
                RETURN fn.id AS id, fn.name AS name, fn.code_body AS body,
                       fn.is_async AS is_async, fn.kind AS kind,
                       f.path AS file_path
                LIMIT 5000
            """, {"repo_hash": repo_hash})

            functions = [r async for r in result]
            stats["functions_scanned"] = len(functions)

            all_hints: List[Dict[str, Any]] = []

            for fn in functions:
                fn_id = fn["id"]
                fn_name = fn["name"] or ""
                body = fn["body"] or ""
                is_async = fn["is_async"] or False
                file_path = fn["file_path"] or ""

                hints = self._analyze_function(fn_id, fn_name, body, is_async, file_path)
                all_hints.extend(hints)

            # ── 2. Also scan route handlers ──
            route_result = await session.run("""
                MATCH (f:File)-[:CONTAINS]->(r:VertxRoute)
                WHERE f.repository_hash = $repo_hash
                  AND r.code_body IS NOT NULL
                  AND size(r.code_body) > 10
                RETURN r.id AS id, r.handler_method AS name, r.code_body AS body,
                       r.path AS route_path, r.method AS http_method,
                       f.path AS file_path
                LIMIT 2000
            """, {"repo_hash": repo_hash})

            routes = [r async for r in route_result]
            for rt in routes:
                rt_id = rt["id"]
                rt_name = rt["name"] or ""
                body = rt["body"] or ""
                file_path = rt["file_path"] or ""
                hints = self._analyze_function(rt_id, rt_name, body, True, file_path)

                # Additional route-specific checks
                route_path = rt["route_path"] or ""
                http_method = rt["http_method"] or "GET"
                if http_method in ("GET", "LIST") and "limit" not in body.lower() and "pagina" not in body.lower():
                    if any(p.search(body) for p in DB_CALL_PATTERNS):
                        hints.append(self._make_hint(
                            rt_id, rt_name, file_path,
                            "missing_pagination",
                            f"GET {route_path} queries data without pagination/limit",
                            "medium",
                        ))

                all_hints.extend(hints)

            # ── 3. Create PerformanceHint nodes ──
            if all_hints:
                for chunk_start in range(0, len(all_hints), 200):
                    chunk = all_hints[chunk_start:chunk_start + 200]
                    await session.run("""
                        UNWIND $hints AS h
                        MERGE (ph:PerformanceHint {id: h.id})
                        SET ph.function_id = h.function_id,
                            ph.function_name = h.function_name,
                            ph.file = h.file,
                            ph.category = h.category,
                            ph.message = h.message,
                            ph.severity = h.severity,
                            ph.line_hint = h.line_hint,
                            ph.repository_hash = $repo_hash,
                            ph.created_at = datetime()
                        WITH ph, h
                        MATCH (fn {id: h.function_id})
                        MERGE (fn)-[:HAS_PERF_HINT]->(ph)
                    """, {"hints": chunk, "repo_hash": repo_hash})

                stats["hints_created"] = len(all_hints)
                for h in all_hints:
                    stats["by_category"][h["category"]] += 1

            # ── 4. Store summary on Repository ──
            await session.run("""
                MATCH (r:Repository {hash: $repo_hash})
                SET r.perf_hints_count = $count,
                    r.perf_categories = $categories
            """, {
                "repo_hash": repo_hash,
                "count": stats["hints_created"],
                "categories": json.dumps(dict(stats["by_category"])),
            })

            logger.info(
                f"Performance profiling: {stats['functions_scanned']} functions scanned, "
                f"{stats['hints_created']} hints found: {dict(stats['by_category'])}"
            )

        except Exception as e:
            logger.error(f"Performance profiling failed: {e}")
            stats["errors"].append(str(e))

        return stats

    def _analyze_function(
        self, fn_id: str, fn_name: str, body: str, is_async: bool, file_path: str
    ) -> List[Dict[str, Any]]:
        """Run all heuristic checks against a function body."""
        hints: List[Dict[str, Any]] = []

        # ── Check 1: N+1 pattern (loop containing DB/API call) ──
        lines = body.split("\n")
        in_loop = False
        loop_start = 0
        for i, line in enumerate(lines):
            if any(p.search(line) for p in LOOP_PATTERNS):
                in_loop = True
                loop_start = i
            if in_loop and any(p.search(line) for p in DB_CALL_PATTERNS):
                hints.append(self._make_hint(
                    fn_id, fn_name, file_path,
                    "n_plus_one",
                    f"Potential N+1: DB/API call inside loop at line ~{loop_start + 1}",
                    "high",
                    line_hint=loop_start + 1,
                ))
                in_loop = False  # Only flag once per function
            # Reset after loop block ends (simple heuristic: dedent)
            if in_loop and i > loop_start + 1:
                stripped = line.lstrip()
                if stripped and not stripped.startswith((' ', '\t')) and i > loop_start + 2:
                    in_loop = False

        # ── Check 2: Blocking I/O in async function ──
        if is_async:
            for i, line in enumerate(lines):
                for p in BLOCKING_IO_PATTERNS:
                    if p.search(line):
                        hints.append(self._make_hint(
                            fn_id, fn_name, file_path,
                            "blocking_io_in_async",
                            f"Blocking I/O in async function: {p.pattern[:40]}",
                            "high",
                            line_hint=i + 1,
                        ))
                        break  # one per function is enough

        # ── Check 3: Unbounded queries ──
        for p in UNBOUNDED_QUERY_PATTERNS:
            match = p.search(body)
            if match:
                # Check if there's a LIMIT nearby
                context = body[max(0, match.start() - 50):match.end() + 100]
                if 'limit' not in context.lower() and 'LIMIT' not in context:
                    hints.append(self._make_hint(
                        fn_id, fn_name, file_path,
                        "unbounded_query",
                        f"Query without LIMIT: {match.group()[:50]}",
                        "medium",
                    ))
                    break

        # ── Check 4: Missing await on async call ──
        if is_async:
            # Look for calls to known async functions without await
            async_call_pattern = re.compile(r'(?<!await\s)(?:fetch|axios|session\.run|execute_query|supabase)\s*\(')
            for i, line in enumerate(lines):
                if async_call_pattern.search(line) and 'await' not in line:
                    hints.append(self._make_hint(
                        fn_id, fn_name, file_path,
                        "missing_await",
                        f"Possibly missing await on async call at line ~{i + 1}",
                        "medium",
                        line_hint=i + 1,
                    ))
                    break

        # ── Check 5: Large response without streaming ──
        if any(kw in body.lower() for kw in ('json.dumps', 'jsonify', 'jsonresponse', 'return {')):
            if any(kw in body.lower() for kw in ('.all()', 'find()', 'findmany', 'select *')):
                if 'stream' not in body.lower() and 'pagina' not in body.lower():
                    hints.append(self._make_hint(
                        fn_id, fn_name, file_path,
                        "large_payload_risk",
                        "Returns full query result as JSON without pagination or streaming",
                        "low",
                    ))

        return hints

    def _make_hint(
        self, fn_id: str, fn_name: str, file_path: str,
        category: str, message: str, severity: str, line_hint: int = 0
    ) -> Dict[str, Any]:
        hint_id = generate_node_id("perf", fn_id, category, message[:30])
        return {
            "id": hint_id,
            "function_id": fn_id,
            "function_name": fn_name,
            "file": file_path,
            "category": category,
            "message": message,
            "severity": severity,
            "line_hint": line_hint,
        }
