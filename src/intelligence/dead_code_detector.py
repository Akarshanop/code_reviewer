"""
Dead Code Detector — Feature 5
Uses the call graph to find functions, classes, and routes that are never
referenced by any other code. Reports them so QA/developers can decide
whether to remove or document them.

Rules:
- A function is "dead" if it has zero incoming CALLS / CALLS_FUNCTION / TESTS edges
  AND is not a route handler, entry point, test function, or __init__/__main__.
- A class is "dead" if none of its methods are called and it's not extended.
- A route is "dead" if it has no incoming CALLS_API edges from any frontend component.
"""
import json
from typing import Dict, List, Any
from collections import defaultdict

from src.core.logger import get_logger

logger = get_logger(__name__)


class DeadCodeDetector:
    """
    Post-ingestion analysis that identifies unreachable code in the graph.
    Results are stored as properties on nodes (dead_code = true, dead_reason = ...)
    and as a summary on the Repository node.
    """

    # Names that are never "dead" even with zero callers
    EXEMPT_NAMES = {
        '__init__', '__main__', 'main', 'setup', 'teardown',
        'configure', 'register', 'bootstrap', 'create_app',
        'app', 'index', 'default',
    }

    async def detect_dead_code(self, session, repo_hash: str) -> Dict[str, Any]:
        stats = {
            "dead_functions": 0,
            "dead_routes": 0,
            "total_functions": 0,
            "total_routes": 0,
            "dead_code_details": [],
            "errors": [],
        }

        try:
            # ══════════════════════════════════════════════════════════════
            # 1. Dead functions: no incoming calls, not a handler/test/entry
            # ══════════════════════════════════════════════════════════════
            dead_fn_result = await session.run("""
                MATCH (parent_file:File {repository_hash: $repo_hash})-[:CONTAINS]->(fn:Function)
                OPTIONAL MATCH (caller)-[r:CALLS|CALLS_FUNCTION]->(fn)
                OPTIONAL MATCH (test)-[t:TESTS]->(fn)
                WITH fn, parent_file,
                     count(r) AS incoming_calls,
                     count(t) AS test_coverage
                RETURN fn.id AS id, fn.name AS name, fn.file AS file,
                       fn.is_handler AS is_handler,
                       COALESCE(fn.is_test, parent_file:TestFile) AS is_test,
                       fn.start_line AS line,
                       incoming_calls, test_coverage
            """, {"repo_hash": repo_hash})

            all_functions = []
            dead_functions = []
            async for rec in dead_fn_result:
                all_functions.append(rec)
                name = rec["name"] or ""
                incoming = rec["incoming_calls"] or 0
                tests = rec["test_coverage"] or 0
                is_handler = rec["is_handler"] or False
                is_test = rec["is_test"] or False

                if incoming == 0 and tests == 0 and not is_handler and not is_test:
                    # Check exempt names
                    lower_name = name.lower()
                    if lower_name not in self.EXEMPT_NAMES and not lower_name.startswith('_'):
                        dead_functions.append(rec)

            stats["total_functions"] = len(all_functions)
            stats["dead_functions"] = len(dead_functions)

            # Mark dead functions in the graph
            if dead_functions:
                dead_fn_ids = [df["id"] for df in dead_functions]
                for chunk_start in range(0, len(dead_fn_ids), 500):
                    chunk = dead_fn_ids[chunk_start:chunk_start + 500]
                    await session.run("""
                        UNWIND $ids AS fnid
                        MATCH (fn:Function {id: fnid})
                        SET fn.dead_code = true,
                            fn.dead_reason = 'no_incoming_calls'
                    """, {"ids": chunk})

                # Collect details for the top 50 most suspicious
                for df in dead_functions[:50]:
                    stats["dead_code_details"].append({
                        "type": "function",
                        "name": df["name"],
                        "file": df["file"],
                        "line": df["line"],
                        "reason": "No incoming calls, not a handler/test",
                    })

            # ══════════════════════════════════════════════════════════════
            # 2. Dead routes: no incoming CALLS_API from frontend
            # ══════════════════════════════════════════════════════════════
            dead_route_result = await session.run("""
                MATCH (route:VertxRoute)
                WHERE EXISTS { MATCH (:File {repository_hash: $repo_hash})-[:CONTAINS]->(route) }
                OPTIONAL MATCH (caller)-[r:CALLS_API]->(route)
                WITH route, count(r) AS api_calls
                RETURN route.id AS id, route.path AS path, route.method AS method,
                       route.file AS file, api_calls
            """, {"repo_hash": repo_hash})

            all_routes = []
            dead_routes = []
            async for rec in dead_route_result:
                all_routes.append(rec)
                if (rec["api_calls"] or 0) == 0:
                    dead_routes.append(rec)

            stats["total_routes"] = len(all_routes)
            stats["dead_routes"] = len(dead_routes)

            if dead_routes:
                dead_route_ids = [dr["id"] for dr in dead_routes]
                for chunk_start in range(0, len(dead_route_ids), 500):
                    chunk = dead_route_ids[chunk_start:chunk_start + 500]
                    await session.run("""
                        UNWIND $ids AS rid
                        MATCH (r:VertxRoute {id: rid})
                        SET r.dead_code = true,
                            r.dead_reason = 'no_frontend_consumers'
                    """, {"ids": chunk})

                for dr in dead_routes[:20]:
                    stats["dead_code_details"].append({
                        "type": "route",
                        "path": dr["path"],
                        "method": dr["method"],
                        "file": dr["file"],
                        "reason": "No frontend components call this API",
                    })

            # ══════════════════════════════════════════════════════════════
            # 3. Store summary on Repository node
            # ══════════════════════════════════════════════════════════════
            await session.run("""
                MATCH (r:Repository {hash: $repo_hash})
                SET r.dead_functions = $dead_fn,
                    r.dead_routes = $dead_rt,
                    r.total_functions = $total_fn,
                    r.total_routes = $total_rt,
                    r.dead_code_ratio = CASE WHEN $total_fn > 0
                        THEN toFloat($dead_fn) / toFloat($total_fn)
                        ELSE 0.0 END
            """, {
                "repo_hash": repo_hash,
                "dead_fn": stats["dead_functions"],
                "dead_rt": stats["dead_routes"],
                "total_fn": stats["total_functions"],
                "total_rt": stats["total_routes"],
            })

            logger.info(
                f"Dead code: {stats['dead_functions']}/{stats['total_functions']} functions, "
                f"{stats['dead_routes']}/{stats['total_routes']} routes"
            )

        except Exception as e:
            logger.error(f"Dead code detection failed: {e}")
            stats["errors"].append(str(e))

        return stats
