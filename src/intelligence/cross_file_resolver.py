"""
Cross-File Call Resolver — Feature 1
Resolves `from X import Y` / `import X` / `require('X')` statements to actual
Function, Class, and File nodes in the graph, creating precise CALLS edges
across file boundaries.

Also handles Feature 2 (Type Flow):
Propagates parameter + return type annotations along CALLS edges so agents
can trace type flow across the codebase.
"""
import json
import hashlib
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict

from src.core.logger import get_logger

logger = get_logger(__name__)


def _node_id(node_type: str, *parts) -> str:
    combined = f"{node_type}:{'|'.join(str(p) for p in parts)}"
    return hashlib.md5(combined.encode()).hexdigest()[:16]


class CrossFileResolver:
    """
    Post-ingestion pass that:
    1. Reads all Function.calls lists from the graph
    2. Reads all File import/from_import data
    3. Resolves `callee_name` → Function node across files
    4. Creates CALLS_CROSS_FILE edges
    5. Propagates type annotations along those edges (TYPE_FLOWS)
    """

    async def resolve(self, session, repo_hash: str) -> Dict[str, Any]:
        stats = {
            "cross_file_calls_created": 0,
            "type_flow_edges": 0,
            "errors": [],
        }

        try:
            # ── Step 1: Build import map (file → imported symbols + source file) ──
            import_map = await self._build_import_map(session, repo_hash)

            # ── Step 2: For every Function, resolve its calls[] across files ──
            func_result = await session.run("""
                MATCH (f:File)-[:CONTAINS]->(fn:Function)
                WHERE fn.calls IS NOT NULL AND size(fn.calls) > 0
                  AND f.repository_hash = $repo_hash
                RETURN fn.id AS fn_id, fn.name AS fn_name, fn.calls AS calls,
                       f.path AS file_path, f.id AS file_id
            """, {"repo_hash": repo_hash})

            func_records = [r async for r in func_result]

            # ── Step 3: Build a global name→node_id index ──
            name_index = await self._build_function_name_index(session, repo_hash)

            # ── Step 4: Create cross-file CALLS edges ──
            cross_pairs: list[dict] = []
            for rec in func_records:
                fn_id = rec["fn_id"]
                fn_name = rec["fn_name"]
                file_path = rec["file_path"] or ""
                calls = rec["calls"] or []

                imported_symbols = import_map.get(file_path, {})

                for callee in calls:
                    # Skip self-calls or same-file (already handled by node_factory)
                    bare_name = callee.split(".")[-1] if "." in callee else callee

                    # 1. Check if callee was imported from another file
                    if bare_name in imported_symbols:
                        target_file = imported_symbols[bare_name]
                        # Look up function in target file
                        target_key = f"{target_file}::{bare_name}"
                        target_id = name_index.get(target_key)
                        if target_id and target_id != fn_id:
                            cross_pairs.append({
                                "caller": fn_id,
                                "callee": target_id,
                                "callee_name": bare_name,
                                "source_file": file_path,
                                "target_file": target_file,
                            })
                    else:
                        # 2. Try global name match (fallback) — same repo, different file
                        for key, tid in name_index.items():
                            if key.endswith(f"::{bare_name}") and tid != fn_id:
                                src_file = key.split("::")[0]
                                if src_file != file_path:
                                    cross_pairs.append({
                                        "caller": fn_id,
                                        "callee": tid,
                                        "callee_name": bare_name,
                                        "source_file": file_path,
                                        "target_file": src_file,
                                    })
                                    break  # first match only to avoid explosion

            # Batch-create edges
            if cross_pairs:
                for chunk_start in range(0, len(cross_pairs), 500):
                    chunk = cross_pairs[chunk_start:chunk_start + 500]
                    await session.run("""
                        UNWIND $pairs AS p
                        MATCH (caller:Function {id: p.caller})
                        MATCH (callee:Function {id: p.callee})
                        MERGE (caller)-[r:CALLS_CROSS_FILE]->(callee)
                        SET r.callee_name = p.callee_name,
                            r.source_file = p.source_file,
                            r.target_file = p.target_file,
                            r.created_at = datetime()
                    """, {"pairs": chunk})
                stats["cross_file_calls_created"] = len(cross_pairs)

            # ── Step 5: Type flow propagation ──
            type_count = await self._propagate_type_flow(session, repo_hash)
            stats["type_flow_edges"] = type_count

            logger.info(
                f"Cross-file resolution: {stats['cross_file_calls_created']} cross-calls, "
                f"{stats['type_flow_edges']} type-flow edges"
            )

        except Exception as e:
            logger.error(f"Cross-file resolution failed: {e}")
            stats["errors"].append(str(e))

        return stats

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    async def _build_import_map(
        self, session, repo_hash: str
    ) -> Dict[str, Dict[str, str]]:
        """
        Returns: { file_path: { symbol_name: source_file_path } }
        Works for Python (from_imports) and JS/TS (imports with source).
        """
        result = await session.run("""
            MATCH (f:File)
            WHERE f.repository_hash = $repo_hash AND f.imports_json IS NOT NULL
            RETURN f.path AS path, f.imports_json AS imports_json
        """, {"repo_hash": repo_hash})

        import_map: Dict[str, Dict[str, str]] = {}
        records = [r async for r in result]

        # Also query IMPORTS edges (already created by enhanced_graph_linker)
        edge_result = await session.run("""
            MATCH (a:File)-[:IMPORTS]->(b:File)
            WHERE a.repository_hash = $repo_hash
            RETURN a.path AS importer, b.path AS imported
        """, {"repo_hash": repo_hash})
        edge_records = [r async for r in edge_result]

        # Build file→file import lookup
        file_imports: Dict[str, Set[str]] = defaultdict(set)
        for rec in edge_records:
            file_imports[rec["importer"]].add(rec["imported"])

        for rec in records:
            path = rec["path"] or ""
            try:
                imports = json.loads(rec["imports_json"]) if isinstance(rec["imports_json"], str) else rec["imports_json"]
            except Exception:
                continue

            symbols: Dict[str, str] = {}
            if isinstance(imports, list):
                for imp in imports:
                    if isinstance(imp, str):
                        # Java-style string imports: "import com.example.Foo"
                        # or Python basic imports: "os", "sys"
                        clean = imp.strip().rstrip(";")
                        if clean.startswith("import "):
                            clean = clean[len("import "):]
                        # Extract the class/module name (last segment)
                        parts = clean.split(".")
                        short_name = parts[-1] if parts else clean
                        if short_name and short_name != "*":
                            # Try to match to known files via IMPORTS edges
                            for imported_file in file_imports.get(path, set()):
                                if short_name.lower() in imported_file.lower():
                                    symbols[short_name] = imported_file
                                    break
                            # Also store the full dotted path for later matching
                            if short_name not in symbols:
                                symbols[short_name] = clean.replace(".", "/")

                    elif isinstance(imp, dict):
                        # Python from_imports: {"module": "services.auth", "name": "AuthService"}
                        name = imp.get("name", "")
                        module = imp.get("module", "") or imp.get("source", "")
                        if name and module:
                            # Try to find the target file in known IMPORTS edges
                            for imported_file in file_imports.get(path, set()):
                                if module.replace(".", "/") in imported_file or name.lower() in imported_file.lower():
                                    symbols[name] = imported_file
                                    break
                            # Fallback: store module path for later matching
                            if name and name not in symbols:
                                symbols[name] = module.replace(".", "/")

                        # JS/TS imports with specifiers
                        specifiers = imp.get("specifiers", [])
                        source = imp.get("source", "")
                        for spec in specifiers:
                            if isinstance(spec, dict):
                                symbols[spec.get("name", "")] = source
                            elif isinstance(spec, str):
                                symbols[spec] = source
                        # Default imports
                        if imp.get("default"):
                            symbols[imp["default"]] = source

            import_map[path] = symbols

        return import_map

    async def _build_function_name_index(
        self, session, repo_hash: str
    ) -> Dict[str, str]:
        """Returns { "file_path::func_name": node_id }"""
        result = await session.run("""
            MATCH (f:File)-[:CONTAINS]->(fn:Function)
            WHERE f.repository_hash = $repo_hash
            RETURN fn.id AS id, fn.name AS name, f.path AS file_path
        """, {"repo_hash": repo_hash})

        index: Dict[str, str] = {}
        async for rec in result:
            name = rec["name"] or ""
            fpath = rec["file_path"] or ""
            if name:
                index[f"{fpath}::{name}"] = rec["id"]
        return index

    async def _propagate_type_flow(self, session, repo_hash: str) -> int:
        """
        Feature 2: Type flow analysis.
        For every CALLS / CALLS_CROSS_FILE edge, if the caller passes typed args
        and the callee has typed parameters, create a TYPE_FLOWS edge capturing
        the type mapping.
        """
        # Get functions with return_type and params, scoped to this repo
        result = await session.run("""
            MATCH (caller:Function)-[r:CALLS|CALLS_CROSS_FILE]->(callee:Function)
            WHERE (caller.return_type IS NOT NULL OR callee.params IS NOT NULL)
              AND caller.repository_hash = $repo_hash
            RETURN caller.id AS cid, caller.return_type AS ret,
                   callee.id AS tid, callee.params AS params
            LIMIT 5000
        """, {"repo_hash": repo_hash})

        type_pairs: list[dict] = []
        async for rec in result:
            ret_type = rec["ret"] or ""
            params_raw = rec["params"] or "[]"
            try:
                params = json.loads(params_raw) if isinstance(params_raw, str) else params_raw
            except Exception:
                params = []

            param_types = [p.get("type", "") for p in params if isinstance(p, dict) and p.get("type")]
            if ret_type or param_types:
                type_pairs.append({
                    "caller": rec["cid"],
                    "callee": rec["tid"],
                    "return_type": ret_type,
                    "param_types": param_types,
                })

        if type_pairs:
            for chunk_start in range(0, len(type_pairs), 500):
                chunk = type_pairs[chunk_start:chunk_start + 500]
                await session.run("""
                    UNWIND $pairs AS p
                    MATCH (caller:Function {id: p.caller})
                    MATCH (callee:Function {id: p.callee})
                    MERGE (caller)-[r:TYPE_FLOWS]->(callee)
                    SET r.return_type = p.return_type,
                        r.param_types = p.param_types,
                        r.created_at = datetime()
                """, {"pairs": chunk})

        return len(type_pairs)
