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
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

from src.core.logger import get_logger

logger = get_logger(__name__)


class CrossFileResolver:
    """
    Post-ingestion pass that:
    1. Reads all Function.calls lists from the graph
    2. Reads all File import/from_import data
    3. Resolves `callee_name` → Function node across files
    4. Creates CALLS_FUNCTION edges (with a guard to skip edges already
       created by EnhancedGraphLinker)
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

            # ── Step 3: Build function indexes (FQN + scoped name) ──
            fqn_index, file_name_index, name_index = await self._build_function_indexes(session, repo_hash)

            # ── Step 4: Create cross-file CALLS edges ──
            seen_pairs: Set[Tuple[str, str]] = set()
            cross_pairs: List[Dict[str, Any]] = []
            for rec in func_records:
                fn_id = rec["fn_id"]
                file_path = rec["file_path"] or ""
                calls = rec["calls"] or []

                imported_symbols = import_map.get(file_path, {})

                for callee in calls:
                    target = self._resolve_call_target(
                        callee=callee,
                        source_file=file_path,
                        imported_symbols=imported_symbols,
                        fqn_index=fqn_index,
                        file_name_index=file_name_index,
                        name_index=name_index,
                    )
                    if not target:
                        continue

                    target_id = target["id"]
                    if target_id == fn_id:
                        continue
                    pair_key = (fn_id, target_id)
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    cross_pairs.append(
                        {
                            "caller": fn_id,
                            "callee": target_id,
                            "callee_name": target["bare_name"],
                            "source_file": file_path,
                            "target_file": target["target_file"],
                            "match_type": target["match_type"],
                            "confidence_score": target["confidence_score"],
                            "speculative": target["speculative"],
                        }
                    )

            if cross_pairs:
                for chunk_start in range(0, len(cross_pairs), 500):
                    chunk = cross_pairs[chunk_start:chunk_start + 500]
                    await session.run("""
                        UNWIND $pairs AS p
                        MATCH (caller:Function {id: p.caller})
                        MATCH (callee:Function {id: p.callee})
                        WHERE NOT (caller)-[:CALLS_FUNCTION]->(callee)
                        MERGE (caller)-[r:CALLS_FUNCTION]->(callee)
                        ON CREATE SET r.created_at = datetime()
                        SET r.callee_name = p.callee_name,
                            r.source_file = p.source_file,
                            r.target_file = p.target_file,
                            r.match_type = p.match_type,
                            r.confidence_score = p.confidence_score,
                            r.speculative = p.speculative,
                            r.last_updated = datetime()
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
    ) -> Dict[str, Dict[str, Any]]:
        """
        Returns: { file_path: { symbol_name: source_file_path } }
        Works for Python (from_imports) and JS/TS (imports with source).
        """
        result = await session.run("""
            MATCH (f:File)
            WHERE f.repository_hash = $repo_hash AND f.imports_json IS NOT NULL
            RETURN f.path AS path, f.imports_json AS imports_json
        """, {"repo_hash": repo_hash})

        import_map: Dict[str, Dict[str, Any]] = {}
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
            wildcard_modules: List[str] = []
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
                        elif short_name == "*":
                            wildcard_modules.append(clean.replace("/", "."))

                    elif isinstance(imp, dict):
                        # Python from_imports: {"module": "services.auth", "name": "AuthService"}
                        name = imp.get("name", "")
                        alias = imp.get("alias", "")
                        module = imp.get("module", "") or imp.get("source", "")
                        if name == "*" and module:
                            wildcard_modules.append(module.replace("/", "."))
                            continue
                        if name and module:
                            # Try to find the target file in known IMPORTS edges
                            for imported_file in file_imports.get(path, set()):
                                if module.replace(".", "/") in imported_file or name.lower() in imported_file.lower():
                                    symbols[name] = imported_file
                                    break
                            # Fallback: store module path for later matching
                            if name and name not in symbols:
                                symbols[name] = module.replace(".", "/")
                            if alias and alias not in symbols:
                                symbols[alias] = module.replace(".", "/")

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

            import_map[path] = {
                "symbols": symbols,
                "wildcards": wildcard_modules,
            }

        return import_map

    async def _build_function_indexes(
        self, session, repo_hash: str
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, List[Dict[str, str]]]]:
        """
        Returns:
          - fqn_index: {fqn: {"id": ..., "file_path": ...}}
          - file_name_index: {"file_path::func_name": node_id}
          - name_index: {func_name: [{"id": ..., "file_path": ..., "fqn": ...}, ...]}
        """
        result = await session.run("""
            MATCH (f:File)-[:CONTAINS]->(fn:Function)
            WHERE f.repository_hash = $repo_hash
            RETURN fn.id AS id, fn.name AS name, fn.fqn AS fqn, f.path AS file_path
        """, {"repo_hash": repo_hash})

        fqn_index: Dict[str, Dict[str, str]] = {}
        file_name_index: Dict[str, str] = {}
        name_index: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        async for rec in result:
            name = rec["name"] or ""
            fpath = rec["file_path"] or ""
            fqn = rec["fqn"] or ""
            node_id = rec["id"]
            if name:
                file_name_index[f"{fpath}::{name}"] = node_id
                name_index[name].append({"id": node_id, "file_path": fpath, "fqn": fqn})
            if fqn:
                fqn_index[fqn] = {"id": node_id, "file_path": fpath}
        return fqn_index, file_name_index, name_index

    def _resolve_call_target(
        self,
        callee: str,
        source_file: str,
        imported_symbols: Dict[str, Any],
        fqn_index: Dict[str, Dict[str, str]],
        file_name_index: Dict[str, str],
        name_index: Dict[str, List[Dict[str, str]]],
    ) -> Optional[Dict[str, Any]]:
        """Resolve a callee to a function target with confidence tiering."""
        raw = (callee or "").strip()
        if not raw:
            return None

        symbols = imported_symbols.get("symbols", {}) if isinstance(imported_symbols, dict) else {}
        wildcards = imported_symbols.get("wildcards", []) if isinstance(imported_symbols, dict) else []
        dotted = "." in raw
        bare_name = raw.split(".")[-1]

        # 1) Direct FQN match (highest confidence)
        if raw in fqn_index:
            target = fqn_index[raw]
            return {
                "id": target["id"],
                "target_file": target["file_path"],
                "bare_name": bare_name,
                "match_type": "fqn_exact",
                "confidence_score": 0.95,
                "speculative": False,
            }

        # 2) Import-scoped resolution (symbol -> module path -> candidate fqn)
        module_name = symbols.get(bare_name, "").replace("/", ".")
        if dotted:
            module_hint = ".".join(raw.split(".")[:-1])
            if module_hint in symbols:
                module_name = symbols[module_hint].replace("/", ".")
            elif module_hint:
                module_name = module_hint
        if module_name:
            candidate_fqn = f"{module_name}.{bare_name}"
            if candidate_fqn in fqn_index:
                target = fqn_index[candidate_fqn]
                return {
                    "id": target["id"],
                    "target_file": target["file_path"],
                    "bare_name": bare_name,
                    "match_type": "import_scoped",
                    "confidence_score": 0.8,
                    "speculative": False,
                }

        # 3) File-scoped match by imported file and function name
        imported_file = symbols.get(bare_name)
        if imported_file:
            file_key = f"{imported_file}::{bare_name}"
            target_id = file_name_index.get(file_key)
            if target_id:
                return {
                    "id": target_id,
                    "target_file": imported_file,
                    "bare_name": bare_name,
                    "match_type": "import_file_scoped",
                    "confidence_score": 0.8,
                    "speculative": False,
                }

        # 4) Wildcard imports (still import-scoped but weaker)
        for module in wildcards:
            candidate_fqn = f"{module}.{bare_name}"
            if candidate_fqn in fqn_index:
                target = fqn_index[candidate_fqn]
                return {
                    "id": target["id"],
                    "target_file": target["file_path"],
                    "bare_name": bare_name,
                    "match_type": "wildcard_import_scoped",
                    "confidence_score": 0.7,
                    "speculative": False,
                }

        # 5) Global fallback (speculative, low confidence)
        candidates = name_index.get(bare_name, [])
        for candidate in candidates:
            if candidate["file_path"] != source_file:
                return {
                    "id": candidate["id"],
                    "target_file": candidate["file_path"],
                    "bare_name": bare_name,
                    "match_type": "global_name_fallback",
                    "confidence_score": 0.4,
                    "speculative": True,
                }
        return None

    async def _propagate_type_flow(self, session, repo_hash: str) -> int:
        """
        Feature 2: Type flow analysis.
        For every CALLS / CALLS_FUNCTION edge, if the caller passes typed args
        and the callee has typed parameters, create a TYPE_FLOWS edge capturing
        the type mapping.
        """
        result = await session.run("""
            MATCH (caller:Function)-[r:CALLS|CALLS_FUNCTION]->(callee:Function)
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
