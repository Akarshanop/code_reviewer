from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

from src.database.neo4j_client import neo4j_client
from src.core.exceptions import GraphDatabaseException
from src.core.logger import get_logger
from src.intelligence.domain_model import domain_engine
from src.intelligence.confidence_engine import ConfidenceEngine, ConfidenceQueryResult, confidence_engine as _default_confidence_engine

logger = get_logger(__name__)

# Every relationship type the ingestion pipeline produces, kept in one place
# so queries can be maintained easily.
_ALL_CODE_RELS = (
    "CALLS|CALLS_FUNCTION|CALLS_ROUTE|CALLS_API|"
    "IMPORTS|EXTENDS|DEPENDS_ON|DEPENDS_ON_SERVICE|"
    "CONTAINS|BELONGS_TO|CONTAINS_MODULE|CONTAINS_FILE|"
    "MAKES_API_CALL|PUBLISHES_TO|CONSUMES_FROM|"
    "USES_DTO|USES_DTO_FIELDS|RENDERS|HANDLED_BY|MOUNTS|"
    "GUARDS|ESCALATES_FAILURE|REQUIRES_CONSISTENCY"
)


class GraphService:
    """
    Service for graph-based code analysis and queries.

    All queries reference the actual relationship types created by
    EdgeLinker + EnhancedGraphLinker + NodeFactory.
    """

    def __init__(self, confidence_engine: Optional[ConfidenceEngine] = None):
        self.neo4j = neo4j_client
        self.confidence = confidence_engine or _default_confidence_engine

    # ─── Entity discovery ─────────────────────────────────────────────

    async def find_related_entities(
        self,
        entity_name: str,
        relationship_types: List[str] = None,
        max_depth: int = 3,
        direction: str = "BOTH",
    ) -> Dict[str, Any]:
        """Find entities related to a given entity by name."""
        rel_filter = ""
        if relationship_types:
            rel_filter = f":{('|'.join(relationship_types))}"

        dir_symbol = {"OUTGOING": "->", "INCOMING": "<-", "BOTH": "-"}.get(direction, "-")

        query = f"""
        MATCH path = (start {{name: $entity_name}})-[*1..{max_depth}{rel_filter}]{dir_symbol}(end)
        WHERE start <> end
        RETURN
            start, end,
            [n IN nodes(path) | labels(n)[0]] AS node_labels,
            [r IN relationships(path) | type(r)] AS relationship_types,
            length(path) AS path_length
        ORDER BY path_length
        LIMIT 150
        """
        try:
            results = await self.neo4j.execute_query(query, {"entity_name": entity_name})

            relationships = defaultdict(list)
            for record in results:
                rel_type = record["relationship_types"][-1] if record["relationship_types"] else "UNKNOWN"
                relationships[rel_type].append({
                    "target": dict(record["end"]),
                    "path_length": record["path_length"],
                    "target_labels": record["node_labels"][-1] if record["node_labels"] else None,
                })

            return {
                "source": dict(results[0]["start"]) if results else None,
                "relationships": dict(relationships),
                "total_relationships": sum(len(v) for v in relationships.values()),
            }
        except Exception as e:
            logger.error("Failed to find related entities", entity=entity_name, error=str(e))
            raise GraphDatabaseException(f"Failed to find related entities: {e}")

    # ─── Impact analysis ──────────────────────────────────────────────

    async def analyze_impact(
        self, file_path: str, change_type: str = "modify"
    ) -> Dict[str, Any]:
        """Analyze the blast-radius of changing a file.

        Now also traverses IMPORTS, CALLS_FUNCTION and EXTENDS edges so that
        cross-file dependencies are visible.
        """
        queries = {
            "direct_dependencies": """
            MATCH (f:File {path: $file_path})
            OPTIONAL MATCH (f)-[:CONTAINS]->(component)
            OPTIONAL MATCH (component)-[r]-(dependent)
            WHERE dependent <> component
            RETURN component, type(r) AS relationship_type, dependent
            LIMIT 50
            """,
            "import_dependents": """
            MATCH (f:File {path: $file_path})
            OPTIONAL MATCH (importer:File)-[:IMPORTS]->(f)
            RETURN importer.path AS importer_path, importer.file_name AS importer_name
            """,
            "function_callers": """
            MATCH (f:File {path: $file_path})-[:CONTAINS]->(fn:Function)
            OPTIONAL MATCH (caller:Function)-[:CALLS_FUNCTION]->(fn)
            OPTIONAL MATCH (caller_file:File)-[:CONTAINS]->(caller)
            RETURN fn.name AS callee, caller.name AS caller_name, caller_file.path AS caller_file
            """,
            "api_calls": """
            MATCH (f:File {path: $file_path})
            OPTIONAL MATCH (f)-[:CONTAINS]->(component)-[:MAKES_API_CALL]->(api:ApiCall)
            OPTIONAL MATCH (api)-[:CALLS_ROUTE]->(route:VertxRoute)
            RETURN component, api, route
            """,
            "eventbus_flows": """
            MATCH (f:File {path: $file_path})
            OPTIONAL MATCH (f)-[:CONTAINS]->(component)-[:PUBLISHES_TO|CONSUMES_FROM]->(eventbus:EventBusNode)
            RETURN component, eventbus
            """,
            "type_dependents": """
            MATCH (f:File {path: $file_path})-[:CONTAINS]->(c:Class)-[:IS_TYPE]->(t:Type)
            OPTIONAL MATCH (sub:Type)-[:SUBTYPES_OF]->(t)
            OPTIONAL MATCH (fn:Function)-[:RETURNS_TYPE]->(sub)
            OPTIONAL MATCH (ff:File)-[:CONTAINS]->(fn)
            RETURN t.name AS type_name, count(DISTINCT sub) AS subtype_count, count(DISTINCT ff) AS impacted_files
            """,
        }

        results: Dict[str, Any] = {}
        for name, query in queries.items():
            try:
                results[name] = await self.neo4j.execute_query(query, {"file_path": file_path})
            except Exception as e:
                logger.warning(f"Impact query '{name}' failed: {e}")
                results[name] = []

        total_components = len(results.get("direct_dependencies", []))
        total_imports = len([r for r in results.get("import_dependents", []) if r.get("importer_path")])
        total_callers = len([r for r in results.get("function_callers", []) if r.get("caller_name")])
        total_apis = len(results.get("api_calls", []))
        total_eventbus = len(results.get("eventbus_flows", []))
        total_type_dependents = len(results.get("type_dependents", []))

        structural_score = min(
            100,
            (total_components * 5) + (total_imports * 8) + (total_callers * 6)
            + (total_apis * 10) + (total_eventbus * 15) + (total_type_dependents * 6),
        )

        behavioral_flags: list[str] = []
        behavioral_boost = 0
        try:
            beh_query = """
            MATCH (f:File {path: $file_path})-[:CONTAINS]->(n)
            WHERE n.behavioral_labels IS NOT NULL
            RETURN n.behavioral_labels AS labels, n.name AS name
            """
            for rec in (await self.neo4j.execute_query(beh_query, {"file_path": file_path}) or []):
                labels = rec.get("labels", [])
                if isinstance(labels, list):
                    behavioral_flags.extend(labels)
                    if "AuthCritical" in labels:
                        behavioral_boost += 20
                    if "TransactionBoundary" in labels:
                        behavioral_boost += 15
                    if "AsyncFanOut" in labels:
                        behavioral_boost += 10
        except Exception:
            pass

        impact_score = min(100, structural_score + behavioral_boost)

        domains = domain_engine.infer_domain_for_node(file_path=file_path)
        domain_names = [d[0] for d in domains[:3]]

        return {
            "file": file_path,
            "impact_score": impact_score,
            "risk_level": (
                "critical" if impact_score > 75
                else "high" if impact_score > 50
                else "medium" if impact_score > 20
                else "low"
            ),
            "breakdown": {
                "components_affected": total_components,
                "import_dependents": total_imports,
                "function_callers": total_callers,
                "apis_affected": total_apis,
                "eventbus_flows_affected": total_eventbus,
                "type_dependents": total_type_dependents,
                "behavioral_boost": behavioral_boost,
            },
            "behavioral_flags": list(set(behavioral_flags)),
            "domains": domain_names,
            "details": results,
        }

    # ─── Cross-repository ─────────────────────────────────────────────

    async def find_cross_repository_dependencies(
        self, repository_name: str = None
    ) -> List[Dict[str, Any]]:
        repo_filter = "WHERE f.repository_name = $repository_name" if repository_name else ""

        query = f"""
        MATCH (f:File)
        {repo_filter}
        MATCH (f)-[:CONTAINS]->(source)
        MATCH (source)-[r]->(target)
        WHERE target.repository_name <> source.repository_name
          OR target.repository_name IS NULL
        RETURN
            source.name AS source_name, labels(source)[0] AS source_type,
            type(r) AS relationship_type,
            target.name AS target_name, labels(target)[0] AS target_type,
            target.repository_name AS target_repository,
            source.repository_name AS source_repository
        ORDER BY source.repository_name, target.repository_name
        LIMIT 200
        """
        params = {"repository_name": repository_name} if repository_name else {}
        results = await self.neo4j.execute_query(query, params)

        deps = defaultdict(list)
        for rec in results:
            key = f"{rec['source_repository']} -> {rec['target_repository'] or 'external'}"
            deps[key].append({
                "source": f"{rec['source_type']}: {rec['source_name']}",
                "target": f"{rec['target_type']}: {rec['target_name']}",
                "relationship": rec["relationship_type"],
            })

        return [{"dependency_path": k, "connections": v, "count": len(v)} for k, v in deps.items()]

    # ─── Similarity (APOC-free) ───────────────────────────────────────

    async def find_similar_components(
        self,
        component_name: str,
        similarity_threshold: float = 0.7,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Jaccard similarity of outgoing relationship types — no APOC needed."""
        query = """
        MATCH (c {name: $component_name})
        MATCH (similar)
        WHERE similar <> c
          AND labels(similar) = labels(c)
          AND similar.name IS NOT NULL
        OPTIONAL MATCH (c)-[r1]->()
        WITH c, similar, collect(DISTINCT type(r1)) AS c_rels
        OPTIONAL MATCH (similar)-[r2]->()
        WITH c, similar, c_rels, collect(DISTINCT type(r2)) AS s_rels
        WITH similar, c_rels, s_rels,
             [x IN c_rels WHERE x IN s_rels] AS intersection
        WHERE size(c_rels) + size(s_rels) - size(intersection) > 0
        WITH similar,
             toFloat(size(intersection)) /
             (size(c_rels) + size(s_rels) - size(intersection)) AS score,
             c_rels
        WHERE score >= $threshold
        RETURN similar, score AS similarity_score, c_rels
        ORDER BY score DESC
        LIMIT $limit
        """
        results = await self.neo4j.execute_query(query, {
            "component_name": component_name,
            "limit": limit,
            "threshold": similarity_threshold,
        })
        return [
            {
                "component": dict(r["similar"]),
                "similarity_score": r["similarity_score"],
                "shared_relationships": r["c_rels"],
            }
            for r in results
        ]

    # ─── Metrics ──────────────────────────────────────────────────────

    async def get_codebase_metrics(self) -> Dict[str, Any]:
        metrics_queries = {
            "repository_stats": """
            MATCH (r:Repository)
            RETURN r.name AS repository, r.repo_type AS type,
                   r.last_ingested AS last_ingested, r.ingestion_status AS status
            """,
            "node_counts": """
            MATCH (n)
            RETURN labels(n)[0] AS type, count(*) AS count
            ORDER BY count DESC
            """,
            "relationship_counts": """
            MATCH ()-[r]->()
            RETURN type(r) AS relationship_type, count(*) AS count
            ORDER BY count DESC
            LIMIT 25
            """,
            "top_connected_components": """
            MATCH (n)
            WHERE n:VertxRoute OR n:ReactComponent OR n:DTO OR n:Function OR n:File
            OPTIONAL MATCH (n)-[r]-()
            WITH n, count(r) AS cnt
            ORDER BY cnt DESC
            LIMIT 15
            RETURN n.name AS name, labels(n)[0] AS type, cnt AS connection_count
            """,
        }

        metrics: Dict[str, Any] = {}
        for name, query in metrics_queries.items():
            try:
                metrics[name] = await self.neo4j.execute_query(query)
            except Exception as e:
                logger.warning(f"Metrics query '{name}' failed: {e}")
                metrics[name] = []

        total_nodes = sum(r.get("count", 0) for r in metrics.get("node_counts", []))
        total_rels = sum(r.get("count", 0) for r in metrics.get("relationship_counts", []))

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "totals": {
                "nodes": total_nodes,
                "relationships": total_rels,
                "repositories": len(metrics.get("repository_stats", [])),
            },
            "breakdown": metrics,
        }

    # ─── Visualization ────────────────────────────────────────────────

    async def visualize_dependency_graph(
        self,
        start_node_id: str,
        max_depth: int = 2,
        relationship_types: List[str] = None,
    ) -> Dict[str, Any]:
        rel_filter = ""
        if relationship_types:
            rel_filter = f":{('|'.join(relationship_types))}"

        query = f"""
        MATCH path = (start)-[*0..{max_depth}{rel_filter}]-(connected)
        WHERE elementId(start) = $start_node_id
        WITH collect(DISTINCT path) AS paths
        UNWIND paths AS path
        UNWIND nodes(path) AS node
        UNWIND relationships(path) AS rel
        RETURN
            collect(DISTINCT {{id: elementId(node), labels: labels(node), properties: properties(node)}}) AS nodes,
            collect(DISTINCT {{
                id: elementId(rel), type: type(rel),
                source: elementId(startNode(rel)),
                target: elementId(endNode(rel))
            }}) AS relationships
        """
        results = await self.neo4j.execute_query(query, {"start_node_id": start_node_id})
        if not results:
            return {"nodes": [], "relationships": []}
        return {"nodes": results[0]["nodes"], "relationships": results[0]["relationships"]}

    # ─── Circular dependency detection ────────────────────────────────

    async def find_circular_dependencies(self) -> List[List[Dict[str, Any]]]:
        """Detect cycles using all meaningful code-flow relationship types."""
        query = """
        MATCH path = (start)-[:CALLS|CALLS_FUNCTION|DEPENDS_ON|DEPENDS_ON_SERVICE|IMPORTS*2..6]->(start)
        RETURN
            [node IN nodes(path) | {name: node.name, type: labels(node)[0]}] AS cycle,
            length(path) AS cycle_length
        ORDER BY cycle_length
        LIMIT 20
        """
        try:
            results = await self.neo4j.execute_query(query)
            return [r["cycle"] for r in results]
        except Exception as e:
            logger.warning(f"Circular dependency detection failed: {e}")
            return []

    # ─── Unused component detection ───────────────────────────────────

    async def find_unused_components(self) -> List[Dict[str, Any]]:
        """Find components with no incoming edges (nobody calls/uses them)."""
        query = """
        MATCH (component)
        WHERE (component:VertxRoute OR component:ReactComponent OR component:DTO OR component:Function)
        AND NOT ()-[:CALLS_ROUTE|CALLS_FUNCTION|CALLS|MAKES_API_CALL|USES_DTO|RENDERS|HANDLED_BY]->(component)
        RETURN
            component.name AS name,
            labels(component)[0] AS type,
            component.file AS file
        ORDER BY type, name
        LIMIT 50
        """
        results = await self.neo4j.execute_query(query)
        return [
            {"name": r["name"], "type": r["type"], "file": r["file"], "status": "unreferenced"}
            for r in results
        ]

    # ─── Module structure ─────────────────────────────────────────────

    async def find_module_structure(
        self, module_name: str, repo_filter: str = None
    ) -> Dict[str, Any]:
        try:
            repo_clause = "AND f.repository_name = $repository" if repo_filter else ""
            query = f"""
            MATCH (f:File)
            WHERE f.path CONTAINS $module_name {repo_clause}
            OPTIONAL MATCH (f)-[:CONTAINS]->(component)
            OPTIONAL MATCH (f)-[:IMPORTS]->(imported:File)
            RETURN
                f.path AS file_path,
                collect(DISTINCT {{name: component.name, type: labels(component)[0]}}) AS components,
                collect(DISTINCT imported.path) AS imports
            LIMIT 100
            """
            params: dict = {"module_name": module_name}
            if repo_filter:
                params["repository"] = repo_filter

            results = await self.neo4j.execute_query(query, params)
            structure: Dict[str, Any] = {"files": [], "components": [], "file_count": 0}
            for rec in results:
                structure["files"].append({
                    "path": rec["file_path"],
                    "components": rec["components"],
                    "imports": rec["imports"],
                })
                structure["components"].extend(rec["components"])
            structure["file_count"] = len(structure["files"])
            return structure
        except Exception as e:
            logger.warning(f"Module structure query failed: {e}")
            return {"files": [], "components": [], "file_count": 0}

    # ─── Impact query (simplified wrapper) ────────────────────────────

    async def query_impact(self, file_path: str) -> Dict[str, Any]:
        try:
            result = await self.analyze_impact(file_path)
            details = result.get("details", {})

            components = []
            for dep in details.get("direct_dependencies", []):
                comp = dep.get("component")
                if comp:
                    name = comp.get("name") if isinstance(comp, dict) else str(comp)
                    if name:
                        components.append(name)

            importers = [
                r["importer_path"] for r in details.get("import_dependents", [])
                if r.get("importer_path")
            ]

            apis = []
            for api_rec in details.get("api_calls", []):
                route = api_rec.get("route")
                if route:
                    path = route.get("path") if isinstance(route, dict) else str(route)
                    if path:
                        apis.append(path)

            eventbus = []
            for eb_rec in details.get("eventbus_flows", []):
                eb = eb_rec.get("eventbus")
                if eb:
                    addr = eb.get("address") if isinstance(eb, dict) else str(eb)
                    if addr:
                        eventbus.append(addr)

            return {
                "components": list(set(components)),
                "import_dependents": importers,
                "apis": list(set(apis)),
                "eventbus": list(set(eventbus)),
                "impact_score": result.get("impact_score", 0),
                "risk_level": result.get("risk_level", "low"),
                "reasons": [
                    f"Affects {len(components)} components",
                    f"{len(importers)} files import this file",
                    f"Impacts {len(apis)} APIs",
                    f"Touches {len(eventbus)} EventBus flows",
                ],
            }
        except Exception as e:
            logger.warning(f"Impact query failed: {e}")
            return {"components": [], "import_dependents": [], "apis": [], "eventbus": [], "reasons": []}

    # ─── API flow tracing ─────────────────────────────────────────────

    async def trace_api_flow(
        self, endpoint: str, include_eventbus: bool = True
    ) -> Dict[str, Any]:
        try:
            query = """
            MATCH (route:VertxRoute)
            WHERE route.path = $endpoint
               OR coalesce(route.full_path, '') = $endpoint
               OR route.path CONTAINS $endpoint
            OPTIONAL MATCH (route)-[:HANDLED_BY]->(handler:Function)
            OPTIONAL MATCH (handler)-[:CALLS|CALLS_FUNCTION]->(downstream)
            OPTIONAL MATCH (downstream)-[:PUBLISHES_TO|CONSUMES_FROM]->(eventbus:EventBusNode)
            RETURN route, handler, downstream, eventbus
            LIMIT 20
            """
            results = await self.neo4j.execute_query(query, {"endpoint": endpoint})

            flow: list[dict] = []
            eventbus_list: list[dict] = []
            seen: set[str] = set()

            for rec in results:
                for key, label in [("route", "VertxRoute"), ("handler", "Function"), ("downstream", "Service/Function")]:
                    node = rec.get(key)
                    if not node:
                        continue
                    node_dict = dict(node) if hasattr(node, "items") else {"name": str(node)}
                    node_id = node_dict.get("name", "") or node_dict.get("path", "")
                    ident = f"{key}:{node_id}"
                    if ident not in seen:
                        seen.add(ident)
                        flow.append({"step": key, "type": label, "data": node_dict})

                if include_eventbus and rec.get("eventbus"):
                    eb = rec["eventbus"]
                    eb_data = dict(eb) if hasattr(eb, "items") else {"address": str(eb)}
                    eventbus_list.append(eb_data)

            return {"endpoint": endpoint, "flow": flow, "eventbus": eventbus_list}
        except Exception as e:
            logger.warning(f"API flow trace failed: {e}")
            return {"endpoint": endpoint, "flow": [], "eventbus": []}

    # ─── Confidence-scored wrappers ────────────────────────────────────

    async def find_related_entities_with_confidence(
        self,
        entity_name: str,
        relationship_types: List[str] = None,
        max_depth: int = 3,
        direction: str = "BOTH",
    ) -> ConfidenceQueryResult:
        """Confidence-scored variant of find_related_entities."""
        rel_filter = ""
        if relationship_types:
            rel_filter = f":{('|'.join(relationship_types))}"
        dir_symbol = {"OUTGOING": "->", "INCOMING": "<-", "BOTH": "-"}.get(direction, "-")

        query = f"""
        MATCH path = (start {{name: $entity_name}})-[*1..{max_depth}{rel_filter}]{dir_symbol}(end)
        WHERE start <> end
        RETURN start, end,
               [n IN nodes(path) | labels(n)[0]] AS node_labels,
               [r IN relationships(path) | type(r)] AS relationship_types,
               length(path) AS path_length
        ORDER BY path_length
        LIMIT 150
        """
        return await self.confidence.graph_query(
            self.neo4j, query, {"entity_name": entity_name},
            description=f"Related entities for '{entity_name}' (depth={max_depth})",
        )

    async def analyze_impact_with_confidence(
        self, file_path: str, change_type: str = "modify"
    ) -> Dict[str, Any]:
        """analyze_impact with per-query confidence scoring.

        Returns the same dict as analyze_impact, but each sub-result
        includes a ConfidenceQueryResult under a `_confidence` key.
        """
        base = await self.analyze_impact(file_path, change_type)

        import_query = """
        MATCH (f:File {path: $file_path})
        OPTIONAL MATCH (importer:File)-[:IMPORTS]->(f)
        RETURN importer.path AS importer_path, importer.file_name AS importer_name
        """
        import_conf = await self.confidence.graph_query(
            self.neo4j, import_query, {"file_path": file_path},
            description=f"Import dependents of {file_path}",
        )

        caller_query = """
        MATCH (f:File {path: $file_path})-[:CONTAINS]->(fn:Function)
        OPTIONAL MATCH (caller:Function)-[:CALLS_FUNCTION]->(fn)
        OPTIONAL MATCH (caller_file:File)-[:CONTAINS]->(caller)
        RETURN fn.name AS callee, caller.name AS caller_name, caller_file.path AS caller_file
        """
        caller_conf = await self.confidence.graph_query(
            self.neo4j, caller_query, {"file_path": file_path},
            description=f"Function callers of {file_path}",
        )

        base["_confidence"] = {
            "import_dependents": import_conf.to_dict(),
            "function_callers": caller_conf.to_dict(),
        }
        return base
