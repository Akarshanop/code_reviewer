"""
Enhanced Graph Linker - Creates rich semantic relationships in Neo4j

Provides capabilities that EdgeLinker does not:
1. File-level IMPORTS edges (traced from real entry points via EntryPointDetector)
2. Cross-file CALLS_FUNCTION edges (scope-aware, using file containment)
3. Class EXTENDS hierarchy edges
4. Service DEPENDS_ON_SERVICE heuristic edges

All operations use batch Cypher (UNWIND) for enterprise-scale performance
and ON CREATE SET for idempotent timestamps.
"""
from typing import Dict, List, Any, Set, Optional
import json
from pathlib import Path
import hashlib

from src.database.neo4j_client import neo4j_client
from src.core.logger import get_logger
from src.data_ingestion.import_resolver import ImportResolver
from src.data_ingestion.entry_point_detector import EntryPointDetector

logger = get_logger(__name__)


class EnhancedGraphLinker:
    """
    Creates rich, semantic relationships in the Neo4j knowledge graph.
    Goes beyond basic file parsing to trace actual code interactions.
    
    All edge creation uses batch UNWIND queries to avoid N+1 performance issues.
    IDs are generated using the same formula as NodeFactory for consistency.
    """
    
    def __init__(self, repo_path: str, repo_hash: str = ""):
        self.repo_path = repo_path
        self.repo_path_obj = Path(repo_path)
        self.repo_hash = repo_hash
        self.import_resolver = ImportResolver(repo_path)
        self.entry_detector = EntryPointDetector(repo_path)
        self.stats = {
            "edges_created": 0,
            "edges_by_type": {},
            "errors": []
        }
    
    def _generate_node_id(self, node_type: str, *identifiers) -> str:
        """Generate deterministic node ID - matches NodeFactory formula exactly."""
        combined = f"{node_type}:{'|'.join(str(i) for i in identifiers)}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    def _to_relative_path(self, absolute_path: str) -> str:
        """Convert absolute path to repo-relative path for consistent ID generation."""
        try:
            return str(Path(absolute_path).relative_to(self.repo_path_obj))
        except ValueError:
            return absolute_path
    
    async def link_imports_in_graph(self, session=None) -> Dict[str, Any]:
        """
        Trace all imports starting from detected entry points and create
        IMPORTS edges in a single batch query.
        """
        use_external_session = session is not None
        
        try:
            logger.info(f"Detecting entry points in {self.repo_path}")
            entry_points = self.entry_detector.detect_entry_points()
            
            all_traced: Dict[str, Set[str]] = {}
            
            if not entry_points:
                logger.warning("No entry points detected, falling back to ImportResolver defaults")
                all_traced = self.import_resolver.trace_entry_point(
                    language=self.entry_detector.language, max_depth=15
                )
            else:
                for entry in entry_points:
                    logger.info(f"Tracing imports from: {entry.file_path} ({entry.detection_method})")
                    traced = self.import_resolver.trace_from_file(
                        entry.file_path, language=entry.language, max_depth=15
                    )
                    for filepath, imports_set in traced.items():
                        all_traced.setdefault(filepath, set()).update(imports_set)
            
            logger.info(f"Traced {len(all_traced)} files with imports")
            
            if not use_external_session:
                async with neo4j_client.session() as session:
                    stats = await self._create_import_edges_batch(session, all_traced)
            else:
                stats = await self._create_import_edges_batch(session, all_traced)
            
            self.stats = stats
            return stats
            
        except Exception as e:
            logger.error(f"Import linking failed: {e}")
            self.stats["errors"].append(str(e))
            return self.stats
    
    async def _create_import_edges_batch(self, session, traced_files: Dict[str, Set[str]]) -> Dict[str, Any]:
        """Create IMPORTS edges using a single batch UNWIND query."""
        stats = {"edges_created": 0, "by_type": {}, "errors": []}
        
        edge_list = []
        for importing_file, imported_files in traced_files.items():
            for imported_file in imported_files:
                edge_list.append({
                    "importer_path": self._to_relative_path(importing_file),
                    "imported_path": self._to_relative_path(imported_file),
                })
        
        if not edge_list:
            return stats
        
        batch_size = 500
        for i in range(0, len(edge_list), batch_size):
            batch = edge_list[i:i + batch_size]
            try:
                query = """
                UNWIND $edges AS edge
                MATCH (importer:File)
                WHERE (importer.path = edge.importer_path
                       OR importer.relative_path = edge.importer_path)
                  AND ($repo_hash = '' OR importer.repository_hash = $repo_hash)
                MATCH (imported:File)
                WHERE (imported.path = edge.imported_path
                       OR imported.relative_path = edge.imported_path)
                  AND ($repo_hash = '' OR imported.repository_hash = $repo_hash)
                MERGE (importer)-[r:IMPORTS]->(imported)
                ON CREATE SET r.created = datetime()
                SET r.last_updated = datetime()
                RETURN count(r) as created_count
                """
                result = await session.run(query, {"edges": batch, "repo_hash": self.repo_hash})
                record = await result.single()
                count = record["created_count"] if record else 0
                stats["edges_created"] += count
            except Exception as e:
                stats["errors"].append(f"Batch IMPORTS failed: {e}")
        
        stats["by_type"]["IMPORTS"] = stats["edges_created"]
        logger.info(f"Created {stats['edges_created']} IMPORTS edges (batch)")
        return stats
    
    async def link_function_calls_in_graph(self, session=None) -> Dict[str, Any]:
        """
        Link functions that call other functions using scope-aware matching.
        Uses the IMPORTS graph to find the correct target function.
        """
        use_external_session = session is not None
        
        try:
            if not use_external_session:
                async with neo4j_client.session() as session:
                    stats = await self._link_function_calls_batch(session)
            else:
                stats = await self._link_function_calls_batch(session)
            
            return stats
            
        except Exception as e:
            logger.error(f"Function call linking failed: {e}")
            return {"edges_created": 0, "errors": [str(e)]}
    
    async def _link_function_calls_batch(self, session) -> Dict[str, Any]:
        """
        Scope-aware function call linking using batch Cypher.
        
        Strategy: For each function with a `calls` list property,
        find target functions that live in files imported by the caller's file,
        or in the same file. This avoids the "global name collision" problem.
        """
        stats = {"edges_created": 0, "by_type": {}, "errors": []}
        
        try:
            query = """
            MATCH (caller_file:File)-[:CONTAINS]->(caller:Function)
            WHERE caller.calls IS NOT NULL AND size(caller.calls) > 0
            UNWIND caller.calls AS callee_name
            OPTIONAL MATCH (caller_file)-[:CONTAINS]->(same_file_target:Function {name: callee_name})
            WHERE same_file_target <> caller
            OPTIONAL MATCH (caller_file)-[:IMPORTS]->(imported_file:File)-[:CONTAINS]->(imported_target:Function {name: callee_name})
            WITH caller, callee_name,
                 COALESCE(same_file_target, imported_target) AS target
            WHERE target IS NOT NULL
            MERGE (caller)-[r:CALLS_FUNCTION]->(target)
            ON CREATE SET r.created = datetime()
            SET r.last_updated = datetime()
            RETURN count(r) as created_count
            """
            result = await session.run(query)
            record = await result.single()
            count = record["created_count"] if record else 0
            stats["edges_created"] = count
            stats["by_type"]["CALLS_FUNCTION"] = count
            logger.info(f"Created {count} CALLS_FUNCTION edges (scope-aware batch)")
        except Exception as e:
            stats["errors"].append(f"Scope-aware function call linking error: {e}")
            logger.warning(f"Scope-aware linking failed, falling back to global: {e}")
            
            try:
                fallback_query = """
                MATCH (caller:Function)
                WHERE caller.calls IS NOT NULL AND size(caller.calls) > 0
                UNWIND caller.calls AS callee_name
                MATCH (target:Function {name: callee_name})
                WHERE target <> caller
                MERGE (caller)-[r:CALLS_FUNCTION]->(target)
                ON CREATE SET r.created = datetime(),
                             r.confidence_score = 0.5,
                             r.match_type = 'global_name'
                SET r.last_updated = datetime()
                RETURN count(r) as created_count
                """
                result = await session.run(fallback_query)
                record = await result.single()
                count = record["created_count"] if record else 0
                stats["edges_created"] = count
                stats["by_type"]["CALLS_FUNCTION"] = count
                logger.info(f"Created {count} CALLS_FUNCTION edges (global fallback)")
            except Exception as fallback_e:
                stats["errors"].append(f"Global fallback also failed: {fallback_e}")
        
        return stats
    
    async def create_type_hierarchy_edges(self, session=None) -> Dict[str, Any]:
        """
        Create class hierarchy and service dependency edges.
        Already uses batch Cypher (single-pass MATCH/MERGE).
        """
        use_external_session = session is not None
        
        try:
            if not use_external_session:
                async with neo4j_client.session() as session:
                    stats = await self._create_type_edges_internal(session)
            else:
                stats = await self._create_type_edges_internal(session)
            
            return stats
            
        except Exception as e:
            logger.error(f"Type hierarchy linking failed: {e}")
            return {"edges_created": 0, "errors": [str(e)]}
    
    async def _create_type_edges_internal(self, session) -> Dict[str, Any]:
        stats = {"edges_created": 0, "by_type": {}, "errors": []}
        
        try:
            extends_query = """
            MATCH (child:Class)
            WHERE child.extends IS NOT NULL AND child.extends <> ''
            WITH child, split(child.extends, ',')[0] as parent_name
            MATCH (parent:Class {name: trim(parent_name)})
            WHERE parent <> child
            MERGE (child)-[r:EXTENDS]->(parent)
            ON CREATE SET r.created = datetime()
            SET r.last_updated = datetime()
            RETURN count(r) as count
            """
            result = await session.run(extends_query)
            async for record in result:
                count = record.get('count', 0)
                stats["edges_created"] += count
                stats["by_type"]["EXTENDS"] = count
            
            service_query = """
            MATCH (s1:Service), (s2:Service)
            WHERE s1.name <> s2.name
              AND s1.repository_hash = s2.repository_hash
              AND s1.file CONTAINS s2.name
            MERGE (s1)-[r:DEPENDS_ON_SERVICE]->(s2)
            ON CREATE SET r.created = datetime()
            SET r.last_updated = datetime()
            RETURN count(r) as count
            """
            result = await session.run(service_query)
            async for record in result:
                count = record.get('count', 0)
                stats["edges_created"] += count
                stats["by_type"]["DEPENDS_ON_SERVICE"] = count
            
            logger.info(f"Created {stats['edges_created']} type hierarchy edges")
        except Exception as e:
            stats["errors"].append(f"Type hierarchy error: {e}")
        
        return stats
    
    async def run_all_enhancements(self, session=None) -> Dict[str, Any]:
        """Run all graph enhancement operations."""
        logger.info("Starting enhanced graph linking (batch mode)...")
        
        use_external_session = session is not None
        
        if not use_external_session:
            async with neo4j_client.session() as session:
                return await self._run_all_internal(session)
        else:
            return await self._run_all_internal(session)
    
    async def _run_all_internal(self, session) -> Dict[str, Any]:
        all_stats = {
            "imports": await self.link_imports_in_graph(session),
            "function_calls": await self.link_function_calls_in_graph(session),
            "type_hierarchy": await self.create_type_hierarchy_edges(session),
            "total_edges_created": 0
        }
        
        for category in ['imports', 'function_calls', 'type_hierarchy']:
            all_stats["total_edges_created"] += all_stats[category].get("edges_created", 0)
        
        logger.info(f"Enhanced graph linking complete. Total edges created: {all_stats['total_edges_created']}")
        return all_stats
