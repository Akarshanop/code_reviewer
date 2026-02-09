"""
Enhanced Graph Linker - Creates rich semantic relationships in Neo4j
Improves graph by:
1. Resolving imports and creating IMPORTS edges
2. Linking function calls (Function -> Function)
3. Linking component API calls (Component -> Route)
4. Creating semantic type relationships
"""
from typing import Dict, List, Any, Set, Optional
import re
from pathlib import Path
import hashlib

from src.database.neo4j_client import neo4j_client
from src.core.logger import get_logger
from src.data_ingestion.import_resolver import ImportResolver

logger = get_logger(__name__)


class EnhancedGraphLinker:
    """
    Creates rich, semantic relationships in the Neo4j knowledge graph.
    Goes beyond basic file parsing to trace actual code interactions.
    """
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.import_resolver = ImportResolver(repo_path)
        self.stats = {
            "edges_created": 0,
            "edges_by_type": {},
            "errors": []
        }
    
    def _generate_node_id(self, node_type: str, *identifiers) -> str:
        """Generate deterministic node ID"""
        combined = f"{node_type}:{'|'.join(str(i) for i in identifiers)}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    async def link_imports_in_graph(self, session=None) -> Dict[str, Any]:
        """
        1. Trace all imports starting from entry points
        2. Create IMPORTS relationships in Neo4j
        3. Return statistics
        """
        use_external_session = session is not None
        
        try:
            # Trace entry point
            logger.info(f"Tracing imports from entry points in {self.repo_path}")
            traced = self.import_resolver.trace_entry_point(language='javascript', max_depth=15)
            
            logger.info(f"Traced {len(traced)} files with imports")
            
            # Create Neo4j relationships
            if not use_external_session:
                async with neo4j_client.session() as session:
                    stats = await self._create_import_edges(session, traced)
            else:
                stats = await self._create_import_edges(session, traced)
            
            self.stats = stats
            return stats
            
        except Exception as e:
            logger.error(f"Import linking failed: {e}")
            self.stats["errors"].append(str(e))
            return self.stats
    
    async def _create_import_edges(self, session, traced_files: Dict[str, Set[str]]) -> Dict[str, Any]:
        """Create IMPORTS edges between files in Neo4j"""
        stats = {
            "edges_created": 0,
            "by_type": {},
            "errors": []
        }
        
        for importing_file, imported_files in traced_files.items():
            for imported_file in imported_files:
                try:
                    # Get file node IDs
                    importing_id = self._generate_node_id("File", importing_file)
                    imported_id = self._generate_node_id("File", imported_file)
                    
                    # Create relationship
                    query = """
                    MATCH (importer:File {id: $importer_id})
                    MATCH (imported:File {id: $imported_id})
                    MERGE (importer)-[r:IMPORTS]->(imported)
                    SET r.created = datetime()
                    RETURN r
                    """
                    
                    await session.run(query, {
                        "importer_id": importing_id,
                        "imported_id": imported_id,
                    })
                    
                    stats["edges_created"] += 1
                    stats["by_type"]["IMPORTS"] = stats["by_type"].get("IMPORTS", 0) + 1
                    
                except Exception as e:
                    stats["errors"].append(f"Failed to create IMPORTS edge: {e}")
        
        logger.info(f"Created {stats['edges_created']} IMPORTS edges")
        return stats
    
    async def link_function_calls_in_graph(self, session=None) -> Dict[str, Any]:
        """
        2. Link functions that call other functions
        Pattern: (Function)-[CALLS_FUNCTION]->(Function)
        
        Extracts from parsed code:
        - functionA() calls functionB()
        - componentA uses serviceB.method()
        """
        use_external_session = session is not None
        
        try:
            if not use_external_session:
                async with neo4j_client.session() as session:
                    stats = await self._link_function_calls_internal(session)
            else:
                stats = await self._link_function_calls_internal(session)
            
            return stats
            
        except Exception as e:
            logger.error(f"Function call linking failed: {e}")
            return {"edges_created": 0, "errors": [str(e)]}
    
    async def _link_function_calls_internal(self, session) -> Dict[str, Any]:
        """Internal method to link function calls"""
        stats = {
            "edges_created": 0,
            "by_type": {},
            "errors": []
        }
        
        try:
            # Query all functions with their call_targets metadata
            query = """
            MATCH (f:Function)
            WHERE f.call_targets_json IS NOT NULL
            RETURN f.id as func_id, f.name as func_name, f.call_targets_json as targets
            LIMIT 1000
            """
            
            result = await session.run(query)
            
            async for record in result:
                func_id = record['func_id']
                func_name = record['func_name']
                targets_json = record['targets']
                
                # Parse targets (should be JSON array of function names)
                try:
                    import json
                    targets = json.loads(targets_json) if isinstance(targets_json, str) else targets_json
                    
                    for target_name in targets:
                        # Find target function by name
                        target_query = """
                        MATCH (target:Function {name: $target_name})
                        WITH target
                        MATCH (source:Function {id: $source_id})
                        MERGE (source)-[r:CALLS_FUNCTION {name: $target_name}]->(target)
                        SET r.created = datetime()
                        RETURN r
                        """
                        
                        await session.run(target_query, {
                            "source_id": func_id,
                            "target_name": target_name
                        })
                        
                        stats["edges_created"] += 1
                        stats["by_type"]["CALLS_FUNCTION"] = stats["by_type"].get("CALLS_FUNCTION", 0) + 1
                        
                except Exception as e:
                    stats["errors"].append(f"Failed to parse targets for {func_name}: {e}")
            
            logger.info(f"Created {stats['edges_created']} CALLS_FUNCTION edges")
            
        except Exception as e:
            stats["errors"].append(f"Function call linking error: {e}")
        
        return stats
    
    async def link_component_api_calls(self, session=None) -> Dict[str, Any]:
        """
        3. Link React components to backend API endpoints they call
        Pattern: (Component)-[CALLS_API {method, endpoint}]->(Route)
        
        Extracts from React code:
        - fetch('/api/users') → links to backend GET /api/users route
        - axios.post('/api/auth') → links to POST /api/auth route
        """
        use_external_session = session is not None
        
        try:
            if not use_external_session:
                async with neo4j_client.session() as session:
                    stats = await self._link_component_apis_internal(session)
            else:
                stats = await self._link_component_apis_internal(session)
            
            return stats
            
        except Exception as e:
            logger.error(f"Component API linking failed: {e}")
            return {"edges_created": 0, "errors": [str(e)]}
    
    async def _link_component_apis_internal(self, session) -> Dict[str, Any]:
        """Internal method to link components to API endpoints"""
        stats = {
            "edges_created": 0,
            "by_type": {},
            "errors": []
        }
        
        try:
            # Find all components with API calls
            query = """
            MATCH (c:Component)
            WHERE c.api_calls_json IS NOT NULL
            RETURN c.id as comp_id, c.name as comp_name, c.api_calls_json as api_calls
            LIMIT 500
            """
            
            result = await session.run(query)
            
            async for record in result:
                comp_id = record['comp_id']
                comp_name = record['comp_name']
                api_calls_json = record['api_calls']
                
                try:
                    import json
                    api_calls = json.loads(api_calls_json) if isinstance(api_calls_json, str) else api_calls_json
                    
                    for api_call in api_calls:
                        method = api_call.get('method', 'GET').upper()
                        endpoint = api_call.get('endpoint', '')
                        
                        # Find matching backend route
                        route_query = """
                        MATCH (r:Route)
                        WHERE r.path CONTAINS $endpoint AND r.method = $method
                        WITH r
                        MATCH (source:Component {id: $comp_id})
                        MERGE (source)-[rel:CALLS_API {
                            endpoint: $endpoint,
                            method: $method
                        }]->(r)
                        SET rel.created = datetime()
                        RETURN rel
                        """
                        
                        await session.run(route_query, {
                            "comp_id": comp_id,
                            "endpoint": endpoint,
                            "method": method
                        })
                        
                        stats["edges_created"] += 1
                        stats["by_type"]["CALLS_API"] = stats["by_type"].get("CALLS_API", 0) + 1
                        
                except Exception as e:
                    stats["errors"].append(f"Failed to process API calls for {comp_name}: {e}")
            
            logger.info(f"Created {stats['edges_created']} CALLS_API edges")
            
        except Exception as e:
            stats["errors"].append(f"Component API linking error: {e}")
        
        return stats
    
    async def create_type_hierarchy_edges(self, session=None) -> Dict[str, Any]:
        """
        4. Create semantic type relationships
        Pattern: (Service)-[IMPLEMENTS {interface}]->(Interface)
        Pattern: (Class)-[EXTENDS]->(BaseClass)
        
        Useful for finding "all services" or "all components implementing X interface"
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
        """Create type hierarchy edges"""
        stats = {
            "edges_created": 0,
            "by_type": {},
            "errors": []
        }
        
        try:
            # Link classes that extend other classes
            extends_query = """
            MATCH (child:Class)
            WHERE child.extends IS NOT NULL
            WITH child, split(child.extends, ',')[0] as parent_name
            MATCH (parent:Class {name: trim(parent_name)})
            MERGE (child)-[r:EXTENDS]->(parent)
            SET r.created = datetime()
            RETURN count(r) as count
            """
            
            result = await session.run(extends_query)
            async for record in result:
                count = record.get('count', 0)
                stats["edges_created"] += count
                stats["by_type"]["EXTENDS"] = count
            
            # Link services by naming pattern
            # If service name contains pattern of another service, likely depends on it
            service_query = """
            MATCH (s1:Service), (s2:Service)
            WHERE s1.name CONTAINS "Service" AND s2.name CONTAINS "Service"
            AND s1.name <> s2.name
            AND toLower(s1.name) CONTAINS toLower(replace(s2.name, "Service", ""))
            MERGE (s1)-[r:DEPENDS_ON_SERVICE]->(s2)
            SET r.created = datetime()
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
        """Run all graph enhancement operations"""
        logger.info("Starting comprehensive graph enhancements...")
        
        use_external_session = session is not None
        
        if not use_external_session:
            async with neo4j_client.session() as session:
                return await self._run_all_internal(session)
        else:
            return await self._run_all_internal(session)
    
    async def _run_all_internal(self, session) -> Dict[str, Any]:
        """Run all enhancements with provided session"""
        all_stats = {
            "imports": await self.link_imports_in_graph(session),
            "function_calls": await self.link_function_calls_in_graph(session),
            "component_apis": await self.link_component_api_calls(session),
            "type_hierarchy": await self.create_type_hierarchy_edges(session),
            "total_edges_created": 0
        }
        
        # Sum up all edges
        for category in ['imports', 'function_calls', 'component_apis', 'type_hierarchy']:
            all_stats["total_edges_created"] += all_stats[category].get("edges_created", 0)
        
        logger.info(f"Graph enhancements complete. Total edges created: {all_stats['total_edges_created']}")
        return all_stats
