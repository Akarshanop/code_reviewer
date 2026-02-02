import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import networkx as nx
from collections import defaultdict
import json

from src.database.neo4j_client import neo4j_client
from src.core.exceptions import GraphDatabaseException
from src.core.logger import get_logger

logger = get_logger(__name__)

class GraphService:
    """Service for graph-based code analysis and queries"""
    
    def __init__(self):
        self.neo4j = neo4j_client
    
    async def find_related_entities(
        self,
        entity_name: str,
        relationship_types: List[str] = None,
        max_depth: int = 3,
        direction: str = "BOTH"
    ) -> Dict[str, Any]:
        """Find entities related to a given entity"""
        rel_filter = ""
        if relationship_types:
            rel_types = "|".join(relationship_types)
            rel_filter = f":{rel_types}"
        
        dir_symbol = {
            "OUTGOING": "->",
            "INCOMING": "<-",
            "BOTH": "-"
        }.get(direction, "-")
        
        query = f"""
        MATCH path = (start {{name: $entity_name}})-[*1..{max_depth}{rel_filter}]{dir_symbol}(end)
        WHERE start <> end
        RETURN 
            start,
            end,
            [n in nodes(path) | labels(n)[0]] as node_labels,
            [r in relationships(path) | type(r)] as relationship_types,
            length(path) as path_length
        ORDER BY path_length
        LIMIT 150
        """
        
        try:
            results = await self.neo4j.execute_query(query, {"entity_name": entity_name})
            
            # Group by relationship type
            relationships = defaultdict(list)
            for record in results:
                rel_type = record["relationship_types"][-1] if record["relationship_types"] else "UNKNOWN"
                relationships[rel_type].append({
                    "target": dict(record["end"]),
                    "path_length": record["path_length"],
                    "target_labels": record["node_labels"][-1] if record["node_labels"] else None
                })
            
            return {
                "source": dict(results[0]["start"]) if results else None,
                "relationships": dict(relationships),
                "total_relationships": sum(len(v) for v in relationships.values())
            }
            
        except Exception as e:
            logger.error("Failed to find related entities", entity=entity_name, error=str(e))
            raise GraphDatabaseException(f"Failed to find related entities: {str(e)}")
    
    async def analyze_impact(
        self,
        file_path: str,
        change_type: str = "modify"
    ) -> Dict[str, Any]:
        """Analyze impact of changing a file"""
        queries = {
            "direct_dependencies": """
            MATCH (f:File {path: $file_path})
            OPTIONAL MATCH (f)-[:CONTAINS]->(component)
            OPTIONAL MATCH (component)-[r]-(dependent)
            WHERE dependent <> component
            RETURN component, type(r) as relationship_type, dependent
            LIMIT 50
            """,
            
            "api_calls": """
            MATCH (f:File {path: $file_path})
            OPTIONAL MATCH (f)-[:CONTAINS]->(component)-[:MAKES_API_CALL]->(api:ApiCall)
            OPTIONAL MATCH (api)-[:CALLS_ROUTE]->(route:VertxRoute)
            RETURN component, api, route
            """,
            
            "eventbus_flows": """
            MATCH (f:File {path: $file_path})
            OPTIONAL MATCH (f)-[:CONTAINS]->(component)-[:PUBLISHES_TO|:CONSUMES_FROM]->(eventbus:EventBusNode)
            OPTIONAL MATCH (eventbus)-[:HAS_HANDLER]->(handler)
            RETURN component, eventbus, handler
            """
        }
        
        results = {}
        for name, query in queries.items():
            try:
                query_results = await self.neo4j.execute_query(query, {"file_path": file_path})
                results[name] = query_results
            except Exception as e:
                logger.warning(f"Failed to execute impact analysis query: {name}", error=str(e))
                results[name] = []
        
        # Calculate impact score
        total_components = len(results.get("direct_dependencies", []))
        total_apis = len(results.get("api_calls", []))
        total_eventbus = len(results.get("eventbus_flows", []))
        
        impact_score = min(100, (total_components * 5) + (total_apis * 10) + (total_eventbus * 15))
        
        return {
            "file": file_path,
            "impact_score": impact_score,
            "risk_level": "high" if impact_score > 50 else "medium" if impact_score > 20 else "low",
            "breakdown": {
                "components_affected": total_components,
                "apis_affected": total_apis,
                "eventbus_flows_affected": total_eventbus
            },
            "details": results
        }
    
    async def find_cross_repository_dependencies(
        self,
        repository_name: str = None
    ) -> List[Dict[str, Any]]:
        """Find dependencies across different repositories"""
        repo_filter = "WHERE f.repository = $repository_name" if repository_name else ""
        
        query = f"""
        MATCH (f:File)
        {repo_filter}
        MATCH (f)-[:CONTAINS]->(source)
        MATCH (source)-[r]->(target)
        WHERE target.repository <> source.repository OR target.repository IS NULL
        RETURN 
            source.name as source_name,
            labels(source)[0] as source_type,
            type(r) as relationship_type,
            target.name as target_name,
            labels(target)[0] as target_type,
            target.repository as target_repository,
            source.repository as source_repository
        ORDER BY source.repository, target.repository
        LIMIT 200
        """
        
        params = {"repository_name": repository_name} if repository_name else {}
        results = await self.neo4j.execute_query(query, params)
        
        # Group by source repository
        dependencies = defaultdict(list)
        for record in results:
            key = f"{record['source_repository']} -> {record['target_repository'] or 'external'}"
            dependencies[key].append({
                "source": f"{record['source_type']}: {record['source_name']}",
                "target": f"{record['target_type']}: {record['target_name']}",
                "relationship": record["relationship_type"]
            })
        
        return [
            {
                "dependency_path": key,
                "connections": connections,
                "count": len(connections)
            }
            for key, connections in dependencies.items()
        ]
    
    async def find_similar_components(
        self,
        component_name: str,
        similarity_threshold: float = 0.7,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find similar components using graph structure similarity"""
        # This uses graph algorithms to find structurally similar components
        query = """
        MATCH (c {name: $component_name})
        MATCH (similar)
        WHERE similar <> c 
          AND labels(similar) = labels(c)
          AND similar.name IS NOT NULL
        
        // Calculate Jaccard similarity of relationships
        OPTIONAL MATCH (c)-[r1]->(neighbor1)
        WITH c, similar, collect(DISTINCT type(r1)) as c_rels
        OPTIONAL MATCH (similar)-[r2]->(neighbor2)
        WITH c, similar, c_rels, collect(DISTINCT type(r2)) as similar_rels
        
        // Calculate Jaccard similarity
        WITH c, similar, c_rels, similar_rels,
             apoc.coll.intersection(c_rels, similar_rels) as intersection,
             apoc.coll.union(c_rels, similar_rels) as union
        
        WHERE size(union) > 0
        
        RETURN 
            similar,
            size(intersection) * 1.0 / size(union) as similarity_score,
            c_rels,
            similar_rels
        
        ORDER BY similarity_score DESC
        LIMIT $limit
        """
        
        results = await self.neo4j.execute_query(query, {
            "component_name": component_name,
            "limit": limit
        })
        
        similar_components = []
        for record in results:
            if record["similarity_score"] >= similarity_threshold:
                similar_components.append({
                    "component": dict(record["similar"]),
                    "similarity_score": record["similarity_score"],
                    "shared_relationships": record["c_rels"]
                })
        
        return similar_components
    
    async def get_codebase_metrics(self) -> Dict[str, Any]:
        """Get comprehensive codebase metrics"""
        metrics_queries = {
            "repository_stats": """
            MATCH (r:Repository)
            RETURN r.name as repository, r.language as language, 
                   r.last_synced as last_synced, r.sync_status as status
            """,
            
            "component_counts": """
            MATCH (n)
            WHERE n:VertxRoute OR n:ReactComponent OR n:DTO OR n:Service OR n:EventBusNode
            RETURN labels(n)[0] as type, count(*) as count
            ORDER BY count DESC
            """,
            
            "relationship_counts": """
            MATCH ()-[r]->()
            RETURN type(r) as relationship_type, count(*) as count
            ORDER BY count DESC
            LIMIT 20
            """,
            
            "top_connected_components": """
            MATCH (n)
            WHERE n:VertxRoute OR n:ReactComponent OR n:DTO
            OPTIONAL MATCH (n)-[r]-()
            RETURN n.name as name, labels(n)[0] as type, count(r) as connection_count
            ORDER BY connection_count DESC
            LIMIT 10
            """,
            
            "api_usage_stats": """
            MATCH (api:ApiCall)
            OPTIONAL MATCH (api)-[:CALLS_ROUTE]->(route)
            RETURN 
                api.method as method,
                count(*) as count,
                collect(DISTINCT route.path)[0..5] as sample_routes
            ORDER BY count DESC
            """
        }
        
        metrics = {}
        for name, query in metrics_queries.items():
            try:
                results = await self.neo4j.execute_query(query)
                metrics[name] = results
            except Exception as e:
                logger.warning(f"Failed to execute metrics query: {name}", error=str(e))
                metrics[name] = []
        
        # Calculate totals
        total_nodes = sum(len(v) for v in metrics.values())
        total_relationships = sum(record.get("count", 0) for record in metrics.get("relationship_counts", []))
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "totals": {
                "nodes": total_nodes,
                "relationships": total_relationships,
                "repositories": len(metrics.get("repository_stats", []))
            },
            "breakdown": metrics
        }
    
    async def visualize_dependency_graph(
        self,
        start_node_id: str,
        max_depth: int = 2,
        relationship_types: List[str] = None
    ) -> Dict[str, Any]:
        """Generate data for dependency graph visualization"""
        rel_filter = ""
        if relationship_types:
            rel_types = "|".join(relationship_types)
            rel_filter = f":{rel_types}"
        
        query = f"""
        MATCH path = (start)-[*0..{max_depth}{rel_filter}]-(connected)
        WHERE elementId(start) = $start_node_id
        WITH collect(DISTINCT path) as paths
        UNWIND paths as path
        UNWIND nodes(path) as node
        UNWIND relationships(path) as rel
        RETURN 
            collect(DISTINCT {{id: elementId(node), labels: labels(node), properties: properties(node)}}) as nodes,
            collect(DISTINCT {{
                id: elementId(rel),
                type: type(rel),
                source: elementId(startNode(rel)),
                target: elementId(endNode(rel))
            }}) as relationships
        """
        
        results = await self.neo4j.execute_query(query, {"start_node_id": start_node_id})
        
        if not results:
            return {"nodes": [], "relationships": []}
        
        return {
            "nodes": results[0]["nodes"],
            "relationships": results[0]["relationships"]
        }
    
    async def find_circular_dependencies(self) -> List[List[Dict[str, Any]]]:
        """Find circular dependencies in the codebase"""
        query = """
        MATCH path = (start)-[:CALLS|:DEPENDS_ON|:USES*]->(start)
        WHERE length(path) > 1
        RETURN 
            [node in nodes(path) | {name: node.name, type: labels(node)[0]}] as cycle,
            length(path) as cycle_length
        ORDER BY cycle_length
        LIMIT 20
        """
        
        try:
            results = await self.neo4j.execute_query(query)
            return [record["cycle"] for record in results]
        except Exception as e:
            logger.warning("Failed to find circular dependencies", error=str(e))
            return []
    
    async def find_unused_components(self) -> List[Dict[str, Any]]:
        """Find components that are not referenced by any other component"""
        query = """
        MATCH (component)
        WHERE (component:VertxRoute OR component:ReactComponent OR component:DTO)
          AND NOT (:File)-[:CONTAINS]->(component)
        RETURN 
            component.name as name,
            labels(component)[0] as type,
            component.file as file
        ORDER BY type, name
        LIMIT 50
        """
        
        results = await self.neo4j.execute_query(query)
        
        unused_components = []
        for record in results:
            unused_components.append({
                "name": record["name"],
                "type": record["type"],
                "file": record["file"],
                "status": "unreferenced"
            })
        
        return unused_components
    
    async def find_module_structure(
        self,
        module_name: str,
        repo_filter: str = None
    ) -> Dict[str, Any]:
        """Find the structure of a module"""
        try:
            repo_clause = "AND f.repository = $repository" if repo_filter else ""
            
            query = f"""
            MATCH (f:File)
            WHERE f.path CONTAINS $module_name {repo_clause}
            OPTIONAL MATCH (f)-[:CONTAINS]->(component)
            RETURN 
                f.path as file_path,
                collect(DISTINCT {{
                    name: component.name,
                    type: labels(component)[0]
                }}) as components
            LIMIT 100
            """
            
            params = {"module_name": module_name}
            if repo_filter:
                params["repository"] = repo_filter
            
            results = await self.neo4j.execute_query(query, params)
            
            # Organize by file
            structure = {
                "files": [],
                "components": [],
                "file_count": 0
            }
            
            for record in results:
                structure["files"].append({
                    "path": record["file_path"],
                    "components": record["components"]
                })
                structure["components"].extend(record["components"])
            
            structure["file_count"] = len(structure["files"])
            
            return structure
            
        except Exception as e:
            logger.warning(f"Failed to find module structure: {e}")
            return {"files": [], "components": [], "file_count": 0}
    
    async def query_impact(self, file_path: str) -> Dict[str, Any]:
        """Query impact of a file change"""
        try:
            result = await self.analyze_impact(file_path)
            
            # Extract affected components/apis/eventbus
            components = []
            apis = []
            eventbus = []
            
            for dep in result.get("details", {}).get("direct_dependencies", []):
                if dep.get("component"):
                    comp = dep.get("component", {})
                    comp_name = comp.get("name") if isinstance(comp, dict) else str(comp)
                    if comp_name:
                        components.append(comp_name)
            
            for api in result.get("details", {}).get("api_calls", []):
                if api.get("route"):
                    route = api.get("route", {})
                    route_path = route.get("path") if isinstance(route, dict) else str(route)
                    if route_path:
                        apis.append(route_path)
            
            for eb in result.get("details", {}).get("eventbus_flows", []):
                if eb.get("eventbus"):
                    eb_data = eb.get("eventbus", {})
                    eb_addr = eb_data.get("address") if isinstance(eb_data, dict) else str(eb_data)
                    if eb_addr:
                        eventbus.append(eb_addr)
            
            return {
                "components": list(set(components)),
                "apis": list(set(apis)),
                "eventbus": list(set(eventbus)),
                "impact_score": result.get("impact_score", 0),
                "risk_level": result.get("risk_level", "low"),
                "reasons": [
                    f"Affects {len(components)} components",
                    f"Impacts {len(apis)} APIs",
                    f"Touches {len(eventbus)} EventBus flows"
                ]
            }
            
        except Exception as e:
            logger.warning(f"Failed to query impact: {e}")
            return {"components": [], "apis": [], "eventbus": [], "reasons": []}
    
    async def trace_api_flow(
        self,
        endpoint: str,
        include_eventbus: bool = True
    ) -> Dict[str, Any]:
        """Trace the complete flow of an API endpoint"""
        try:
            query = """
            MATCH (route:VertxRoute)
            WHERE route.path = $endpoint OR route.path CONTAINS $endpoint
            OPTIONAL MATCH (route)-[:HANDLED_BY]->(handler)
            OPTIONAL MATCH (handler)-[:CALLS]->(service)
            OPTIONAL MATCH (service)-[:PUBLISHES_TO|:CONSUMES_FROM]->(eventbus:EventBusNode)
            RETURN route, handler, service, eventbus
            LIMIT 20
            """
            
            results = await self.neo4j.execute_query(query, {"endpoint": endpoint})
            
            flow = []
            eventbus_interactions = []
            
            for record in results:
                if record.get("route"):
                    route = record["route"]
                    route_data = dict(route) if hasattr(route, 'items') else {"path": str(route)}
                    flow.append({
                        "step": "route",
                        "type": "VertxRoute",
                        "data": route_data
                    })
                
                if record.get("handler"):
                    handler = record["handler"]
                    handler_data = dict(handler) if hasattr(handler, 'items') else {"name": str(handler)}
                    flow.append({
                        "step": "handler",
                        "type": "Handler",
                        "data": handler_data
                    })
                
                if record.get("service"):
                    service = record["service"]
                    service_data = dict(service) if hasattr(service, 'items') else {"name": str(service)}
                    flow.append({
                        "step": "service",
                        "type": "Service",
                        "data": service_data
                    })
                
                if include_eventbus and record.get("eventbus"):
                    eb = record["eventbus"]
                    eb_data = dict(eb) if hasattr(eb, 'items') else {"address": str(eb)}
                    eventbus_interactions.append(eb_data)
            
            return {
                "endpoint": endpoint,
                "flow": flow,
                "eventbus": eventbus_interactions
            }
            
        except Exception as e:
            logger.warning(f"Failed to trace API flow: {e}")
            return {"endpoint": endpoint, "flow": [], "eventbus": []}