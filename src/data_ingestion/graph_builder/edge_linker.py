"""
Enterprise Edge Linker for Neo4j Knowledge Graph
Creates relationships between code artifacts: API calls to routes, EventBus flows, DTO usages
"""
import logging
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
import re
from datetime import datetime

from src.database.neo4j_client import neo4j_client
from src.core.logger import get_logger

logger = get_logger(__name__)


class EdgeLinker:
    """
    Creates and manages relationships between nodes in the code knowledge graph.
    Handles cross-language linking (React → Vert.x) and EventBus flow mapping.
    """
    
    def __init__(self):
        self.stats = {
            "edges_created": 0,
            "edges_updated": 0,
            "errors": 0
        }
    
    async def link_all_edges(self, session=None) -> Dict[str, Any]:
        """
        Link all types of edges between code artifacts.
        
        Args:
            session: Optional Neo4j session. If not provided, creates new session.
            
        Returns:
            Statistics about created relationships
        """
        stats = {
            "relationships_created": 0,
            "by_type": {},
            "errors": []
        }
        
        # Use provided session or create new one
        use_external_session = session is not None
        
        try:
            if not use_external_session:
                async with neo4j_client.session() as session:
                    stats = await self._link_all_edges_internal(session, stats)
            else:
                stats = await self._link_all_edges_internal(session, stats)
            
            logger.info(
                "All edges linked successfully",
                total_relationships=stats["relationships_created"],
                by_type=stats["by_type"]
            )
            
        except Exception as e:
            logger.error("Edge linking failed", error=str(e))
            stats["errors"].append(str(e))
        
        return stats
    
    async def _link_all_edges_internal(self, session, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Internal method to link all edges with provided session"""

        # 1. Compose mount points (app.use) to downstream routes
        mounts_result = await self._link_mounts_to_routes(session)
        stats["relationships_created"] += mounts_result.get("created", 0)
        stats["by_type"]["MOUNTS"] = mounts_result.get("created", 0)

        # 2. Link API calls to Vert.x routes
        api_result = await self._link_api_calls_to_routes(session)
        stats["relationships_created"] += api_result.get("created", 0)
        stats["by_type"]["CALLS_ROUTE"] = api_result.get("created", 0)
        
        # 3. Link routes to handler functions (best-effort)
        handler_result = await self._link_routes_to_functions(session)
        stats["relationships_created"] += handler_result.get("created", 0)
        stats["by_type"]["HANDLED_BY"] = handler_result.get("created", 0)

        # 4. Link EventBus producers to consumers
        eventbus_result = await self._link_eventbus_producers_consumers(session)
        stats["relationships_created"] += eventbus_result.get("created", 0)
        stats["by_type"]["PUBLISHES_TO"] = eventbus_result.get("created", 0)
        
        # 5. Link DTO usages
        dto_result = await self._link_dto_usages(session)
        stats["relationships_created"] += dto_result.get("created", 0)
        stats["by_type"]["USES_DTO"] = dto_result.get("created", 0)
        
        # 6. Link component hierarchies
        component_result = await self._link_component_hierarchies(session)
        stats["relationships_created"] += component_result.get("created", 0)
        stats["by_type"]["RENDERS"] = component_result.get("created", 0)
        
        # 7. Link services to their dependencies
        service_result = await self._link_service_dependencies(session)
        stats["relationships_created"] += service_result.get("created", 0)
        stats["by_type"]["DEPENDS_ON"] = service_result.get("created", 0)
        
        return stats

    async def _link_mounts_to_routes(self, session) -> Dict[str, Any]:
        """Link app.use('/prefix', router) mounts to routes within mounted modules and compute full_path."""
        query = """
        MATCH (mount:VertxRoute)
        WHERE mount.method = 'USE'
          AND mount.path IS NOT NULL AND mount.path <> ''
          AND mount.mount_target_files IS NOT NULL
          AND size(mount.mount_target_files) > 0
        UNWIND mount.mount_target_files AS target_file
        MATCH (f:File {path: target_file})-[:CONTAINS]->(route:VertxRoute)
        WHERE route.method <> 'USE'
        MERGE (mount)-[m:MOUNTS]->(route)
        SET m.created_at = coalesce(m.created_at, datetime()),
            m.last_updated = datetime(),
            route.full_path = CASE
                WHEN mount.path ENDS WITH '/' AND route.path STARTS WITH '/' THEN substring(mount.path, 0, size(mount.path) - 1) + route.path
                WHEN NOT mount.path ENDS WITH '/' AND NOT route.path STARTS WITH '/' THEN mount.path + '/' + route.path
                ELSE mount.path + route.path
            END,
            route.last_updated = datetime()
        RETURN count(m) AS relationships_created
        """

        try:
            result = await session.run(query)
            record = await result.single()
            created = record["relationships_created"] if record else 0
            return {"created": created}
        except Exception as e:
            logger.error("Failed to link mounts to routes", error=str(e))
            return {"created": 0, "error": str(e)}

    async def _link_routes_to_functions(self, session) -> Dict[str, Any]:
        """Link VertxRoute nodes to Function nodes when handler_method matches function name (same file)."""
        query = """
        MATCH (route:VertxRoute)
        WHERE route.handler_method IS NOT NULL AND route.handler_method <> ''
        MATCH (fn:Function)
        WHERE fn.file = route.file
          AND (
            route.handler_method = fn.name
            OR route.handler_method ENDS WITH '.' + fn.name
          )
        MERGE (route)-[r:HANDLED_BY]->(fn)
        SET r.created_at = coalesce(r.created_at, datetime()),
            r.last_updated = datetime()
        RETURN count(r) AS relationships_created
        """

        try:
            result = await session.run(query)
            record = await result.single()
            created = record["relationships_created"] if record else 0
            return {"created": created}
        except Exception as e:
            logger.error("Failed to link routes to functions", error=str(e))
            return {"created": 0, "error": str(e)}
    
    async def _link_api_calls_to_routes(self, session) -> Dict[str, Any]:
        """Link React API calls to Vert.x routes with path normalization"""
        
        query = """
        // Find all API calls and routes
        MATCH (api:ApiCall)
        WHERE api.url IS NOT NULL AND api.url <> ''
           MATCH (route:VertxRoute)
           WHERE (route.path IS NOT NULL AND route.path <> '')
        
        // Normalize and match paths
        WITH api, route,
             // Remove common prefixes for matching
             replace(replace(replace(api.url, '/api/v1', ''), '/api/v2', ''), '/api', '') AS normalized_api_url,
               replace(replace(replace(coalesce(route.full_path, route.path), '/api/v1', ''), '/api/v2', ''), '/api', '') AS normalized_route_path
        
        // Match routes - exact match or pattern match
        WHERE normalized_api_url = normalized_route_path
           OR normalized_route_path CONTAINS normalized_api_url
           OR normalized_api_url CONTAINS normalized_route_path
           OR (api.method = route.method AND (
               coalesce(route.full_path, route.path) = api.url 
               OR replace(api.url, '/api/', '/') = coalesce(route.full_path, route.path)
               OR api.url ENDS WITH coalesce(route.full_path, route.path)
           ))
        
        // Create relationship with metadata
        MERGE (api)-[r:CALLS_ROUTE]->(route)
        SET r.method_match = (api.method = route.method OR api.method IS NULL),
            r.created_at = datetime(),
            r.last_updated = datetime()
        
        RETURN count(r) as relationships_created
        """
        
        try:
            result = await session.run(query)
            record = await result.single()
            created = record["relationships_created"] if record else 0
            
            logger.debug(f"Linked {created} API calls to routes")
            return {"created": created}
            
        except Exception as e:
            logger.error("Failed to link API calls to routes", error=str(e))
            return {"created": 0, "error": str(e)}
    
    async def _link_eventbus_producers_consumers(self, session) -> Dict[str, Any]:
        """Link EventBus producers to consumers by address"""
        
        query = """
        // Find all EventBus publishers
        MATCH (producer:EventBusNode)
        WHERE producer.type IN ['publish', 'send', 'request', 'producer']
        
        // Find consumers with matching address
        MATCH (consumer:EventBusNode)
        WHERE consumer.type IN ['consume', 'consumer', 'handler']
          AND consumer.address = producer.address
          AND consumer <> producer
        
        // Create bidirectional relationships
        MERGE (producer)-[r1:PUBLISHES_TO]->(consumer)
        SET r1.address = producer.address,
            r1.created_at = datetime()
        
        MERGE (consumer)-[r2:CONSUMES_FROM]->(producer)
        SET r2.address = consumer.address,
            r2.created_at = datetime()
        
        RETURN count(DISTINCT r1) + count(DISTINCT r2) as relationships_created
        """
        
        try:
            result = await session.run(query)
            record = await result.single()
            created = record["relationships_created"] if record else 0
            
            logger.debug(f"Linked {created} EventBus relationships")
            return {"created": created}
            
        except Exception as e:
            logger.error("Failed to link EventBus producers/consumers", error=str(e))
            return {"created": 0, "error": str(e)}
    
    async def _link_dto_usages(self, session) -> Dict[str, Any]:
        """Link DTO usages across backend and frontend"""
        
        query = """
        // Find DTOs
        MATCH (dto:DTO)
        WHERE dto.name IS NOT NULL
        
        // Find routes that might use DTOs (in file or handler)
        OPTIONAL MATCH (route:VertxRoute)
        WHERE route.file CONTAINS dto.name 
           OR route.handler_method CONTAINS dto.name
        
        FOREACH (_ IN CASE WHEN route IS NOT NULL THEN [1] ELSE [] END |
            MERGE (route)-[r:USES_DTO]->(dto)
            SET r.created_at = datetime()
        )
        
        // Find React components that might use DTOs (in props)
        WITH dto
        OPTIONAL MATCH (comp:ReactComponent)
        WHERE ANY(prop IN comp.props WHERE prop CONTAINS dto.name)
        
        FOREACH (_ IN CASE WHEN comp IS NOT NULL THEN [1] ELSE [] END |
            MERGE (comp)-[r:USES_DTO_FIELDS]->(dto)
            SET r.created_at = datetime()
        )
        
        // Find services that reference DTOs
        WITH dto
        OPTIONAL MATCH (service:Service)
        WHERE service.file CONTAINS dto.name
        
        FOREACH (_ IN CASE WHEN service IS NOT NULL THEN [1] ELSE [] END |
            MERGE (service)-[r:USES_DTO]->(dto)
            SET r.created_at = datetime()
        )
        
        RETURN count(dto) as dtos_processed
        """
        
        # Also run a count query
        count_query = """
        MATCH ()-[r:USES_DTO|USES_DTO_FIELDS]->()
        RETURN count(r) as relationships_created
        """
        
        try:
            await session.run(query)
            count_result = await session.run(count_query)
            record = await count_result.single()
            created = record["relationships_created"] if record else 0
            
            logger.debug(f"Linked {created} DTO usage relationships")
            return {"created": created}
            
        except Exception as e:
            logger.error("Failed to link DTO usages", error=str(e))
            return {"created": 0, "error": str(e)}
    
    async def _link_component_hierarchies(self, session) -> Dict[str, Any]:
        """Link React component parent-child relationships"""
        
        query = """
        // Find components and their potential children
        MATCH (parent:ReactComponent)
        MATCH (child:ReactComponent)
        WHERE parent.name <> child.name
          AND parent.file IS NOT NULL
        
        // Read the parent's file to check for child usage
        // (This is a simplified check - in production you'd parse JSX)
        WITH parent, child
        WHERE parent.file CONTAINS child.name
           OR (parent.file ENDS WITH '.tsx' AND 
               EXISTS { 
                   MATCH (f:File {path: parent.file})
                   WHERE f.path CONTAINS child.name 
               })
        
        MERGE (parent)-[r:RENDERS]->(child)
        SET r.created_at = datetime()
        
        RETURN count(r) as relationships_created
        """
        
        try:
            result = await session.run(query)
            record = await result.single()
            created = record["relationships_created"] if record else 0
            
            logger.debug(f"Linked {created} component hierarchy relationships")
            return {"created": created}
            
        except Exception as e:
            logger.error("Failed to link component hierarchies", error=str(e))
            return {"created": 0, "error": str(e)}
    
    async def _link_service_dependencies(self, session) -> Dict[str, Any]:
        """Link services to their dependencies (other services, DTOs, etc.)"""
        
        query = """
        // Find services in the same repository
        MATCH (service1:Service)
        MATCH (service2:Service)
        WHERE service1 <> service2
          AND service1.repository_hash = service2.repository_hash
          AND service1.file CONTAINS service2.name
        
        MERGE (service1)-[r:DEPENDS_ON]->(service2)
        SET r.created_at = datetime()
        
        RETURN count(r) as relationships_created
        """
        
        try:
            result = await session.run(query)
            record = await result.single()
            created = record["relationships_created"] if record else 0
            
            logger.debug(f"Linked {created} service dependency relationships")
            return {"created": created}
            
        except Exception as e:
            logger.error("Failed to link service dependencies", error=str(e))
            return {"created": 0, "error": str(e)}
    
    async def find_cross_service_dependencies(self, service_name: str) -> List[Dict[str, Any]]:
        """Find all dependencies for a given service"""
        
        query = """
        MATCH (service:Service {name: $service_name})
        OPTIONAL MATCH (service)-[:DEPENDS_ON]->(dependency:Service)
        OPTIONAL MATCH (service)-[:USES_DTO]->(dto:DTO)
        OPTIONAL MATCH (service)<-[:CALLS_ROUTE]-(caller:ApiCall)
        OPTIONAL MATCH (caller)<-[:MAKES_API_CALL]-(component:ReactComponent)
        
        RETURN 
            service,
            collect(DISTINCT dependency) as service_dependencies,
            collect(DISTINCT dto) as dto_dependencies,
            collect(DISTINCT component) as frontend_callers
        """
        
        try:
            async with neo4j_client.session() as session:
                result = await session.run(query, {"service_name": service_name})
                records = await result.data()
                
                dependencies = []
                for record in records:
                    if record.get("service_dependencies"):
                        for dep in record["service_dependencies"]:
                            if dep:
                                dependencies.append({
                                    "type": "Service",
                                    "name": dep.get("name"),
                                    "relationship": "DEPENDS_ON"
                                })
                    
                    if record.get("dto_dependencies"):
                        for dto in record["dto_dependencies"]:
                            if dto:
                                dependencies.append({
                                    "type": "DTO",
                                    "name": dto.get("name"),
                                    "relationship": "USES_DTO"
                                })
                    
                    if record.get("frontend_callers"):
                        for comp in record["frontend_callers"]:
                            if comp:
                                dependencies.append({
                                    "type": "ReactComponent",
                                    "name": comp.get("name"),
                                    "relationship": "CALLED_BY"
                                })
                
                return dependencies
                
        except Exception as e:
            logger.error(f"Failed to find dependencies for service: {service_name}", error=str(e))
            return []
    
    async def get_api_flow(self, api_url: str) -> Dict[str, Any]:
        """Get the complete flow for an API endpoint"""
        
        query = """
        MATCH (api:ApiCall {url: $api_url})
        OPTIONAL MATCH (component:ReactComponent)-[:MAKES_API_CALL]->(api)
        OPTIONAL MATCH (api)-[:CALLS_ROUTE]->(route:VertxRoute)
        OPTIONAL MATCH (route)-[:USES_DTO]->(dto:DTO)
        OPTIONAL MATCH (route)<-[:PUBLISHES_TO]-(eventbus:EventBusNode)
        
        RETURN api, component, route, dto, eventbus
        """
        
        try:
            async with neo4j_client.session() as session:
                result = await session.run(query, {"api_url": api_url})
                records = await result.data()
                
                flow = {
                    "api_call": None,
                    "frontend_component": None,
                    "backend_route": None,
                    "dtos_used": [],
                    "eventbus_flows": []
                }
                
                for record in records:
                    if record.get("api"):
                        flow["api_call"] = dict(record["api"])
                    if record.get("component"):
                        flow["frontend_component"] = dict(record["component"])
                    if record.get("route"):
                        flow["backend_route"] = dict(record["route"])
                    if record.get("dto"):
                        flow["dtos_used"].append(dict(record["dto"]))
                    if record.get("eventbus"):
                        flow["eventbus_flows"].append(dict(record["eventbus"]))
                
                return flow
                
        except Exception as e:
            logger.error(f"Failed to get API flow for: {api_url}", error=str(e))
            return {}
    
    async def get_linking_statistics(self) -> Dict[str, Any]:
        """Get statistics about all relationships in the graph"""
        
        query = """
        MATCH ()-[r]->()
        RETURN type(r) as relationship_type, count(r) as count
        ORDER BY count DESC
        """
        
        try:
            async with neo4j_client.session() as session:
                result = await session.run(query)
                records = await result.data()
                
                total = sum(record["count"] for record in records)
                
                return {
                    "total_relationships": total,
                    "by_type": {
                        record["relationship_type"]: record["count"]
                        for record in records
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error("Failed to get linking statistics", error=str(e))
            return {"total_relationships": 0, "by_type": {}}