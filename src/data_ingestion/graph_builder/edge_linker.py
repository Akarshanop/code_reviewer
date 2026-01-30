# edge_linker.py - IMPROVED VERSION
import logging
from typing import Dict, List, Any
from urllib.parse import urlparse
import re

logger = logging.getLogger(__name__)

class EdgeLinker:
    
    def __init__(self):
        pass  # Neo4j driver will be passed as parameter in async methods
        
    async def link_all_edges(self):
        """Link all types of edges between services"""
        await self._link_api_calls_to_routes()
        await self._link_eventbus_producers_consumers()
        await self._link_dto_usages()
        await self._link_component_hierarchies()
        
        logger.info("All edges linked successfully")
    
    async def _link_api_calls_to_routes(self):
        """Link React API calls to Vert.x routes"""
        query = """
        // Find React components with API calls
        MATCH (comp:ReactComponent)-[:HAS_API_CALL]->(api:ApiCall)
        
        // Find Vert.x routes with matching paths
        MATCH (route:VertxRoute)
        WHERE route.path = api.url 
           OR route.path = replace(api.url, '/api/v1/', '/api/')
           OR route.path = replace(api.url, '/api/', '/')
           OR api.url CONTAINS route.path
        
        // Create relationship with metadata
        MERGE (api)-[r:CALLS_ROUTE {
            method_match: route.method = api.method OR api.method IS NULL,
            path_similarity: apoc.text.jaroWinklerDistance(api.url, route.path)
        }]->(route)
        
        RETURN count(r) as relationships_created
        """
        
        result = await self.neo4j_client.execute_query(query)
        logger.info(f"Linked {result[0]['relationships_created']} API calls to routes")
    
    async def _link_eventbus_producers_consumers(self):
        """Link EventBus producers to consumers"""
        query = """
        // Find all EventBus producers and consumers
        MATCH (producer:VertxMethod {type: 'eventbus_publish'})
        MATCH (consumer:VertxMethod {type: 'eventbus_consume'})
        WHERE producer.address = consumer.address
        
        // Create bidirectional relationships
        MERGE (producer)-[r1:PUBLISHES_TO]->(consumer)
        MERGE (consumer)-[r2:CONSUMES_FROM]->(producer)
        
        RETURN count(r1) + count(r2) as relationships_created
        """
        
        result = await self.neo4j_client.execute_query(query)
        logger.info(f"Linked {result[0]['relationships_created']} EventBus relationships")
    
    async def _link_dto_usages(self):
        """Link DTO usages across layers"""
        query = """
        // Find DTO classes
        MATCH (dto:DTO)
        
        // Find Vert.x methods that use DTOs as parameters or return types
        MATCH (method:VertxMethod)
        WHERE method.signature CONTAINS dto.name 
           OR method.return_type = dto.name
        
        MERGE (method)-[r:USES_DTO]->(dto)
        
        WITH dto
        
        // Find React components that might use DTO fields
        MATCH (comp:ReactComponent)
        WHERE ANY(prop IN comp.props WHERE prop.type CONTAINS dto.name)
        
        MERGE (comp)-[r2:USES_DTO_FIELDS]->(dto)
        
        RETURN count(r) + count(r2) as relationships_created
        """
        
        result = await self.neo4j_client.execute_query(query)
        logger.info(f"Linked {result[0]['relationships_created']} DTO relationships")
    
    async def _link_component_hierarchies(self):
        """Link React component parent-child relationships"""
        query = """
        // Find components that render other components
        MATCH (parent:ReactComponent), (child:ReactComponent)
        WHERE parent.name <> child.name 
          AND parent.jsx_content CONTAINS child.name
        
        MERGE (parent)-[r:RENDERS]->(child)
        SET r.occurrences = size(split(parent.jsx_content, child.name)) - 1
        
        RETURN count(r) as relationships_created
        """
        
        result = await self.neo4j_client.execute_query(query)
        logger.info(f"Linked {result[0]['relationships_created']} component hierarchies")
    
    async def find_cross_service_dependencies(self, service_name: str) -> List[Dict[str, Any]]:
        """Find all dependencies for a given service"""
        query = """
        MATCH (service:VertxService {name: $service_name})
        OPTIONAL MATCH (service)-[:CONTAINS_METHOD]->(method:VertxMethod)
        OPTIONAL MATCH (method)-[:CALLS_ROUTE|:PUBLISHES_TO|:USES_DTO]->(dependency)
        OPTIONAL MATCH (dependency)<-[:CALLS_ROUTE|:CONSUMES_FROM|:USES_DTO_FIELDS]-(frontend:ReactComponent)
        
        RETURN service, method, dependency, frontend
        """
        
        results = await self.neo4j_client.execute_query(query, {"service_name": service_name})
        
        # Process results into a structured format
        dependencies = []
        for record in results:
            if record.get("dependency"):
                dep = {
                    "type": list(record["dependency"].labels)[0],
                    "name": record["dependency"]["name"],
                    "relationship": self._get_relationship_type(record)
                }
                if dep not in dependencies:
                    dependencies.append(dep)
        
        return dependencies
    
    def _get_relationship_type(self, record) -> str:
        """Determine relationship type from record"""
        relationships = record.get("_relationships", [])
        if relationships:
            return type(relationships[0]).__name__
        return "unknown"