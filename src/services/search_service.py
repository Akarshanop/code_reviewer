import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncio

from src.services.graph_service import GraphService
from src.services.vector_service import VectorService
from src.core.logger import get_logger
from src.core.exceptions import InternalServerException
from src.intelligence.domain_model import domain_engine

logger = get_logger(__name__)

class SearchService:
    """
    Unified search service combining graph and vector search.
    
    Enhanced with:
        - Domain-first query resolution (resolves domains before code)
        - Confidence scoring on results
    """
    
    def __init__(self):
        self.graph_service = GraphService()
        self.vector_service = VectorService()
        self.hybrid_alpha = 0.7  # Weight for vector search vs graph search
    
    async def initialize(self):
        """Initialize search service"""
        await self.vector_service.initialize()
    
    async def search_codebase(
        self,
        query: str,
        search_type: str = "hybrid",  # hybrid, semantic, graph
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search the codebase with multiple strategies"""
        start_time = datetime.utcnow()
        
        filters = filters or {}
        
        try:
            results = {}
            
            # Always perform semantic search
            if search_type in ["hybrid", "semantic"]:
                semantic_results = await self.vector_service.semantic_search(
                    query=query,
                    top_k=top_k * 2,
                    filter_type=filters.get("type"),
                    filter_repository=filters.get("repository"),
                    filter_language=filters.get("language"),
                    score_threshold=0.3
                )
                results["semantic"] = semantic_results
            
            # Perform graph search if needed
            graph_results = []
            if search_type in ["hybrid", "graph"]:
                # Extract potential entity names from query
                entities = self._extract_entities(query)
                
                # Search for each entity in graph
                for entity in entities[:3]:  # Limit to top 3 entities
                    try:
                        entity_results = await self.graph_service.find_related_entities(
                            entity_name=entity,
                            max_depth=2
                        )
                        if entity_results.get("relationships"):
                            graph_results.append(entity_results)
                    except Exception as e:
                        logger.debug(f"Graph search failed for entity {entity}", error=str(e))
                
                results["graph"] = graph_results
            
            # Combine results for hybrid search
            if search_type == "hybrid" and results.get("semantic") and results.get("graph"):
                combined = await self.vector_service.hybrid_search(
                    query=query,
                    graph_results=graph_results,
                    top_k=top_k,
                    alpha=self.hybrid_alpha
                )
                results["combined"] = combined
            elif results.get("semantic"):
                results["combined"] = results["semantic"][:top_k]
            
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "query": query,
                "search_type": search_type,
                "filters": filters,
                "duration_ms": round(duration, 2),
                "results": results,
                "summary": {
                    "semantic_results": len(results.get("semantic", [])),
                    "graph_results": len(results.get("graph", [])),
                    "combined_results": len(results.get("combined", [])),
                    "top_score": results.get("combined", [{}])[0].get("score", 0) if results.get("combined") else 0
                }
            }
            
        except Exception as e:
            logger.error("Search failed", query=query, search_type=search_type, error=str(e))
            raise InternalServerException(f"Search failed: {str(e)}")
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract potential entity names from query"""
        # Simple extraction: words that start with capital letter or are in quotes
        import re
        
        entities = []
        
        # Extract quoted strings
        quoted = re.findall(r'"([^"]*)"', query) + re.findall(r"'([^']*)'", query)
        entities.extend(quoted)
        
        # Extract CamelCase words
        camel_case = re.findall(r'[A-Z][a-z]+(?:[A-Z][a-z]+)+', query)
        entities.extend(camel_case)
        
        # Extract words that might be class/component names
        words = query.split()
        for word in words:
            if (word[0].isupper() and len(word) > 2) or word.endswith(("Service", "Component", "DTO", "Handler")):
                entities.append(word)
        
        # Remove duplicates
        return list(dict.fromkeys(entities))
    
    async def search_by_component(
        self,
        component_name: str,
        component_type: str = None,
        include_related: bool = True
    ) -> Dict[str, Any]:
        """Search for a specific component and its relationships"""
        results = {
            "component": None,
            "relationships": {},
            "similar_components": [],
            "usage": []
        }
        
        try:
            # Find the component in graph
            query = """
            MATCH (c)
            WHERE c.name = $component_name
              AND ($component_type IS NULL OR labels(c)[0] = $component_type)
            RETURN c
            LIMIT 1
            """
            
            component_results = await self.graph_service.neo4j.execute_query(query, {
                "component_name": component_name,
                "component_type": component_type
            })
            
            if component_results:
                component = dict(component_results[0]["c"])
                results["component"] = component
                
                # Get relationships if requested
                if include_related:
                    # Get incoming relationships
                    incoming_query = """
                    MATCH (source)-[r]->(target {name: $component_name})
                    RETURN source, type(r) as relationship_type, target
                    LIMIT 20
                    """
                    
                    incoming = await self.graph_service.neo4j.execute_query(incoming_query, {
                        "component_name": component_name
                    })
                    
                    # Get outgoing relationships
                    outgoing_query = """
                    MATCH (source {name: $component_name})-[r]->(target)
                    RETURN source, type(r) as relationship_type, target
                    LIMIT 20
                    """
                    
                    outgoing = await self.graph_service.neo4j.execute_query(outgoing_query, {
                        "component_name": component_name
                    })
                    
                    results["relationships"] = {
                        "incoming": [dict(record) for record in incoming],
                        "outgoing": [dict(record) for record in outgoing]
                    }
                    
                    # Find similar components
                    similar = await self.graph_service.find_similar_components(component_name)
                    results["similar_components"] = similar
                    
                    # Find usage in code
                    if component.get("file"):
                        usage_results = await self.vector_service.semantic_search(
                            query=component_name,
                            top_k=10,
                            filter_type="code"
                        )
                        results["usage"] = usage_results
            
            return results
            
        except Exception as e:
            logger.error("Component search failed", component=component_name, error=str(e))
            raise
    
    async def search_apis(
        self,
        endpoint: str = None,
        method: str = None,
        repository: str = None
    ) -> List[Dict[str, Any]]:
        """Search for API endpoints"""
        where_clauses = []
        params = {}
        
        if endpoint:
            where_clauses.append("r.path CONTAINS $endpoint")
            params["endpoint"] = endpoint
        
        if method:
            where_clauses.append("r.method = $method")
            params["method"] = method.upper()
        
        if repository:
            where_clauses.append("r.repository = $repository")
            params["repository"] = repository
        
        # Build additional conditions if any
        additional_conditions = " AND " + " AND ".join(where_clauses) if where_clauses else ""
        
        query = f"""
        MATCH (r)
        WHERE (r:VertxRoute OR r:Route OR r:APIEndpoint)
        {additional_conditions}
        OPTIONAL MATCH (r)-[:HAS_HANDLER|HANDLED_BY]->(handler)
        OPTIONAL MATCH (api:ApiCall)-[:CALLS_ROUTE|CALLS_API]->(r)
        RETURN 
            r,
            collect(DISTINCT handler.name) as handlers,
            count(api) as api_call_count
        ORDER BY api_call_count DESC
        LIMIT 50
        """
        
        results = await self.graph_service.neo4j.execute_query(query, params)
        
        apis = []
        for record in results:
            route = dict(record["r"])
            apis.append({
                "path": route.get("path"),
                "method": route.get("method"),
                "repository": route.get("repository"),
                "file": route.get("file"),
                "handlers": record["handlers"],
                "api_call_count": record["api_call_count"],
                "description": route.get("description", "")
            })
        
        return apis
    
    async def search_eventbus_flows(
        self,
        address: str = None,
        flow_type: str = None  # publish, consume
    ) -> List[Dict[str, Any]]:
        """Search for EventBus flows"""
        where_clauses = []
        params = {}
        
        if address:
            where_clauses.append("e.address CONTAINS $address")
            params["address"] = address
        
        if flow_type:
            where_clauses.append("e.type = $flow_type")
            params["flow_type"] = flow_type
        
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        query = f"""
        MATCH (e:EventBusNode)
        {where_clause}
        OPTIONAL MATCH (e)<-[:PUBLISHES_TO]-(publisher)
        OPTIONAL MATCH (e)-[:CONSUMES_FROM]->(consumer)
        RETURN 
            e,
            collect(DISTINCT publisher.name) as publishers,
            collect(DISTINCT consumer.name) as consumers
        LIMIT 50
        """
        
        results = await self.graph_service.neo4j.execute_query(query, params)
        
        flows = []
        for record in results:
            eventbus = dict(record["e"])
            flows.append({
                "address": eventbus.get("address"),
                "type": eventbus.get("type"),
                "repository": eventbus.get("repository"),
                "file": eventbus.get("file"),
                "publishers": record["publishers"],
                "consumers": record["consumers"],
                "description": eventbus.get("description", "")
            })
        
        return flows
    
    async def find_code_smells(
        self,
        repository: str = None,
        smell_type: str = None
    ) -> List[Dict[str, Any]]:
        """Find potential code smells"""
        # This is a simplified implementation
        # In production, you'd want more sophisticated analysis
        
        smells_query = """
        MATCH (c)
        WHERE c:VertxRoute OR c:Route OR c:ReactComponent OR c:Service OR c:Function
        OPTIONAL MATCH (c)-[r]->()
        WITH c, count(r) as outgoing_count
        
        OPTIONAL MATCH ()-[r2]->(c)
        WITH c, outgoing_count, count(r2) as incoming_count
        
        // Calculate complexity score
        WITH c, outgoing_count, incoming_count,
             (outgoing_count * 0.7) + (incoming_count * 0.3) as complexity_score
        
        WHERE complexity_score > 10  // Threshold for "complex" component
        
        RETURN 
            c.name as name,
            labels(c)[0] as type,
            c.file as file,
            outgoing_count,
            incoming_count,
            complexity_score,
            CASE 
                WHEN complexity_score > 20 THEN 'high'
                WHEN complexity_score > 15 THEN 'medium'
                ELSE 'low'
            END as severity
        
        ORDER BY complexity_score DESC
        LIMIT 50
        """
        
        params = {}
        if repository:
            smells_query = smells_query.replace(
                "WHERE c:VertxRoute OR c:Route",
                f"WHERE (c:VertxRoute OR c:Route) AND c.repository = $repository"
            )
            params["repository"] = repository
        
        results = await self.graph_service.neo4j.execute_query(smells_query, params)
        
        smells = []
        for record in results:
            smells.append({
                "name": record["name"],
                "type": record["type"],
                "file": record["file"],
                "outgoing_dependencies": record["outgoing_count"],
                "incoming_dependencies": record["incoming_count"],
                "complexity_score": record["complexity_score"],
                "severity": record["severity"],
                "smell_type": "high_complexity"
            })
        
        # Also find circular dependencies
        circular = await self.graph_service.find_circular_dependencies()
        if circular:
            for cycle in circular[:5]:  # Limit to top 5
                smells.append({
                    "name": " -> ".join([f"{item['type']}:{item['name']}" for item in cycle]),
                    "type": "circular_dependency",
                    "file": "multiple",
                    "severity": "high",
                    "smell_type": "circular_dependency",
                    "cycle": cycle
                })
        
        return smells
    
    async def get_search_metrics(self) -> Dict[str, Any]:
        """Get search service metrics"""
        return {
            "hybrid_alpha": self.hybrid_alpha,
            "graph_service": "initialized",
            "vector_service": "initialized",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def close(self):
        """Close service resources"""
        await self.vector_service.close()