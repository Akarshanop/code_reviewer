import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncio

from src.services.graph_service import GraphService
from src.services.vector_service import VectorService
from src.core.logger import get_logger
from src.core.exceptions import InternalServerException
from src.intelligence.domain_model import domain_engine
from src.intelligence.subgraph_compression import SubgraphCompressionEngine

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
        self.subgraph_compression = SubgraphCompressionEngine()
        self.hybrid_alpha = 0.7  # Backward compatibility default
        self.intent_alpha = {
            "structural": 0.1,
            "semantic": 0.8,
            "impact": 0.0,
            "temporal": 0.0,
        }
    
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
            intent = self._detect_query_intent(query, search_type)
            if intent == "structural":
                results = await self._run_structural_query(query=query, top_k=top_k, filters=filters)
            elif intent == "impact":
                results = await self._run_impact_query(query=query, top_k=top_k)
            elif intent == "temporal":
                results = await self._run_temporal_query(query=query, top_k=top_k)
            else:
                results = await self._run_semantic_query(query=query, top_k=top_k, filters=filters)
            
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "query": query,
                "search_type": search_type,
                "intent": intent,
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

    def _detect_query_intent(self, query: str, search_type: str) -> str:
        """Route user query to structural / semantic / impact / temporal path."""
        if search_type == "graph":
            return "structural"
        if search_type == "semantic":
            return "semantic"
        q = (query or "").lower()
        temporal_terms = ("last week", "last month", "recently", "changed in", "timeline", "history")
        impact_terms = ("impact", "blast radius", "what breaks", "affected", "dependents")
        structural_terms = ("connected", "dependency graph", "call graph", "path between", "how is")
        if any(term in q for term in temporal_terms):
            return "temporal"
        if any(term in q for term in impact_terms):
            return "impact"
        if any(term in q for term in structural_terms):
            return "structural"
        return "semantic"

    async def _run_semantic_query(
        self, query: str, top_k: int, filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        semantic_results = await self.vector_service.semantic_search(
            query=query,
            top_k=top_k * 2,
            filter_type=filters.get("type"),
            filter_repository=filters.get("repository"),
            filter_language=filters.get("language"),
            score_threshold=0.3,
        )
        graph_results, graph_candidates = await self._collect_graph_candidates(query, max_entities=3)
        alpha = self.intent_alpha["semantic"]
        if semantic_results and graph_candidates:
            combined = await self.vector_service.hybrid_search(
                query=query,
                graph_results=graph_candidates,
                top_k=top_k,
                alpha=alpha,
            )
        else:
            combined = semantic_results[:top_k]
        compressed = self.subgraph_compression.compress(
            graph_results=graph_results,
            graph_candidates=graph_candidates,
            max_nodes=min(50, top_k * 5),
        )
        return {
            "semantic": semantic_results,
            "graph": graph_results,
            "graph_candidates": graph_candidates,
            "combined": combined,
            "compressed_subgraph": compressed,
            "intent_alpha": alpha,
        }

    async def _run_structural_query(
        self, query: str, top_k: int, filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        graph_results, graph_candidates = await self._collect_graph_candidates(query, max_entities=5)
        semantic_results = await self.vector_service.semantic_search(
            query=query,
            top_k=max(5, top_k),
            filter_type=filters.get("type"),
            filter_repository=filters.get("repository"),
            filter_language=filters.get("language"),
            score_threshold=0.35,
        )
        alpha = self.intent_alpha["structural"]
        combined = await self.vector_service.hybrid_search(
            query=query,
            graph_results=graph_candidates,
            top_k=top_k,
            alpha=alpha,
        ) if graph_candidates else semantic_results[:top_k]
        compressed = self.subgraph_compression.compress(
            graph_results=graph_results,
            graph_candidates=graph_candidates,
            max_nodes=min(50, top_k * 5),
        )
        return {
            "semantic": semantic_results,
            "graph": graph_results,
            "graph_candidates": graph_candidates,
            "combined": combined,
            "compressed_subgraph": compressed,
            "intent_alpha": alpha,
        }

    async def _run_impact_query(self, query: str, top_k: int) -> Dict[str, Any]:
        entities = await self._extract_entities(query)
        target = entities[0] if entities else query.strip()
        impacts: List[Dict[str, Any]] = []
        try:
            impact = await self.graph_service.analyze_impact(file_path=target, change_type="modify")
            impacts.append(impact)
        except Exception:
            for entity in entities[:top_k]:
                try:
                    impacts.append(await self.graph_service.analyze_impact(file_path=entity, change_type="modify"))
                except Exception:
                    continue
        return {
            "semantic": [],
            "graph": impacts,
            "graph_candidates": [],
            "combined": impacts[:top_k],
            "intent_alpha": self.intent_alpha["impact"],
        }

    async def _run_temporal_query(self, query: str, top_k: int) -> Dict[str, Any]:
        temporal_query = """
        MATCH (c:ChangeEvent)
        WHERE c.timestamp IS NOT NULL
        RETURN c
        ORDER BY c.timestamp DESC
        LIMIT $top_k
        """
        rows = await self.graph_service.neo4j.execute_query(temporal_query, {"top_k": top_k})
        events = [dict(r.get("c", {})) for r in rows] if rows else []
        return {
            "semantic": [],
            "graph": events,
            "graph_candidates": [],
            "combined": events,
            "intent_alpha": self.intent_alpha["temporal"],
            "query_template": "temporal_change_event",
        }

    async def _collect_graph_candidates(self, query: str, max_entities: int = 3) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        graph_results: List[Dict[str, Any]] = []
        graph_candidates: List[Dict[str, Any]] = []
        entities = await self._extract_entities(query)
        for entity in entities[:max_entities]:
            try:
                entity_results = await self.graph_service.find_related_entities(
                    entity_name=entity,
                    max_depth=2,
                )
                if entity_results.get("relationships"):
                    graph_results.append(entity_results)
                    graph_candidates.extend(self._flatten_graph_results(entity_results))
            except Exception as e:
                logger.debug(f"Graph search failed for entity {entity}", error=str(e))
        return graph_results, graph_candidates
    
    async def _extract_entities(self, query: str) -> List[str]:
        """Extract entity names using vocabulary lookup against the graph, falling back to CamelCase regex."""
        import re

        found = await self._extract_entities_from_vocab(query)
        if found:
            return found
        return self._extract_entities_camelcase(query)

    async def _extract_entities_from_vocab(self, query: str) -> List[str]:
        """Match query tokens against known graph node names via Neo4j."""
        import re

        words = re.findall(r'\b\w+\b', query)
        if not words:
            return []

        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "need", "must", "ought",
            "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
            "they", "them", "their", "this", "that", "these", "those", "what",
            "which", "who", "whom", "how", "when", "where", "why", "if", "then",
            "and", "but", "or", "nor", "not", "no", "so", "for", "yet", "both",
            "from", "to", "in", "on", "at", "by", "with", "about", "of", "as",
            "into", "like", "through", "after", "before", "between", "out", "up",
            "all", "each", "every", "any", "some", "add", "get", "set", "show",
            "find", "search", "list", "create", "update", "delete", "remove",
            "new", "old", "make", "use", "see", "tell", "give", "take",
        }

        filtered = [w for w in words if w.lower() not in stop_words and len(w) > 2]
        if not filtered:
            return []

        candidates = []
        for length in range(min(3, len(filtered)), 0, -1):
            for i in range(len(filtered) - length + 1):
                candidates.append(" ".join(filtered[i:i + length]))

        try:
            from src.database.neo4j_client import neo4j_client
            if not neo4j_client._initialized:
                await neo4j_client.initialize()

            query_str = """
            UNWIND $phrases AS phrase
            MATCH (n)
            WHERE toLower(n.name) = toLower(phrase)
            RETURN DISTINCT n.name AS name
            LIMIT 10
            """
            results = await neo4j_client.execute_query(query_str, {"phrases": candidates[:30]})
            return [r["name"] for r in (results or []) if r.get("name")]
        except Exception as e:
            logger.debug(f"Vocab entity lookup failed: {e}")
            return []

    def _extract_entities_camelcase(self, query: str) -> List[str]:
        """Extract potential entity names from query using CamelCase regex and heuristics."""
        import re

        entities = []

        quoted = re.findall(r'"([^"]*)"', query) + re.findall(r"'([^']*)'", query)
        entities.extend(quoted)

        camel_case = re.findall(r'[A-Z][a-z]+(?:[A-Z][a-z]+)+', query)
        entities.extend(camel_case)

        words = query.split()
        for word in words:
            if len(word) > 2 and (word[0].isupper() or word.endswith(("Service", "Component", "DTO", "Handler"))):
                entities.append(word)

        return list(dict.fromkeys(entities))

    def _flatten_graph_results(self, entity_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert GraphService grouped result into entity-level candidates
        for hybrid fusion.
        """
        flattened: List[Dict[str, Any]] = []
        relationships = entity_results.get("relationships", {}) or {}

        for rel_type, targets in relationships.items():
            for item in targets:
                target = item.get("target", {}) if isinstance(item, dict) else {}
                if not isinstance(target, dict) or not target:
                    continue
                target_id = target.get("id")
                if not target_id:
                    continue

                path_length = item.get("path_length", 1) if isinstance(item, dict) else 1
                try:
                    path_len_val = max(1, int(path_length))
                except Exception:
                    path_len_val = 1

                flattened.append({
                    "id": target_id,
                    "score": 1.0 / path_len_val,
                    "name": target.get("name", ""),
                    "type": target.get("type", "graph_node"),
                    "repository": target.get("repository", target.get("repository_name", "")),
                    "file": target.get("file", ""),
                    "metadata": {
                        "relationship_type": rel_type,
                        "path_length": path_len_val,
                    },
                })

        return flattened
    
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