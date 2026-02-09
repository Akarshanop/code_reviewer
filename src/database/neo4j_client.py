import logging
from typing import Any, Dict, List, Optional, Union
from neo4j import AsyncGraphDatabase, AsyncSession, AsyncTransaction
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime

from config.settings import settings
from src.core.exceptions import GraphDatabaseException
from src.core.logger import get_logger

logger = get_logger(__name__)

class Neo4jClient:
    """Enterprise-grade Neo4j client with connection pooling and retry logic"""
    
    def __init__(self):
        self.driver = None
        self._initialized = False
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize connection pool"""
        if not self._initialized:
            async with self._lock:
                if not self._initialized:
                    try:
                        self.driver = AsyncGraphDatabase.driver(
                            settings.NEO4J_URI,
                            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                            max_connection_pool_size=settings.NEO4J_MAX_CONNECTION_POOL_SIZE,
                            connection_acquisition_timeout=30.0,
                            connection_timeout=15.0,
                            max_connection_lifetime=3600,
                            keep_alive=True,
                        )
                        
                        # Verify connectivity
                        await self.verify_connectivity()
                        
                        # Run constraints in background (don't block startup)
                        logger.info("Scheduling constraints creation in background...")
                        asyncio.create_task(self._ensure_constraints_background())
                        
                        self._initialized = True
                        logger.info(
                            "Neo4j client initialized",
                            uri=settings.NEO4J_URI[:30] + "..." if len(settings.NEO4J_URI) > 30 else settings.NEO4J_URI,
                            pool_size=settings.NEO4J_MAX_CONNECTION_POOL_SIZE
                        )
                        
                    except Exception as e:
                        logger.error("Failed to initialize Neo4j client", error=str(e))
                        raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ServiceUnavailable, ConnectionError))
    )
    async def verify_connectivity(self):
        """Verify Neo4j connectivity with retry"""
        try:
            await self.driver.verify_connectivity()
            logger.info("Neo4j connectivity verified")
        except Exception as e:
            logger.error("Neo4j connectivity check failed", error=str(e))
            raise
    
    async def _ensure_constraints_background(self):
        """Create constraints in background - don't block startup"""
        try:
            await asyncio.sleep(2)  # Give server time to start
            await self.ensure_constraints()
            logger.info("Background constraints creation completed")
        except Exception as e:
            logger.warning(f"Background constraints creation failed: {e}")
    
    async def ensure_constraints(self):
        """Create Neo4j constraints and indexes"""
        constraints = [
            # Unique constraints
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Repository) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:VertxRoute) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:ReactComponent) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:DTO) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:EventBusNode) REQUIRE n.id IS UNIQUE",
            
            # Indexes for common queries
            "CREATE INDEX IF NOT EXISTS FOR (n:VertxRoute) ON (n.path)",
            "CREATE INDEX IF NOT EXISTS FOR (n:ReactComponent) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:ApiCall) ON (n.url)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Module) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:File) ON (n.path)",

            # --- Intelligence Layer indexes ---
            # Behavioral classification
            "CREATE INDEX IF NOT EXISTS FOR (n:Function) ON (n.behavioral_labels)",
            "CREATE INDEX IF NOT EXISTS FOR (n:VertxRoute) ON (n.behavioral_labels)",
            # Temporal metadata
            "CREATE INDEX IF NOT EXISTS FOR (n:File) ON (n.volatility_score)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Function) ON (n.volatility_score)",
            # Domain inference
            "CREATE INDEX IF NOT EXISTS FOR (n:File) ON (n.domain)",
            # Evidence-weighted edges (composite index on confidence)
            "CREATE INDEX IF NOT EXISTS FOR ()-[r:CALLS_ROUTE]-() ON (r.confidence_score)",
            "CREATE INDEX IF NOT EXISTS FOR ()-[r:GUARDS]-() ON (r.confidence_score)",
            # Domain node type
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Domain) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Capability) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:PRD) REQUIRE n.id IS UNIQUE",
        ]
        
        # Fulltext index created separately (can be slow)
        fulltext_index = "CREATE FULLTEXT INDEX node_search IF NOT EXISTS FOR (n:VertxRoute|ReactComponent|DTO|Service) ON EACH [n.name, n.description]"
        
        async with self.session() as session:
            for i, constraint in enumerate(constraints):
                try:
                    logger.debug(f"Creating constraint {i+1}/{len(constraints)}")
                    await session.run(constraint)
                except Exception as e:
                    logger.warning(f"Failed to create constraint/index: {constraint[:50]}...", error=str(e))
            
            # Try fulltext index separately with timeout
            try:
                logger.info("Creating fulltext search index (may take a moment)...")
                await asyncio.wait_for(session.run(fulltext_index), timeout=30.0)
                logger.info("Fulltext index created")
            except asyncio.TimeoutError:
                logger.warning("Fulltext index creation timed out - will be created in background")
            except Exception as e:
                logger.warning(f"Failed to create fulltext index: {e}")
    
    @asynccontextmanager
    async def session(self, database: str = None, access_mode: str = "write"):
        """Get a Neo4j session with automatic cleanup"""
        if not self._initialized:
            await self.initialize()
        
        session: AsyncSession = self.driver.session(
            database=database or settings.NEO4J_DATABASE,
            default_access_mode=access_mode.upper()
        )
        
        try:
            yield session
        finally:
            await session.close()
    
    @asynccontextmanager
    async def transaction(self, session: AsyncSession):
        """Get a transaction with automatic commit/rollback"""
        tx: AsyncTransaction = await session.begin_transaction()
        try:
            yield tx
            await tx.commit()
        except Exception as e:
            await tx.rollback()
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((Neo4jError, ConnectionError))
    )
    async def execute_query(
        self,
        query: str,
        parameters: Dict[str, Any] = None,
        database: str = None,
        timeout: int = 30
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query with retry logic"""
        start_time = datetime.utcnow()
        
        async with self.session(database=database) as session:
            try:
                result = await session.run(query, parameters or {}, timeout=timeout)
                records = await result.data()
                
                duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                logger.debug(
                    "Neo4j query executed",
                    query=query[:100] + "..." if len(query) > 100 else query,
                    duration_ms=round(duration, 2),
                    record_count=len(records)
                )
                
                return records
                
            except Neo4jError as e:
                duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                logger.error(
                    "Neo4j query failed",
                    query=query[:100] + "..." if len(query) > 100 else query,
                    error=str(e),
                    duration_ms=round(duration, 2)
                )
                raise GraphDatabaseException(
                    f"Neo4j query failed: {str(e)}",
                    query=query
                )
    
    async def execute_write_query(
        self,
        query: str,
        parameters: Dict[str, Any] = None,
        database: str = None
    ) -> List[Dict[str, Any]]:
        """Execute write query with transaction"""
        async with self.session(database=database, access_mode="write") as session:
            async with self.transaction(session) as tx:
                result = await tx.run(query, parameters or {})
                return await result.data()
    
    async def batch_execute(
        self,
        queries: List[Dict[str, Any]],
        batch_size: int = 100,
        database: str = None
    ) -> List[List[Dict[str, Any]]]:
        """Execute multiple queries in batches"""
        results = []
        
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[self.execute_query(q["query"], q.get("parameters"), database) for q in batch],
                return_exceptions=True
            )
            
            # Handle any exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(
                        "Batch query failed",
                        query_index=i + j,
                        error=str(result)
                    )
                    raise result
            
            results.extend(batch_results)
        
        return results
    
    async def find_node(self, label: str, properties: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a single node by properties"""
        where_clauses = [f"n.{key} = ${key}" for key in properties.keys()]
        query = f"""
        MATCH (n:{label})
        WHERE {' AND '.join(where_clauses)}
        RETURN n
        LIMIT 1
        """
        
        results = await self.execute_query(query, properties)
        return results[0]["n"] if results else None
    
    async def find_related_nodes(
        self,
        start_node_id: str,
        relationship_type: str = None,
        target_label: str = None,
        direction: str = "OUTGOING",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Find nodes related to a starting node"""
        rel_dir = "->" if direction == "OUTGOING" else "<-"
        rel_pattern = f"-[r:{relationship_type}]-" if relationship_type else "-[r]-"
        
        query = f"""
        MATCH (start)
        WHERE elementId(start) = $start_node_id
        MATCH (start){rel_pattern}{rel_dir}(target{':' + target_label if target_label else ''})
        RETURN target, type(r) as relationship_type
        LIMIT $limit
        """
        
        return await self.execute_query(query, {
            "start_node_id": start_node_id,
            "limit": limit
        })
    
    async def get_graph_metrics(self) -> Dict[str, Any]:
        """Get Neo4j graph metrics"""
        queries = {
            "node_counts": """
            MATCH (n)
            RETURN labels(n)[0] as label, count(*) as count
            ORDER BY count DESC
            """,
            "relationship_counts": """
            MATCH ()-[r]->()
            RETURN type(r) as type, count(*) as count
            ORDER BY count DESC
            """,
            "database_info": """
            CALL db.info() YIELD name, value
            RETURN name, value
            """
        }
        
        results = {}
        for name, query in queries.items():
            try:
                results[name] = await self.execute_query(query)
            except Exception as e:
                logger.warning(f"Failed to get metric: {name}", error=str(e))
                results[name] = []
        
        return results
    
    async def health_check(self) -> bool:
        """Check Neo4j connectivity and health"""
        try:
            if not self._initialized:
                await self.initialize()
            
            await self.driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error("Neo4j health check failed", error=str(e))
            return False
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics (nodes, relationships, etc.)"""
        try:
            # Get total node count
            node_query = "MATCH (n) RETURN count(n) as total_nodes"
            node_result = await self.execute_query(node_query)
            total_nodes = node_result[0]["total_nodes"] if node_result else 0
            
            # Get total relationship count
            rel_query = "MATCH ()-[r]->() RETURN count(r) as total_relationships"
            rel_result = await self.execute_query(rel_query)
            total_relationships = rel_result[0]["total_relationships"] if rel_result else 0
            
            # Get node counts by label
            label_query = """
            MATCH (n)
            RETURN labels(n)[0] as label, count(*) as count
            ORDER BY count DESC
            LIMIT 10
            """
            label_result = await self.execute_query(label_query)
            
            # Get relationship counts by type
            rel_type_query = """
            MATCH ()-[r]->()
            RETURN type(r) as type, count(*) as count
            ORDER BY count DESC
            LIMIT 10
            """
            rel_type_result = await self.execute_query(rel_type_query)
            
            return {
                "total_nodes": total_nodes,
                "total_relationships": total_relationships,
                "nodes_by_label": {r["label"]: r["count"] for r in label_result},
                "relationships_by_type": {r["type"]: r["count"] for r in rel_type_result}
            }
        except Exception as e:
            logger.error("Failed to get graph stats", error=str(e))
            return {
                "total_nodes": 0,
                "total_relationships": 0,
                "nodes_by_label": {},
                "relationships_by_type": {}
            }
    
    async def close(self):
        """Close the Neo4j driver"""
        if self.driver:
            await self.driver.close()
            self._initialized = False
            logger.info("Neo4j client closed")
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

# Singleton instance
neo4j_client = Neo4jClient()