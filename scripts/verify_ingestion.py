#!/usr/bin/env python3
"""
Script to test data ingestion and verify both Neo4j and Qdrant are populated
"""
import asyncio
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.database.neo4j_client import Neo4jClient
from src.database.qdrant_client import QdrantClient
from src.core.logger import get_logger

logger = get_logger(__name__)


async def count_neo4j_data():
    """Count nodes and relationships in Neo4j"""
    neo4j_client = Neo4jClient()
    
    try:
        await neo4j_client.initialize()
        
        # Count nodes
        node_result = await neo4j_client.execute_query(
            "MATCH (n) RETURN count(n) as total, "
            "countIf(n:Repository) as repositories, "
            "countIf(n:File) as files, "
            "countIf(n:Class) as classes, "
            "countIf(n:Method) as methods, "
            "countIf(n:Function) as functions"
        )
        
        # Count relationships
        rel_result = await neo4j_client.execute_query(
            "MATCH ()-[r]->() RETURN count(r) as total, "
            "countIf(type(r)='DEPENDS_ON') as depends_on, "
            "countIf(type(r)='CALLS') as calls, "
            "countIf(type(r)='CONTAINS') as contains"
        )
        
        if node_result and rel_result:
            nodes = node_result[0]
            rels = rel_result[0]
            
            logger.info("=" * 60)
            logger.info("NEO4J STATISTICS")
            logger.info("=" * 60)
            logger.info(f"Total Nodes: {nodes.get('total', 0)}")
            logger.info(f"  ├─ Repositories: {nodes.get('repositories', 0)}")
            logger.info(f"  ├─ Files: {nodes.get('files', 0)}")
            logger.info(f"  ├─ Classes: {nodes.get('classes', 0)}")
            logger.info(f"  ├─ Methods: {nodes.get('methods', 0)}")
            logger.info(f"  └─ Functions: {nodes.get('functions', 0)}")
            logger.info("")
            logger.info(f"Total Relationships: {rels.get('total', 0)}")
            logger.info(f"  ├─ DEPENDS_ON: {rels.get('depends_on', 0)}")
            logger.info(f"  ├─ CALLS: {rels.get('calls', 0)}")
            logger.info(f"  └─ CONTAINS: {rels.get('contains', 0)}")
            logger.info("=" * 60)
            
            return {
                "nodes": nodes.get('total', 0),
                "relationships": rels.get('total', 0),
                "status": "healthy" if nodes.get('total', 0) > 0 else "empty"
            }
        else:
            logger.warning("No results from Neo4j queries")
            return {"nodes": 0, "relationships": 0, "status": "error"}
            
    except Exception as e:
        logger.error(f"Failed to count Neo4j data: {str(e)}", exc_info=True)
        return {"nodes": 0, "relationships": 0, "status": "error"}
    finally:
        await neo4j_client.close()


async def count_qdrant_data():
    """Count vectors in Qdrant"""
    qdrant_client = QdrantClient()
    
    try:
        await qdrant_client.initialize()
        
        collection_name = settings.QDRANT_COLLECTION_NAME
        collection_info = await qdrant_client.client.get_collection(collection_name)
        
        point_count = collection_info.points_count
        vector_size = collection_info.config.params.vectors.size if hasattr(collection_info.config.params, 'vectors') else 0
        
        logger.info("=" * 60)
        logger.info("QDRANT STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Collection: {collection_name}")
        logger.info(f"Total Vectors: {point_count}")
        logger.info(f"Vector Dimension: {vector_size}")
        
        if point_count > 0:
            logger.info(f"Status: ✓ POPULATED ({point_count} embeddings)")
        else:
            logger.info(f"Status: ✗ EMPTY (no embeddings)")
        
        logger.info("=" * 60)
        
        return {
            "vectors": point_count,
            "dimension": vector_size,
            "status": "populated" if point_count > 0 else "empty"
        }
        
    except Exception as e:
        logger.error(f"Failed to count Qdrant data: {str(e)}", exc_info=True)
        return {"vectors": 0, "dimension": 0, "status": "error"}
    finally:
        if qdrant_client.client:
            await qdrant_client.client.close()


async def verify_ingestion():
    """Verify ingestion by checking both databases"""
    logger.info("Starting ingestion verification...")
    logger.info("")
    
    neo4j_stats = await count_neo4j_data()
    logger.info("")
    qdrant_stats = await count_qdrant_data()
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("INGESTION VERIFICATION SUMMARY")
    logger.info("=" * 60)
    
    if neo4j_stats['nodes'] > 0:
        logger.info(f"✓ Neo4j is POPULATED: {neo4j_stats['nodes']} nodes, {neo4j_stats['relationships']} relationships")
    else:
        logger.info(f"✗ Neo4j is EMPTY: No code structure data ingested")
    
    if qdrant_stats['vectors'] > 0:
        logger.info(f"✓ Qdrant is POPULATED: {qdrant_stats['vectors']} code embeddings")
    else:
        logger.info(f"✗ Qdrant is EMPTY: No code embeddings generated")
    
    logger.info("=" * 60)
    
    # Detailed report
    if neo4j_stats['nodes'] == 0 and qdrant_stats['vectors'] == 0:
        logger.error("✗ INGESTION FAILED: Neither Neo4j nor Qdrant received data")
        logger.error("   This suggests the ingestion pipeline was not triggered or failed")
        return False
    elif neo4j_stats['nodes'] > 0 and qdrant_stats['vectors'] == 0:
        logger.warning("⚠ PARTIAL INGESTION: Code structure in Neo4j, but no embeddings in Qdrant")
        logger.warning("   This is the bug we just fixed! Check if embeddings are being generated...")
        return False
    elif neo4j_stats['nodes'] > 0 and qdrant_stats['vectors'] > 0:
        logger.info("✓ FULL INGESTION SUCCESS: Both Neo4j and Qdrant populated!")
        logger.info("   The fix is working correctly!")
        return True
    else:
        logger.error("? UNEXPECTED STATE: Qdrant populated but Neo4j empty")
        return False


async def main():
    """Main verification routine"""
    success = await verify_ingestion()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
