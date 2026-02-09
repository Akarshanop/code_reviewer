#!/usr/bin/env python3
"""
Script to clear Neo4j and Qdrant databases for fresh ingestion testing
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.database.neo4j_client import Neo4jClient
from src.database.qdrant_client import QdrantClient
from src.core.logger import get_logger

logger = get_logger(__name__)


async def clear_neo4j():
    """Clear all nodes and relationships from Neo4j"""
    logger.info("=" * 60)
    logger.info("CLEARING NEO4J DATABASE")
    logger.info("=" * 60)
    
    neo4j_client = Neo4jClient()
    
    try:
        await neo4j_client.initialize()
        logger.info(f"✓ Connected to Neo4j: {settings.NEO4J_URI}")
        
        # Delete all relationships first
        logger.info("Deleting all relationships...")
        result = await neo4j_client.execute_query(
            "MATCH ()-[r]->() DELETE r RETURN count(r) as deleted"
        )
        rel_count = result[0].get("deleted", 0) if result else 0
        logger.info(f"✓ Deleted {rel_count} relationships")
        
        # Delete all nodes
        logger.info("Deleting all nodes...")
        result = await neo4j_client.execute_query(
            "MATCH (n) DELETE n RETURN count(n) as deleted"
        )
        node_count = result[0].get("deleted", 0) if result else 0
        logger.info(f"✓ Deleted {node_count} nodes")
        
        # Verify cleanup
        result = await neo4j_client.execute_query(
            "MATCH (n) RETURN count(n) as node_count"
        )
        remaining = result[0].get("node_count", 0) if result else 0
        
        if remaining == 0:
            logger.info("✓ Neo4j database is now EMPTY")
            return True
        else:
            logger.error(f"✗ Neo4j still has {remaining} nodes")
            return False
            
    except Exception as e:
        logger.error(f"✗ Failed to clear Neo4j: {str(e)}", exc_info=True)
        return False
    finally:
        await neo4j_client.close()


async def clear_qdrant():
    """Clear all vectors from Qdrant collection"""
    logger.info("=" * 60)
    logger.info("CLEARING QDRANT DATABASE")
    logger.info("=" * 60)
    
    qdrant_client = QdrantClient()
    
    try:
        await qdrant_client.initialize()
        logger.info(f"✓ Connected to Qdrant: {settings.QDRANT_URL}")
        
        # Delete the collection
        collection_name = settings.QDRANT_COLLECTION_NAME
        logger.info(f"Deleting collection '{collection_name}'...")
        
        try:
            await qdrant_client.client.delete_collection(collection_name)
            logger.info(f"✓ Deleted collection '{collection_name}'")
        except Exception as e:
            logger.warning(f"Collection may not exist or delete failed: {str(e)}")
        
        # Recreate the collection
        logger.info(f"Recreating collection '{collection_name}'...")
        await qdrant_client.ensure_collection()
        logger.info(f"✓ Recreated collection '{collection_name}'")
        
        # Verify it's empty
        collection_info = await qdrant_client.client.get_collection(collection_name)
        point_count = collection_info.points_count
        
        if point_count == 0:
            logger.info("✓ Qdrant collection is now EMPTY")
            return True
        else:
            logger.error(f"✗ Qdrant collection still has {point_count} points")
            return False
            
    except Exception as e:
        logger.error(f"✗ Failed to clear Qdrant: {str(e)}", exc_info=True)
        return False
    finally:
        if qdrant_client.client:
            await qdrant_client.client.close()


async def main():
    """Main cleanup routine"""
    logger.info("Starting database cleanup...")
    
    # Clear both databases
    neo4j_ok = await clear_neo4j()
    qdrant_ok = await clear_qdrant()
    
    # Summary
    logger.info("=" * 60)
    logger.info("CLEANUP SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Neo4j:  {'✓ CLEARED' if neo4j_ok else '✗ FAILED'}")
    logger.info(f"Qdrant: {'✓ CLEARED' if qdrant_ok else '✗ FAILED'}")
    logger.info("=" * 60)
    
    if neo4j_ok and qdrant_ok:
        logger.info("✓ All databases cleared successfully!")
        logger.info("You can now run ingestion tests.")
        return 0
    else:
        logger.error("✗ Some databases failed to clear")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
