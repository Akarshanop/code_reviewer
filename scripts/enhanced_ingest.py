"""
Re-ingest repositories with enhanced relationship linking.

This script:
1. Clears existing data
2. Ingests both repos with fixed function extraction
3. Creates cross-repo links (frontend -> backend API calls)
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.neo4j_client import neo4j_client
from src.database.qdrant_client import qdrant_client
from src.data_ingestion.smart_graph_builder import SmartGraphBuilder
from src.services.embedding_service import embedding_service
from src.core.logger import get_logger

logger = get_logger(__name__)

REPOS = {
    "backend": {
        "path": "data/repos/intergration-studio-be_af864019b1b6",
        "name": "intergration-studio-be_af864019b1b6"
    },
    "frontend": {
        "path": "data/repos/ui-integration-studio_4fb665ac5745", 
        "name": "ui-integration-studio_4fb665ac5745"
    }
}


async def clear_databases():
    """Clear Neo4j and Qdrant databases"""
    print("\n=== Clearing Databases ===")
    
    # Clear Neo4j
    await neo4j_client.initialize()
    async with neo4j_client.session() as session:
        result = await session.run("MATCH (n) DETACH DELETE n")
        await result.consume()
    print("✓ Neo4j cleared")
    
    # Clear Qdrant
    await qdrant_client.initialize()
    collections = await qdrant_client.list_collections()
    if qdrant_client._collection_name in collections:
        await qdrant_client.client.delete_collection(qdrant_client._collection_name)
    await qdrant_client.ensure_collection()
    print("✓ Qdrant cleared")


async def ingest_repo(repo_info: dict) -> dict:
    """Ingest a single repository"""
    base_path = Path(__file__).parent.parent
    repo_path = base_path / repo_info["path"]
    
    print(f"\n=== Ingesting: {repo_info['name']} ===")
    print(f"Path: {repo_path}")
    
    builder = SmartGraphBuilder(str(repo_path), repo_info["name"])
    stats = await builder.build_graph()
    
    print(f"  Files processed: {stats['files_processed']}")
    print(f"  Nodes created: {stats['nodes_created']}")
    print(f"  Edges created: {stats['edges_created']}")
    print(f"  Routes found: {stats['routes_found']}")
    print(f"  Functions found: {stats['functions_found']}")
    
    return stats


async def create_cross_repo_links():
    """Create links between frontend and backend repos"""
    print("\n=== Creating Cross-Repository Links ===")
    
    query = """
    // First, find all backend routes
    MATCH (backendRoute:Route)
    WHERE backendRoute.repository CONTAINS 'intergration-studio-be'
    WITH backendRoute.path as routePath, backendRoute
    
    // Then find frontend files that might call these routes
    MATCH (frontendFile:File)
    WHERE frontendFile.repository CONTAINS 'ui-integration-studio'
    AND (frontendFile.path CONTAINS 'api' OR frontendFile.path CONTAINS 'service' 
         OR frontendFile.path CONTAINS 'hook' OR frontendFile.path CONTAINS 'Action')
    
    // Create link if route path appears in file
    WITH frontendFile, backendRoute, routePath
    WHERE frontendFile.imports_json CONTAINS routePath
       OR frontendFile.path CONTAINS 'Api'
    
    MERGE (frontendFile)-[r:CALLS_API]->(backendRoute)
    RETURN count(r) as links_created
    """
    
    try:
        result = await neo4j_client.execute_query(query)
        print(f"  Cross-repo links created: {result[0]['links_created'] if result else 0}")
    except Exception as e:
        logger.warning(f"Cross-repo linking failed: {e}")


async def link_routes_to_controllers():
    """Create HANDLED_BY relationships from routes to their handler functions"""
    print("\n=== Linking Routes to Controllers ===")
    
    # Get routes with handlers
    routes_query = """
    MATCH (r:Route)
    WHERE r.handler IS NOT NULL AND r.handler <> ''
    RETURN r.id as route_id, r.handler as handler, r.path as path, r.repository as repo
    """
    routes = await neo4j_client.execute_query(routes_query)
    
    links_created = 0
    for route in routes:
        handler = route['handler']
        repo = route['repo']
        
        # Find function with this name in the same repo
        func_query = """
        MATCH (f:Function)
        WHERE f.name = $handler AND f.repository = $repo
        RETURN f.id as func_id
        """
        funcs = await neo4j_client.execute_query(func_query, {'handler': handler, 'repo': repo})
        
        if funcs:
            # Create HANDLED_BY relationship
            link_query = """
            MATCH (r {id: $route_id})
            MATCH (f {id: $func_id})
            MERGE (r)-[:HANDLED_BY]->(f)
            RETURN count(*) as created
            """
            await neo4j_client.execute_query(link_query, {
                'route_id': route['route_id'],
                'func_id': funcs[0]['func_id']
            })
            links_created += 1
    
    print(f"  Route->Handler links created: {links_created}")


async def generate_embeddings():
    """Generate embeddings for key code elements"""
    print("\n=== Generating Embeddings ===")
    
    await embedding_service.initialize()
    
    # Get all functions and routes for embedding
    query = """
    MATCH (n)
    WHERE n:Function OR n:Route OR n:Class
    RETURN n.id as id, n.name as name, labels(n) as labels,
           n.file_path as file_path, n.repository as repo,
           COALESCE(n.description, '') as description,
           COALESCE(n.path, '') as path,
           COALESCE(n.method, '') as method
    LIMIT 500
    """
    
    nodes = await neo4j_client.execute_query(query)
    print(f"  Found {len(nodes)} nodes to embed")
    
    if not nodes:
        return
    
    # Prepare texts for embedding
    from qdrant_client import models
    points = []
    
    for i, node in enumerate(nodes):
        # Create meaningful text for embedding
        labels = node['labels']
        if 'Route' in labels:
            text = f"API Route: {node['method']} {node['path']} in {node['file_path']}"
        elif 'Function' in labels:
            text = f"Function: {node['name']} in {node['file_path']}. {node['description']}"
        else:
            text = f"Class: {node['name']} in {node['file_path']}"
        
        try:
            embedding = await embedding_service.generate_embedding(text)
            
            points.append(models.PointStruct(
                id=i,
                vector=embedding,
                payload={
                    'node_id': node['id'],
                    'name': node['name'],
                    'type': labels[0] if labels else 'unknown',
                    'file_path': node['file_path'],
                    'repository': node['repo'],
                    'text': text
                }
            ))
        except Exception as e:
            logger.warning(f"Failed to embed node {node['id']}: {e}")
    
    # Upsert to Qdrant
    if points:
        await qdrant_client.upsert_vectors(points)
        print(f"  Embeddings stored: {len(points)}")


async def print_final_stats():
    """Print final graph statistics"""
    print("\n=== Final Graph Statistics ===")
    
    # Node counts
    result = await neo4j_client.execute_query(
        "MATCH (n) RETURN labels(n)[0] as type, count(*) as count ORDER BY count DESC"
    )
    print("\nNodes by type:")
    for r in result:
        print(f"  {r['type']}: {r['count']}")
    
    # Relationship counts
    result = await neo4j_client.execute_query(
        "MATCH ()-[r]->() RETURN type(r) as type, count(*) as count ORDER BY count DESC"
    )
    print("\nRelationships by type:")
    for r in result:
        print(f"  {r['type']}: {r['count']}")
    
    # Total counts
    nodes = await neo4j_client.execute_query("MATCH (n) RETURN count(n) as count")
    rels = await neo4j_client.execute_query("MATCH ()-[r]->() RETURN count(r) as count")
    
    print(f"\n✓ Total nodes: {nodes[0]['count']}")
    print(f"✓ Total relationships: {rels[0]['count']}")


async def main():
    """Main ingestion flow"""
    print("=" * 60)
    print("Enhanced Repository Ingestion")
    print("=" * 60)
    
    try:
        # Step 1: Clear databases
        await clear_databases()
        
        # Step 2: Ingest backend repo
        await ingest_repo(REPOS["backend"])
        
        # Step 3: Ingest frontend repo
        await ingest_repo(REPOS["frontend"])
        
        # Step 4: Link routes to controllers
        await link_routes_to_controllers()
        
        # Step 5: Create cross-repo links
        await create_cross_repo_links()
        
        # Step 6: Generate embeddings
        await generate_embeddings()
        
        # Step 7: Print final stats
        await print_final_stats()
        
        print("\n✅ Ingestion complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        await neo4j_client.driver.close() if neo4j_client.driver else None
        await qdrant_client.close() if qdrant_client.client else None


if __name__ == "__main__":
    asyncio.run(main())
