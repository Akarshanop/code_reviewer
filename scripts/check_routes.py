#!/usr/bin/env python3
"""Script to check routes in the graph database."""
import asyncio
import sys
sys.path.insert(0, '/Users/akarshan.tyagi/Desktop/Code_reviewer/backend')

from src.database.neo4j_client import neo4j_client

async def check_routes():
    await neo4j_client.initialize()
    
    # Check VertxRoute nodes
    routes = await neo4j_client.execute_query('''
        MATCH (r:VertxRoute)
        RETURN r.method as method, r.path as path, r.handler as handler, r.repository_name as repo
        LIMIT 30
    ''')
    print('=== VERTX ROUTES ===')
    for r in routes:
        print(f'{r["method"]} {r["path"]} -> {r["handler"]} ({r["repo"]})')
    
    # Check if we have code stored in VertxRoute
    route_with_code = await neo4j_client.execute_query('''
        MATCH (r:VertxRoute)
        RETURN keys(r) as keys
        LIMIT 1
    ''')
    print('\n=== VERTX ROUTE PROPERTIES ===')
    for r in route_with_code:
        print(r["keys"])
    
    # Check Functions in route files
    funcs = await neo4j_client.execute_query('''
        MATCH (f:Function)
        WHERE f.file_path CONTAINS 'route'
        RETURN f.name as name, f.file_path as file
        LIMIT 20
    ''')
    print('\n=== FUNCTIONS IN ROUTE FILES ===')
    for f in funcs:
        print(f'{f["name"]} in {f["file"]}')
    
    # Check what's in Qdrant for routes
    print('\n=== CHECKING QDRANT FOR ROUTE CONTENT ===')
    from src.database.qdrant_client import qdrant_client
    await qdrant_client.initialize()
    
    # Search for router.get in vectors
    from src.services.embedding_service import EmbeddingService
    embedding_service = EmbeddingService()
    
    query_embedding = await embedding_service.generate_embeddings("router.get router.post Express.js routes API endpoints")
    
    results = await qdrant_client.search_similar(
        query_vector=query_embedding,
        limit=5,
        score_threshold=0.3
    )
    
    print(f'Found {len(results)} results for route search')
    for r in results:
        print(f'\nScore: {r["score"]:.3f}')
        print(f'File: {r["metadata"].get("file_path", "N/A")}')
        print(f'Type: {r["metadata"].get("type", "N/A")}')
        code = r["metadata"].get("code", "")[:300] if r["metadata"].get("code") else "No code"
        print(f'Code preview: {code}')

if __name__ == "__main__":
    asyncio.run(check_routes())
