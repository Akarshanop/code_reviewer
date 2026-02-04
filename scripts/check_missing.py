"""Check what's missing in the graph"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.neo4j_client import neo4j_client


async def check():
    await neo4j_client.initialize()
    
    # Check for API calls relationships
    result = await neo4j_client.execute_query(
        """MATCH ()-[r]->() 
        WHERE type(r) IN ['CALLS_API', 'CALLS', 'CALLS_ROUTE', 'USES_API', 'HANDLED_BY'] 
        RETURN type(r) as type, count(*) as count"""
    )
    print('=== API Call & Handler Relationships ===')
    if result:
        for r in result:
            print(f'  {r["type"]}: {r["count"]}')
    else:
        print('  NONE FOUND - need to add these!')
    
    # Check routes for handlers
    result = await neo4j_client.execute_query(
        """MATCH (r:Route) 
        WHERE r.handler IS NOT NULL AND r.handler <> ''
        RETURN r.path as path, r.method as method, r.handler as handler LIMIT 10"""
    )
    print('\n=== Routes with handlers (should link to functions) ===')
    for r in result:
        print(f'  {r["method"]} {r["path"]} -> {r["handler"]}')
    
    # Check controllers
    result = await neo4j_client.execute_query(
        """MATCH (f:File)-[:CONTAINS]->(func:Function)
        WHERE f.path CONTAINS 'controllers'
        RETURN f.path as file, func.name as function LIMIT 10"""
    )
    print('\n=== Controller Functions ===')
    for r in result:
        print(f'  {r["file"]} -> {r["function"]}')
    
    # Check if backend repo has functions
    result = await neo4j_client.execute_query(
        """MATCH (f:File)-[:CONTAINS]->(func:Function)
        WHERE f.repository CONTAINS 'intergration-studio-be'
        RETURN f.path as file, func.name as function LIMIT 10"""
    )
    print('\n=== Backend Functions ===')
    if result:
        for r in result:
            print(f'  {r["file"]} -> {r["function"]}')
    else:
        print('  NONE - backend functions not detected!')


if __name__ == "__main__":
    asyncio.run(check())
