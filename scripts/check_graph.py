"""Check the current state of the Neo4j graph"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.neo4j_client import neo4j_client


async def check():
    await neo4j_client.initialize()
    
    # Check Repository -> File -> Route chain
    print('=== Repository -> Files (BELONGS_TO) ===')
    result = await neo4j_client.execute_query(
        'MATCH (repo:Repository)<-[:BELONGS_TO]-(f:File) RETURN repo.name as repo, f.path as file LIMIT 5'
    )
    for r in result:
        print(f'  {r["repo"]} <- {r["file"]}')
    
    print('\n=== Files -> Routes (EXPOSES) ===')
    result = await neo4j_client.execute_query(
        'MATCH (f:File)-[:EXPOSES]->(r:Route) RETURN f.path as file, r.path as route, r.method as method LIMIT 5'
    )
    for r in result:
        print(f'  {r["file"]} -> {r["method"]} {r["route"]}')
    
    print('\n=== Files -> Functions (CONTAINS) ===')
    result = await neo4j_client.execute_query(
        'MATCH (f:File)-[:CONTAINS]->(func:Function) RETURN f.path as file, func.name as function LIMIT 5'
    )
    for r in result:
        print(f'  {r["file"]} -> {r["function"]}')
    
    print('\n=== File Imports ===')
    result = await neo4j_client.execute_query(
        'MATCH (f1:File)-[i:IMPORTS]->(f2:File) RETURN f1.path as from_file, f2.path as to_file LIMIT 5'
    )
    for r in result:
        print(f'  {r["from_file"]} imports {r["to_file"]}')
    
    # Full path from repo to function
    print('\n=== Full Path: Repo -> File -> Function ===')
    result = await neo4j_client.execute_query(
        'MATCH (repo:Repository)<-[:BELONGS_TO]-(f:File)-[:CONTAINS]->(func:Function) RETURN repo.name as repo, f.path as file, func.name as function LIMIT 5'
    )
    for r in result:
        print(f'  {r["repo"]} / {r["file"]} / {r["function"]}')
    
    # Total counts  
    print('\n=== Summary ===')
    total_nodes = await neo4j_client.execute_query('MATCH (n) RETURN count(n) as count')
    total_rels = await neo4j_client.execute_query('MATCH ()-[r]->() RETURN count(r) as count')
    print(f'Total nodes: {total_nodes[0]["count"]}')
    print(f'Total relationships: {total_rels[0]["count"]}')
    
    # By type
    print('\n=== Nodes by Type ===')
    result = await neo4j_client.execute_query('MATCH (n) RETURN labels(n)[0] as type, count(*) as count ORDER BY count DESC')
    for r in result:
        print(f'  {r["type"]}: {r["count"]}')
    
    print('\n=== Relationships by Type ===')
    result = await neo4j_client.execute_query('MATCH ()-[r]->() RETURN type(r) as type, count(*) as count ORDER BY count DESC')
    for r in result:
        print(f'  {r["type"]}: {r["count"]}')


if __name__ == "__main__":
    asyncio.run(check())
