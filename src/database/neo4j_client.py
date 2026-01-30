import logging
from neo4j import AsyncGraphDatabase, WRITE_ACCESS
from fastapi import Depends
from config.settings import settings

logger = logging.getLogger(__name__)

async def get_neo4j_driver():
    driver = AsyncGraphDatabase.driver(
        settings.NEO4J_URI,   # neo4j+s://b017a44d.databases.neo4j.io
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        max_connection_lifetime=60 * 30,
        connection_timeout=30,
        keep_alive=True,
    )

    try:
        await driver.verify_connectivity()
        logger.info("Neo4j connectivity verified.")
        yield driver
    finally:
        await driver.close()
        logger.info("Neo4j driver closed.")


async def test_neo4j_connection(driver=Depends(get_neo4j_driver)):
    async with driver.session(
        database=settings.NEO4J_DATABASE,
        default_access_mode=WRITE_ACCESS
    ) as session:
        logger.info("Testing Neo4j connection...")
        result = await session.run("RETURN 1 AS ok")
        record = await result.single()
        return {"neo4j": record["ok"]}
