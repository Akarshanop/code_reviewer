import logging
from src.database.neo4j_client import get_neo4j_driver

logger = logging.getLogger(__name__)

class EdgeLinker:
    async def link_edges(self, driver, backend_parsed: dict, frontend_parsed: dict):
        async with driver.session() as session:
            # Link React API call -> Vert.x Route
            for api_call in frontend_parsed.get('api_calls', []):
                await session.run(
                    """
                    MATCH (f:ReactComponent {uses_api: $url})
                    MATCH (b:VertxRoute {path: $url})
                    MERGE (f)-[:CALLS_API]->(b)
                    """,
                    url=api_call['url']
                )
            # Add EventBus producer-consumer links, DTO usages
        logger.info("Linked cross-language edges")