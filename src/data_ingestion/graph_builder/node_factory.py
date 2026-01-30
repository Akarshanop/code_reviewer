from src.database.neo4j_client import get_neo4j_driver
from src.services.vector_service import VectorService
import logging

logger = logging.getLogger(__name__)

class NodeFactory:
    async def create_nodes(self, driver, parsed_data: dict, file_path: str):
        async with driver.session() as session:
            # Example: Create Route node
            for route in parsed_data.get('routes', []):
                await session.run(
                    "MERGE (n:VertxRoute {path: $path, file: $file})",
                    path=route['path'], file=file_path
                )
            vector_service = VectorService()
            await vector_service.upsert_vector(route['path'], {'type': 'route'})
        logger.info(f"Created nodes from {file_path}")