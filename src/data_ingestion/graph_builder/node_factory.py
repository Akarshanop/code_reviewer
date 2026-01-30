import logging
from src.database.neo4j_client import get_neo4j_driver

logger = logging.getLogger(__name__)

class NodeFactory:
    async def create_nodes(self, driver, parsed_data: dict, file_path: str):
        async with driver.session() as session:
            try:
                # Routes
                for route in parsed_data.get('routes', []):
                    await session.run(
                        """
                        MERGE (n:VertxRoute {
                            path: $path,
                            method: $method,
                            handler_method: $handler_method,
                            file: $file
                        })
                        """,
                        path=route.path,
                        method=route.method,
                        handler_method=route.handler_method,
                        file=file_path
                    )

                # EventBus
                for eb in parsed_data.get('eventbus', []):
                    await session.run(
                        """
                        MERGE (n:EventBusNode {
                            address: $address,
                            type: $type,
                            handler_method: $handler_method,
                            file: $file
                        })
                        """,
                        address=eb.address,
                        type=eb.type,
                        handler_method=eb.handler_method,
                        file=file_path
                    )

                for dto in parsed_data.get('dtos', []):
                    await session.run(
                        """
                        MERGE (n:DTO {
                            name: $name,
                            fields: $fields,
                            is_data_object: $is_data_object,
                            file: $file
                        })
                        """,
                        name=dto.name,
                        fields=[f'{f["type"]} {f["name"]}' for f in dto.fields],  # Serialize fields
                        is_data_object=dto.is_data_object,
                        file=file_path
                    )

                for service in parsed_data.get('services', []) + parsed_data.get('configs', []):
                    await session.run(
                        """
                        MERGE (n:Service {
                            name: $name,
                            type: $type,
                            file: $file
                        })
                        """,
                        name=service['name'],
                        type='Service' if 'Service' in service['name'] else 'Config',
                        file=file_path
                    )

                for comp in parsed_data.get('components', []):
                    await session.run(
                        """
                        MERGE (n:ReactComponent {
                            name: $name,
                            props: $props,
                            hooks: $hooks,
                            file: $file
                        })
                        """,
                        name=comp.name,
                        props=[f'{p["name"]}: {p["type"]}' for p in comp.props],
                        hooks=comp.hooks,
                        file=file_path
                    )

                for api in parsed_data.get('api_calls', []):
                    await session.run(
                        """
                        MERGE (n:ApiCall {
                            url: $url,
                            method: $method,
                            hook: $hook,
                            component: $component,
                            file: $file
                        })
                        """,
                        url=api.url,
                        method=api.method,
                        hook=api.hook,
                        component=api.component,
                        file=file_path
                    )

                vector_service = VectorService()
                for route in parsed_data.get('routes', []):
                    await vector_service.upsert_vector(route.path, {'type': 'route', 'file': file_path})
                for eb in parsed_data.get('eventbus', []):
                    await vector_service.upsert_vector(eb.address, {'type': 'eventbus', 'file': file_path})
                for dto in parsed_data.get('dtos', []):
                    await vector_service.upsert_vector(dto.name, {'type': 'dto', 'file': file_path})

                logger.info(f"Created nodes from {file_path}")
            except Exception as e:
                logger.error(f"Node creation failed for {file_path}: {str(e)}")
                raise