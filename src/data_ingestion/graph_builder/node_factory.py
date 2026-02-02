"""
Enterprise Node Factory for Neo4j Knowledge Graph
Creates nodes for code artifacts: Routes, EventBus, DTOs, Components, Services
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib
import json

from src.database.neo4j_client import neo4j_client
from src.core.logger import get_logger

logger = get_logger(__name__)


class NodeFactory:
    """
    Factory for creating Neo4j nodes from parsed code artifacts.
    Supports Vert.x backend and React frontend patterns.
    """
    
    def __init__(self):
        self.stats = {
            "nodes_created": 0,
            "nodes_updated": 0,
            "errors": 0
        }
    
    def _generate_node_id(self, node_type: str, *identifiers) -> str:
        """Generate a unique, deterministic node ID"""
        combined = f"{node_type}:{'|'.join(str(i) for i in identifiers)}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    async def create_nodes(self, session, parsed_data: Dict[str, Any], 
                          file_path: str, repository: str = None,
                          repository_hash: str = None) -> Dict[str, Any]:
        """
        Create all nodes from parsed file data.
        
        Args:
            session: Neo4j async session
            parsed_data: Dictionary containing parsed code artifacts
            file_path: Path to the source file
            repository: Repository URL
            repository_hash: Repository hash identifier
            
        Returns:
            Statistics about created nodes
        """
        stats = {
            "created": 0,
            "updated": 0,
            "errors": [],
            "by_type": {}
        }
        
        try:
            # Create File node first
            file_node_id = self._generate_node_id("File", file_path)
            file_query = """
            MERGE (f:File {id: $id})
            SET f.path = $path,
                f.repository = $repository,
                f.repository_hash = $repository_hash,
                f.language = $language,
                f.last_updated = datetime()
            RETURN f
            """
            
            await session.run(file_query, {
                "id": file_node_id,
                "path": file_path,
                "repository": repository or "",
                "repository_hash": repository_hash or "",
                "language": parsed_data.get("language", self._detect_language(file_path))
            })
            stats["created"] += 1
            stats["by_type"]["File"] = 1
            
            # Create Vert.x Routes
            routes = parsed_data.get('routes', [])
            for route in routes:
                try:
                    route_result = await self._create_route_node(
                        session, route, file_path, file_node_id, repository_hash
                    )
                    stats["created"] += route_result.get("created", 0)
                    stats["by_type"]["VertxRoute"] = stats["by_type"].get("VertxRoute", 0) + 1
                except Exception as e:
                    stats["errors"].append(f"Route error: {str(e)}")
            
            # Create EventBus nodes
            eventbus_nodes = parsed_data.get('eventbus', [])
            for eb in eventbus_nodes:
                try:
                    eb_result = await self._create_eventbus_node(
                        session, eb, file_path, file_node_id, repository_hash
                    )
                    stats["created"] += eb_result.get("created", 0)
                    stats["by_type"]["EventBusNode"] = stats["by_type"].get("EventBusNode", 0) + 1
                except Exception as e:
                    stats["errors"].append(f"EventBus error: {str(e)}")
            
            # Create DTO nodes
            dtos = parsed_data.get('dtos', [])
            for dto in dtos:
                try:
                    dto_result = await self._create_dto_node(
                        session, dto, file_path, file_node_id, repository_hash
                    )
                    stats["created"] += dto_result.get("created", 0)
                    stats["by_type"]["DTO"] = stats["by_type"].get("DTO", 0) + 1
                except Exception as e:
                    stats["errors"].append(f"DTO error: {str(e)}")
            
            # Create Service nodes
            services = parsed_data.get('services', []) + parsed_data.get('configs', [])
            for service in services:
                try:
                    service_result = await self._create_service_node(
                        session, service, file_path, file_node_id, repository_hash
                    )
                    stats["created"] += service_result.get("created", 0)
                    stats["by_type"]["Service"] = stats["by_type"].get("Service", 0) + 1
                except Exception as e:
                    stats["errors"].append(f"Service error: {str(e)}")
            
            # Create React Component nodes
            components = parsed_data.get('components', [])
            for comp in components:
                try:
                    comp_result = await self._create_component_node(
                        session, comp, file_path, file_node_id, repository_hash
                    )
                    stats["created"] += comp_result.get("created", 0)
                    stats["by_type"]["ReactComponent"] = stats["by_type"].get("ReactComponent", 0) + 1
                except Exception as e:
                    stats["errors"].append(f"Component error: {str(e)}")
            
            # Create API Call nodes
            api_calls = parsed_data.get('api_calls', [])
            for api in api_calls:
                try:
                    api_result = await self._create_api_call_node(
                        session, api, file_path, file_node_id, repository_hash
                    )
                    stats["created"] += api_result.get("created", 0)
                    stats["by_type"]["ApiCall"] = stats["by_type"].get("ApiCall", 0) + 1
                except Exception as e:
                    stats["errors"].append(f"ApiCall error: {str(e)}")
            
            logger.info(
                f"Nodes created from {file_path}",
                total=stats["created"],
                by_type=stats["by_type"]
            )
            
        except Exception as e:
            logger.error(f"Node creation failed for {file_path}: {str(e)}")
            stats["errors"].append(str(e))
        
        return stats
    
    async def _create_route_node(self, session, route: Dict[str, Any], 
                                file_path: str, file_node_id: str,
                                repository_hash: str) -> Dict[str, Any]:
        """Create a Vert.x Route node"""
        # Handle different route data formats
        if hasattr(route, 'path'):  # Dataclass
            path = route.path
            method = route.method
            handler = route.handler_method
        elif isinstance(route, dict):
            path = route.get('path') or route.get('route_info', {}).get('path', '')
            method = route.get('method') or route.get('route_info', {}).get('method', 'GET')
            handler = route.get('handler_method') or route.get('name', '')
        else:
            return {"created": 0}
        
        node_id = self._generate_node_id("Route", path, method, file_path)
        
        query = """
        MERGE (r:VertxRoute {id: $id})
        SET r.path = $path,
            r.method = $method,
            r.handler_method = $handler_method,
            r.file = $file,
            r.repository_hash = $repository_hash,
            r.last_updated = datetime()
        WITH r
        MATCH (f:File {id: $file_id})
        MERGE (f)-[:CONTAINS]->(r)
        RETURN r
        """
        
        await session.run(query, {
            "id": node_id,
            "path": path,
            "method": method,
            "handler_method": handler,
            "file": file_path,
            "file_id": file_node_id,
            "repository_hash": repository_hash or ""
        })
        
        return {"created": 1}
    
    async def _create_eventbus_node(self, session, eb: Dict[str, Any],
                                   file_path: str, file_node_id: str,
                                   repository_hash: str) -> Dict[str, Any]:
        """Create an EventBus node"""
        # Handle different data formats
        if hasattr(eb, 'address'):  # Dataclass
            address = eb.address
            eb_type = eb.type
            handler = eb.handler_method
        elif isinstance(eb, dict):
            address = eb.get('address', '')
            eb_type = eb.get('type', 'unknown')
            handler = eb.get('handler_method') or eb.get('name', '')
        else:
            return {"created": 0}
        
        node_id = self._generate_node_id("EventBus", address, eb_type, file_path)
        
        query = """
        MERGE (e:EventBusNode {id: $id})
        SET e.address = $address,
            e.type = $type,
            e.handler_method = $handler_method,
            e.file = $file,
            e.repository_hash = $repository_hash,
            e.last_updated = datetime()
        WITH e
        MATCH (f:File {id: $file_id})
        MERGE (f)-[:CONTAINS]->(e)
        RETURN e
        """
        
        await session.run(query, {
            "id": node_id,
            "address": address,
            "type": eb_type,
            "handler_method": handler,
            "file": file_path,
            "file_id": file_node_id,
            "repository_hash": repository_hash or ""
        })
        
        return {"created": 1}
    
    async def _create_dto_node(self, session, dto: Dict[str, Any],
                              file_path: str, file_node_id: str,
                              repository_hash: str) -> Dict[str, Any]:
        """Create a DTO node"""
        # Handle different data formats
        if hasattr(dto, 'name'):  # Dataclass
            name = dto.name
            fields = dto.fields
            is_data_object = dto.is_data_object
        elif isinstance(dto, dict):
            name = dto.get('name', '')
            fields = dto.get('fields', [])
            is_data_object = dto.get('is_data_object', False)
        else:
            return {"created": 0}
        
        # Serialize fields
        fields_serialized = [
            f"{f.get('type', 'unknown')} {f.get('name', 'unknown')}" 
            if isinstance(f, dict) else str(f)
            for f in fields
        ]
        
        node_id = self._generate_node_id("DTO", name, file_path)
        
        query = """
        MERGE (d:DTO {id: $id})
        SET d.name = $name,
            d.fields = $fields,
            d.fields_json = $fields_json,
            d.is_data_object = $is_data_object,
            d.file = $file,
            d.repository_hash = $repository_hash,
            d.last_updated = datetime()
        WITH d
        MATCH (f:File {id: $file_id})
        MERGE (f)-[:CONTAINS]->(d)
        RETURN d
        """
        
        await session.run(query, {
            "id": node_id,
            "name": name,
            "fields": fields_serialized,
            "fields_json": json.dumps(fields),
            "is_data_object": is_data_object,
            "file": file_path,
            "file_id": file_node_id,
            "repository_hash": repository_hash or ""
        })
        
        return {"created": 1}
    
    async def _create_service_node(self, session, service: Dict[str, Any],
                                  file_path: str, file_node_id: str,
                                  repository_hash: str) -> Dict[str, Any]:
        """Create a Service/Config node"""
        name = service.get('name', '')
        service_type = service.get('type', 'Service')
        
        if 'Service' in name:
            service_type = 'Service'
        elif 'Config' in name:
            service_type = 'Config'
        elif 'Handler' in name:
            service_type = 'Handler'
        
        node_id = self._generate_node_id("Service", name, file_path)
        
        query = """
        MERGE (s:Service {id: $id})
        SET s.name = $name,
            s.type = $type,
            s.file = $file,
            s.repository_hash = $repository_hash,
            s.last_updated = datetime()
        WITH s
        MATCH (f:File {id: $file_id})
        MERGE (f)-[:CONTAINS]->(s)
        RETURN s
        """
        
        await session.run(query, {
            "id": node_id,
            "name": name,
            "type": service_type,
            "file": file_path,
            "file_id": file_node_id,
            "repository_hash": repository_hash or ""
        })
        
        return {"created": 1}
    
    async def _create_component_node(self, session, comp: Any,
                                    file_path: str, file_node_id: str,
                                    repository_hash: str) -> Dict[str, Any]:
        """Create a React Component node"""
        # Handle different data formats
        if hasattr(comp, 'name'):  # Dataclass/ComponentInfo
            name = comp.name
            props = comp.props if hasattr(comp, 'props') else []
            hooks = comp.hooks if hasattr(comp, 'hooks') else []
        elif isinstance(comp, dict):
            name = comp.get('name', '')
            props = comp.get('props', [])
            hooks = comp.get('hooks', [])
        else:
            return {"created": 0}
        
        # Serialize props
        props_serialized = [
            f"{p.get('name', 'unknown')}: {p.get('type', 'any')}" 
            if isinstance(p, dict) else str(p)
            for p in props
        ]
        
        node_id = self._generate_node_id("Component", name, file_path)
        
        query = """
        MERGE (c:ReactComponent {id: $id})
        SET c.name = $name,
            c.props = $props,
            c.hooks = $hooks,
            c.file = $file,
            c.repository_hash = $repository_hash,
            c.last_updated = datetime()
        WITH c
        MATCH (f:File {id: $file_id})
        MERGE (f)-[:CONTAINS]->(c)
        RETURN c
        """
        
        await session.run(query, {
            "id": node_id,
            "name": name,
            "props": props_serialized,
            "hooks": hooks if isinstance(hooks, list) else [],
            "file": file_path,
            "file_id": file_node_id,
            "repository_hash": repository_hash or ""
        })
        
        return {"created": 1}
    
    async def _create_api_call_node(self, session, api: Any,
                                   file_path: str, file_node_id: str,
                                   repository_hash: str) -> Dict[str, Any]:
        """Create an API Call node"""
        # Handle different data formats
        if hasattr(api, 'url'):  # Dataclass/ApiCall
            url = api.url
            method = api.method if hasattr(api, 'method') else 'GET'
            hook = api.hook if hasattr(api, 'hook') else ''
            component = api.component if hasattr(api, 'component') else ''
        elif isinstance(api, dict):
            url = api.get('url', '')
            method = api.get('method', 'GET')
            hook = api.get('hook', '')
            component = api.get('component', '')
        else:
            return {"created": 0}
        
        node_id = self._generate_node_id("ApiCall", url, method, file_path)
        
        query = """
        MERGE (a:ApiCall {id: $id})
        SET a.url = $url,
            a.method = $method,
            a.hook = $hook,
            a.component = $component,
            a.file = $file,
            a.repository_hash = $repository_hash,
            a.last_updated = datetime()
        WITH a
        MATCH (f:File {id: $file_id})
        MERGE (f)-[:CONTAINS]->(a)
        
        // Link to component if exists
        WITH a
        OPTIONAL MATCH (c:ReactComponent {name: $component})
        FOREACH (_ IN CASE WHEN c IS NOT NULL THEN [1] ELSE [] END |
            MERGE (c)-[:MAKES_API_CALL]->(a)
        )
        
        RETURN a
        """
        
        await session.run(query, {
            "id": node_id,
            "url": url,
            "method": method,
            "hook": hook,
            "component": component,
            "file": file_path,
            "file_id": file_node_id,
            "repository_hash": repository_hash or ""
        })
        
        return {"created": 1}
    
    def _detect_language(self, file_path: str) -> str:
        """Detect language from file extension"""
        ext = file_path.lower().split('.')[-1] if '.' in file_path else ''
        
        language_map = {
            'java': 'java',
            'kt': 'kotlin',
            'kts': 'kotlin',
            'ts': 'typescript',
            'tsx': 'typescript',
            'js': 'javascript',
            'jsx': 'javascript',
            'py': 'python',
            'json': 'json',
            'yaml': 'yaml',
            'yml': 'yaml',
            'xml': 'xml',
            'properties': 'properties',
            'sql': 'sql'
        }
        
        return language_map.get(ext, 'unknown')