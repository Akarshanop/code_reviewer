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
                          repository_hash: str = None,
                          repo_type: str = None,
                          repository_name: str = None,
                          entry_file: str = None) -> Dict[str, Any]:
        """
        Create all nodes from parsed file data.
        
        Args:
            session: Neo4j async session
            parsed_data: Dictionary containing parsed code artifacts
            file_path: Path to the source file
            repository: Repository URL
            repository_hash: Repository hash identifier
            repo_type: Type of repository ('backend' or 'frontend')
            repository_name: Name of the repository
            entry_file: Entry point file path for the repository
            
        Returns:
            Statistics about created nodes
        """
        # Extract required metadata
        file_name = file_path.split('/')[-1] if file_path else ''
        relative_path = file_path  # Will be set by caller with proper relative path
        stats = {
            "created": 0,
            "updated": 0,
            "errors": [],
            "by_type": {}
        }
        
        try:
            # Create File node first with REQUIRED metadata
            file_node_id = self._generate_node_id("File", file_path)
            
            # Detect if this is an entry file
            is_entry_file = self._is_entry_file(file_path, repo_type)
            file_labels = "File:EntryFile" if is_entry_file else "File"
            
            file_query = f"""
            MERGE (f:{file_labels} {{id: $id}})
            SET f.path = $path,
                f.repository = $repository,
                f.repository_hash = $repository_hash,
                f.language = $language,
                f.imports_json = $imports_json,
                f.repository_name = $repository_name,
                f.file_name = $file_name,
                f.relative_path = $relative_path,
                f.entry_file = $entry_file,
                f.repo_type = $repo_type,
                f.is_entry_file = $is_entry_file,
                f.last_updated = datetime()
            RETURN f
            """
            
            await session.run(file_query, {
                "id": file_node_id,
                "path": file_path,
                "repository": repository or "",
                "repository_hash": repository_hash or "",
                "language": parsed_data.get("language", self._detect_language(file_path)),
                "imports_json": json.dumps(parsed_data.get("imports", [])) if parsed_data.get("imports") else "[]",
                "repository_name": repository_name or "",
                "file_name": file_name,
                "relative_path": relative_path,
                "entry_file": entry_file or "",
                "repo_type": repo_type or "",
                "is_entry_file": is_entry_file,
            })
            stats["created"] += 1
            stats["by_type"]["File"] = 1
            
            # Create Vert.x Routes
            routes = parsed_data.get('routes', [])
            for route in routes:
                try:
                    route_result = await self._create_route_node(
                        session, route, file_path, file_node_id, repository_hash,
                        repo_type=repo_type, repository_name=repository_name,
                        entry_file=entry_file, file_name=file_name, relative_path=relative_path
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
                        session, eb, file_path, file_node_id, repository_hash,
                        repo_type=repo_type, repository_name=repository_name,
                        entry_file=entry_file, file_name=file_name, relative_path=relative_path
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
                        session, dto, file_path, file_node_id, repository_hash,
                        repo_type=repo_type, repository_name=repository_name,
                        entry_file=entry_file, file_name=file_name, relative_path=relative_path
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
                        session, service, file_path, file_node_id, repository_hash,
                        repo_type=repo_type, repository_name=repository_name,
                        entry_file=entry_file, file_name=file_name, relative_path=relative_path
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
                        session, comp, file_path, file_node_id, repository_hash,
                        repo_type=repo_type, repository_name=repository_name,
                        entry_file=entry_file, file_name=file_name, relative_path=relative_path
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
                        session, api, file_path, file_node_id, repository_hash,
                        repo_type=repo_type, repository_name=repository_name,
                        entry_file=entry_file, file_name=file_name, relative_path=relative_path
                    )
                    stats["created"] += api_result.get("created", 0)
                    stats["by_type"]["ApiCall"] = stats["by_type"].get("ApiCall", 0) + 1
                except Exception as e:
                    stats["errors"].append(f"ApiCall error: {str(e)}")

            # Create Function nodes (JS/TS) + same-file call graph edges
            functions = parsed_data.get('functions', [])
            function_id_by_name: Dict[str, str] = {}
            for fn in functions:
                try:
                    fn_result = await self._create_function_node(
                        session, fn, file_path, file_node_id, repository_hash,
                        repo_type=repo_type, repository_name=repository_name,
                        entry_file=entry_file, file_name=file_name, relative_path=relative_path
                    )
                    if fn_result.get("created"):
                        stats["created"] += 1
                        stats["by_type"]["Function"] = stats["by_type"].get("Function", 0) + 1
                    if fn_result.get("id") and fn_result.get("name"):
                        function_id_by_name[fn_result["name"]] = fn_result["id"]
                except Exception as e:
                    stats["errors"].append(f"Function error: {str(e)}")

            try:
                pairs: list[dict[str, str]] = []
                for fn in functions:
                    if not isinstance(fn, dict):
                        continue
                    caller_name = (fn.get("name") or "").strip()
                    caller_id = function_id_by_name.get(caller_name)
                    if not caller_id:
                        continue
                    for callee_name in (fn.get("calls") or []):
                        callee_id = function_id_by_name.get(callee_name)
                        if callee_id and callee_id != caller_id:
                            pairs.append({"caller": caller_id, "callee": callee_id})

                if pairs:
                    await session.run(
                        """
                        UNWIND $pairs AS p
                        MATCH (caller:Function {id: p.caller})
                        MATCH (callee:Function {id: p.callee})
                        MERGE (caller)-[r:CALLS]->(callee)
                        SET r.created_at = coalesce(r.created_at, datetime()),
                            r.last_updated = datetime(),
                            r.file = $file
                        """,
                        {"pairs": pairs, "file": file_path},
                    )
            except Exception as e:
                stats["errors"].append(f"Function CALLS edge error: {str(e)}")
            
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
                                repository_hash: str,
                                repo_type: str = None,
                                repository_name: str = None,
                                entry_file: str = None,
                                file_name: str = None,
                                relative_path: str = None) -> Dict[str, Any]:
        """Create a Vert.x Route node with REQUIRED metadata"""
        # Handle different route data formats
        if hasattr(route, 'path'):  # Dataclass
            path = route.path
            method = route.method
            handler = route.handler_method
        elif isinstance(route, dict):
            path = route.get('path') or route.get('route_info', {}).get('path', '')
            method = route.get('method') or route.get('route_info', {}).get('method', 'GET')
            handler = route.get('handler_method') or route.get('name', '')

            if not handler and isinstance(route.get('handlers'), list) and route['handlers']:
                first = route['handlers'][0]
                if isinstance(first, dict) and first.get('type') == 'reference' and first.get('name'):
                    handler = first.get('name')
                elif isinstance(first, dict) and first.get('type') == 'inline':
                    handler = 'inline_handler'

            mount_target = route.get('mount_target')
            mount_target_kind = route.get('mount_target_kind')
            mount_target_files = route.get('mount_target_files') or []
        else:
            return {"created": 0}
        
        node_id = self._generate_node_id("Route", path, method, file_path)
        
        query = """
        MERGE (r:VertxRoute {id: $id})
        SET r.path = $path,
            r.method = $method,
            r.handler_method = $handler_method,
            r.file = $file,
            r.file_path = $file,
            r.mount_target = $mount_target,
            r.mount_target_kind = $mount_target_kind,
            r.mount_target_files = $mount_target_files,
            r.repository_hash = $repository_hash,
            r.repository_name = $repository_name,
            r.file_name = $file_name,
            r.relative_path = $relative_path,
            r.entry_file = $entry_file,
            r.repo_type = $repo_type,
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
            "mount_target": mount_target if isinstance(route, dict) else None,
            "mount_target_kind": mount_target_kind if isinstance(route, dict) else None,
            "mount_target_files": mount_target_files if isinstance(route, dict) else [],
            "repository_hash": repository_hash or "",
            "repository_name": repository_name or "",
            "file_name": file_name or "",
            "relative_path": relative_path or "",
            "entry_file": entry_file or "",
            "repo_type": repo_type or ""
        })
        
        return {"created": 1}
    
    async def _create_eventbus_node(self, session, eb: Dict[str, Any],
                                   file_path: str, file_node_id: str,
                                   repository_hash: str,
                                   repo_type: str = None,
                                   repository_name: str = None,
                                   entry_file: str = None,
                                   file_name: str = None,
                                   relative_path: str = None) -> Dict[str, Any]:
        """Create an EventBus node with REQUIRED metadata"""
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
            e.repository_name = $repository_name,
            e.file_name = $file_name,
            e.relative_path = $relative_path,
            e.entry_file = $entry_file,
            e.repo_type = $repo_type,
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
            "repository_hash": repository_hash or "",
            "repository_name": repository_name or "",
            "file_name": file_name or "",
            "relative_path": relative_path or "",
            "entry_file": entry_file or "",
            "repo_type": repo_type or ""
        })
        
        return {"created": 1}

    async def _create_function_node(
        self,
        session,
        fn: Dict[str, Any],
        file_path: str,
        file_node_id: str,
        repository_hash: str,
        repo_type: str = None,
        repository_name: str = None,
        entry_file: str = None,
        file_name: str = None,
        relative_path: str = None,
    ) -> Dict[str, Any]:
        """Create a Function node with REQUIRED metadata. For backend, add HandlerFunction label if route handler."""
        if not isinstance(fn, dict):
            return {"created": 0}

        name = (fn.get("name") or "").strip()
        if not name:
            return {"created": 0}

        kind = fn.get("type") or "function"
        is_async = bool(fn.get("is_async"))
        start_line = fn.get("start_line")
        end_line = fn.get("end_line")
        
        # Check if this is a handler function for backend repos
        is_handler = fn.get("is_handler", False) or fn.get("is_route_handler", False)
        labels = "Function:HandlerFunction" if (repo_type == "backend" and is_handler) else "Function"

        node_id = self._generate_node_id("Function", repository_hash or "", file_path, name)

        query = f"""
        MERGE (fn:{labels} {{id: $id}})
        SET fn.name = $name,
            fn.kind = $kind,
            fn.is_async = $is_async,
            fn.start_line = $start_line,
            fn.end_line = $end_line,
            fn.file = $file,
            fn.file_path = $file,
            fn.repository_hash = $repository_hash,
            fn.repository_name = $repository_name,
            fn.file_name = $file_name,
            fn.relative_path = $relative_path,
            fn.entry_file = $entry_file,
            fn.repo_type = $repo_type,
            fn.is_handler = $is_handler,
            fn.last_updated = datetime()
        WITH fn
        MATCH (f:File {{id: $file_id}})
        MERGE (f)-[:CONTAINS]->(fn)
        RETURN fn
        """

        await session.run(
            query,
            {
                "id": node_id,
                "name": name,
                "kind": kind,
                "is_async": is_async,
                "start_line": start_line,
                "end_line": end_line,
                "file": file_path,
                "file_id": file_node_id,
                "repository_hash": repository_hash or "",
                "repository_name": repository_name or "",
                "file_name": file_name or "",
                "relative_path": relative_path or "",
                "entry_file": entry_file or "",
                "repo_type": repo_type or "",
                "is_handler": is_handler,
            },
        )

        return {"created": 1, "id": node_id, "name": name}
    
    async def _create_dto_node(self, session, dto: Dict[str, Any],
                              file_path: str, file_node_id: str,
                              repository_hash: str,
                              repo_type: str = None,
                              repository_name: str = None,
                              entry_file: str = None,
                              file_name: str = None,
                              relative_path: str = None) -> Dict[str, Any]:
        """Create a DTO node with REQUIRED metadata"""
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
            d.repository_name = $repository_name,
            d.file_name = $file_name,
            d.relative_path = $relative_path,
            d.entry_file = $entry_file,
            d.repo_type = $repo_type,
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
            "repository_hash": repository_hash or "",
            "repository_name": repository_name or "",
            "file_name": file_name or "",
            "relative_path": relative_path or "",
            "entry_file": entry_file or "",
            "repo_type": repo_type or ""
        })
        
        return {"created": 1}
    
    async def _create_service_node(self, session, service: Dict[str, Any],
                                  file_path: str, file_node_id: str,
                                  repository_hash: str,
                                  repo_type: str = None,
                                  repository_name: str = None,
                                  entry_file: str = None,
                                  file_name: str = None,
                                  relative_path: str = None) -> Dict[str, Any]:
        """Create a Service/Config node with REQUIRED metadata"""
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
            s.repository_name = $repository_name,
            s.file_name = $file_name,
            s.relative_path = $relative_path,
            s.entry_file = $entry_file,
            s.repo_type = $repo_type,
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
            "repository_hash": repository_hash or "",
            "repository_name": repository_name or "",
            "file_name": file_name or "",
            "relative_path": relative_path or "",
            "entry_file": entry_file or "",
            "repo_type": repo_type or ""
        })
        
        return {"created": 1}
    
    async def _create_component_node(self, session, comp: Any,
                                    file_path: str, file_node_id: str,
                                    repository_hash: str,
                                    repo_type: str = None,
                                    repository_name: str = None,
                                    entry_file: str = None,
                                    file_name: str = None,
                                    relative_path: str = None) -> Dict[str, Any]:
        """Create a React Component node with REQUIRED metadata"""
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
            c.repository_name = $repository_name,
            c.file_name = $file_name,
            c.relative_path = $relative_path,
            c.entry_file = $entry_file,
            c.repo_type = $repo_type,
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
            "repository_hash": repository_hash or "",
            "repository_name": repository_name or "",
            "file_name": file_name or "",
            "relative_path": relative_path or "",
            "entry_file": entry_file or "",
            "repo_type": repo_type or ""
        })
        
        return {"created": 1}
    
    async def _create_api_call_node(self, session, api: Any,
                                   file_path: str, file_node_id: str,
                                   repository_hash: str,
                                   repo_type: str = None,
                                   repository_name: str = None,
                                   entry_file: str = None,
                                   file_name: str = None,
                                   relative_path: str = None) -> Dict[str, Any]:
        """Create an API Call node with REQUIRED metadata"""
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
            a.repository_name = $repository_name,
            a.file_name = $file_name,
            a.relative_path = $relative_path,
            a.entry_file = $entry_file,
            a.repo_type = $repo_type,
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
            "repository_hash": repository_hash or "",
            "repository_name": repository_name or "",
            "file_name": file_name or "",
            "relative_path": relative_path or "",
            "entry_file": entry_file or "",
            "repo_type": repo_type or ""
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
    
    def _is_entry_file(self, file_path: str, repo_type: str = None) -> bool:
        """
        Detect if a file is an entry point based on repo type and filename patterns.
        
        Backend entry files: index.js, app.js, server.js, main.js, index.ts, app.ts, etc.
        Frontend entry files: main.jsx, main.tsx, index.jsx, index.tsx, App.jsx, App.tsx
        """
        if not file_path:
            return False
        
        file_name = file_path.split('/')[-1].lower()
        
        # Backend entry patterns
        backend_entry_patterns = [
            'index.js', 'index.ts', 'app.js', 'app.ts',
            'server.js', 'server.ts', 'main.js', 'main.ts',
            'application.java', 'main.java', 'app.java'
        ]
        
        # Frontend entry patterns
        frontend_entry_patterns = [
            'main.jsx', 'main.tsx', 'index.jsx', 'index.tsx',
            'app.jsx', 'app.tsx', 'main.js', 'main.ts',
            'index.js', 'index.ts'
        ]
        
        if repo_type == 'backend':
            return file_name in backend_entry_patterns
        elif repo_type == 'frontend':
            return file_name in frontend_entry_patterns
        else:
            # Check both patterns if repo_type is unknown
            return file_name in backend_entry_patterns or file_name in frontend_entry_patterns