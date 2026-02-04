"""
Smart Graph Builder for Code Intelligence Platform

Builds a rich, connected knowledge graph by:
1. Starting from detected entry points (index.js, app.js)
2. Following imports/requires to discover connected files
3. Extracting routes → controllers → services → models relationships
4. Creating semantic edges (CALLS, IMPORTS, USES, EXPOSES)

This produces a much better graph than naive file-by-file parsing.
"""
import asyncio
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from src.core.logger import get_logger
from src.database.neo4j_client import neo4j_client
from src.data_ingestion.entry_point_detector import EntryPointDetector
from src.data_ingestion.parsers.parser_coordinator import ParserCoordinator, ParsedFile

logger = get_logger(__name__)


@dataclass
class GraphNode:
    """Represents a node to be created in Neo4j"""
    id: str
    labels: List[str]
    properties: Dict[str, Any]


@dataclass
class GraphEdge:
    """Represents an edge to be created in Neo4j"""
    source_id: str
    target_id: str
    relationship: str
    properties: Dict[str, Any] = field(default_factory=dict)


class SmartGraphBuilder:
    """
    Builds a rich knowledge graph by intelligently traversing the codebase
    starting from entry points and following code flow.
    """
    
    def __init__(self, repo_path: str, repo_name: str):
        """
        Initialize the smart graph builder.
        
        Args:
            repo_path: Absolute path to repository
            repo_name: Name/identifier for the repository
        """
        self.repo_path = Path(repo_path)
        self.repo_name = repo_name
        self.repo_hash = hashlib.md5(repo_name.encode()).hexdigest()[:12]
        
        # Initialize components
        self.entry_detector = EntryPointDetector(repo_path)
        self.parser = ParserCoordinator()
        
        # Track what we've processed
        self.processed_files: Set[str] = set()
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'nodes_created': 0,
            'edges_created': 0,
            'routes_found': 0,
            'functions_found': 0,
            'components_found': 0,
            'imports_resolved': 0,
            'errors': []
        }
    
    def _generate_id(self, node_type: str, *parts) -> str:
        """Generate deterministic node ID"""
        combined = f"{node_type}:{self.repo_hash}:{'|'.join(str(p) for p in parts)}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    async def build_graph(self) -> Dict[str, Any]:
        """
        Build the complete knowledge graph.
        
        Process:
        1. Detect entry points
        2. Parse entry points first
        3. Follow imports recursively
        4. Create nodes and edges
        5. Persist to Neo4j
        
        Returns:
            Statistics about the graph building process
        """
        logger.info(f"Starting smart graph build for {self.repo_name}")
        start_time = datetime.now()
        
        try:
            # Step 1: Create repository node
            await self._create_repository_node()
            
            # Step 2: Detect entry points
            entry_points = self.entry_detector.detect_entry_points()
            logger.info(f"Detected {len(entry_points)} entry points")
            
            if not entry_points:
                logger.warning("No entry points detected, falling back to full scan")
                entry_points = []
            
            # Step 3: Get optimal parsing order
            parsing_order = self.entry_detector.get_parsing_order()
            logger.info(f"Parsing order contains {len(parsing_order)} files")
            
            # Step 4: Parse files in order and build graph
            for file_path in parsing_order:
                if file_path not in self.processed_files:
                    await self._process_file(file_path)
            
            # Step 5: Scan remaining files (not in parsing order)
            await self._scan_remaining_files()
            
            # Step 6: Create all edges (import links, function calls, etc.)
            await self._create_semantic_edges()
            
            # Step 7: Persist to Neo4j
            await self._persist_to_neo4j()
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            self.stats['duration_seconds'] = duration
            
            logger.info(
                "Graph build complete",
                files=self.stats['files_processed'],
                nodes=self.stats['nodes_created'],
                edges=self.stats['edges_created'],
                duration=f"{duration:.2f}s"
            )
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Graph build failed: {e}")
            self.stats['errors'].append(str(e))
            raise
    
    async def _create_repository_node(self):
        """Create the repository root node"""
        repo_id = self._generate_id('Repository', self.repo_name)
        
        self.nodes[repo_id] = GraphNode(
            id=repo_id,
            labels=['Repository'],
            properties={
                'name': self.repo_name,
                'hash': self.repo_hash,
                'path': str(self.repo_path),
                'language': self.entry_detector.language,
                'created_at': datetime.now().isoformat(),
            }
        )
        
        self.repo_node_id = repo_id
    
    async def _process_file(self, file_path: str) -> Optional[ParsedFile]:
        """
        Parse a single file and create graph nodes.
        
        Args:
            file_path: Absolute path to file
            
        Returns:
            ParsedFile result or None if failed
        """
        if file_path in self.processed_files:
            return None
        
        self.processed_files.add(file_path)
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse file (sync method)
            parsed = self.parser.parse_file(Path(file_path), content)
            
            if parsed.error:
                logger.debug(f"Parse error for {file_path}: {parsed.error}")
                return None
            
            # Create file node
            await self._create_file_node(file_path, parsed)
            
            # Create function nodes
            for func in parsed.functions:
                await self._create_function_node(file_path, func)
            
            # Create class nodes
            for cls in parsed.classes:
                await self._create_class_node(file_path, cls)
            
            # Create route nodes (Express routes)
            for route in parsed.routes:
                await self._create_route_node(file_path, route)
            
            # Create component nodes (React)
            if hasattr(parsed, 'components'):
                for comp in getattr(parsed, 'components', []):
                    await self._create_component_node(file_path, comp)
            
            # Track imports for edge creation
            await self._track_imports(file_path, parsed.imports)
            
            self.stats['files_processed'] += 1
            
            return parsed
            
        except Exception as e:
            logger.debug(f"Error processing {file_path}: {e}")
            self.stats['errors'].append(f"{file_path}: {e}")
            return None
    
    async def _create_file_node(self, file_path: str, parsed: ParsedFile):
        """Create a File node"""
        relative_path = str(Path(file_path).relative_to(self.repo_path))
        file_id = self._generate_id('File', relative_path)
        
        self.nodes[file_id] = GraphNode(
            id=file_id,
            labels=['File'],
            properties={
                'path': relative_path,
                'full_path': file_path,
                'language': parsed.language,
                'framework': parsed.framework,
                'imports_json': json.dumps(parsed.imports[:50]),  # Limit size
                'exports_json': json.dumps(parsed.exports[:50]),
                'repository': self.repo_name,
                'repository_hash': self.repo_hash,
            }
        )
        
        # Link file to repository
        self.edges.append(GraphEdge(
            source_id=file_id,
            target_id=self.repo_node_id,
            relationship='BELONGS_TO'
        ))
    
    async def _create_function_node(self, file_path: str, func: Dict[str, Any]):
        """Create a Function node"""
        func_name = func.get('name', 'anonymous')
        relative_path = str(Path(file_path).relative_to(self.repo_path))
        func_id = self._generate_id('Function', relative_path, func_name, func.get('start_line', 0))
        
        # Determine if this is a service, controller, middleware, etc.
        labels = ['Function']
        
        # Check file path to add more specific labels
        lower_path = file_path.lower()
        if '/services/' in lower_path or 'service' in func_name.lower():
            labels.append('Service')
        elif '/controllers/' in lower_path or 'controller' in func_name.lower():
            labels.append('Controller')
        elif '/middleware/' in lower_path:
            labels.append('Middleware')
        elif '/models/' in lower_path or 'model' in func_name.lower():
            labels.append('Model')
        elif '/utils/' in lower_path or '/helpers/' in lower_path:
            labels.append('Utility')
        
        self.nodes[func_id] = GraphNode(
            id=func_id,
            labels=labels,
            properties={
                'name': func_name,
                'file_path': relative_path,
                'start_line': func.get('start_line', 0),
                'end_line': func.get('end_line', 0),
                'is_async': func.get('is_async', False),
                'is_exported': func.get('is_exported', False),
                'parameters_json': json.dumps(func.get('parameters', [])),
                'description': func.get('description', '')[:500],
                'repository': self.repo_name,
            }
        )
        
        # Link to file
        file_id = self._generate_id('File', relative_path)
        self.edges.append(GraphEdge(
            source_id=file_id,
            target_id=func_id,
            relationship='CONTAINS'
        ))
        
        self.stats['functions_found'] += 1
    
    async def _create_class_node(self, file_path: str, cls: Dict[str, Any]):
        """Create a Class node"""
        class_name = cls.get('name', 'AnonymousClass')
        relative_path = str(Path(file_path).relative_to(self.repo_path))
        class_id = self._generate_id('Class', relative_path, class_name)
        
        labels = ['Class']
        
        # Add service/model labels based on naming
        lower_name = class_name.lower()
        if 'service' in lower_name:
            labels.append('Service')
        elif 'controller' in lower_name:
            labels.append('Controller')
        elif 'model' in lower_name or 'schema' in lower_name:
            labels.append('Model')
        
        self.nodes[class_id] = GraphNode(
            id=class_id,
            labels=labels,
            properties={
                'name': class_name,
                'file_path': relative_path,
                'extends': cls.get('extends', ''),
                'implements_json': json.dumps(cls.get('implements', [])),
                'methods_json': json.dumps([m.get('name', '') for m in cls.get('methods', [])]),
                'repository': self.repo_name,
            }
        )
        
        # Link to file
        file_id = self._generate_id('File', relative_path)
        self.edges.append(GraphEdge(
            source_id=file_id,
            target_id=class_id,
            relationship='CONTAINS'
        ))
    
    async def _create_route_node(self, file_path: str, route: Dict[str, Any]):
        """Create a Route node for Express/API routes"""
        method = route.get('method', 'GET').upper()
        path = route.get('path', route.get('route', '/unknown'))
        relative_path = str(Path(file_path).relative_to(self.repo_path))
        route_id = self._generate_id('Route', method, path)
        
        self.nodes[route_id] = GraphNode(
            id=route_id,
            labels=['Route', 'APIEndpoint'],
            properties={
                'method': method,
                'path': path,
                'full_path': f"{method} {path}",
                'file_path': relative_path,
                'handler': route.get('handler', ''),
                'middleware_json': json.dumps(route.get('middleware', [])),
                'description': route.get('description', ''),
                'repository': self.repo_name,
            }
        )
        
        # Link to file
        file_id = self._generate_id('File', relative_path)
        self.edges.append(GraphEdge(
            source_id=file_id,
            target_id=route_id,
            relationship='EXPOSES'
        ))
        
        # Link to handler function if we can identify it
        handler_name = route.get('handler', '')
        if handler_name:
            # We'll create this edge in _create_semantic_edges
            pass
        
        self.stats['routes_found'] += 1
    
    async def _create_component_node(self, file_path: str, comp: Dict[str, Any]):
        """Create a ReactComponent node"""
        comp_name = comp.get('name', 'AnonymousComponent')
        relative_path = str(Path(file_path).relative_to(self.repo_path))
        comp_id = self._generate_id('Component', relative_path, comp_name)
        
        self.nodes[comp_id] = GraphNode(
            id=comp_id,
            labels=['ReactComponent', 'Component'],
            properties={
                'name': comp_name,
                'file_path': relative_path,
                'hooks_json': json.dumps(comp.get('hooks', [])),
                'props_json': json.dumps(comp.get('props', [])),
                'api_calls_json': json.dumps(comp.get('api_calls', [])),
                'repository': self.repo_name,
            }
        )
        
        # Link to file
        file_id = self._generate_id('File', relative_path)
        self.edges.append(GraphEdge(
            source_id=file_id,
            target_id=comp_id,
            relationship='CONTAINS'
        ))
        
        self.stats['components_found'] += 1
    
    async def _track_imports(self, file_path: str, imports: List):
        """Track imports for later edge creation"""
        relative_path = str(Path(file_path).relative_to(self.repo_path))
        file_id = self._generate_id('File', relative_path)
        
        for imp in imports:
            # Handle both string imports and dict imports
            if isinstance(imp, dict):
                import_path = imp.get('source', imp.get('path', imp.get('module', '')))
            else:
                import_path = str(imp)
            
            if not import_path:
                continue
                
            # Try to resolve import to actual file
            resolved = self._resolve_import(file_path, import_path)
            if resolved:
                target_relative = str(Path(resolved).relative_to(self.repo_path))
                target_id = self._generate_id('File', target_relative)
                
                self.edges.append(GraphEdge(
                    source_id=file_id,
                    target_id=target_id,
                    relationship='IMPORTS',
                    properties={'import_path': import_path}
                ))
                
                self.stats['imports_resolved'] += 1
    
    def _resolve_import(self, from_file: str, import_path: str) -> Optional[str]:
        """Resolve import path to actual file path"""
        if not import_path or not isinstance(import_path, str):
            return None
            
        if not import_path.startswith('.'):
            return None  # External package
        
        from_dir = Path(from_file).parent
        
        # Try different resolutions
        candidates = [
            from_dir / import_path,
            from_dir / f"{import_path}.js",
            from_dir / f"{import_path}.ts",
            from_dir / f"{import_path}.jsx",
            from_dir / f"{import_path}.tsx",
            from_dir / import_path / 'index.js',
            from_dir / import_path / 'index.ts',
        ]
        
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.exists():
                return str(resolved)
        
        return None
    
    async def _scan_remaining_files(self):
        """Scan any files not yet processed"""
        extensions = ['.js', '.ts', '.jsx', '.tsx', '.py']
        
        for ext in extensions:
            for file_path in self.repo_path.rglob(f'*{ext}'):
                # Skip node_modules, dist, build directories
                if any(skip in str(file_path) for skip in ['node_modules', 'dist', 'build', '.git', '__pycache__']):
                    continue
                
                if str(file_path) not in self.processed_files:
                    await self._process_file(str(file_path))
    
    async def _create_semantic_edges(self):
        """Create semantic edges between nodes"""
        # Link routes to their handler functions
        for node_id, node in self.nodes.items():
            if 'Route' in node.labels:
                handler = node.properties.get('handler', '')
                if handler:
                    # Find function with this name
                    for other_id, other_node in self.nodes.items():
                        if 'Function' in other_node.labels and other_node.properties.get('name') == handler:
                            self.edges.append(GraphEdge(
                                source_id=node_id,
                                target_id=other_id,
                                relationship='HANDLED_BY'
                            ))
                            break
    
    async def _persist_to_neo4j(self):
        """Persist all nodes and edges to Neo4j"""
        async with neo4j_client.session() as session:
            # Create nodes in batches
            node_list = list(self.nodes.values())
            batch_size = 100
            
            for i in range(0, len(node_list), batch_size):
                batch = node_list[i:i+batch_size]
                await self._create_node_batch(session, batch)
            
            self.stats['nodes_created'] = len(self.nodes)
            
            # Create edges in batches
            for i in range(0, len(self.edges), batch_size):
                batch = self.edges[i:i+batch_size]
                await self._create_edge_batch(session, batch)
            
            self.stats['edges_created'] = len(self.edges)
    
    async def _create_node_batch(self, session, nodes: List[GraphNode]):
        """Create a batch of nodes"""
        for node in nodes:
            labels_str = ':'.join(node.labels)
            
            # Build properties string
            props = {k: v for k, v in node.properties.items() if v is not None}
            props['id'] = node.id
            
            query = f"""
            MERGE (n:{labels_str} {{id: $id}})
            SET n += $props
            """
            
            await session.run(query, {'id': node.id, 'props': props})
    
    async def _create_edge_batch(self, session, edges: List[GraphEdge]):
        """Create a batch of edges"""
        for edge in edges:
            query = f"""
            MATCH (source {{id: $source_id}})
            MATCH (target {{id: $target_id}})
            MERGE (source)-[r:{edge.relationship}]->(target)
            SET r += $props
            """
            
            await session.run(query, {
                'source_id': edge.source_id,
                'target_id': edge.target_id,
                'props': edge.properties
            })
