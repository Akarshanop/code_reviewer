# java_parser.py - IMPROVED VERSION
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from tree_sitter import Node

logger = logging.getLogger(__name__)

@dataclass
class RouteInfo:
    path: str
    method: str
    handler_method: str
    consumes: List[str] = None
    produces: List[str] = None

@dataclass
class EventBusInfo:
    address: str
    type: str 
    handler_method: str

@dataclass
class DTOInfo:
    name: str
    fields: List[Dict[str, str]]
    is_data_object: bool = False

class JavaParser:
    """Enhanced Java parser specifically for Vert.x patterns"""
    
    def __init__(self):
        from tree_sitter_languages import get_language, get_parser
        self.language = get_language('java')
        self.parser = get_parser('java')
        self.route_patterns = {
            '@Route': self._extract_vertx_route,
            '@GetMapping': self._extract_spring_route,
            '@PostMapping': self._extract_spring_route,
            '@PutMapping': self._extract_spring_route,
            '@DeleteMapping': self._extract_spring_route,
        }
    
    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse a Java file and extract Vert.x specific patterns"""
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        tree = self.parser.parse(bytes(code, 'utf-8'))
        
        results = {
            'file_path': str(file_path),
            'package': self._extract_package(tree.root_node, code),
            'imports': self._extract_imports(tree.root_node, code),
            'routes': [],
            'eventbus': [],
            'dtos': [],
            'services': [],
            'configs': []
        }
        
        # Walk the tree and extract patterns
        self._walk_tree(tree.root_node, code, results)
        
        logger.info(
            f"Parsed Java file: {file_path} "
            f"(routes: {len(results['routes'])}, "
            f"eventbus: {len(results['eventbus'])}, "
            f"dtos: {len(results['dtos'])})"
        )
        
        return results
    
    def _walk_tree(self, node: Node, code: str, results: Dict):
        """Recursively walk the AST and extract patterns"""
        
        # Check for class declarations
        if node.type == 'class_declaration':
            class_info = self._extract_class_info(node, code)
            if class_info:
                if 'DTO' in class_info['name'] or class_info.get('is_data_object'):
                    results['dtos'].append(class_info)
                elif 'Service' in class_info['name'] or 'Handler' in class_info['name']:
                    results['services'].append(class_info)
                elif 'Config' in class_info['name']:
                    results['configs'].append(class_info)
        
        # Check for method declarations with annotations
        elif node.type == 'method_declaration':
            method_info = self._extract_method_info(node, code)
            if method_info:
                if method_info.get('is_route'):
                    results['routes'].append(method_info)
                elif method_info.get('is_eventbus'):
                    results['eventbus'].append(method_info)
        
        # Recursively process children
        for child in node.children:
            self._walk_tree(child, code, results)
    
    def _extract_vertx_route(self, annotation_node: Node, code: str) -> Optional[Dict]:
        """Extract Vert.x route information"""
        annotation_text = code[annotation_node.start_byte:annotation_node.end_byte]
        
        # Extract path and method from @Route annotation
        path_match = re.search(r'path\s*=\s*["\']([^"\']+)["\']', annotation_text)
        method_match = re.search(r'method\s*=\s*(HttpMethod\.\w+)', annotation_text)
        
        if path_match:
            return {
                'path': path_match.group(1),
                'method': method_match.group(1).split('.')[1] if method_match else 'GET',
                'annotation': '@Route'
            }
        return None
    
    def _extract_method_info(self, node: Node, code: str) -> Optional[Dict]:
        """Extract method information and annotations"""
        method_text = code[node.start_byte:node.end_byte]
        
        # Extract method name
        method_name_node = node.child_by_field_name('name')
        if not method_name_node:
            return None
        
        method_name = code[method_name_node.start_byte:method_name_node.end_byte]
        
        # Look for annotations on this method
        annotations = []
        for child in node.children:
            if child.type == 'annotation':
                annotation_text = code[child.start_byte:child.end_byte]
                annotations.append(annotation_text)
                
                # Check if it's a route annotation
                for pattern, extractor in self.route_patterns.items():
                    if pattern in annotation_text:
                        route_info = extractor(child, code)
                        if route_info:
                            return {
                                'name': method_name,
                                'is_route': True,
                                'route_info': route_info,
                                'annotations': annotations
                            }
                
                # Check if it's an EventBus annotation
                if '@Consume' in annotation_text or '@Publish' in annotation_text:
                    address_match = re.search(r'["\']([^"\']+)["\']', annotation_text)
                    if address_match:
                        return {
                            'name': method_name,
                            'is_eventbus': True,
                            'address': address_match.group(1),
                            'type': 'consume' if '@Consume' in annotation_text else 'publish',
                            'annotations': annotations
                        }
        
        return {
            'name': method_name,
            'annotations': annotations
        }
    
    def _extract_class_info(self, node: Node, code: str) -> Optional[Dict]:
        """Extract class information"""
        class_name_node = node.child_by_field_name('name')
        if not class_name_node:
            return None
        
        class_name = code[class_name_node.start_byte:class_name_node.end_byte]
        
        # Check for @DataObject annotation
        class_text = code[node.start_byte:node.end_byte]
        is_data_object = '@DataObject' in class_text
        
        # Extract fields
        fields = []
        body_node = node.child_by_field_name('body')
        if body_node:
            for child in body_node.children:
                if child.type == 'field_declaration':
                    field_text = code[child.start_byte:child.end_byte]
                    # Simple field extraction
                    field_match = re.search(r'(\w+)\s+(\w+)\s*;', field_text)
                    if field_match:
                        fields.append({
                            'type': field_match.group(1),
                            'name': field_match.group(2)
                        })
        
        return {
            'name': class_name,
            'is_data_object': is_data_object,
            'fields': fields
        }
    
    def _extract_package(self, root: Node, code: str) -> str:
        """Extract package declaration"""
        package_node = root.child_by_field_name('package')
        if package_node:
            return code[package_node.start_byte:package_node.end_byte].replace('package ', '').strip(';')
        return ''
    
    def _extract_imports(self, root: Node, code: str) -> List[str]:
        """Extract import statements"""
        imports = []
        for child in root.children:
            if child.type == 'import_declaration':
                import_text = code[child.start_byte:child.end_byte]
                imports.append(import_text.strip())
        return imports