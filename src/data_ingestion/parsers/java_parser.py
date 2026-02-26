"""
Enhanced Java Parser using tree-sitter AST.
Supports Vert.x, Spring Boot, Micronaut, and generic Java patterns.
Extracts: classes, methods with bodies, fields (via AST), routes, EventBus,
          call graphs, error handling, control flow.
"""
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from tree_sitter import Node

logger = logging.getLogger(__name__)

# Maximum chars stored per method/class body
_MAX_BODY_CHARS = 8000

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
    """Enhanced Java parser for Vert.x, Spring Boot, Micronaut, and generic Java."""
    
    # FIX 7: Spring Boot + Micronaut annotation sets
    ROUTE_ANNOTATIONS = {
        # Vert.x
        '@Route',
        # Spring Boot
        '@GetMapping', '@PostMapping', '@PutMapping', '@DeleteMapping',
        '@PatchMapping', '@RequestMapping',
        # Micronaut
        '@Get', '@Post', '@Put', '@Delete', '@Patch',
        # JAX-RS
        '@GET', '@POST', '@PUT', '@DELETE', '@PATCH', '@Path',
    }
    
    SPRING_METHOD_MAP = {
        'GetMapping': 'GET', 'PostMapping': 'POST', 'PutMapping': 'PUT',
        'DeleteMapping': 'DELETE', 'PatchMapping': 'PATCH',
        'Get': 'GET', 'Post': 'POST', 'Put': 'PUT', 'Delete': 'DELETE',
        'Patch': 'PATCH',
    }

    ROUTER_HTTP_METHOD_MAP = {
        'get': 'GET',
        'post': 'POST',
        'put': 'PUT',
        'delete': 'DELETE',
        'patch': 'PATCH',
        'route': 'ANY',
    }
    
    def __init__(self):
        from tree_sitter_languages import get_language, get_parser
        self.language = get_language('java')
        self.parser = get_parser('java')
    
    def _text(self, node: Node, code: str) -> str:
        """Get text of a tree-sitter node."""
        return code[node.start_byte:node.end_byte]
    
    def _find_all(self, node: Node, type_name: str) -> List[Node]:
        """Find all descendant nodes of a given type."""
        found = []
        if node.type == type_name:
            found.append(node)
        for child in node.children:
            found.extend(self._find_all(child, type_name))
        return found
    
    def parse(self, file_path: str, content: str = None) -> Dict[str, Any]:
        """Parse a Java file. Primary interface used by ParserCoordinator."""
        if content is None:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        return self._parse_content(str(file_path), content)

    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse a Java file from a Path object."""
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        return self._parse_content(str(file_path), code)

    def _parse_content(self, file_path: str, code: str) -> Dict[str, Any]:
        """Internal: parse Java source code string."""
        tree = self.parser.parse(bytes(code, 'utf-8'))
        
        results: Dict[str, Any] = {
            'file_path': str(file_path),
            'package': self._extract_package(tree.root_node, code),
            'imports': self._extract_imports(tree.root_node, code),
            'classes': [],
            'functions': [],
            'routes': [],
            'eventbus': [],
            'dtos': [],
            'services': [],
            'configs': [],
            'error_handling': [],
            'control_flow': [],
        }
        
        # Walk the tree and extract patterns
        self._walk_tree(tree.root_node, code, results)

        # Extract Vert.x programmatic routes like router.post("/x").handler(...)
        detected_programmatic_routes = self._extract_vertx_programmatic_routes(tree.root_node, code)
        if detected_programmatic_routes:
            existing = {
                (
                    str(r.get('path', '')).strip(),
                    str((r.get('method') or 'GET')).upper(),
                    str(r.get('handler_method', '')).strip(),
                )
                for r in results['routes']
                if isinstance(r, dict)
            }
            for route in detected_programmatic_routes:
                route_key = (
                    str(route.get('path', '')).strip(),
                    str((route.get('method') or 'GET')).upper(),
                    str(route.get('handler_method', '')).strip(),
                )
                if route_key not in existing:
                    results['routes'].append(route)
                    existing.add(route_key)
        
        logger.info(
            f"Parsed Java file: {file_path} "
            f"(classes: {len(results['classes'])}, functions: {len(results['functions'])}, "
            f"routes: {len(results['routes'])}, eventbus: {len(results['eventbus'])}, "
            f"dtos: {len(results['dtos'])})"
        )
        
        return results
    
    def _walk_tree(self, node: Node, code: str, results: Dict):
        """Recursively walk the AST and extract patterns."""
        
        if node.type == 'class_declaration':
            class_info = self._extract_class_info(node, code)
            if class_info:
                results['classes'].append(class_info)
                # Also classify into specialized lists
                name = class_info['name']
                if 'DTO' in name or class_info.get('is_data_object'):
                    results['dtos'].append(class_info)
                elif any(k in name for k in ('Service', 'Handler', 'Controller', 'Resource')):
                    results['services'].append(class_info)
                elif 'Config' in name or 'Configuration' in name:
                    results['configs'].append(class_info)
        
        elif node.type == 'method_declaration':
            method_info = self._extract_method_info(node, code)
            if method_info:
                package_name = results.get("package", "")
                class_name = self._find_enclosing_class_name(node, code) or ""
                method_info["class_name"] = class_name
                if package_name and class_name:
                    method_info["fqn"] = f"{package_name}.{class_name}.{method_info['name']}"
                elif class_name:
                    method_info["fqn"] = f"{class_name}.{method_info['name']}"
                elif package_name:
                    method_info["fqn"] = f"{package_name}.{method_info['name']}"
                else:
                    method_info["fqn"] = method_info["name"]
                results['functions'].append(method_info)
                if method_info.get('is_route'):
                    results['routes'].append(method_info)
                elif method_info.get('is_eventbus'):
                    results['eventbus'].append(method_info)
        
        # FIX 2: Error handling
        elif node.type == 'try_statement':
            results['error_handling'].append(self._extract_try_info(node, code))
        
        # FIX 6: Control flow
        elif node.type == 'if_statement':
            branches = 1
            current = node
            while True:
                alt = current.child_by_field_name('alternative')
                if alt and alt.type == 'if_statement':
                    branches += 1
                    current = alt
                elif alt:
                    branches += 1  # else
                    break
                else:
                    break
            results['control_flow'].append({
                'type': 'if',
                'line': node.start_point[0] + 1,
                'branches': branches,
            })
        elif node.type == 'switch_expression' or node.type == 'switch_statement':
            cases = len(self._find_all(node, 'switch_label'))
            results['control_flow'].append({
                'type': 'switch',
                'line': node.start_point[0] + 1,
                'cases': cases,
            })
        elif node.type in ('for_statement', 'enhanced_for_statement'):
            results['control_flow'].append({
                'type': 'for',
                'line': node.start_point[0] + 1,
            })
        elif node.type == 'while_statement':
            results['control_flow'].append({
                'type': 'while',
                'line': node.start_point[0] + 1,
            })
        
        for child in node.children:
            self._walk_tree(child, code, results)
    
    # =========================================================================
    # FIX 7: Hardened method extraction (tree-sitter, not regex)
    # =========================================================================
    
    def _extract_method_info(self, node: Node, code: str) -> Optional[Dict]:
        """Extract method information using tree-sitter fields (not regex)."""
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None
        method_name = self._text(name_node, code)
        
        # FIX 7: Extract return type via tree-sitter
        return_type = None
        type_node = node.child_by_field_name('type')
        if type_node:
            return_type = self._text(type_node, code)
        
        # FIX 7: Extract parameters via tree-sitter (not regex)
        parameters = []
        params_node = node.child_by_field_name('parameters')
        if params_node:
            for param in params_node.children:
                if param.type == 'formal_parameter':
                    p_type_node = param.child_by_field_name('type')
                    p_name_node = param.child_by_field_name('name')
                    if p_type_node and p_name_node:
                        parameters.append({
                            'type': self._text(p_type_node, code),
                            'name': self._text(p_name_node, code),
                        })
        
        # FIX 1: Get full method body
        body = self._text(node, code)[:_MAX_BODY_CHARS]
        
        # FIX 4: Extract method calls
        calls = self._extract_calls(node, code)
        
        # FIX 2: Error handling within method
        error_handling = []
        for try_node in self._find_all(node, 'try_statement'):
            error_handling.append(self._extract_try_info(try_node, code))
        
        # Extract annotations
        annotations = []
        is_route = False
        is_eventbus = False
        route_path = ""
        route_method = "GET"
        eb_address = ""
        eb_type = ""
        
        for child in node.children:
            if child.type in ('annotation', 'marker_annotation'):
                ann_text = self._text(child, code)
                annotations.append(ann_text)
                
                # Check for route annotations (Vert.x, Spring, Micronaut, JAX-RS)
                for route_ann in self.ROUTE_ANNOTATIONS:
                    if route_ann in ann_text:
                        is_route = True
                        route_path, route_method = self._parse_route_annotation(ann_text)
                        break
                
                # EventBus
                if '@Consume' in ann_text or '@Publish' in ann_text:
                    is_eventbus = True
                    addr_match = re.search(r'["\']([^"\']+)["\']', ann_text)
                    if addr_match:
                        eb_address = addr_match.group(1)
                    eb_type = 'consume' if '@Consume' in ann_text else 'publish'
        
        result: Dict[str, Any] = {
            'name': method_name,
            'return_type': return_type,
            'parameters': parameters,
            'annotations': annotations,
            'body': body,
            'calls': calls,
            'error_handling': error_handling,
            'start_line': node.start_point[0] + 1,
            'end_line': node.end_point[0] + 1,
            'is_route': is_route,
            'is_eventbus': is_eventbus,
        }
        
        if is_route:
            result['path'] = route_path
            result['method'] = route_method
            result['handler_method'] = method_name
            result['route_info'] = {'path': route_path, 'method': route_method}
        
        if is_eventbus:
            result['address'] = eb_address
            result['type'] = eb_type
            result['handler_method'] = method_name
        
        return result
    
    def _parse_route_annotation(self, ann_text: str) -> tuple:
        """Parse route path and HTTP method from any annotation format."""
        # Determine HTTP method from annotation name
        method = "GET"
        for ann_name, http_method in self.SPRING_METHOD_MAP.items():
            if ann_name in ann_text:
                method = http_method
                break
        if '@RequestMapping' in ann_text:
            m_match = re.search(r'method\s*=\s*RequestMethod\.(\w+)', ann_text)
            if m_match:
                method = m_match.group(1)
        
        # Extract path (works for value="...", path="...", or just "...")
        path = ""
        path_match = re.search(r'(?:value|path)\s*=\s*["\']([^"\']+)["\']', ann_text)
        if path_match:
            path = path_match.group(1)
        else:
            # Try bare string: @GetMapping("/users")
            bare_match = re.search(r'\(\s*["\']([^"\']+)["\']\s*\)', ann_text)
            if bare_match:
                path = bare_match.group(1)
        
        return path, method

    def _extract_vertx_programmatic_routes(self, root: Node, code: str) -> List[Dict[str, Any]]:
        """Extract Vert.x programmatic routes from method_invocation chains."""
        routes: List[Dict[str, Any]] = []

        for call_node in self._find_all(root, 'method_invocation'):
            name_node = call_node.child_by_field_name('name')
            if not name_node:
                continue

            call_name = self._text(name_node, code)
            call_name_lower = call_name.lower()
            object_node = call_node.child_by_field_name('object')
            object_text = self._text(object_node, code) if object_node else ''
            if 'router' not in object_text.lower():
                continue

            args_node = call_node.child_by_field_name('arguments')
            expression_text = self._get_enclosing_expression_text(call_node, code)
            handler_method = self._extract_handler_method_from_expression(expression_text)

            if call_name_lower in self.ROUTER_HTTP_METHOD_MAP and args_node is not None:
                path = self._extract_first_string_argument(args_node, code)
                if not path:
                    continue

                method = self.ROUTER_HTTP_METHOD_MAP[call_name_lower]
                if call_name_lower == 'route':
                    explicit_method = self._extract_http_method_from_expression(expression_text)
                    if explicit_method:
                        method = explicit_method

                routes.append({
                    'path': path,
                    'method': method,
                    'handler_method': handler_method or call_name,
                    'start_line': call_node.start_point[0] + 1,
                    'end_line': call_node.end_point[0] + 1,
                    'route_info': {'path': path, 'method': method},
                    'is_route': True,
                })
                continue

            if call_name == 'mountSubRouter' and args_node is not None:
                prefix = self._extract_first_string_argument(args_node, code)
                target_router = self._extract_argument_text(args_node, code, 1)
                if not prefix:
                    continue

                routes.append({
                    'path': prefix,
                    'method': 'MOUNT',
                    'handler_method': self._normalize_handler_name(target_router) or 'mountSubRouter',
                    'start_line': call_node.start_point[0] + 1,
                    'end_line': call_node.end_point[0] + 1,
                    'route_info': {'path': prefix, 'method': 'MOUNT'},
                    'is_route': True,
                })

        return routes

    def _extract_first_string_argument(self, args_node: Node, code: str) -> str:
        """Extract the first string literal argument from an argument list."""
        args_text = self._text(args_node, code)
        match = re.search(r'["\']([^"\']+)["\']', args_text)
        return match.group(1).strip() if match else ''

    def _extract_argument_text(self, args_node: Node, code: str, index: int) -> str:
        """Extract nth argument text from tree-sitter argument_list node."""
        arg_nodes = [
            child for child in args_node.children
            if child.type not in {',', '(', ')'}
        ]
        if index < 0 or index >= len(arg_nodes):
            return ''
        return self._text(arg_nodes[index], code).strip()

    def _get_enclosing_expression_text(self, node: Node, code: str) -> str:
        """Get containing expression text for chained calls like .handler(...)."""
        current = node
        while current.parent is not None:
            current = current.parent
            if current.type == 'expression_statement':
                return self._text(current, code)
        return self._text(node, code)

    def _extract_http_method_from_expression(self, expression_text: str) -> str:
        """Extract HttpMethod.X from chain expressions."""
        match = re.search(r'\.method\s*\(\s*HttpMethod\.(\w+)\s*\)', expression_text)
        return match.group(1).upper() if match else ''

    def _extract_handler_method_from_expression(self, expression_text: str) -> str:
        """Extract handler method from .handler(...) chain."""
        match = re.search(r'\.handler\s*\(\s*([^)]+)\)', expression_text)
        if not match:
            return ''
        return self._normalize_handler_name(match.group(1))

    def _normalize_handler_name(self, handler_expr: str) -> str:
        """Normalize handler references to a concise method-ish name."""
        if not handler_expr:
            return ''

        expr = handler_expr.strip()
        if '::' in expr:
            return expr.split('::')[-1].strip()
        if '(' in expr:
            expr = expr.split('(')[0].strip()
        if '.' in expr:
            return expr.split('.')[-1].strip()
        return expr
    
    # =========================================================================
    # FIX 7: Hardened class extraction
    # =========================================================================
    
    def _extract_class_info(self, node: Node, code: str) -> Optional[Dict]:
        """Extract class information using tree-sitter (not regex)."""
        class_name_node = node.child_by_field_name('name')
        if not class_name_node:
            return None
        class_name = self._text(class_name_node, code)
        
        # Check annotations (before the class)
        class_text = self._text(node, code)
        annotations = [self._text(c, code) for c in node.children if c.type in ('annotation', 'marker_annotation')]
        is_data_object = any('@DataObject' in a for a in annotations)
        
        # Superclass / interfaces
        superclass = None
        interfaces = []
        sc_node = node.child_by_field_name('superclass')
        if sc_node:
            superclass = self._text(sc_node, code).replace('extends ', '').strip()
        for child in node.children:
            if child.type == 'super_interfaces':
                interfaces = [self._text(c, code) for c in child.children if c.type == 'type_identifier']
        
        # FIX 7: Extract fields via tree-sitter (not regex)
        fields = []
        methods = []
        body_node = node.child_by_field_name('body')
        if body_node:
            for child in body_node.children:
                if child.type == 'field_declaration':
                    f_type = child.child_by_field_name('type')
                    for declarator in child.children:
                        if declarator.type == 'variable_declarator':
                            f_name = declarator.child_by_field_name('name')
                            if f_type and f_name:
                                fields.append({
                                    'type': self._text(f_type, code),
                                    'name': self._text(f_name, code),
                                })
                elif child.type == 'method_declaration':
                    m_name = child.child_by_field_name('name')
                    if m_name:
                        methods.append(self._text(m_name, code))
        
        return {
            'name': class_name,
            'superclass': superclass,
            'interfaces': interfaces,
            'annotations': annotations,
            'is_data_object': is_data_object,
            'fields': fields,
            'methods': methods,
            'body': class_text[:_MAX_BODY_CHARS],
            'start_line': node.start_point[0] + 1,
            'end_line': node.end_point[0] + 1,
        }
    
    # =========================================================================
    # FIX 4: Call graph extraction
    # =========================================================================
    
    def _extract_calls(self, node: Node, code: str) -> List[str]:
        """Extract all method invocations within a node."""
        calls: Set[str] = set()
        for call_node in self._find_all(node, 'method_invocation'):
            name_node = call_node.child_by_field_name('name')
            if name_node:
                obj_node = call_node.child_by_field_name('object')
                if obj_node:
                    calls.add(f"{self._text(obj_node, code)}.{self._text(name_node, code)}")
                else:
                    calls.add(self._text(name_node, code))
        return sorted(calls)
    
    # =========================================================================
    # FIX 2: Error handling extraction
    # =========================================================================
    
    def _extract_try_info(self, node: Node, code: str) -> Dict[str, Any]:
        """Extract try/catch/finally info from a try_statement node."""
        info: Dict[str, Any] = {
            'line': node.start_point[0] + 1,
            'end_line': node.end_point[0] + 1,
            'has_finally': False,
            'exceptions_caught': [],
        }
        for child in node.children:
            if child.type == 'catch_clause':
                catch_param = child.child_by_field_name('parameter') or child.child_by_field_name('body')
                exc_type = "Exception"
                # Try to get the catch type
                for cc in child.children:
                    if cc.type == 'catch_formal_parameter':
                        for ccc in cc.children:
                            if ccc.type in ('type_identifier', 'scoped_type_identifier'):
                                exc_type = self._text(ccc, code)
                                break
                info['exceptions_caught'].append({'type': exc_type, 'line': child.start_point[0] + 1})
            elif child.type == 'finally_clause':
                info['has_finally'] = True
        return info
    
    def _extract_package(self, root: Node, code: str) -> str:
        """Extract package declaration"""
        for child in root.children:
            if child.type == 'package_declaration':
                for sc in child.children:
                    if sc.type == 'scoped_identifier' or sc.type == 'identifier':
                        return self._text(sc, code)
        return ''
    
    def _extract_imports(self, root: Node, code: str) -> List[str]:
        """Extract import statements"""
        imports = []
        for child in root.children:
            if child.type == 'import_declaration':
                import_text = self._text(child, code)
                imports.append(import_text.strip())
        return imports

    def _find_enclosing_class_name(self, node: Node, code: str) -> str:
        """Find class declaration name that contains this node."""
        current = node.parent
        while current is not None:
            if current.type == "class_declaration":
                class_name_node = current.child_by_field_name("name")
                if class_name_node is not None:
                    return self._text(class_name_node, code)
            current = current.parent
        return ""