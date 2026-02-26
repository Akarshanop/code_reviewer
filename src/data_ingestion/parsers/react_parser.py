"""
Unified JavaScript/TypeScript Parser for Code Intelligence Platform.

Supports:
- JavaScript (.js) - Express.js backends, Node.js modules
- JSX (.jsx) - React components
- TypeScript (.ts) - TypeScript modules
- TSX (.tsx) - TypeScript React components

Extracts:
- Functions, classes, methods
- Express.js routes, middleware, controllers
- React components, hooks, API calls
- MongoDB/Mongoose models and schemas
- Imports and exports
"""

from pathlib import Path
from typing import Any, Optional
import re

from src.core.logger import get_logger

logger = get_logger(__name__)


class TsReactParser:
    """
    Unified parser for JavaScript, TypeScript, JSX, and TSX files.
    Uses tree-sitter with automatic language detection based on file extension.
    """

    def __init__(self):
        """Initialize both JavaScript and TypeScript parsers."""
        try:
            from tree_sitter_languages import get_language, get_parser
            
            # Initialize JavaScript parser
            self.js_language = get_language('javascript')
            self.js_parser = get_parser('javascript')
            
            # Initialize TypeScript parser
            self.ts_language = get_language('typescript')
            self.ts_parser = get_parser('typescript')
            
            # Default to JavaScript
            self.parser = self.js_parser
            self.language = self.js_language
            
            logger.info("TsReactParser initialized with JavaScript and TypeScript support")
            
        except Exception as e:
            logger.error(f"Failed to initialize TsReactParser: {e}")
            raise

    def _select_parser_for_file(self, file_path: str) -> None:
        """Select the appropriate parser based on file extension."""
        ext = Path(file_path).suffix.lower()
        
        if ext in ('.ts', '.tsx'):
            self.parser = self.ts_parser
            self.language = self.ts_language
            logger.debug(f"Using TypeScript parser for {file_path}")
        else:
            # Use JavaScript parser for .js, .jsx, .mjs, .cjs
            self.parser = self.js_parser
            self.language = self.js_language
            logger.debug(f"Using JavaScript parser for {file_path}")

    def parse(self, file_path: str, content: str) -> dict[str, Any]:
        """
        Parse a JavaScript/TypeScript file and extract code elements.
        
        Args:
            file_path: Path to the file being parsed
            content: The file content as a string
            
        Returns:
            Dictionary containing extracted code elements
        """
        try:
            # Select appropriate parser
            self._select_parser_for_file(file_path)
            
            # Parse the content
            tree = self.parser.parse(content.encode('utf-8'))
            root_node = tree.root_node
            
            # Extract all code elements
            result = {
                'file_path': file_path,
                'language': self._get_language_name(file_path),
                'imports': self._extract_imports(root_node, content),
                'exports': self._extract_exports(root_node, content),
                'functions': self._extract_functions(root_node, content),
                'classes': self._extract_classes(root_node, content),
                'components': self._extract_components(root_node, content, file_path),
                'routes': self._extract_express_routes(root_node, content, file_path),
                'middleware': self._extract_middleware(root_node, content),
                'models': self._extract_models(root_node, content),
                'hooks': self._extract_hooks(root_node, content),
                'api_calls': self._extract_api_calls(root_node, content),
                'variables': self._extract_variables(root_node, content),
            }
            
            logger.info(f"Parsed {file_path}: {len(result['functions'])} functions, "
                       f"{len(result['classes'])} classes, {len(result['routes'])} routes, "
                       f"{len(result['components'])} components")
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return {
                'file_path': file_path,
                'language': self._get_language_name(file_path),
                'error': str(e),
                'imports': [],
                'exports': [],
                'functions': [],
                'classes': [],
                'components': [],
                'routes': [],
                'middleware': [],
                'models': [],
                'hooks': [],
                'api_calls': [],
                'variables': [],
            }

    def _get_language_name(self, file_path: str) -> str:
        """Get the language name based on file extension."""
        ext = Path(file_path).suffix.lower()
        mapping = {
            '.js': 'javascript',
            '.jsx': 'javascript_react',
            '.mjs': 'javascript_module',
            '.cjs': 'javascript_commonjs',
            '.ts': 'typescript',
            '.tsx': 'typescript_react',
        }
        return mapping.get(ext, 'javascript')

    def _get_node_text(self, node, content: str) -> str:
        """Extract text content from a tree-sitter node."""
        try:
            return content[node.start_byte:node.end_byte]
        except Exception:
            return ""

    def _get_node_location(self, node) -> dict:
        """Get the location info for a node."""
        return {
            'start_line': node.start_point[0] + 1,
            'end_line': node.end_point[0] + 1,
            'start_col': node.start_point[1],
            'end_col': node.end_point[1],
        }

    # =========================================================================
    # Import/Export Extraction
    # =========================================================================

    def _extract_imports(self, root_node, content: str) -> list[dict]:
        """Extract all import statements."""
        imports = []
        
        # Handle ES6 imports
        for node in self._find_nodes_by_type(root_node, 'import_statement'):
            import_info = {
                'type': 'es6',
                'source': '',
                'specifiers': [],
                'default': None,
                'namespace': None,
                **self._get_node_location(node),
                'raw': self._get_node_text(node, content),
            }
            
            for child in node.children:
                if child.type == 'string':
                    import_info['source'] = self._get_node_text(child, content).strip('"\'')
                elif child.type == 'import_clause':
                    self._parse_import_clause(child, content, import_info)
            
            imports.append(import_info)
        
        # Handle CommonJS require
        for node in self._find_nodes_by_type(root_node, 'call_expression'):
            text = self._get_node_text(node, content)
            if 'require(' in text:
                match = re.search(r'require\([\'"]([^\'"]+)[\'"]\)', text)
                if match:
                    imports.append({
                        'type': 'commonjs',
                        'source': match.group(1),
                        'specifiers': [],
                        **self._get_node_location(node),
                        'raw': text,
                    })
        
        return imports

    def _parse_import_clause(self, node, content: str, import_info: dict) -> None:
        """Parse import clause for specifiers."""
        for child in node.children:
            if child.type == 'identifier':
                import_info['default'] = self._get_node_text(child, content)
            elif child.type == 'named_imports':
                for spec in child.children:
                    if spec.type == 'import_specifier':
                        name = None
                        alias = None
                        for part in spec.children:
                            if part.type == 'identifier':
                                if name is None:
                                    name = self._get_node_text(part, content)
                                else:
                                    alias = self._get_node_text(part, content)
                        if name:
                            import_info['specifiers'].append({
                                'name': name,
                                'alias': alias,
                            })
            elif child.type == 'namespace_import':
                for part in child.children:
                    if part.type == 'identifier':
                        import_info['namespace'] = self._get_node_text(part, content)

    def _extract_exports(self, root_node, content: str) -> list[dict]:
        """Extract all export statements."""
        exports = []
        
        # Named exports
        for node in self._find_nodes_by_type(root_node, 'export_statement'):
            export_info = {
                'type': 'named',
                'names': [],
                'default': False,
                **self._get_node_location(node),
                'raw': self._get_node_text(node, content),
            }
            
            text = self._get_node_text(node, content)
            if 'export default' in text:
                export_info['type'] = 'default'
                export_info['default'] = True
            
            # Extract exported names
            for child in node.children:
                if child.type == 'identifier':
                    export_info['names'].append(self._get_node_text(child, content))
                elif child.type == 'function_declaration':
                    for subchild in child.children:
                        if subchild.type == 'identifier':
                            export_info['names'].append(self._get_node_text(subchild, content))
                            break
                elif child.type == 'class_declaration':
                    for subchild in child.children:
                        if subchild.type == 'identifier':
                            export_info['names'].append(self._get_node_text(subchild, content))
                            break
            
            exports.append(export_info)
        
        # CommonJS module.exports
        for node in self._find_nodes_by_type(root_node, 'assignment_expression'):
            text = self._get_node_text(node, content)
            if 'module.exports' in text or 'exports.' in text:
                exports.append({
                    'type': 'commonjs',
                    'names': [],
                    **self._get_node_location(node),
                    'raw': text,
                })
        
        return exports

    # =========================================================================
    # Function Extraction
    # =========================================================================

    def _extract_functions(self, root_node, content: str) -> list[dict]:
        """Extract all functions including arrow functions."""
        functions = []
        
        # Regular function declarations
        for node in self._find_nodes_by_type(root_node, 'function_declaration'):
            func = self._parse_function_node(node, content)
            if func:
                functions.append(func)
        
        # Function expressions and arrow functions in variable declarations
        for node in self._find_nodes_by_type(root_node, 'variable_declarator'):
            func = self._parse_variable_function(node, content)
            if func:
                functions.append(func)
        
        # Arrow functions in assignments
        for node in self._find_nodes_by_type(root_node, 'assignment_expression'):
            func = self._parse_assignment_function(node, content)
            if func:
                functions.append(func)
        
        return functions

    def _parse_function_node(self, node, content: str) -> Optional[dict]:
        """Parse a function declaration node."""
        name = None
        params = []
        is_async = False
        is_generator = False
        
        text = self._get_node_text(node, content)
        is_async = 'async ' in text[:20]
        is_generator = 'function*' in text[:20]
        
        for child in node.children:
            if child.type == 'identifier':
                name = self._get_node_text(child, content)
            elif child.type == 'formal_parameters':
                params = self._parse_parameters(child, content)
        
        if name:
            return {
                'name': name,
                'type': 'function',
                'params': params,
                'is_async': is_async,
                'is_generator': is_generator,
                'calls': self._extract_call_names_within(node, content),
                **self._get_node_location(node),
                'body': self._get_node_text(node, content),
            }
        return None

    def _parse_variable_function(self, node, content: str) -> Optional[dict]:
        """Parse a function assigned to a variable."""
        name = None
        func_node = None
        
        for child in node.children:
            if child.type == 'identifier':
                name = self._get_node_text(child, content)
            elif child.type in ('arrow_function', 'function_expression', 'function'):
                func_node = child
        
        if name and func_node:
            params = []
            is_async = False
            
            text = self._get_node_text(func_node, content)
            is_async = text.strip().startswith('async')
            
            for child in func_node.children:
                if child.type == 'formal_parameters':
                    params = self._parse_parameters(child, content)
                elif child.type == 'identifier':
                    # Single parameter arrow function
                    params = [{'name': self._get_node_text(child, content), 'type': None}]
            
            return {
                'name': name,
                'type': 'arrow_function' if func_node.type == 'arrow_function' else 'function_expression',
                'params': params,
                'is_async': is_async,
                'calls': self._extract_call_names_within(func_node, content),
                **self._get_node_location(node),
                'body': self._get_node_text(func_node, content),
            }
        return None

    def _parse_assignment_function(self, node, content: str) -> Optional[dict]:
        """Parse a function in an assignment expression."""
        left = None
        right = None
        
        for child in node.children:
            if child.type in ('identifier', 'member_expression'):
                left = self._get_node_text(child, content)
            elif child.type in ('arrow_function', 'function_expression'):
                right = child
        
        if left and right:
            params = []
            for child in right.children:
                if child.type == 'formal_parameters':
                    params = self._parse_parameters(child, content)
            
            return {
                'name': left,
                'type': 'assigned_function',
                'params': params,
                'calls': self._extract_call_names_within(right, content),
                **self._get_node_location(node),
                'body': self._get_node_text(right, content),
            }
        return None

    def _extract_call_names_within(self, node, content: str) -> list[str]:
        """Extract called function identifiers (best-effort) within a subtree."""
        callees: set[str] = set()
        if node is None:
            return []

        for call in self._find_nodes_by_type(node, 'call_expression'):
            callee_node = call.child_by_field_name('function') if hasattr(call, 'child_by_field_name') else None
            if callee_node is None:
                for ch in call.children:
                    if ch.type in ('identifier', 'member_expression', 'subscript_expression'):
                        callee_node = ch
                        break

            if callee_node is None:
                continue

            if callee_node.type == 'identifier':
                name = self._get_node_text(callee_node, content).strip()
                if name:
                    callees.add(name)
            elif callee_node.type == 'member_expression':
                # Capture `obj.method()` as `method` (useful within a file/module)
                prop = callee_node.child_by_field_name('property') if hasattr(callee_node, 'child_by_field_name') else None
                name = self._get_node_text(prop, content).strip() if prop else ''
                if name:
                    callees.add(name)

        return sorted(callees)

    def _parse_parameters(self, node, content: str) -> list[dict]:
        """Parse function parameters."""
        params = []
        
        for child in node.children:
            if child.type == 'identifier':
                params.append({
                    'name': self._get_node_text(child, content),
                    'type': None,
                    'default': None,
                })
            elif child.type == 'assignment_pattern':
                # Parameter with default value
                name = None
                default = None
                for subchild in child.children:
                    if subchild.type == 'identifier' and name is None:
                        name = self._get_node_text(subchild, content)
                    elif name is not None:
                        default = self._get_node_text(subchild, content)
                if name:
                    params.append({
                        'name': name,
                        'type': None,
                        'default': default,
                    })
            elif child.type == 'rest_pattern':
                # Rest parameter
                for subchild in child.children:
                    if subchild.type == 'identifier':
                        params.append({
                            'name': '...' + self._get_node_text(subchild, content),
                            'type': None,
                            'default': None,
                        })
            elif child.type == 'object_pattern':
                # Destructured object parameter
                params.append({
                    'name': self._get_node_text(child, content),
                    'type': 'destructured_object',
                    'default': None,
                })
            elif child.type == 'array_pattern':
                # Destructured array parameter
                params.append({
                    'name': self._get_node_text(child, content),
                    'type': 'destructured_array',
                    'default': None,
                })
        
        return params

    # =========================================================================
    # Class Extraction
    # =========================================================================

    def _extract_classes(self, root_node, content: str) -> list[dict]:
        """Extract class declarations."""
        classes = []
        
        for node in self._find_nodes_by_type(root_node, 'class_declaration'):
            class_info = self._parse_class_node(node, content)
            if class_info:
                classes.append(class_info)
        
        # Class expressions
        for node in self._find_nodes_by_type(root_node, 'class'):
            if node.parent and node.parent.type == 'variable_declarator':
                class_info = self._parse_class_expression(node, content)
                if class_info:
                    classes.append(class_info)
        
        return classes

    def _parse_class_node(self, node, content: str) -> Optional[dict]:
        """Parse a class declaration node."""
        name = None
        extends = None
        methods = []
        properties = []
        
        for child in node.children:
            if child.type == 'identifier':
                name = self._get_node_text(child, content)
            elif child.type == 'class_heritage':
                for subchild in child.children:
                    if subchild.type == 'identifier':
                        extends = self._get_node_text(subchild, content)
            elif child.type == 'class_body':
                methods, properties = self._parse_class_body(child, content)
        
        if name:
            return {
                'name': name,
                'extends': extends,
                'methods': methods,
                'properties': properties,
                **self._get_node_location(node),
            }
        return None

    def _parse_class_expression(self, node, content: str) -> Optional[dict]:
        """Parse a class expression assigned to a variable."""
        # Get name from parent variable_declarator
        name = None
        if node.parent:
            for sibling in node.parent.children:
                if sibling.type == 'identifier':
                    name = self._get_node_text(sibling, content)
                    break
        
        methods = []
        properties = []
        extends = None
        
        for child in node.children:
            if child.type == 'class_heritage':
                for subchild in child.children:
                    if subchild.type == 'identifier':
                        extends = self._get_node_text(subchild, content)
            elif child.type == 'class_body':
                methods, properties = self._parse_class_body(child, content)
        
        if name:
            return {
                'name': name,
                'extends': extends,
                'methods': methods,
                'properties': properties,
                **self._get_node_location(node),
            }
        return None

    def _parse_class_body(self, node, content: str) -> tuple[list, list]:
        """Parse class body for methods and properties."""
        methods = []
        properties = []
        
        for child in node.children:
            if child.type == 'method_definition':
                method = self._parse_method_definition(child, content)
                if method:
                    methods.append(method)
            elif child.type in ('field_definition', 'public_field_definition'):
                prop = self._parse_field_definition(child, content)
                if prop:
                    properties.append(prop)
        
        return methods, properties

    def _parse_method_definition(self, node, content: str) -> Optional[dict]:
        """Parse a method definition."""
        name = None
        params = []
        is_static = False
        is_async = False
        kind = 'method'
        
        text = self._get_node_text(node, content)
        is_static = 'static ' in text[:20]
        is_async = 'async ' in text[:30]
        
        if 'get ' in text[:15]:
            kind = 'getter'
        elif 'set ' in text[:15]:
            kind = 'setter'
        
        for child in node.children:
            if child.type == 'property_identifier':
                name = self._get_node_text(child, content)
            elif child.type == 'formal_parameters':
                params = self._parse_parameters(child, content)
        
        if name:
            return {
                'name': name,
                'kind': kind,
                'params': params,
                'is_static': is_static,
                'is_async': is_async,
                **self._get_node_location(node),
            }
        return None

    def _parse_field_definition(self, node, content: str) -> Optional[dict]:
        """Parse a field definition."""
        name = None
        value = None
        
        for child in node.children:
            if child.type == 'property_identifier':
                name = self._get_node_text(child, content)
            elif name is not None:
                value = self._get_node_text(child, content)
        
        if name:
            return {
                'name': name,
                'value': value,
                **self._get_node_location(node),
            }
        return None

    # =========================================================================
    # React Component Extraction
    # =========================================================================

    def _extract_components(self, root_node, content: str, file_path: str) -> list[dict]:
        """Extract React components (functional and class-based)."""
        components = []
        
        # Check if this looks like a React file
        ext = Path(file_path).suffix.lower()
        has_jsx = ext in ('.jsx', '.tsx')
        has_react_import = 'react' in content.lower()[:500]
        
        if not (has_jsx or has_react_import):
            return components
        
        # Functional components (arrow functions returning JSX)
        for node in self._find_nodes_by_type(root_node, 'variable_declarator'):
            comp = self._extract_arrow_component(node, content)
            if comp:
                components.append(comp)
        
        # Functional components (function declarations returning JSX)
        for node in self._find_nodes_by_type(root_node, 'function_declaration'):
            comp = self._extract_function_component(node, content)
            if comp:
                components.append(comp)
        
        # Class components
        for node in self._find_nodes_by_type(root_node, 'class_declaration'):
            comp = self._extract_class_component(node, content)
            if comp:
                components.append(comp)
        
        return components

    def _extract_arrow_component(self, node, content: str) -> Optional[dict]:
        """Extract a functional React component from an arrow function."""
        name = None
        has_jsx = False
        props = []
        hooks = []
        
        for child in node.children:
            if child.type == 'identifier':
                name = self._get_node_text(child, content)
            elif child.type == 'arrow_function':
                # AST-based JSX check
                has_jsx = self._contains_jsx(child)
                
                # Extract props from parameters
                for subchild in child.children:
                    if subchild.type == 'formal_parameters':
                        props = self._extract_component_props(subchild, content)
                
                # Find hooks used
                hooks = self._find_hooks_in_node(child, content)
        
        if name and has_jsx and name[0].isupper():
            return {
                'name': name,
                'type': 'functional',
                'props': props,
                'hooks': hooks,
                **self._get_node_location(node),
            }
        return None

    def _extract_function_component(self, node, content: str) -> Optional[dict]:
        """Extract a functional React component from a function declaration."""
        name = None
        props = []
        hooks = []

        has_jsx = self._contains_jsx(node)
        
        if not has_jsx:
            return None
        
        for child in node.children:
            if child.type == 'identifier':
                name = self._get_node_text(child, content)
            elif child.type == 'formal_parameters':
                props = self._extract_component_props(child, content)
        
        hooks = self._find_hooks_in_node(node, content)
        
        if name and name[0].isupper():
            return {
                'name': name,
                'type': 'functional',
                'props': props,
                'hooks': hooks,
                **self._get_node_location(node),
            }
        return None

    def _extract_class_component(self, node, content: str) -> Optional[dict]:
        """Extract a class-based React component."""
        name = None
        extends = None
        
        for child in node.children:
            if child.type == 'identifier':
                name = self._get_node_text(child, content)
            elif child.type == 'class_heritage':
                heritage_text = self._get_node_text(child, content)
                if 'Component' in heritage_text or 'PureComponent' in heritage_text:
                    extends = 'React.Component'
        
        if name and extends:
            return {
                'name': name,
                'type': 'class',
                'extends': extends,
                **self._get_node_location(node),
            }
        return None

    def _extract_component_props(self, node, content: str) -> list[dict]:
        """Extract component props from parameters."""
        props = []
        
        for child in node.children:
            if child.type == 'object_pattern':
                # Destructured props
                for subchild in child.children:
                    if subchild.type == 'shorthand_property_identifier_pattern':
                        props.append({
                            'name': self._get_node_text(subchild, content),
                            'type': None,
                        })
                    elif subchild.type == 'pair_pattern':
                        for part in subchild.children:
                            if part.type == 'property_identifier':
                                props.append({
                                    'name': self._get_node_text(part, content),
                                    'type': None,
                                })
                                break
            elif child.type == 'identifier':
                props.append({
                    'name': self._get_node_text(child, content),
                    'type': 'props_object',
                })
        
        return props

    def _find_hooks_in_node(self, node, content: str) -> list[str]:
        """Find React hooks used within a node."""
        hooks: set[str] = set()
        for call in self._find_nodes_by_type(node, 'call_expression'):
            callee = call.child_by_field_name('function') if hasattr(call, 'child_by_field_name') else None
            if callee is None:
                continue
            if callee.type == "identifier":
                name = self._get_node_text(callee, content).strip()
            elif callee.type == "member_expression":
                prop = callee.child_by_field_name('property') if hasattr(callee, 'child_by_field_name') else None
                name = self._get_node_text(prop, content).strip() if prop else ""
            else:
                name = ""
            if name.startswith("use") and len(name) > 3 and name[3:4].isupper():
                hooks.add(name)
        return sorted(hooks)

    # =========================================================================
    # Express.js Route Extraction
    # =========================================================================

    def _collect_express_identifiers(self, root_node, content: str) -> tuple[set[str], set[str]]:
        """Collect likely Express app and router variable names within a file."""
        app_names: set[str] = {"app"}
        router_names: set[str] = {"router"}

        for node in self._find_nodes_by_type(root_node, "variable_declarator"):
            name = None
            value_node = None
            for child in node.children:
                if child.type == "identifier" and name is None:
                    name = self._get_node_text(child, content)
                elif child.type in ("call_expression", "new_expression"):
                    value_node = child

            if not name or not value_node:
                continue

            value_text = self._get_node_text(value_node, content).lower()

            # Express app patterns: const api = express();
            if re.search(r"\bexpress\s*\(\s*\)", value_text):
                app_names.add(name)

            # Router patterns: const r = express.Router(); const r = Router();
            if re.search(r"\bexpress\.router\s*\(", value_text) or re.search(
                r"\brouter\s*\(", value_text
            ):
                router_names.add(name)

        return app_names, router_names

    def _extract_express_routes(self, root_node, content: str, file_path: str) -> list[dict]:
        """Extract Express.js route definitions."""
        routes = []

        app_names, router_names = self._collect_express_identifiers(root_node, content)
        require_map = self._collect_commonjs_requires(content)
        
        # Find method calls like app.get(), router.post(), etc.
        for node in self._find_nodes_by_type(root_node, 'call_expression'):
            route = self._parse_express_route(node, content, app_names, router_names)
            if route:
                routes.append(route)

        # Also capture mount points: app.use('/prefix', router)
        for node in self._find_nodes_by_type(root_node, 'call_expression'):
            mount_route = self._parse_express_use_mount(
                node,
                content,
                app_names,
                router_names,
                require_map=require_map,
                current_file_path=file_path,
            )
            if mount_route:
                routes.append(mount_route)
        
        return routes

    def _parse_express_use_mount(
        self,
        node,
        content: str,
        app_names: set[str],
        router_names: set[str],
        require_map: Optional[dict[str, str]] = None,
        current_file_path: Optional[str] = None,
    ) -> Optional[dict]:
        """Treat app/router .use('/prefix', ...) as a route-like mount (method=USE)."""
        callee = node.child_by_field_name('function') if hasattr(node, 'child_by_field_name') else None
        if callee is None:
            for child in node.children:
                if child.type in ('member_expression', 'subscript_expression', 'identifier'):
                    callee = child
                    break

        if not callee or callee.type != 'member_expression':
            return None

        obj_node = callee.child_by_field_name('object') if hasattr(callee, 'child_by_field_name') else None
        prop_node = callee.child_by_field_name('property') if hasattr(callee, 'child_by_field_name') else None
        prop = (self._get_node_text(prop_node, content) if prop_node else '').lower()
        if prop != 'use':
            return None

        obj_text = (self._get_node_text(obj_node, content) if obj_node else '').strip()
        if not obj_text or obj_text not in (app_names | router_names):
            return None

        base_path = self._extract_first_string_argument(node, content)
        if not base_path:
            return None

        mount_target = None
        mount_target_kind = None
        require_map = require_map or {}

        # Extract the second argument to `.use('/prefix', <target>)`
        args_node = None
        for child in node.children:
            if child.type == 'arguments':
                args_node = child
                break

        if args_node:
            meaningful = [c for c in args_node.children if c.type not in ('(', ')', ',')]
            if len(meaningful) >= 2:
                target_node = meaningful[1]
                if target_node.type == 'identifier':
                    ident = self._get_node_text(target_node, content).strip()
                    mount_target_kind = 'identifier'
                    mount_target = require_map.get(ident)
                    if mount_target:
                        mount_target_kind = 'require'
                elif target_node.type == 'call_expression':
                    # Common: app.use('/x', require('./routes/x'))
                    callee = target_node.child_by_field_name('function') if hasattr(target_node, 'child_by_field_name') else None
                    callee_text = self._get_node_text(callee, content).strip() if callee else ''
                    if callee_text == 'require':
                        module_path = self._extract_first_string_argument(target_node, content)
                        if module_path:
                            mount_target_kind = 'require'
                            mount_target = module_path

        if mount_target:
            mount_target = mount_target.strip().strip('"\'`')
            if mount_target.startswith('./'):
                mount_target = mount_target[2:]

        mount_target_files: list[str] = []
        if mount_target and current_file_path:
            mount_target_files = self._resolve_mount_target_files(current_file_path, mount_target)

        return {
            'method': 'USE',
            'path': base_path,
            'mount_target': mount_target,
            'mount_target_kind': mount_target_kind,
            'mount_target_files': mount_target_files,
            'handlers': [],
            'middleware': [],
            **self._get_node_location(node),
        }

    def _resolve_mount_target_files(self, current_file_path: str, mount_target: str) -> list[str]:
        """Resolve `require('./x')`/`./x` module paths to likely file paths (best-effort)."""
        try:
            current = Path(current_file_path)
            base_dir = current.parent
            raw = mount_target.replace('\\', '/').lstrip('./')
            p = (base_dir / raw).as_posix()
            # Normalize any leading './'
            if p.startswith('./'):
                p = p[2:]

            candidates: list[str] = []
            if re.search(r"\.(js|cjs|mjs|ts|tsx|jsx)$", p):
                candidates.append(p)
            else:
                candidates.append(f"{p}.js")
                candidates.append(f"{p}.cjs")
                candidates.append(f"{p}.mjs")
                candidates.append(f"{p}/index.js")
                candidates.append(f"{p}/index.cjs")
                candidates.append(f"{p}/index.mjs")

            # Deduplicate while preserving order
            seen: set[str] = set()
            out: list[str] = []
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    out.append(c)
            return out
        except Exception:
            return []

    def _collect_commonjs_requires(self, content: str) -> dict[str, str]:
        """Collect `const X = require('./path')` assignments (best-effort)."""
        require_map: dict[str, str] = {}

        # Covers common patterns:
        #   const foo = require('./routes/foo')
        #   let foo = require("./routes/foo")
        #   var foo = require(`./routes/foo`)
        pattern = re.compile(
            r"\b(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*require\(\s*(?P<q>['\"`])(?P<path>[^'\"`]+)(?P=q)\s*\)",
            re.MULTILINE,
        )
        for m in pattern.finditer(content):
            name = m.group('name')
            path = m.group('path')
            if not name or not path:
                continue
            if path.startswith('./'):
                path = path[2:]
            require_map[name] = path

        return require_map

    def _parse_express_route(
        self,
        node,
        content: str,
        app_names: set[str],
        router_names: set[str],
    ) -> Optional[dict]:
        """Parse an Express route call expression."""
        http_methods = ['get', 'post', 'put', 'patch', 'delete', 'options', 'head', 'all']

        callee = node.child_by_field_name('function') if hasattr(node, 'child_by_field_name') else None
        if callee is None:
            for child in node.children:
                if child.type in ('member_expression', 'subscript_expression', 'identifier'):
                    callee = child
                    break

        if callee is None:
            return None

        # Handle direct member calls: app.get('/x', ...), myRouter.post('/y', ...)
        if callee.type == 'member_expression':
            obj_node = callee.child_by_field_name('object') if hasattr(callee, 'child_by_field_name') else None
            prop_node = callee.child_by_field_name('property') if hasattr(callee, 'child_by_field_name') else None
            if obj_node is None or prop_node is None:
                # Fallback: first significant child is object, last is property
                significant = [c for c in callee.children if c.type not in ('.',)]
                if len(significant) >= 2:
                    obj_node, prop_node = significant[0], significant[-1]

            method = (self._get_node_text(prop_node, content) if prop_node else '').lower()
            if method not in http_methods:
                return None

            obj_text = (self._get_node_text(obj_node, content) if obj_node else '').strip()

            # Direct: identifier.get(...)
            if obj_node and obj_node.type == 'identifier' and obj_text in (app_names | router_names):
                return self._extract_route_details(node, content, method.upper())

            # Chained: app.route('/x').get(...), router.route('/x').post(...)
            if obj_node and obj_node.type == 'call_expression':
                inner = obj_node
                inner_callee = inner.child_by_field_name('function') if hasattr(inner, 'child_by_field_name') else None
                if inner_callee is None:
                    for ch in inner.children:
                        if ch.type in ('member_expression', 'identifier'):
                            inner_callee = ch
                            break

                if inner_callee and inner_callee.type == 'member_expression':
                    inner_obj = inner_callee.child_by_field_name('object') if hasattr(inner_callee, 'child_by_field_name') else None
                    inner_prop = inner_callee.child_by_field_name('property') if hasattr(inner_callee, 'child_by_field_name') else None

                    inner_prop_text = (self._get_node_text(inner_prop, content) if inner_prop else '').lower()
                    inner_obj_text = (self._get_node_text(inner_obj, content) if inner_obj else '').strip()

                    if inner_prop_text == 'route' and inner_obj and inner_obj.type == 'identifier' and inner_obj_text in (app_names | router_names):
                        base_path = self._extract_first_string_argument(inner, content)
                        if base_path:
                            return self._extract_route_details(node, content, method.upper(), path_override=base_path)

        return None

    def _extract_first_string_argument(self, call_expr, content: str) -> Optional[str]:
        """Extract first string/template_string argument from a call_expression."""
        args_node = None
        for child in call_expr.children:
            if child.type == 'arguments':
                args_node = child
                break

        if not args_node:
            return None

        for child in args_node.children:
            if child.type in ('(', ')', ','):
                continue
            if child.type == 'string':
                return self._get_node_text(child, content).strip('"\'`')
            if child.type == 'template_string':
                return self._get_node_text(child, content)

        return None

    def _extract_route_details(
        self,
        node,
        content: str,
        method: str,
        path_override: Optional[str] = None,
    ) -> Optional[dict]:
        """Extract route path and handler details."""
        path = path_override
        handlers = []
        middleware = []
        
        args_node = None
        for child in node.children:
            if child.type == 'arguments':
                args_node = child
                break
        
        if args_node:
            arg_index = 0
            for child in args_node.children:
                if child.type in ('(', ')', ','):
                    continue
                if child.type == 'string':
                    if path is None:
                        path = self._get_node_text(child, content).strip('"\'`')
                elif child.type == 'template_string':
                    if path is None:
                        path = self._get_node_text(child, content)
                elif child.type in ('identifier', 'arrow_function', 'function_expression', 'member_expression'):
                    handler_text = self._get_node_text(child, content)
                    if arg_index == 0 and path is None:
                        # This might be a path variable
                        path = f'${{{handler_text}}}'
                    elif child.type in ('arrow_function', 'function_expression'):
                        handlers.append({
                            'type': 'inline',
                            'code': handler_text[:200],  # Truncate long handlers
                        })
                    else:
                        handlers.append({
                            'type': 'reference',
                            'name': handler_text,
                        })
                elif child.type == 'array':
                    # Array of middleware/handlers
                    for elem in child.children:
                        if elem.type == 'identifier':
                            middleware.append(self._get_node_text(elem, content))
                arg_index += 1
        
        if path:
            return {
                'method': method,
                'path': path,
                'handlers': handlers,
                'middleware': middleware,
                **self._get_node_location(node),
            }
        return None

    # =========================================================================
    # Middleware Extraction
    # =========================================================================

    def _extract_middleware(self, root_node, content: str) -> list[dict]:
        """Extract Express.js middleware definitions."""
        middleware = []
        
        for node in self._find_nodes_by_type(root_node, 'call_expression'):
            mw = self._parse_middleware_use(node, content)
            if mw:
                middleware.append(mw)
        
        return middleware

    def _parse_middleware_use(self, node, content: str) -> Optional[dict]:
        """Parse app.use() middleware calls."""
        callee = node.child_by_field_name('function') if hasattr(node, 'child_by_field_name') else None
        if callee is None or callee.type != 'member_expression':
            return None

        obj_node = callee.child_by_field_name('object') if hasattr(callee, 'child_by_field_name') else None
        prop_node = callee.child_by_field_name('property') if hasattr(callee, 'child_by_field_name') else None
        prop = (self._get_node_text(prop_node, content) if prop_node else '').lower()
        obj_text = (self._get_node_text(obj_node, content) if obj_node else '').strip()
        if prop != 'use' or obj_text not in ('app', 'router', 'Router'):
            return None
        
        path = None
        middlewares = []
        
        for child in node.children:
            if child.type == 'arguments':
                for arg in child.children:
                    if arg.type == 'string':
                        path = self._get_node_text(arg, content).strip('"\'')
                    elif arg.type == 'identifier':
                        middlewares.append(self._get_node_text(arg, content))
                    elif arg.type == 'call_expression':
                        middlewares.append(self._get_node_text(arg, content))
                    elif arg.type in ('arrow_function', 'function_expression'):
                        middlewares.append('inline_middleware')
        
        if middlewares:
            return {
                'path': path,
                'middlewares': middlewares,
                **self._get_node_location(node),
            }
        return None

    def _contains_jsx(self, node) -> bool:
        """AST-level JSX presence check (JSX/TSX)."""
        jsx_nodes = (
            'jsx_element',
            'jsx_self_closing_element',
            'jsx_fragment',
        )
        for jsx_type in jsx_nodes:
            if self._find_nodes_by_type(node, jsx_type):
                return True
        return False

    # =========================================================================
    # MongoDB/Mongoose Model Extraction
    # =========================================================================

    def _extract_models(self, root_node, content: str) -> list[dict]:
        """Extract MongoDB/Mongoose model definitions."""
        models = []
        
        # Find mongoose.model() calls
        for node in self._find_nodes_by_type(root_node, 'call_expression'):
            model = self._parse_mongoose_model(node, content)
            if model:
                models.append(model)
        
        # Find Schema definitions
        for node in self._find_nodes_by_type(root_node, 'new_expression'):
            schema = self._parse_mongoose_schema(node, content)
            if schema:
                models.append(schema)
        
        return models

    def _parse_mongoose_model(self, node, content: str) -> Optional[dict]:
        """Parse mongoose.model() call."""
        text = self._get_node_text(node, content)
        
        if 'mongoose.model' not in text and '.model(' not in text:
            return None
        
        model_name = None
        
        for child in node.children:
            if child.type == 'arguments':
                for arg in child.children:
                    if arg.type == 'string':
                        model_name = self._get_node_text(arg, content).strip('"\'')
                        break
        
        if model_name:
            return {
                'type': 'mongoose_model',
                'name': model_name,
                **self._get_node_location(node),
            }
        return None

    def _parse_mongoose_schema(self, node, content: str) -> Optional[dict]:
        """Parse new Schema() or new mongoose.Schema() calls."""
        text = self._get_node_text(node, content)
        
        if 'Schema' not in text:
            return None
        
        # Try to find the variable name this schema is assigned to
        parent = node.parent
        schema_name = None
        
        while parent:
            if parent.type == 'variable_declarator':
                for child in parent.children:
                    if child.type == 'identifier':
                        schema_name = self._get_node_text(child, content)
                        break
                break
            parent = parent.parent
        
        if schema_name:
            return {
                'type': 'mongoose_schema',
                'name': schema_name,
                **self._get_node_location(node),
            }
        return None

    # =========================================================================
    # React Hooks Extraction
    # =========================================================================

    def _extract_hooks(self, root_node, content: str) -> list[dict]:
        """Extract custom React hooks."""
        hooks = []
        
        # Find functions starting with 'use'
        for node in self._find_nodes_by_type(root_node, 'function_declaration'):
            name = None
            for child in node.children:
                if child.type == 'identifier':
                    name = self._get_node_text(child, content)
                    break
            
            if name and name.startswith('use') and name[3:4].isupper():
                hooks.append({
                    'name': name,
                    'type': 'custom_hook',
                    **self._get_node_location(node),
                })
        
        # Arrow function hooks
        for node in self._find_nodes_by_type(root_node, 'variable_declarator'):
            name = None
            is_function = False
            
            for child in node.children:
                if child.type == 'identifier':
                    name = self._get_node_text(child, content)
                elif child.type in ('arrow_function', 'function_expression'):
                    is_function = True
            
            if name and is_function and name.startswith('use') and name[3:4].isupper():
                hooks.append({
                    'name': name,
                    'type': 'custom_hook',
                    **self._get_node_location(node),
                })
        
        return hooks

    # =========================================================================
    # API Call Extraction
    # =========================================================================

    def _extract_api_calls(self, root_node, content: str) -> list[dict]:
        """Extract API calls (fetch, axios, etc.)."""
        api_calls = []
        
        for node in self._find_nodes_by_type(root_node, 'call_expression'):
            api_call = self._parse_api_call(node, content)
            if api_call:
                api_calls.append(api_call)
        
        return api_calls

    def _parse_api_call(self, node, content: str) -> Optional[dict]:
        """Parse fetch or axios API calls."""
        text = self._get_node_text(node, content)
        
        # Check for common API call patterns
        api_patterns = [
            (r'\bfetch\s*\(', 'fetch'),
            (r'\baxios\.(get|post|put|patch|delete|request)\s*\(', 'axios'),
            (r'\baxios\s*\(', 'axios'),
            (r'\b\$\.(ajax|get|post)\s*\(', 'jquery'),
            (r'\bhttp\.(get|post|put|patch|delete)\s*\(', 'angular_http'),
        ]
        
        for pattern, api_type in api_patterns:
            if re.search(pattern, text):
                url = None
                method = 'GET'
                
                # Try to extract URL
                for child in node.children:
                    if child.type == 'arguments':
                        for arg in child.children:
                            if arg.type == 'string':
                                url = self._get_node_text(arg, content).strip('"\'`')
                                break
                            elif arg.type == 'template_string':
                                url = self._get_node_text(arg, content)
                                break
                
                # Try to extract method from axios calls
                method_match = re.search(r'\.(get|post|put|patch|delete)\s*\(', text, re.I)
                if method_match:
                    method = method_match.group(1).upper()
                
                return {
                    'type': api_type,
                    'method': method,
                    'url': url,
                    **self._get_node_location(node),
                }
        
        return None

    # =========================================================================
    # Variable Extraction
    # =========================================================================

    def _extract_variables(self, root_node, content: str) -> list[dict]:
        """Extract important variable declarations."""
        variables = []
        
        for node in self._find_nodes_by_type(root_node, 'variable_declaration'):
            var_kind = 'let'
            
            for child in node.children:
                if child.type in ('const', 'let', 'var'):
                    var_kind = child.type
            
            for child in node.children:
                if child.type == 'variable_declarator':
                    var = self._parse_variable_declarator(child, content, var_kind)
                    if var:
                        variables.append(var)
        
        return variables

    def _parse_variable_declarator(self, node, content: str, kind: str) -> Optional[dict]:
        """Parse a variable declarator."""
        name = None
        value_type = None
        
        for child in node.children:
            if child.type == 'identifier':
                name = self._get_node_text(child, content)
            elif child.type in ('arrow_function', 'function_expression'):
                return None  # Skip functions, handled elsewhere
            elif child.type == 'object':
                value_type = 'object'
            elif child.type == 'array':
                value_type = 'array'
            elif child.type == 'string':
                value_type = 'string'
            elif child.type == 'number':
                value_type = 'number'
            elif child.type == 'true' or child.type == 'false':
                value_type = 'boolean'
            elif child.type == 'call_expression':
                value_type = 'call_result'
            elif child.type == 'new_expression':
                value_type = 'instance'
        
        if name and value_type:
            return {
                'name': name,
                'kind': kind,
                'value_type': value_type,
                **self._get_node_location(node),
            }
        return None

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _find_nodes_by_type(self, node, node_type: str) -> list:
        """Recursively find all nodes of a specific type."""
        results = []
        
        if node.type == node_type:
            results.append(node)
        
        for child in node.children:
            results.extend(self._find_nodes_by_type(child, node_type))
        
        return results

    def parse_file(self, file_path: str) -> dict[str, Any]:
        """
        Parse a file by reading its content.
        This method maintains backward compatibility.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            Dictionary containing extracted code elements
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"File not found: {file_path}")
                return {'error': f'File not found: {file_path}'}
            
            content = path.read_text(encoding='utf-8')
            return self.parse(file_path, content)
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return {'error': str(e)}
