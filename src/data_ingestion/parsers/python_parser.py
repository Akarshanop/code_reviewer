"""
Python AST Parser using tree-sitter
Extracts classes, functions, decorators, imports, FastAPI/Django/Flask patterns,
function bodies, error handling, call graphs, and control flow.
"""
import ast
import textwrap
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field

from src.core.logger import get_logger

logger = get_logger(__name__)

# Maximum characters stored for a single function/method/class body
_MAX_BODY_CHARS = 8000


@dataclass
class PythonClass:
    """Extracted Python class information"""
    name: str
    file_path: str
    line_number: int
    bases: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    is_dataclass: bool = False
    is_pydantic: bool = False


@dataclass
class PythonFunction:
    """Extracted Python function information"""
    name: str
    file_path: str
    line_number: int
    decorators: List[str] = field(default_factory=list)
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    is_async: bool = False
    is_method: bool = False
    class_name: Optional[str] = None


@dataclass
class APIRoute:
    """Extracted API route (FastAPI, Flask, Django)"""
    path: str
    method: str
    function_name: str
    file_path: str
    line_number: int
    framework: str  # fastapi, flask, django
    decorators: List[str] = field(default_factory=list)
    parameters: List[Dict[str, Any]] = field(default_factory=list)


class PythonParser:
    """
    Python parser using AST for code analysis.
    Supports FastAPI, Django, Flask framework detection.
    """
    
    # Framework detection patterns
    FASTAPI_DECORATORS = {
        'get', 'post', 'put', 'delete', 'patch', 'options', 'head',
        'api_route', 'router.get', 'router.post', 'router.put', 
        'router.delete', 'router.patch', 'app.get', 'app.post',
        'app.put', 'app.delete', 'app.patch'
    }
    
    FLASK_DECORATORS = {
        'route', 'app.route', 'bp.route', 'blueprint.route'
    }
    
    DJANGO_PATTERNS = {
        'path', 'url', 're_path', 'include'
    }
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def parse(self, file_path: str, content: str = None) -> Dict[str, Any]:
        """
        Parse a Python file and extract code elements.
        
        Args:
            file_path: Path to the Python file
            content: Optional pre-loaded content
        
        Returns:
            Dictionary with extracted classes, functions, routes, imports, etc.
        """
        path = Path(file_path)
        
        # Load content if not provided
        if content is None:
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {e}")
                return self._empty_result()
        
        # Store source lines for body extraction
        self._source_lines = content.splitlines(keepends=True)
        self._source = content
        
        # Parse AST
        try:
            tree = ast.parse(content, filename=str(path))
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return self._empty_result()
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return self._empty_result()
        
        result = {
            "file_path": str(path),
            "classes": [],
            "functions": [],
            "routes": [],
            "imports": [],
            "from_imports": [],
            "dependencies": [],
            "decorators": [],
            "constants": [],
            "error_handling": [],
            "control_flow": [],
            "framework": "generic",
            "metadata": {}
        }
        
        # Extract elements
        self._extract_imports(tree, result)
        self._extract_classes(tree, str(path), result, content)
        self._extract_functions(tree, str(path), result, content)
        self._extract_constants(tree, result)
        self._detect_framework(result)
        self._extract_routes(tree, str(path), result)
        self._extract_error_handling(tree, str(path), result)
        self._extract_control_flow(tree, str(path), result)
        
        return result
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            "classes": [],
            "functions": [],
            "routes": [],
            "imports": [],
            "from_imports": [],
            "dependencies": [],
            "decorators": [],
            "constants": [],
            "error_handling": [],
            "control_flow": [],
            "framework": "generic",
            "metadata": {}
        }
    
    def _get_source_segment(self, node: ast.AST) -> str:
        """Extract the source code of an AST node (function/class body)."""
        try:
            return ast.get_source_segment(self._source, node) or ""
        except Exception:
            # Fallback: use line numbers
            try:
                start = node.lineno - 1
                end = getattr(node, 'end_lineno', start + 1)
                return "".join(self._source_lines[start:end])
            except Exception:
                return ""
    
    def _extract_imports(self, tree: ast.Module, result: Dict[str, Any]):
        """Extract import statements"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result["imports"].append({
                        "module": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno
                    })
                    # Top-level package as dependency
                    pkg = alias.name.split('.')[0]
                    if pkg not in result["dependencies"]:
                        result["dependencies"].append(pkg)
                        
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    result["from_imports"].append({
                        "module": module,
                        "name": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno
                    })
                    # Top-level package as dependency
                    if module:
                        pkg = module.split('.')[0]
                        if pkg not in result["dependencies"]:
                            result["dependencies"].append(pkg)
    
    def _extract_classes(self, tree: ast.Module, file_path: str, result: Dict[str, Any],
                         content: str = ""):
        """Extract class definitions with bodies and method details"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Get base classes
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        bases.append(f"{self._get_attr_chain(base)}")
                
                # Get decorators
                decorators = self._get_decorators(node)
                
                # Get methods with details (not just names)
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_body = self._get_source_segment(item)
                        method_calls = self._extract_calls_from_node(item)
                        methods.append({
                            "name": item.name,
                            "is_async": isinstance(item, ast.AsyncFunctionDef),
                            "params": self._get_parameters(item),
                            "return_type": self._get_type_annotation(item.returns) if item.returns else None,
                            "line_number": item.lineno,
                            "end_line": getattr(item, 'end_lineno', None),
                            "body": method_body[:_MAX_BODY_CHARS] if method_body else "",
                            "calls": method_calls,
                            "docstring": ast.get_docstring(item) or "",
                        })
                
                # Get docstring
                docstring = ast.get_docstring(node)
                
                # Get class body
                class_body = self._get_source_segment(node)
                
                # Check for special class types
                is_dataclass = any('dataclass' in d for d in decorators)
                is_pydantic = any(b in ('BaseModel', 'BaseSettings') for b in bases)
                
                class_info = {
                    "name": node.name,
                    "file_path": file_path,
                    "line_number": node.lineno,
                    "end_line": getattr(node, 'end_lineno', None),
                    "bases": bases,
                    "decorators": decorators,
                    "methods": methods,
                    "docstring": docstring[:500] if docstring else None,
                    "body": class_body[:_MAX_BODY_CHARS] if class_body else "",
                    "is_dataclass": is_dataclass,
                    "is_pydantic": is_pydantic,
                    "is_abstract": any('ABC' in b or 'abstract' in b.lower() for b in bases)
                }
                
                result["classes"].append(class_info)
    
    def _extract_functions(self, tree: ast.Module, file_path: str, result: Dict[str, Any],
                           content: str = ""):
        """Extract function definitions with bodies, calls, and error handling"""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip methods (handled in class extraction)
                if self._is_method(node, tree):
                    continue
                
                # Get decorators
                decorators = self._get_decorators(node)
                
                # Get parameters
                parameters = self._get_parameters(node)
                
                # Get return type
                return_type = None
                if node.returns:
                    return_type = self._get_type_annotation(node.returns)
                
                # Get docstring
                docstring = ast.get_docstring(node)
                
                # FIX 1: Get function body
                body = self._get_source_segment(node)
                
                # FIX 4: Extract function calls
                calls = self._extract_calls_from_node(node)
                
                # FIX 2: Extract error handling within this function
                error_handling = self._extract_error_handling_from_node(node)
                
                func_info = {
                    "name": node.name,
                    "file_path": file_path,
                    "line_number": node.lineno,
                    "end_line": getattr(node, 'end_lineno', None),
                    "decorators": decorators,
                    "parameters": parameters,
                    "return_type": return_type,
                    "docstring": docstring[:500] if docstring else None,
                    "body": body[:_MAX_BODY_CHARS] if body else "",
                    "calls": calls,
                    "error_handling": error_handling,
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "is_private": node.name.startswith('_'),
                    "is_dunder": node.name.startswith('__') and node.name.endswith('__')
                }
                
                result["functions"].append(func_info)
                result["decorators"].extend(decorators)
    
    def _extract_constants(self, tree: ast.Module, result: Dict[str, Any]):
        """Extract module-level constants"""
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Check if it's a constant (UPPER_CASE)
                        if target.id.isupper():
                            value = self._get_constant_value(node.value)
                            result["constants"].append({
                                "name": target.id,
                                "value": value,
                                "line": node.lineno
                            })
    
    def _extract_routes(self, tree: ast.Module, file_path: str, result: Dict[str, Any]):
        """Extract API routes (FastAPI, Flask, Django)"""
        framework = result.get("framework", "generic")
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                decorators = self._get_decorators(node)
                
                for dec in node.decorator_list:
                    route_info = self._parse_route_decorator(dec, node, file_path, framework)
                    if route_info:
                        result["routes"].append(route_info)
    
    def _parse_route_decorator(self, decorator: ast.expr, func: ast.AST, 
                               file_path: str, framework: str) -> Optional[Dict[str, Any]]:
        """Parse a route decorator and extract route information"""
        if isinstance(decorator, ast.Call):
            func_name = self._get_decorator_name(decorator.func)
            
            # FastAPI routes
            if framework == "fastapi" or func_name in self.FASTAPI_DECORATORS:
                return self._parse_fastapi_route(decorator, func, file_path)
            
            # Flask routes
            if framework == "flask" or func_name in self.FLASK_DECORATORS:
                return self._parse_flask_route(decorator, func, file_path)
        
        elif isinstance(decorator, ast.Attribute):
            attr_name = decorator.attr.lower()
            
            # FastAPI decorator without call (e.g., @router.get)
            if attr_name in ('get', 'post', 'put', 'delete', 'patch'):
                return {
                    "path": "/",
                    "method": attr_name.upper(),
                    "function_name": func.name,
                    "file_path": file_path,
                    "line_number": func.lineno,
                    "framework": "fastapi"
                }
        
        return None
    
    def _parse_fastapi_route(self, decorator: ast.Call, func: ast.AST, 
                            file_path: str) -> Dict[str, Any]:
        """Parse FastAPI route decorator"""
        func_name = self._get_decorator_name(decorator.func)
        
        # Determine HTTP method
        method_map = {
            'get': 'GET', 'post': 'POST', 'put': 'PUT',
            'delete': 'DELETE', 'patch': 'PATCH',
            'options': 'OPTIONS', 'head': 'HEAD'
        }
        
        method = 'GET'  # default
        for key, val in method_map.items():
            if key in func_name.lower():
                method = val
                break
        
        # Get path from first argument
        path = "/"
        if decorator.args:
            first_arg = decorator.args[0]
            if isinstance(first_arg, ast.Constant):
                path = str(first_arg.value)
        
        # Check for path in keywords
        for kw in decorator.keywords:
            if kw.arg == 'path' and isinstance(kw.value, ast.Constant):
                path = str(kw.value.value)
        
        return {
            "path": path,
            "method": method,
            "function_name": func.name,
            "file_path": file_path,
            "line_number": func.lineno,
            "framework": "fastapi",
            "is_async": isinstance(func, ast.AsyncFunctionDef)
        }
    
    def _parse_flask_route(self, decorator: ast.Call, func: ast.AST, 
                          file_path: str) -> Dict[str, Any]:
        """Parse Flask route decorator"""
        # Get path from first argument
        path = "/"
        if decorator.args:
            first_arg = decorator.args[0]
            if isinstance(first_arg, ast.Constant):
                path = str(first_arg.value)
        
        # Get methods from keyword
        methods = ["GET"]
        for kw in decorator.keywords:
            if kw.arg == 'methods' and isinstance(kw.value, ast.List):
                methods = []
                for elt in kw.value.elts:
                    if isinstance(elt, ast.Constant):
                        methods.append(str(elt.value).upper())
        
        return {
            "path": path,
            "method": methods[0] if len(methods) == 1 else methods,
            "function_name": func.name,
            "file_path": file_path,
            "line_number": func.lineno,
            "framework": "flask"
        }
    
    def _detect_framework(self, result: Dict[str, Any]):
        """Detect the web framework used"""
        imports = [i["module"] for i in result["imports"]]
        from_imports = [i["module"] for i in result["from_imports"]]
        all_imports = imports + from_imports
        
        if any("fastapi" in imp.lower() for imp in all_imports):
            result["framework"] = "fastapi"
        elif any("flask" in imp.lower() for imp in all_imports):
            result["framework"] = "flask"
        elif any("django" in imp.lower() for imp in all_imports):
            result["framework"] = "django"
        elif any("starlette" in imp.lower() for imp in all_imports):
            result["framework"] = "starlette"
        elif any("tornado" in imp.lower() for imp in all_imports):
            result["framework"] = "tornado"
        elif any("aiohttp" in imp.lower() for imp in all_imports):
            result["framework"] = "aiohttp"
    
    def _get_decorators(self, node: ast.AST) -> List[str]:
        """Get decorator names from a node"""
        decorators = []
        for dec in getattr(node, 'decorator_list', []):
            name = self._get_decorator_name(dec)
            if name:
                decorators.append(name)
        return decorators
    
    def _get_decorator_name(self, node: ast.expr) -> str:
        """Get the name of a decorator"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_attr_chain(node)
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        return ""
    
    def _get_attr_chain(self, node: ast.Attribute) -> str:
        """Get the full attribute chain (e.g., 'app.route')"""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return '.'.join(reversed(parts))
    
    def _get_parameters(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract function parameters"""
        params = []
        
        args = node.args
        defaults = args.defaults
        num_defaults = len(defaults)
        num_args = len(args.args)
        
        for i, arg in enumerate(args.args):
            param = {
                "name": arg.arg,
                "type": self._get_type_annotation(arg.annotation) if arg.annotation else None,
                "has_default": i >= (num_args - num_defaults)
            }
            params.append(param)
        
        # *args
        if args.vararg:
            params.append({
                "name": f"*{args.vararg.arg}",
                "type": self._get_type_annotation(args.vararg.annotation) if args.vararg.annotation else None,
                "is_vararg": True
            })
        
        # **kwargs
        if args.kwarg:
            params.append({
                "name": f"**{args.kwarg.arg}",
                "type": self._get_type_annotation(args.kwarg.annotation) if args.kwarg.annotation else None,
                "is_kwarg": True
            })
        
        return params
    
    def _get_type_annotation(self, node: ast.expr) -> Optional[str]:
        """Convert type annotation AST to string"""
        if node is None:
            return None
        
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Subscript):
            value = self._get_type_annotation(node.value)
            slice_val = self._get_type_annotation(node.slice)
            return f"{value}[{slice_val}]"
        elif isinstance(node, ast.Attribute):
            return self._get_attr_chain(node)
        elif isinstance(node, ast.Tuple):
            elements = [self._get_type_annotation(e) for e in node.elts]
            return f"({', '.join(filter(None, elements))})"
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            # Union type (X | Y)
            left = self._get_type_annotation(node.left)
            right = self._get_type_annotation(node.right)
            return f"{left} | {right}"
        
        return None
    
    def _get_constant_value(self, node: ast.expr) -> Any:
        """Get the value of a constant node"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # Python 3.7 compatibility
            return node.n
        elif isinstance(node, ast.Str):  # Python 3.7 compatibility
            return node.s
        elif isinstance(node, ast.List):
            return [self._get_constant_value(e) for e in node.elts]
        elif isinstance(node, ast.Dict):
            return {
                self._get_constant_value(k): self._get_constant_value(v)
                for k, v in zip(node.keys, node.values) if k is not None
            }
        return None
    
    def _is_method(self, func_node: ast.AST, tree: ast.Module) -> bool:
        """Check if a function is a method (inside a class)"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if item is func_node:
                        return True
        return False
    
    # =========================================================================
    # FIX 4: Call graph extraction
    # =========================================================================
    
    def _extract_calls_from_node(self, node: ast.AST) -> List[str]:
        """Extract all function/method call names within an AST node."""
        calls: Set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                name = self._get_call_name(child.func)
                if name:
                    calls.add(name)
        return sorted(calls)
    
    def _get_call_name(self, node: ast.expr) -> Optional[str]:
        """Get the name of a function being called."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_attr_chain(node)
        return None
    
    # =========================================================================
    # FIX 2: Error handling extraction
    # =========================================================================
    
    def _extract_error_handling_from_node(self, node: ast.AST) -> List[Dict[str, Any]]:
        """Extract try/except/finally patterns within a single function/method."""
        handlers = []
        for child in ast.walk(node):
            if isinstance(child, ast.Try):
                handler_info: Dict[str, Any] = {
                    "line": child.lineno,
                    "has_finally": len(child.finalbody) > 0,
                    "has_else": len(child.orelse) > 0,
                    "exceptions_caught": [],
                }
                for h in child.handlers:
                    exc_type = None
                    if h.type:
                        exc_type = self._get_type_annotation(h.type)
                    handler_info["exceptions_caught"].append({
                        "type": exc_type or "Exception",
                        "name": h.name,
                        "line": h.lineno,
                    })
                handlers.append(handler_info)
            # Also capture bare raise statements
            elif isinstance(child, ast.Raise):
                exc_type = None
                if child.exc:
                    if isinstance(child.exc, ast.Call):
                        exc_type = self._get_call_name(child.exc.func)
                    elif isinstance(child.exc, ast.Name):
                        exc_type = child.exc.id
                handlers.append({
                    "line": child.lineno,
                    "type": "raise",
                    "exception": exc_type,
                })
        return handlers
    
    def _extract_error_handling(self, tree: ast.Module, file_path: str, result: Dict[str, Any]):
        """FIX 2: Extract all error handling patterns in the file."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                info: Dict[str, Any] = {
                    "file": file_path,
                    "line": node.lineno,
                    "end_line": getattr(node, 'end_lineno', None),
                    "has_finally": len(node.finalbody) > 0,
                    "has_else": len(node.orelse) > 0,
                    "exceptions_caught": [],
                }
                for h in node.handlers:
                    exc_type = self._get_type_annotation(h.type) if h.type else "Exception"
                    info["exceptions_caught"].append({
                        "type": exc_type,
                        "name": h.name,
                        "line": h.lineno,
                    })
                result["error_handling"].append(info)
    
    # =========================================================================
    # FIX 6: Control flow extraction
    # =========================================================================
    
    def _extract_control_flow(self, tree: ast.Module, file_path: str, result: Dict[str, Any]):
        """FIX 6: Extract control flow structures for QA edge-case generation."""
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Count elif branches
                branch_count = 1  # the if itself
                current = node
                while current.orelse and len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                    branch_count += 1
                    current = current.orelse[0]
                if current.orelse:
                    branch_count += 1  # final else
                
                result["control_flow"].append({
                    "type": "if",
                    "line": node.lineno,
                    "branches": branch_count,
                    "has_else": bool(node.orelse),
                })
            elif isinstance(node, ast.For):
                result["control_flow"].append({
                    "type": "for",
                    "line": node.lineno,
                    "has_else": bool(node.orelse),
                })
            elif isinstance(node, ast.While):
                result["control_flow"].append({
                    "type": "while",
                    "line": node.lineno,
                    "has_else": bool(node.orelse),
                })
            elif hasattr(ast, 'Match') and isinstance(node, ast.Match):
                result["control_flow"].append({
                    "type": "match",
                    "line": node.lineno,
                    "cases": len(node.cases),
                })
            elif isinstance(node, ast.Assert):
                result["control_flow"].append({
                    "type": "assert",
                    "line": node.lineno,
                })
    
    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse a file (convenience method matching other parsers)"""
        return self.parse(str(file_path))


# Singleton instance
python_parser = PythonParser()
