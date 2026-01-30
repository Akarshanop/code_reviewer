# ts_react_parser.py - IMPROVED VERSION
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ApiCall:
    url: str
    method: str = 'GET'
    hook: str = ''  # 'useQuery', 'useMutation', 'fetch', 'axios'
    variables: List[str] = None
    component: str = ''

@dataclass
class ComponentInfo:
    name: str
    props: List[Dict[str, str]]
    hooks: List[str]
    api_calls: List[ApiCall]

class TsReactParser:
    """Enhanced TypeScript/React parser for modern patterns"""
    
    def __init__(self):
        from tree_sitter_languages import get_language, get_parser
        self.language = get_language('typescript')
        self.parser = get_parser('typescript')
        self.api_patterns = {
            'fetch': self._extract_fetch_call,
            'axios': self._extract_axios_call,
            'useQuery': self._extract_tanstack_query,
            'useMutation': self._extract_tanstack_mutation,
        }
    
    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse a TypeScript/React file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        tree = self.parser.parse(bytes(code, 'utf-8'))
        
        results = {
            'file_path': str(file_path),
            'components': [],
            'api_calls': [],
            'imports': self._extract_imports(tree.root_node, code),
            'hooks_used': [],
            'state_management': self._detect_state_management(code)
        }
        
        # Extract components
        self._extract_components(tree.root_node, code, results)
        
        # Extract standalone API calls
        self._extract_api_calls(tree.root_node, code, results)
        
        logger.info(
            f"Parsed React file: {file_path} "
            f"(components: {len(results['components'])}, "
            f"api_calls: {len(results['api_calls'])})"
        )
        
        return results
    
    def _extract_components(self, node, code: str, results: Dict):
        """Extract React components from the AST"""
        
        # Look for function components
        if node.type == 'function_declaration':
            comp_info = self._extract_function_component(node, code)
            if comp_info:
                results['components'].append(comp_info)
        
        # Look for arrow function components
        elif node.type == 'variable_declaration':
            comp_info = self._extract_arrow_component(node, code)
            if comp_info:
                results['components'].append(comp_info)
        
        # Recursively process children
        for child in node.children:
            self._extract_components(child, code, results)
    
    def _extract_function_component(self, node, code: str) -> Optional[ComponentInfo]:
        """Extract information from a function component"""
        
        # Get function name
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None
        
        component_name = code[name_node.start_byte:name_node.end_byte]
        
        # Check if it's a React component (returns JSX)
        body_node = node.child_by_field_name('body')
        if not body_node:
            return None
        
        body_text = code[body_node.start_byte:body_node.end_byte]
        
        # Look for JSX return
        if 'return' in body_text and ('<' in body_text or 'React.createElement' in body_text):
            props = self._extract_props(node, code)
            hooks = self._extract_hooks(body_text)
            api_calls = self._extract_api_calls_from_body(body_text, component_name)
            
            return ComponentInfo(
                name=component_name,
                props=props,
                hooks=hooks,
                api_calls=api_calls
            )
        
        return None
    
    def _extract_props(self, node, code: str) -> List[Dict[str, str]]:
        """Extract component props"""
        props = []
        
        # Look for parameters
        parameters_node = node.child_by_field_name('parameters')
        if parameters_node:
            param_text = code[parameters_node.start_byte:parameters_node.end_byte]
            # Extract destructured props like { userId, name }
            destructured = re.findall(r'{([^}]+)}', param_text)
            if destructured:
                for props_str in destructured:
                    for prop in props_str.split(','):
                        prop = prop.strip()
                        if prop and ':' in prop:
                            name, type_ = prop.split(':', 1)
                            props.append({'name': name.strip(), 'type': type_.strip()})
                        elif prop:
                            props.append({'name': prop, 'type': 'any'})
        
        return props
    
    def _extract_hooks(self, body_text: str) -> List[str]:
        """Extract React hooks used"""
        hooks = []
        hook_patterns = [
            r'useState\s*\(',
            r'useEffect\s*\(',
            r'useContext\s*\(',
            r'useReducer\s*\(',
            r'useCallback\s*\(',
            r'useMemo\s*\(',
            r'useRef\s*\(',
            r'useQuery\s*\(',
            r'useMutation\s*\(',
        ]
        
        for pattern in hook_patterns:
            if re.search(pattern, body_text):
                hook_name = pattern.replace(r'\s*\(', '').replace(r'\\', '')
                hooks.append(hook_name)
        
        return hooks
    
    def _extract_api_calls_from_body(self, body_text: str, component_name: str) -> List[ApiCall]:
        """Extract API calls from component body"""
        api_calls = []
        
        # Look for fetch calls
        fetch_matches = re.finditer(r'fetch\s*\(\s*["\']([^"\']+)["\']', body_text)
        for match in fetch_matches:
            api_calls.append(ApiCall(
                url=match.group(1),
                method=self._extract_fetch_method(body_text, match.start()),
                hook='fetch',
                component=component_name
            ))
        
        # Look for axios calls
        axios_matches = re.finditer(r'axios\.(get|post|put|delete)\s*\(\s*["\']([^"\']+)["\']', body_text)
        for match in axios_matches:
            api_calls.append(ApiCall(
                url=match.group(2),
                method=match.group(1).upper(),
                hook='axios',
                component=component_name
            ))
        
        # Look for TanStack Query
        query_matches = re.finditer(r'useQuery\s*\(\s*\{[^}]*key:\s*\[\s*["\']([^"\']+)["\']', body_text)
        for match in query_matches:
            api_calls.append(ApiCall(
                url=match.group(1),
                method='GET',
                hook='useQuery',
                component=component_name
            ))
        
        return api_calls
    
    def _extract_fetch_method(self, text: str, start_idx: int) -> str:
        """Extract HTTP method from fetch call"""
        # Default to GET
        method = 'GET'
        
        # Look for method in fetch options
        substr = text[start_idx:start_idx + 100]  # Look ahead 100 chars
        method_match = re.search(r'method\s*:\s*["\'](\w+)["\']', substr)
        if method_match:
            method = method_match.group(1).upper()
        
        return method
    
    def _detect_state_management(self, code: str) -> List[str]:
        """Detect state management libraries used"""
        libraries = []
        
        if 'redux' in code.lower() or 'createSlice' in code:
            libraries.append('redux')
        if 'mobx' in code.lower() or '@observer' in code:
            libraries.append('mobx')
        if 'recoil' in code.lower() or 'RecoilRoot' in code:
            libraries.append('recoil')
        if 'zustand' in code.lower():
            libraries.append('zustand')
        if 'context' in code.lower() and 'createContext' in code:
            libraries.append('context')
        
        return libraries
    
    def _extract_imports(self, root, code: str) -> List[str]:
        """Extract import statements"""
        imports = []
        for child in root.children:
            if child.type == 'import_statement':
                import_text = code[child.start_byte:child.end_byte]
                imports.append(import_text.strip())
        return imports