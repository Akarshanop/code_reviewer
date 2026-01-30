import logging
from tree_sitter import Language, Parser
from tree_sitter_languages import get_language, get_parser
from pathlib import Path

logger = logging.getLogger(__name__)

class TsReactParser:
    def __init__(self):
        self.language = get_language('javascript')  # Works for TS with JSX
        self.parser = get_parser('javascript')

    def parse_file(self, file_path: Path) -> dict:
        with open(file_path, 'r') as f:
            code = f.read()
        tree = self.parser.parse(bytes(code, 'utf8'))
        # Query for fetch/API calls, components
        query = self.language.query("""
            (call_expression
                function: (identifier) @func
                arguments: (arguments
                    (string) @url))  # Targets fetch('/api/...')
            (jsx_element
                open_tag: (jsx_opening_element name: (identifier) @component_name))
            """)
        captures = query.captures(tree.root_node)
        results = {
            'api_calls': [],  # e.g., {'url': '/api/users', 'component': 'UserList'}
            'components': []  # e.g., {'name': 'UserProfile', 'uses_dto': 'UserDTO'}
        }
        # Process (pseudo)
        for node, tag in captures:
            if tag == 'func' and node.text.decode() == 'fetch':
                url_node = captures[next(i for i, (n, t) in enumerate(captures) if t == 'url')][0]
                results['api_calls'].append({'url': url_node.text.decode().strip("'\"")})
        logger.info(f"Parsed TS file: {file_path} with {len(results['api_calls'])} calls")
        return results