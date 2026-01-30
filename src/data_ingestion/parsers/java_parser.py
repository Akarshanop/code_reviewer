import logging
from tree_sitter import Language, Parser
from tree_sitter_languages import get_language, get_parser 

logger = logging.getLogger(__name__)

class JavaParser:
    def __init__(self):
        self.language = get_language('java') 
        self.parser = get_parser('java')

    def parse_file(self, file_path: Path) -> dict:
        with open(file_path, 'r') as f:
            code = f.read()
        tree = self.parser.parse(bytes(code, 'utf8'))
        query = self.language.query("""
            (annotation
                name: (identifier) @anno_name
                arguments: (annotation_argument_list
                    (string_literal) @arg_value))
            (class_declaration
                name: (identifier) @class_name)
            """)  # Expand for EventBus, DTO fields
        captures = query.captures(tree.root_node)
        results = {
            'routes': [],  # e.g., {'path': '/api/users', 'method': 'GET'}
            'eventbus': [],  # e.g., {'address': 'payment.process', 'type': 'consume'}
            'dtos': []  # e.g., {'class': 'UserDTO', 'fields': ['id', 'name']}
        }
        for node, tag in captures:
            if tag == 'anno_name' and node.text.decode() == 'Route':
                results['routes'].append({'path': captures[next(i for i, (n, t) in enumerate(captures) if t == 'arg_value')][0].text.decode()})
        logger.info(f"Parsed Java file: {file_path} with {len(results['routes'])} routes")
        return results