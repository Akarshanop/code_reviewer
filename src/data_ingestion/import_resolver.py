"""
Import Resolver for Code Intelligence Platform
Resolves relative imports to absolute file paths and tracks dependencies
"""
import os
import re
from pathlib import Path
from typing import Dict, Set, Optional, List, Tuple
from dataclasses import dataclass
from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ImportInfo:
    """Represents a resolved import"""
    import_statement: str  # Original: "import X from '../services/auth'"
    imported_name: str    # What is imported: "X"
    source_path: str      # Relative path: "../services/auth"
    resolved_path: Optional[str]  # Absolute: "/backend/services/auth.js"
    is_external: bool     # Is it external package like "lodash"?
    export_names: List[str] = None  # What this module exports


class ImportResolver:
    """
    Resolves imports across JavaScript/TypeScript/Python files.
    Handles:
    - Relative paths: ../services, ./utils/helper
    - Index files: ./services → ./services/index.js
    - External packages: express, lodash, react
    - File extensions: resolves .js, .ts, .jsx, .tsx, etc.
    """
    
    # Default entry points per language
    ENTRY_POINTS = {
        'javascript': ['index.js', 'index.ts', 'src/index.js', 'src/main.js'],
        'typescript': ['index.ts', 'src/index.ts', 'src/main.ts'],
        'python': ['__init__.py', 'main.py', 'app.py'],
    }
    
    # Resolvable extensions
    EXTENSIONS = {
        'javascript': ['.js', '.mjs', '.cjs', '.json'],
        'typescript': ['.ts', '.tsx', '.d.ts', '.js', '.json'],
        'python': ['.py'],
    }
    
    def __init__(self, repo_path: str):
        """
        Initialize resolver for a repository.
        
        Args:
            repo_path: Root path of the repository
        """
        self.repo_path = Path(repo_path)
        self.external_packages: Set[str] = set()
        self.resolved_imports: Dict[str, List[ImportInfo]] = {}
        self.file_exports: Dict[str, List[str]] = {}
        
    def resolve_import(self, importing_file: str, import_statement: str, 
                      language: str = 'javascript') -> Optional[ImportInfo]:
        """
        Resolve a single import statement.
        
        Args:
            importing_file: Path of file doing the importing
            import_statement: The import statement text
            language: 'javascript', 'typescript', or 'python'
            
        Returns:
            ImportInfo with resolved path, or None if external/unresolvable
        """
        try:
            # Extract import info from statement
            parsed = self._parse_import_statement(import_statement, language)
            if not parsed:
                return None
                
            imported_name, source_path = parsed
            
            # Check if external package
            if self._is_external_package(source_path):
                self.external_packages.add(source_path)
                return ImportInfo(
                    import_statement=import_statement,
                    imported_name=imported_name,
                    source_path=source_path,
                    resolved_path=None,
                    is_external=True
                )
            
            # Resolve relative path
            resolved = self._resolve_relative_path(
                importing_file, source_path, language
            )
            
            if resolved:
                return ImportInfo(
                    import_statement=import_statement,
                    imported_name=imported_name,
                    source_path=source_path,
                    resolved_path=str(resolved),
                    is_external=False
                )
                
            return None
            
        except Exception as e:
            logger.debug(f"Failed to resolve import: {import_statement}", error=str(e))
            return None
    
    def _parse_import_statement(self, statement: str, language: str) -> Optional[Tuple[str, str]]:
        """
        Parse import statement and extract name and source.
        
        Returns:
            (imported_name, source_path) or None if not parseable
        """
        statement = statement.strip()
        
        if language in ('javascript', 'typescript'):
            # ES6: import X from 'path'
            match = re.search(r"import\s+(?:{[^}]*}|[a-zA-Z_]\w*|\*\s+as\s+\w+)\s+from\s+['\"]([^'\"]+)['\"]", statement)
            if match:
                source = match.group(1)
                # Extract first imported name (simplified)
                name_match = re.search(r"import\s+(?:{[^}]*}|([a-zA-Z_]\w*))", statement)
                name = name_match.group(1) if name_match else 'default'
                return (name, source)
            
            # CommonJS: require('path')
            match = re.search(r"require\(['\"]([^'\"]+)['\"]\)", statement)
            if match:
                source = match.group(1)
                # Get variable name
                var_match = re.search(r"(?:const|var|let)\s+([a-zA-Z_]\w*)", statement)
                name = var_match.group(1) if var_match else 'default'
                return (name, source)
                
        elif language == 'python':
            # from path import X or import path
            match = re.search(r"from\s+([a-zA-Z0-9_./]+)\s+import\s+(\w+)", statement)
            if match:
                return (match.group(2), match.group(1).replace('.', '/'))
            
            match = re.search(r"import\s+([a-zA-Z0-9_./]+)", statement)
            if match:
                source = match.group(1).replace('.', '/')
                name = source.split('/')[-1]
                return (name, source)
        
        return None
    
    def _is_external_package(self, source_path: str) -> bool:
        """Check if source is an external package (not relative path)"""
        # External packages don't start with . or /
        return not source_path.startswith(('.', '/'))
    
    def _resolve_relative_path(self, importing_file: str, source_path: str, 
                               language: str) -> Optional[Path]:
        """
        Resolve relative import path to absolute file path.
        
        Handles:
        - ../services → /path/to/services/index.js
        - ./helper → /path/to/helper.js
        - services/auth → /path/to/services/auth.js
        """
        # Get directory of importing file
        import_dir = Path(importing_file).parent
        
        # Resolve relative path
        resolved = (import_dir / source_path).resolve()
        
        # Try different extensions
        extensions = self.EXTENSIONS.get(language, ['.js', '.ts', '.py'])
        
        # 1. Try exact path
        if resolved.exists():
            return resolved
        
        # 2. Try with extensions
        for ext in extensions:
            if (resolved.parent / f"{resolved.name}{ext}").exists():
                return resolved.parent / f"{resolved.name}{ext}"
        
        # 3. Try index file
        index_name = f"index{extensions[0]}"
        if (resolved / index_name).exists():
            return resolved / index_name
        
        # 4. Try alternative index names for TypeScript
        if language == 'typescript':
            for alt_index in [f"index.ts", f"index.tsx", f"index.d.ts"]:
                if (resolved / alt_index).exists():
                    return resolved / alt_index
        
        return None
    
    def trace_entry_point(self, language: str = 'javascript', 
                         max_depth: int = 10) -> Dict[str, Set[str]]:
        """
        Trace all imports starting from entry point.
        
        Returns:
            Dict mapping each file to its imported modules
        """
        entry_points = self.ENTRY_POINTS.get(language, [])
        
        traced_files: Dict[str, Set[str]] = {}
        queue = []
        visited = set()
        
        # Find entry point
        for ep in entry_points:
            ep_path = self.repo_path / ep
            if ep_path.exists():
                queue.append((str(ep_path), 0))
                logger.info(f"Found entry point: {ep}")
                break
        
        if not queue:
            logger.warning(f"No entry point found for {language}")
            return traced_files
        
        # BFS traversal
        while queue:
            file_path, depth = queue.pop(0)
            
            if file_path in visited or depth > max_depth:
                continue
                
            visited.add(file_path)
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                imports = self._extract_imports_from_content(content, language)
                file_imports = set()
                
                for import_stmt in imports:
                    resolved = self.resolve_import(file_path, import_stmt, language)
                    if resolved and not resolved.is_external:
                        file_imports.add(resolved.resolved_path)
                        # Add to queue for traversal
                        queue.append((resolved.resolved_path, depth + 1))
                
                traced_files[file_path] = file_imports
                
            except Exception as e:
                logger.debug(f"Error tracing {file_path}: {e}")
        
        return traced_files
    
    def _extract_imports_from_content(self, content: str, language: str) -> List[str]:
        """Extract all import statements from file content"""
        imports = []
        
        if language in ('javascript', 'typescript'):
            # ES6 imports
            imports.extend(re.findall(
                r"import\s+(?:{[^}]*}|[a-zA-Z_*]\w*(?:\s+as\s+\w+)?|\*\s+as\s+\w+)\s+from\s+['\"]([^'\"]+)['\"]",
                content
            ))
            
            # CommonJS requires
            requires = re.findall(r"require\(['\"]([^'\"]+)['\"]\)", content)
            imports.extend(requires)
            
        elif language == 'python':
            imports.extend(re.findall(r"from\s+([a-zA-Z0-9_.]+)\s+import", content))
            imports.extend(re.findall(r"import\s+([a-zA-Z0-9_.]+)", content))
        
        return imports
    
    def get_statistics(self) -> Dict:
        """Get statistics about resolved imports"""
        return {
            "external_packages": len(self.external_packages),
            "external_packages_list": sorted(list(self.external_packages)),
            "total_resolved_imports": sum(len(v) for v in self.resolved_imports.values()),
            "files_with_imports": len(self.resolved_imports),
        }
