"""
Import Resolver for Code Intelligence Platform
Resolves relative imports to absolute file paths and tracks dependencies.
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
    import_statement: str
    imported_name: str
    source_path: str
    resolved_path: Optional[str]
    is_external: bool
    export_names: List[str] = None


class ImportResolver:
    """
    Resolves imports across JavaScript/TypeScript/Python files.
    Handles:
    - Relative paths: ../services, ./utils/helper
    - Absolute project paths: src/core/logger (Python), @/utils (TS aliases)
    - Index files: ./services → ./services/index.js
    - External packages: express, lodash, react
    - Dynamic imports: import('./heavy')
    - Re-exports: export { X } from './module'
    """

    ENTRY_POINTS = {
        'javascript': ['index.js', 'index.ts', 'src/index.js', 'src/main.js'],
        'typescript': ['index.ts', 'src/index.ts', 'src/main.ts'],
        'python': ['__init__.py', 'main.py', 'app.py'],
    }

    EXTENSIONS = {
        'javascript': ['.js', '.jsx', '.mjs', '.cjs', '.json'],
        'typescript': ['.ts', '.tsx', '.d.ts', '.js', '.jsx', '.json'],
        'python': ['.py'],
    }

    _ALL_CODE_EXTS = {'.js', '.jsx', '.mjs', '.cjs', '.ts', '.tsx', '.py'}

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.external_packages: Set[str] = set()
        self.resolved_imports: Dict[str, List[ImportInfo]] = {}
        self.file_exports: Dict[str, List[str]] = {}

    # ── Public API ────────────────────────────────────────────────────

    def resolve_import(self, importing_file: str, import_statement: str,
                       language: str = 'javascript') -> Optional[ImportInfo]:
        try:
            parsed = self._parse_import_statement(import_statement, language)
            if not parsed:
                return None
            imported_name, source_path = parsed

            if self._is_external_package(source_path, language):
                self.external_packages.add(source_path)
                return ImportInfo(
                    import_statement=import_statement,
                    imported_name=imported_name,
                    source_path=source_path,
                    resolved_path=None,
                    is_external=True,
                )

            resolved = self._resolve_path(importing_file, source_path, language)
            if resolved:
                return ImportInfo(
                    import_statement=import_statement,
                    imported_name=imported_name,
                    source_path=source_path,
                    resolved_path=str(resolved),
                    is_external=False,
                )
            return None
        except Exception as e:
            logger.debug(f"Failed to resolve import: {import_statement} — {e}")
            return None

    def trace_from_file(self, file_path: str, language: str = 'javascript',
                        max_depth: int = 10) -> Dict[str, Set[str]]:
        """BFS traversal of imports starting from *file_path*."""
        traced_files: Dict[str, Set[str]] = {}
        queue: list[tuple[str, int]] = [(file_path, 0)]
        visited: set[str] = set()

        while queue:
            current_path, depth = queue.pop(0)
            if current_path in visited or depth > max_depth:
                continue
            visited.add(current_path)

            try:
                if not os.path.exists(current_path):
                    continue
                with open(current_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                source_paths = self._extract_imports_from_content(content, language)
                file_imports: set[str] = set()

                for source_path in source_paths:
                    if self._is_external_package(source_path, language):
                        self.external_packages.add(source_path)
                        continue

                    resolved = self._resolve_path(current_path, source_path, language)
                    if resolved and resolved.exists():
                        resolved_str = str(resolved)
                        file_imports.add(resolved_str)
                        queue.append((resolved_str, depth + 1))

                traced_files[current_path] = file_imports
            except Exception as e:
                logger.debug(f"Error tracing {current_path}: {e}")

        return traced_files

    def trace_entry_point(self, language: str = 'javascript',
                          max_depth: int = 10) -> Dict[str, Set[str]]:
        for ep in self.ENTRY_POINTS.get(language, []):
            ep_path = self.repo_path / ep
            if ep_path.exists():
                return self.trace_from_file(str(ep_path), language, max_depth)
        logger.warning(f"No entry point found for {language}")
        return {}

    def get_statistics(self) -> Dict:
        return {
            "external_packages": len(self.external_packages),
            "external_packages_list": sorted(list(self.external_packages)),
            "total_resolved_imports": sum(len(v) for v in self.resolved_imports.values()),
            "files_with_imports": len(self.resolved_imports),
        }

    # ── Import extraction ─────────────────────────────────────────────

    def _extract_imports_from_content(self, content: str, language: str) -> List[str]:
        """Extract import source paths from file content."""
        imports: list[str] = []

        if language in ('javascript', 'typescript'):
            # Static ES6: import X from 'path'
            imports.extend(re.findall(
                r"import\s+(?:{[^}]*}|[a-zA-Z_*]\w*(?:\s+as\s+\w+)?|\*\s+as\s+\w+)\s+from\s+['\"]([^'\"]+)['\"]",
                content,
            ))
            # Re-exports: export { X } from 'path'  /  export * from 'path'
            imports.extend(re.findall(
                r"export\s+(?:{[^}]*}|\*)\s+from\s+['\"]([^'\"]+)['\"]",
                content,
            ))
            # Side-effect imports: import 'path'
            imports.extend(re.findall(
                r"^import\s+['\"]([^'\"]+)['\"];?\s*$",
                content, re.MULTILINE,
            ))
            # CommonJS require
            imports.extend(re.findall(r"require\(['\"]([^'\"]+)['\"]\)", content))
            # Dynamic import()
            imports.extend(re.findall(r"import\(['\"]([^'\"]+)['\"]\)", content))

        elif language == 'python':
            from_imports = re.findall(r"from\s+([a-zA-Z0-9_.]+)\s+import", content)
            plain_imports = re.findall(r"^import\s+([a-zA-Z0-9_.]+)", content, re.MULTILINE)
            for mod in from_imports + plain_imports:
                imports.append(mod.replace('.', '/'))

        return imports

    # ── Statement parsing (for resolve_import) ────────────────────────

    def _parse_import_statement(self, statement: str, language: str) -> Optional[Tuple[str, str]]:
        statement = statement.strip()

        if language in ('javascript', 'typescript'):
            match = re.search(
                r"import\s+(?:{[^}]*}|[a-zA-Z_]\w*|\*\s+as\s+\w+)\s+from\s+['\"]([^'\"]+)['\"]",
                statement,
            )
            if match:
                name_m = re.search(r"import\s+(?:{[^}]*}|([a-zA-Z_]\w*))", statement)
                return (name_m.group(1) if name_m and name_m.group(1) else 'default', match.group(1))
            match = re.search(r"require\(['\"]([^'\"]+)['\"]\)", statement)
            if match:
                var_m = re.search(r"(?:const|var|let)\s+([a-zA-Z_]\w*)", statement)
                return (var_m.group(1) if var_m else 'default', match.group(1))

        elif language == 'python':
            match = re.search(r"from\s+([a-zA-Z0-9_./]+)\s+import\s+(\w+)", statement)
            if match:
                return (match.group(2), match.group(1).replace('.', '/'))
            match = re.search(r"import\s+([a-zA-Z0-9_./]+)", statement)
            if match:
                source = match.group(1).replace('.', '/')
                return (source.split('/')[-1], source)

        return None

    # ── External package detection ────────────────────────────────────

    def _is_external_package(self, source_path: str, language: str = 'javascript') -> bool:
        """Return True if source_path refers to an external (npm / PyPI / stdlib) package."""
        if source_path.startswith(('.', '/')):
            return False

        candidate = self.repo_path / source_path
        extensions = self.EXTENSIONS.get(language, list(self._ALL_CODE_EXTS))

        if candidate.exists() or candidate.is_dir():
            return False
        for ext in extensions:
            if candidate.with_suffix(ext).exists():
                return False
        if (candidate / '__init__.py').exists():
            return False
        for idx in ('index.js', 'index.ts', 'index.tsx'):
            if (candidate / idx).exists():
                return False

        return True

    # ── Path resolution ───────────────────────────────────────────────

    def _resolve_path(self, importing_file: str, source_path: str,
                      language: str) -> Optional[Path]:
        """Try to resolve *source_path* to an existing file.

        Resolution order:
        1. Relative to the importing file's directory (handles ./  ../ )
        2. Relative to the repository root (handles Python's `from src.x import y`)
        """
        extensions = self.EXTENSIONS.get(language, ['.js', '.ts', '.py'])

        # 1. Relative resolution
        import_dir = Path(importing_file).parent
        result = self._try_resolve(import_dir / source_path, extensions, language)
        if result:
            return result

        # 2. Repo-root resolution (critical for Python absolute imports)
        result = self._try_resolve(self.repo_path / source_path, extensions, language)
        if result:
            return result

        return None

    def _try_resolve(self, resolved: Path, extensions: list[str],
                     language: str) -> Optional[Path]:
        """Attempt to resolve a candidate path with extension / index fallback."""
        resolved = resolved.resolve()

        if resolved.is_file():
            return resolved

        for ext in extensions:
            candidate = resolved.parent / f"{resolved.name}{ext}"
            if candidate.exists():
                return candidate

        index_name = f"index{extensions[0]}"
        if (resolved / index_name).exists():
            return resolved / index_name

        if language == 'typescript':
            for alt in ('index.ts', 'index.tsx', 'index.d.ts'):
                if (resolved / alt).exists():
                    return resolved / alt

        if language == 'python':
            if (resolved / '__init__.py').exists():
                return resolved / '__init__.py'

        return None
