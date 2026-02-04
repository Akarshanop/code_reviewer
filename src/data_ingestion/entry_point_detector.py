"""
Smart Entry Point Detector for Code Intelligence Platform

Auto-detects entry points by:
1. Reading package.json "main" field (Node.js)
2. Reading pyproject.toml or setup.py (Python)
3. Falling back to common naming conventions
4. Allowing .env override for edge cases

Supports: JavaScript, TypeScript, Python, Java
"""
import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EntryPoint:
    """Represents a detected entry point"""
    file_path: str          # Absolute path to entry file
    entry_type: str         # 'main', 'routes', 'app', 'server', etc.
    confidence: float       # 0.0-1.0 how confident we are
    detection_method: str   # How we found it: 'package.json', 'convention', 'env'
    language: str           # 'javascript', 'typescript', 'python', 'java'


class EntryPointDetector:
    """
    Automatically detects entry points for any codebase.
    Follows the principle: "Convention over Configuration"
    """
    
    # Common entry point patterns by language
    ENTRY_PATTERNS = {
        'javascript': [
            # Node.js server patterns
            ('index.js', 'main', 0.9),
            ('app.js', 'app', 0.9),
            ('server.js', 'server', 0.95),
            ('main.js', 'main', 0.85),
            ('src/index.js', 'main', 0.9),
            ('src/app.js', 'app', 0.9),
            ('src/server.js', 'server', 0.95),
            # Express specific
            ('bin/www', 'server', 0.8),
            # React/Frontend patterns
            ('src/index.jsx', 'frontend', 0.9),
            ('src/main.jsx', 'frontend', 0.9),
            ('src/App.jsx', 'app', 0.85),
        ],
        'typescript': [
            ('index.ts', 'main', 0.9),
            ('app.ts', 'app', 0.9),
            ('server.ts', 'server', 0.95),
            ('main.ts', 'main', 0.85),
            ('src/index.ts', 'main', 0.9),
            ('src/index.tsx', 'frontend', 0.9),
            ('src/main.tsx', 'frontend', 0.9),
            ('src/App.tsx', 'app', 0.85),
        ],
        'python': [
            ('main.py', 'main', 0.9),
            ('app.py', 'app', 0.95),
            ('__main__.py', 'main', 0.9),
            ('run.py', 'server', 0.85),
            ('manage.py', 'django', 0.95),
            ('wsgi.py', 'server', 0.8),
            ('asgi.py', 'server', 0.8),
            ('src/main.py', 'main', 0.9),
            ('src/app.py', 'app', 0.9),
        ],
        'java': [
            ('src/main/java/**/Application.java', 'spring', 0.95),
            ('src/main/java/**/Main.java', 'main', 0.9),
            ('src/main/java/**/App.java', 'app', 0.85),
        ],
    }
    
    # Route/Controller patterns to find after entry point
    ROUTE_PATTERNS = {
        'javascript': [
            'routes/**/*.js',
            'router/**/*.js',
            'api/**/*.js',
            'controllers/**/*.js',
        ],
        'typescript': [
            'routes/**/*.ts',
            'router/**/*.ts',
            'api/**/*.ts',
            'controllers/**/*.ts',
        ],
        'python': [
            'routes/**/*.py',
            'api/**/*.py',
            'views/**/*.py',
            'endpoints/**/*.py',
        ],
    }
    
    def __init__(self, repo_path: str):
        """
        Initialize detector for a repository.
        
        Args:
            repo_path: Absolute path to repository root
        """
        self.repo_path = Path(repo_path)
        self.detected_entries: List[EntryPoint] = []
        self.language = self._detect_primary_language()
        
    def _detect_primary_language(self) -> str:
        """Detect primary language from project files"""
        if (self.repo_path / 'package.json').exists():
            if (self.repo_path / 'tsconfig.json').exists():
                return 'typescript'
            return 'javascript'
        
        if any((self.repo_path / f).exists() for f in ['pyproject.toml', 'setup.py', 'requirements.txt']):
            return 'python'
        
        if (self.repo_path / 'pom.xml').exists() or (self.repo_path / 'build.gradle').exists():
            return 'java'
        
        return 'javascript'
    
    def detect_entry_points(self) -> List[EntryPoint]:
        """
        Detect all entry points for the repository.
        
        Returns:
            List of EntryPoint objects sorted by confidence
        """
        self.detected_entries = []
        
        # 1. Check for .env override first
        env_entries = self._check_env_override()
        if env_entries:
            self.detected_entries.extend(env_entries)
            logger.info(f"Found {len(env_entries)} entry points from .env")
        
        # 2. Check package.json / pyproject.toml
        config_entries = self._check_config_files()
        if config_entries:
            self.detected_entries.extend(config_entries)
            logger.info(f"Found {len(config_entries)} entry points from config files")
        
        # 3. Check common patterns
        pattern_entries = self._check_common_patterns()
        if pattern_entries:
            # Only add if not already found
            existing_paths = {e.file_path for e in self.detected_entries}
            for entry in pattern_entries:
                if entry.file_path not in existing_paths:
                    self.detected_entries.append(entry)
        
        # Sort by confidence (highest first)
        self.detected_entries.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(
            f"Detected {len(self.detected_entries)} entry points for {self.language} project",
            entries=[e.file_path for e in self.detected_entries[:5]]
        )
        
        return self.detected_entries
    
    def _check_env_override(self) -> List[EntryPoint]:
        """Check .env for explicit entry point configuration"""
        entries = []
        env_file = self.repo_path / '.env'
        
        if not env_file.exists():
            return entries
        
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith('ENTRY_POINT='):
                        # Format: ENTRY_POINT=index.js or ENTRY_POINT=src/app.ts
                        entry_path = line.split('=', 1)[1].strip().strip('"\'')
                        full_path = self.repo_path / entry_path
                        
                        if full_path.exists():
                            entries.append(EntryPoint(
                                file_path=str(full_path),
                                entry_type='main',
                                confidence=1.0,  # Explicit config = 100% confidence
                                detection_method='env',
                                language=self.language
                            ))
        except Exception as e:
            logger.debug(f"Error reading .env: {e}")
        
        return entries
    
    def _check_config_files(self) -> List[EntryPoint]:
        """Check package.json, pyproject.toml for entry points"""
        entries = []
        
        # JavaScript/TypeScript: Check package.json
        package_json = self.repo_path / 'package.json'
        if package_json.exists():
            try:
                with open(package_json, 'r') as f:
                    pkg = json.load(f)
                
                # Check "main" field
                if 'main' in pkg:
                    main_path = self.repo_path / pkg['main']
                    if main_path.exists():
                        entries.append(EntryPoint(
                            file_path=str(main_path),
                            entry_type='main',
                            confidence=0.95,
                            detection_method='package.json:main',
                            language=self.language
                        ))
                
                # Check "scripts.start" for entry hint
                if 'scripts' in pkg and 'start' in pkg['scripts']:
                    start_script = pkg['scripts']['start']
                    # Parse "node index.js" or "node src/server.js"
                    if 'node ' in start_script:
                        parts = start_script.split('node ')[-1].split()[0]
                        entry_file = self.repo_path / parts
                        if entry_file.exists() and str(entry_file) not in [e.file_path for e in entries]:
                            entries.append(EntryPoint(
                                file_path=str(entry_file),
                                entry_type='server',
                                confidence=0.9,
                                detection_method='package.json:scripts.start',
                                language=self.language
                            ))
                            
            except Exception as e:
                logger.debug(f"Error parsing package.json: {e}")
        
        # Python: Check pyproject.toml
        pyproject = self.repo_path / 'pyproject.toml'
        if pyproject.exists():
            try:
                import tomllib
                with open(pyproject, 'rb') as f:
                    toml_data = tomllib.load(f)
                
                # Check for entry points in [project.scripts] or [tool.poetry.scripts]
                scripts = toml_data.get('project', {}).get('scripts', {})
                scripts.update(toml_data.get('tool', {}).get('poetry', {}).get('scripts', {}))
                
                for name, entry in scripts.items():
                    # Format: "module:function" or "module.submodule:function"
                    if ':' in entry:
                        module_path = entry.split(':')[0].replace('.', '/') + '.py'
                        full_path = self.repo_path / module_path
                        if full_path.exists():
                            entries.append(EntryPoint(
                                file_path=str(full_path),
                                entry_type=name,
                                confidence=0.95,
                                detection_method='pyproject.toml:scripts',
                                language='python'
                            ))
            except Exception as e:
                logger.debug(f"Error parsing pyproject.toml: {e}")
        
        return entries
    
    def _check_common_patterns(self) -> List[EntryPoint]:
        """Check for common entry point file patterns"""
        entries = []
        patterns = self.ENTRY_PATTERNS.get(self.language, [])
        
        for pattern, entry_type, confidence in patterns:
            # Handle glob patterns
            if '*' in pattern:
                matches = list(self.repo_path.glob(pattern))
                for match in matches:
                    entries.append(EntryPoint(
                        file_path=str(match),
                        entry_type=entry_type,
                        confidence=confidence,
                        detection_method='convention',
                        language=self.language
                    ))
            else:
                # Direct file path
                full_path = self.repo_path / pattern
                if full_path.exists():
                    entries.append(EntryPoint(
                        file_path=str(full_path),
                        entry_type=entry_type,
                        confidence=confidence,
                        detection_method='convention',
                        language=self.language
                    ))
        
        return entries
    
    def get_route_directories(self) -> List[str]:
        """
        Get directories that likely contain routes/controllers.
        These should be parsed with high priority after entry points.
        """
        route_dirs = []
        patterns = self.ROUTE_PATTERNS.get(self.language, [])
        
        for pattern in patterns:
            # Get directory part of pattern
            dir_pattern = pattern.split('/**')[0] if '/**' in pattern else pattern.rsplit('/', 1)[0]
            dir_path = self.repo_path / dir_pattern
            
            if dir_path.exists() and dir_path.is_dir():
                route_dirs.append(str(dir_path))
        
        return route_dirs
    
    def get_parsing_order(self) -> List[str]:
        """
        Get recommended file parsing order for optimal graph construction.
        
        Returns:
            List of file paths in recommended parsing order:
            1. Entry points (index.js, app.js)
            2. Route files
            3. Controller files
            4. Service files
            5. Model files
            6. Utility files
        """
        parsing_order = []
        
        # 1. Entry points first
        for entry in self.detected_entries:
            if entry.file_path not in parsing_order:
                parsing_order.append(entry.file_path)
        
        # 2. Route directories
        route_dirs = self.get_route_directories()
        for route_dir in route_dirs:
            for ext in ['.js', '.ts', '.jsx', '.tsx', '.py']:
                for f in Path(route_dir).rglob(f'*{ext}'):
                    if str(f) not in parsing_order:
                        parsing_order.append(str(f))
        
        # 3. Other important directories in order
        priority_dirs = ['controllers', 'services', 'models', 'middleware', 'utils', 'helpers', 'lib']
        
        for dir_name in priority_dirs:
            dir_path = self.repo_path / dir_name
            if dir_path.exists():
                for ext in ['.js', '.ts', '.jsx', '.tsx', '.py']:
                    for f in dir_path.rglob(f'*{ext}'):
                        if str(f) not in parsing_order:
                            parsing_order.append(str(f))
        
        return parsing_order
    
    def to_dict(self) -> Dict[str, Any]:
        """Export detection results as dictionary"""
        return {
            'repo_path': str(self.repo_path),
            'language': self.language,
            'entry_points': [
                {
                    'file': e.file_path,
                    'type': e.entry_type,
                    'confidence': e.confidence,
                    'method': e.detection_method,
                }
                for e in self.detected_entries
            ],
            'route_directories': self.get_route_directories(),
            'parsing_order_count': len(self.get_parsing_order()),
        }
