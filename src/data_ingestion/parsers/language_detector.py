"""
Automatic Language and Parser Detection System
Detects file types by extension and routes to appropriate parsers
"""
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field

from src.core.logger import get_logger

logger = get_logger(__name__)


class LanguageType(str, Enum):
    """Supported language types"""
    JAVA = "java"
    KOTLIN = "kotlin"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    PYTHON = "python"
    GO = "go"
    RUST = "rust"
    CSHARP = "csharp"
    CPP = "cpp"
    C = "c"
    RUBY = "ruby"
    PHP = "php"
    SCALA = "scala"
    SWIFT = "swift"
    # Config/Data files
    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    TOML = "toml"
    PROPERTIES = "properties"
    # Documentation
    MARKDOWN = "markdown"
    HTML = "html"
    CSS = "css"
    SCSS = "scss"
    # Other
    SQL = "sql"
    GRAPHQL = "graphql"
    PROTO = "protobuf"
    DOCKERFILE = "dockerfile"
    SHELL = "shell"
    UNKNOWN = "unknown"


class FrameworkType(str, Enum):
    """Detected framework types"""
    # Java frameworks
    VERTX = "vertx"
    SPRING = "spring"
    QUARKUS = "quarkus"
    MICRONAUT = "micronaut"
    # JavaScript/TypeScript frameworks
    REACT = "react"
    VUE = "vue"
    ANGULAR = "angular"
    NEXTJS = "nextjs"
    NESTJS = "nestjs"
    EXPRESS = "express"
    FASTIFY = "fastify"
    # Python frameworks
    FASTAPI = "fastapi"
    DJANGO = "django"
    FLASK = "flask"
    # Other
    GENERIC = "generic"


@dataclass
class FileAnalysis:
    """Result of file analysis"""
    file_path: Path
    language: LanguageType
    framework: FrameworkType = FrameworkType.GENERIC
    is_test_file: bool = False
    is_config_file: bool = False
    is_generated: bool = False
    is_parseable: bool = True
    parser_type: str = "generic"
    metadata: Dict[str, Any] = field(default_factory=dict)


class LanguageDetector:
    """
    Intelligent language and framework detector.
    Analyzes files by extension and content to determine the best parser.
    """
    
    # Extension to language mapping
    EXTENSION_MAP: Dict[str, LanguageType] = {
        # Java/JVM
        ".java": LanguageType.JAVA,
        ".kt": LanguageType.KOTLIN,
        ".kts": LanguageType.KOTLIN,
        ".scala": LanguageType.SCALA,
        ".groovy": LanguageType.JAVA,  # Treat as Java-like
        
        # JavaScript/TypeScript
        ".js": LanguageType.JAVASCRIPT,
        ".jsx": LanguageType.JAVASCRIPT,
        ".ts": LanguageType.TYPESCRIPT,
        ".tsx": LanguageType.TYPESCRIPT,
        ".mjs": LanguageType.JAVASCRIPT,
        ".cjs": LanguageType.JAVASCRIPT,
        
        # Python
        ".py": LanguageType.PYTHON,
        ".pyi": LanguageType.PYTHON,
        ".pyx": LanguageType.PYTHON,
        
        # Go
        ".go": LanguageType.GO,
        
        # Rust
        ".rs": LanguageType.RUST,
        
        # C/C++
        ".c": LanguageType.C,
        ".h": LanguageType.C,
        ".cpp": LanguageType.CPP,
        ".cxx": LanguageType.CPP,
        ".cc": LanguageType.CPP,
        ".hpp": LanguageType.CPP,
        ".hxx": LanguageType.CPP,
        
        # C#
        ".cs": LanguageType.CSHARP,
        
        # Ruby
        ".rb": LanguageType.RUBY,
        ".rake": LanguageType.RUBY,
        
        # PHP
        ".php": LanguageType.PHP,
        
        # Swift
        ".swift": LanguageType.SWIFT,
        
        # Config/Data
        ".json": LanguageType.JSON,
        ".yaml": LanguageType.YAML,
        ".yml": LanguageType.YAML,
        ".xml": LanguageType.XML,
        ".toml": LanguageType.TOML,
        ".properties": LanguageType.PROPERTIES,
        ".ini": LanguageType.PROPERTIES,
        ".conf": LanguageType.PROPERTIES,
        ".env": LanguageType.PROPERTIES,
        
        # Documentation/Web
        ".md": LanguageType.MARKDOWN,
        ".markdown": LanguageType.MARKDOWN,
        ".html": LanguageType.HTML,
        ".htm": LanguageType.HTML,
        ".css": LanguageType.CSS,
        ".scss": LanguageType.SCSS,
        ".sass": LanguageType.SCSS,
        ".less": LanguageType.CSS,
        
        # Database/API
        ".sql": LanguageType.SQL,
        ".graphql": LanguageType.GRAPHQL,
        ".gql": LanguageType.GRAPHQL,
        ".proto": LanguageType.PROTO,
        
        # Shell/Scripts
        ".sh": LanguageType.SHELL,
        ".bash": LanguageType.SHELL,
        ".zsh": LanguageType.SHELL,
        ".fish": LanguageType.SHELL,
    }
    
    # Special filename detection
    SPECIAL_FILES: Dict[str, Tuple[LanguageType, bool]] = {
        "Dockerfile": (LanguageType.DOCKERFILE, True),
        "docker-compose.yml": (LanguageType.YAML, True),
        "docker-compose.yaml": (LanguageType.YAML, True),
        ".dockerignore": (LanguageType.UNKNOWN, True),
        ".gitignore": (LanguageType.UNKNOWN, True),
        "Makefile": (LanguageType.SHELL, True),
        "Jenkinsfile": (LanguageType.JAVA, True),  # Groovy-based
        "build.gradle": (LanguageType.JAVA, True),
        "build.gradle.kts": (LanguageType.KOTLIN, True),
        "pom.xml": (LanguageType.XML, True),
        "package.json": (LanguageType.JSON, True),
        "tsconfig.json": (LanguageType.JSON, True),
        "webpack.config.js": (LanguageType.JAVASCRIPT, True),
        "vite.config.js": (LanguageType.JAVASCRIPT, True),
        "vite.config.ts": (LanguageType.TYPESCRIPT, True),
        "requirements.txt": (LanguageType.UNKNOWN, True),
        "pyproject.toml": (LanguageType.TOML, True),
        "Cargo.toml": (LanguageType.TOML, True),
        "go.mod": (LanguageType.GO, True),
        "go.sum": (LanguageType.GO, True),
    }
    
    # Test file patterns
    TEST_PATTERNS = [
        "test", "spec", "tests", "specs", "__tests__", "__test__",
        "_test", "_spec", "Test", "Spec"
    ]
    
    # Generated file patterns
    GENERATED_PATTERNS = [
        "generated", "auto-generated", ".gen.", "_gen.", 
        "node_modules", "dist", "build", "target", "__pycache__"
    ]
    
    # Framework detection patterns (content-based)
    FRAMEWORK_PATTERNS: Dict[LanguageType, Dict[str, List[str]]] = {
        LanguageType.JAVA: {
            FrameworkType.VERTX.value: [
                "io.vertx", "vertx-web", "Router", "RoutingContext",
                "@Route", "EventBus", "Verticle"
            ],
            FrameworkType.SPRING.value: [
                "org.springframework", "@RestController", "@Service",
                "@Autowired", "@GetMapping", "@PostMapping"
            ],
            FrameworkType.QUARKUS.value: [
                "io.quarkus", "@Path", "@GET", "@POST", "@Inject"
            ],
            FrameworkType.MICRONAUT.value: [
                "io.micronaut", "@Controller", "@Get", "@Post"
            ],
        },
        LanguageType.TYPESCRIPT: {
            FrameworkType.REACT.value: [
                "react", "useState", "useEffect", "jsx", "tsx",
                "React.FC", "React.Component"
            ],
            FrameworkType.VUE.value: [
                "vue", "defineComponent", "ref(", "reactive(",
                "<template>", "<script setup>"
            ],
            FrameworkType.ANGULAR.value: [
                "@angular", "@Component", "@Injectable", "@NgModule"
            ],
            FrameworkType.NEXTJS.value: [
                "next/", "getServerSideProps", "getStaticProps",
                "useRouter", "next.config"
            ],
            FrameworkType.NESTJS.value: [
                "@nestjs", "@Controller", "@Injectable", "@Module"
            ],
        },
        LanguageType.JAVASCRIPT: {
            FrameworkType.REACT.value: [
                "react", "useState", "useEffect", "jsx"
            ],
            FrameworkType.EXPRESS.value: [
                "express", "app.get", "app.post", "app.use", "Router()"
            ],
            FrameworkType.FASTIFY.value: [
                "fastify", "Fastify()", "fastify.register"
            ],
        },
        LanguageType.PYTHON: {
            FrameworkType.FASTAPI.value: [
                "fastapi", "FastAPI", "@app.get", "@app.post", "Depends"
            ],
            FrameworkType.DJANGO.value: [
                "django", "from django", "models.Model", "views"
            ],
            FrameworkType.FLASK.value: [
                "flask", "Flask", "@app.route", "Blueprint"
            ],
        },
    }
    
    # Parser routing
    PARSER_MAP: Dict[LanguageType, str] = {
        LanguageType.JAVA: "java",
        LanguageType.KOTLIN: "java",  # Similar structure
        LanguageType.TYPESCRIPT: "react",  # React/TS parser
        LanguageType.JAVASCRIPT: "react",  # Same parser
        LanguageType.PYTHON: "python",
        LanguageType.GO: "go",
        LanguageType.JSON: "generic",
        LanguageType.YAML: "generic",
        LanguageType.XML: "generic",
    }
    
    def __init__(self):
        self._cache: Dict[str, FileAnalysis] = {}
    
    def detect(self, file_path: Path, content: Optional[str] = None) -> FileAnalysis:
        """
        Detect language, framework, and appropriate parser for a file.
        
        Args:
            file_path: Path to the file
            content: Optional file content (to avoid re-reading)
        
        Returns:
            FileAnalysis with detection results
        """
        path_str = str(file_path)
        
        # Check cache
        if path_str in self._cache:
            return self._cache[path_str]
        
        # Detect language from extension/filename
        language = self._detect_language(file_path)
        
        # Check if it's a special file type
        is_config = self._is_config_file(file_path, language)
        is_test = self._is_test_file(file_path)
        is_generated = self._is_generated_file(file_path)
        
        # Detect framework from content
        framework = FrameworkType.GENERIC
        if content and language in self.FRAMEWORK_PATTERNS:
            framework = self._detect_framework(content, language)
        
        # Determine parser
        parser_type = self._get_parser_type(language, framework)
        is_parseable = self._is_parseable(language)
        
        analysis = FileAnalysis(
            file_path=file_path,
            language=language,
            framework=framework,
            is_test_file=is_test,
            is_config_file=is_config,
            is_generated=is_generated,
            is_parseable=is_parseable,
            parser_type=parser_type,
            metadata={
                "extension": file_path.suffix.lower(),
                "filename": file_path.name,
            }
        )
        
        # Cache the result
        self._cache[path_str] = analysis
        
        return analysis
    
    def _detect_language(self, file_path: Path) -> LanguageType:
        """Detect language from file extension or special filename"""
        filename = file_path.name
        
        # Check special files first
        if filename in self.SPECIAL_FILES:
            return self.SPECIAL_FILES[filename][0]
        
        # Check extension
        extension = file_path.suffix.lower()
        if extension in self.EXTENSION_MAP:
            return self.EXTENSION_MAP[extension]
        
        return LanguageType.UNKNOWN
    
    def _detect_framework(self, content: str, language: LanguageType) -> FrameworkType:
        """Detect framework from file content"""
        if language not in self.FRAMEWORK_PATTERNS:
            return FrameworkType.GENERIC
        
        patterns = self.FRAMEWORK_PATTERNS[language]
        scores: Dict[str, int] = {}
        
        for framework, keywords in patterns.items():
            score = sum(1 for kw in keywords if kw in content)
            if score > 0:
                scores[framework] = score
        
        if scores:
            best_match = max(scores, key=scores.get)
            return FrameworkType(best_match)
        
        return FrameworkType.GENERIC
    
    def _is_config_file(self, file_path: Path, language: LanguageType) -> bool:
        """Check if the file is a configuration file"""
        filename = file_path.name.lower()
        
        # Check special files
        if file_path.name in self.SPECIAL_FILES:
            return self.SPECIAL_FILES[file_path.name][1]
        
        # Config-like extensions
        config_extensions = {".json", ".yaml", ".yml", ".xml", ".toml", 
                           ".properties", ".ini", ".conf", ".env"}
        if file_path.suffix.lower() in config_extensions:
            return True
        
        # Config-like names
        config_patterns = ["config", "settings", "env", "application", 
                         "properties", ".rc", "webpack", "vite", "babel",
                         "eslint", "prettier", "tsconfig", "jest"]
        return any(p in filename for p in config_patterns)
    
    def _is_test_file(self, file_path: Path) -> bool:
        """Check if the file is a test file"""
        path_str = str(file_path).lower()
        return any(p in path_str for p in self.TEST_PATTERNS)
    
    def _is_generated_file(self, file_path: Path) -> bool:
        """Check if the file is auto-generated"""
        path_str = str(file_path).lower()
        return any(p in path_str for p in self.GENERATED_PATTERNS)
    
    def _get_parser_type(self, language: LanguageType, framework: FrameworkType) -> str:
        """Determine which parser to use"""
        # Special cases based on framework
        if framework == FrameworkType.REACT:
            return "react"
        if framework in (FrameworkType.VERTX, FrameworkType.SPRING):
            return "java"
        
        # Default based on language
        return self.PARSER_MAP.get(language, "generic")
    
    def _is_parseable(self, language: LanguageType) -> bool:
        """Check if we have a parser for this language"""
        parseable_languages = {
            LanguageType.JAVA, LanguageType.KOTLIN,
            LanguageType.TYPESCRIPT, LanguageType.JAVASCRIPT,
            LanguageType.PYTHON,
            LanguageType.JSON, LanguageType.YAML, LanguageType.XML,
        }
        return language in parseable_languages
    
    def analyze_repository(self, repo_path: Path, 
                          exclude_patterns: List[str] = None) -> Dict[str, Any]:
        """
        Analyze all files in a repository and return statistics.
        
        Args:
            repo_path: Path to repository root
            exclude_patterns: Patterns to exclude
        
        Returns:
            Dictionary with repository analysis
        """
        exclude_patterns = exclude_patterns or [
            "**/node_modules/**", "**/target/**", "**/build/**",
            "**/.git/**", "**/dist/**", "**/__pycache__/**",
            "**/vendor/**", "**/.idea/**", "**/.vscode/**"
        ]
        
        stats = {
            "total_files": 0,
            "parseable_files": 0,
            "languages": {},
            "frameworks": {},
            "test_files": 0,
            "config_files": 0,
            "files_by_parser": {},
        }
        
        import fnmatch
        
        for file_path in repo_path.rglob("*"):
            if not file_path.is_file():
                continue
            
            # Check exclusions
            rel_path = str(file_path.relative_to(repo_path))
            if any(fnmatch.fnmatch(rel_path, p) for p in exclude_patterns):
                continue
            
            stats["total_files"] += 1
            
            # Analyze file
            try:
                analysis = self.detect(file_path)
                
                # Update language stats
                lang = analysis.language.value
                stats["languages"][lang] = stats["languages"].get(lang, 0) + 1
                
                # Update framework stats
                if analysis.framework != FrameworkType.GENERIC:
                    fw = analysis.framework.value
                    stats["frameworks"][fw] = stats["frameworks"].get(fw, 0) + 1
                
                # Update parser stats
                parser = analysis.parser_type
                stats["files_by_parser"][parser] = stats["files_by_parser"].get(parser, 0) + 1
                
                if analysis.is_parseable:
                    stats["parseable_files"] += 1
                if analysis.is_test_file:
                    stats["test_files"] += 1
                if analysis.is_config_file:
                    stats["config_files"] += 1
                    
            except Exception as e:
                logger.warning(f"Failed to analyze {file_path}: {e}")
        
        # Determine primary language and framework
        if stats["languages"]:
            stats["primary_language"] = max(stats["languages"], key=stats["languages"].get)
        if stats["frameworks"]:
            stats["primary_framework"] = max(stats["frameworks"], key=stats["frameworks"].get)
        
        return stats
    
    def clear_cache(self):
        """Clear the detection cache"""
        self._cache.clear()


# Singleton instance
language_detector = LanguageDetector()
