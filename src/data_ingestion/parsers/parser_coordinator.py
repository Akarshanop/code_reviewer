"""
Parser Coordinator - Unified parsing interface
Routes files to appropriate parsers based on automatic language detection
"""
from pathlib import Path
from typing import Dict, List, Any, Optional, Type
from dataclasses import dataclass, field
import asyncio

from src.data_ingestion.parsers.language_detector import (
    LanguageDetector, language_detector, LanguageType, FrameworkType, FileAnalysis
)
from src.data_ingestion.parsers.java_parser import JavaParser
from src.data_ingestion.parsers.react_parser import TsReactParser
from src.data_ingestion.parsers.generic_parser import GenericParser
from src.data_ingestion.parsers.python_parser import PythonParser
from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedFile:
    """Result of parsing a single file"""
    file_path: str
    language: str
    framework: str
    parser_used: str
    ast_nodes: List[Dict[str, Any]] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    classes: List[Dict[str, Any]] = field(default_factory=list)
    functions: List[Dict[str, Any]] = field(default_factory=list)
    routes: List[Dict[str, Any]] = field(default_factory=list)
    api_calls: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_content: Optional[str] = None
    error: Optional[str] = None


@dataclass
class RepositoryParseResult:
    """Result of parsing an entire repository"""
    repo_path: str
    total_files: int = 0
    parsed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    primary_language: Optional[str] = None
    primary_framework: Optional[str] = None
    languages_detected: Dict[str, int] = field(default_factory=dict)
    frameworks_detected: Dict[str, int] = field(default_factory=dict)
    files: List[ParsedFile] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_path": self.repo_path,
            "total_files": self.total_files,
            "parsed_files": self.parsed_files,
            "failed_files": self.failed_files,
            "skipped_files": self.skipped_files,
            "primary_language": self.primary_language,
            "primary_framework": self.primary_framework,
            "languages": self.languages_detected,
            "frameworks": self.frameworks_detected,
        }


class ParserCoordinator:
    """
    Coordinates parsing across multiple languages and frameworks.
    Automatically detects file types and routes to appropriate parsers.
    """
    
    # Default exclusion patterns for repository scanning
    DEFAULT_EXCLUDE_PATTERNS = [
        "**/node_modules/**",
        "**/target/**",
        "**/build/**",
        "**/dist/**",
        "**/.git/**",
        "**/__pycache__/**",
        "**/vendor/**",
        "**/.idea/**",
        "**/.vscode/**",
        "**/coverage/**",
        "**/.gradle/**",
        "**/bin/**",
        "**/obj/**",
        "**/*.min.js",
        "**/*.min.css",
        "**/*.map",
        "**/*.lock",
        "**/package-lock.json",
        "**/yarn.lock",
        "**/pnpm-lock.yaml",
    ]
    
    def __init__(self):
        self.detector = language_detector
        self._parsers: Dict[str, Any] = {}
        self._initialize_parsers()
    
    def _initialize_parsers(self):
        """Initialize available parsers"""
        try:
            self._parsers["java"] = JavaParser()
            logger.info("Java parser initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Java parser: {e}")
        
        try:
            self._parsers["react"] = TsReactParser()
            logger.info("React/TypeScript parser initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize React parser: {e}")
        
        try:
            self._parsers["generic"] = GenericParser()
            logger.info("Generic parser initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Generic parser: {e}")
        
        try:
            self._parsers["python"] = PythonParser()
            logger.info("Python parser initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Python parser: {e}")
    
    def get_parser(self, parser_type: str):
        """Get a parser by type"""
        return self._parsers.get(parser_type)
    
    def parse_file(self, file_path: Path, content: Optional[str] = None) -> ParsedFile:
        """
        Parse a single file with automatic language detection.
        
        Args:
            file_path: Path to the file
            content: Optional pre-loaded content
        
        Returns:
            ParsedFile with extracted information
        """
        # Read content if not provided
        if content is None:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                return ParsedFile(
                    file_path=str(file_path),
                    language="unknown",
                    framework="unknown",
                    parser_used="none",
                    error=f"Failed to read file: {e}"
                )
        
        # Detect language and framework
        analysis = self.detector.detect(file_path, content)
        
        result = ParsedFile(
            file_path=str(file_path),
            language=analysis.language.value,
            framework=analysis.framework.value,
            parser_used=analysis.parser_type,
            metadata={
                "is_test": analysis.is_test_file,
                "is_config": analysis.is_config_file,
                "is_generated": analysis.is_generated,
                "extension": analysis.metadata.get("extension", ""),
            }
        )
        
        # Skip non-parseable files
        if not analysis.is_parseable:
            result.parser_used = "skipped"
            return result
        
        # Get appropriate parser
        parser = self._parsers.get(analysis.parser_type)
        if not parser:
            result.parser_used = "generic"
            parser = self._parsers.get("generic")
        
        if not parser:
            result.error = f"No parser available for {analysis.parser_type}"
            return result
        
        # Parse based on parser type
        try:
            if analysis.parser_type == "java":
                result = self._parse_java(parser, file_path, content, result)
            elif analysis.parser_type == "react":
                result = self._parse_react(parser, file_path, content, result)
            elif analysis.parser_type == "generic":
                result = self._parse_generic(parser, file_path, content, result)
            elif analysis.parser_type == "python":
                result = self._parse_python(parser, file_path, content, result)
            else:
                # Fallback to generic
                result = self._parse_generic(
                    self._parsers.get("generic"), file_path, content, result
                )
        except Exception as e:
            result.error = f"Parse error: {e}"
            logger.error(f"Failed to parse {file_path}: {e}")
        
        return result
    
    def _parse_java(self, parser: JavaParser, file_path: Path, 
                    content: str, result: ParsedFile) -> ParsedFile:
        """Parse a Java file"""
        parsed = parser.parse(str(file_path), content)
        
        if parsed:
            result.classes = parsed.get("classes", [])
            result.functions = parsed.get("methods", [])
            result.routes = parsed.get("routes", [])
            result.imports = parsed.get("imports", [])
            result.dependencies = parsed.get("dependencies", [])
            result.metadata.update({
                "verticles": parsed.get("verticles", []),
                "eventbus": parsed.get("eventbus", []),
                "dtos": parsed.get("dtos", []),
                "services": parsed.get("services", []),
            })
        
        return result
    
    def _parse_react(self, parser: TsReactParser, file_path: Path,
                     content: str, result: ParsedFile) -> ParsedFile:
        """Parse a TypeScript/React file"""
        parsed = parser.parse(str(file_path), content)
        
        if parsed:
            result.classes = parsed.get("classes", []) + parsed.get("components", [])
            result.functions = parsed.get("functions", []) + parsed.get("hooks", [])
            result.routes = parsed.get("routes", [])
            result.api_calls = parsed.get("api_calls", [])
            result.imports = parsed.get("imports", [])
            result.exports = parsed.get("exports", [])
            result.metadata.update({
                "components": parsed.get("components", []),
                "models": parsed.get("models", []),
                "middleware": parsed.get("middleware", []),
                "state": parsed.get("state", []),
                "props": parsed.get("props", []),
            })
        
        return result
    
    def _parse_generic(self, parser: GenericParser, file_path: Path,
                       content: str, result: ParsedFile) -> ParsedFile:
        """Parse a config/generic file"""
        parsed = parser.parse(str(file_path), content)
        
        if parsed:
            result.metadata.update({
                "parsed_data": parsed.get("data", {}),
                "format": parsed.get("format", "unknown"),
            })
            
            # Extract dependencies from package.json
            if "dependencies" in parsed.get("data", {}):
                result.dependencies = list(parsed["data"]["dependencies"].keys())
            if "devDependencies" in parsed.get("data", {}):
                result.dependencies.extend(list(parsed["data"]["devDependencies"].keys()))
        
        return result
    
    def _parse_python(self, parser: PythonParser, file_path: Path,
                      content: str, result: ParsedFile) -> ParsedFile:
        """Parse a Python file"""
        parsed = parser.parse(str(file_path), content)
        
        if parsed:
            result.classes = parsed.get("classes", [])
            result.functions = parsed.get("functions", [])
            result.routes = parsed.get("routes", [])
            result.imports = parsed.get("imports", [])
            result.dependencies = parsed.get("dependencies", [])
            result.metadata.update({
                "framework": parsed.get("framework", "generic"),
                "constants": parsed.get("constants", []),
                "from_imports": parsed.get("from_imports", []),
            })
        
        return result
    
    def parse_repository(self, repo_path: Path, 
                        exclude_patterns: List[str] = None,
                        max_files: int = None) -> RepositoryParseResult:
        """
        Parse all files in a repository.
        
        Args:
            repo_path: Path to repository root
            exclude_patterns: Additional patterns to exclude
            max_files: Maximum number of files to parse (for testing)
        
        Returns:
            RepositoryParseResult with all parsed files
        """
        import fnmatch
        
        exclude = list(self.DEFAULT_EXCLUDE_PATTERNS)
        if exclude_patterns:
            exclude.extend(exclude_patterns)
        
        result = RepositoryParseResult(repo_path=str(repo_path))
        file_count = 0
        
        logger.info(f"Starting repository parse: {repo_path}")
        
        for file_path in repo_path.rglob("*"):
            if not file_path.is_file():
                continue
            
            # Check exclusions
            rel_path = str(file_path.relative_to(repo_path))
            if any(fnmatch.fnmatch(rel_path, p) for p in exclude):
                continue
            
            result.total_files += 1
            
            # Check file limit
            if max_files and file_count >= max_files:
                break
            
            # Parse file
            try:
                parsed = self.parse_file(file_path)
                
                if parsed.error:
                    result.failed_files += 1
                    result.errors.append({
                        "file": str(file_path),
                        "error": parsed.error
                    })
                elif parsed.parser_used == "skipped":
                    result.skipped_files += 1
                else:
                    result.parsed_files += 1
                    result.files.append(parsed)
                
                # Update language stats
                lang = parsed.language
                result.languages_detected[lang] = result.languages_detected.get(lang, 0) + 1
                
                # Update framework stats
                if parsed.framework != "generic":
                    fw = parsed.framework
                    result.frameworks_detected[fw] = result.frameworks_detected.get(fw, 0) + 1
                
                file_count += 1
                
            except Exception as e:
                result.failed_files += 1
                result.errors.append({
                    "file": str(file_path),
                    "error": str(e)
                })
        
        # Determine primary language and framework
        if result.languages_detected:
            # Filter out non-code languages for primary
            code_languages = {
                k: v for k, v in result.languages_detected.items()
                if k not in ("json", "yaml", "xml", "markdown", "unknown")
            }
            if code_languages:
                result.primary_language = max(code_languages, key=code_languages.get)
            else:
                result.primary_language = max(result.languages_detected, key=result.languages_detected.get)
        
        if result.frameworks_detected:
            result.primary_framework = max(result.frameworks_detected, key=result.frameworks_detected.get)
        
        logger.info(
            f"Repository parse complete: {result.parsed_files}/{result.total_files} files parsed, "
            f"primary language: {result.primary_language}, framework: {result.primary_framework}"
        )
        
        return result
    
    async def parse_repository_async(self, repo_path: Path,
                                     exclude_patterns: List[str] = None,
                                     max_files: int = None,
                                     batch_size: int = 50) -> RepositoryParseResult:
        """
        Parse repository asynchronously in batches.
        
        Args:
            repo_path: Path to repository root
            exclude_patterns: Additional patterns to exclude
            max_files: Maximum files to parse
            batch_size: Files to process per batch
        
        Returns:
            RepositoryParseResult
        """
        import fnmatch
        
        exclude = list(self.DEFAULT_EXCLUDE_PATTERNS)
        if exclude_patterns:
            exclude.extend(exclude_patterns)
        
        # Collect all files first
        all_files = []
        for file_path in repo_path.rglob("*"):
            if not file_path.is_file():
                continue
            rel_path = str(file_path.relative_to(repo_path))
            if any(fnmatch.fnmatch(rel_path, p) for p in exclude):
                continue
            all_files.append(file_path)
            if max_files and len(all_files) >= max_files:
                break
        
        result = RepositoryParseResult(
            repo_path=str(repo_path),
            total_files=len(all_files)
        )
        
        # Process in batches
        for i in range(0, len(all_files), batch_size):
            batch = all_files[i:i + batch_size]
            
            # Parse batch (could be parallelized further)
            for file_path in batch:
                try:
                    parsed = self.parse_file(file_path)
                    
                    if parsed.error:
                        result.failed_files += 1
                        result.errors.append({"file": str(file_path), "error": parsed.error})
                    elif parsed.parser_used == "skipped":
                        result.skipped_files += 1
                    else:
                        result.parsed_files += 1
                        result.files.append(parsed)
                    
                    lang = parsed.language
                    result.languages_detected[lang] = result.languages_detected.get(lang, 0) + 1
                    
                    if parsed.framework != "generic":
                        fw = parsed.framework
                        result.frameworks_detected[fw] = result.frameworks_detected.get(fw, 0) + 1
                        
                except Exception as e:
                    result.failed_files += 1
                    result.errors.append({"file": str(file_path), "error": str(e)})
            
            # Yield control
            await asyncio.sleep(0)
        
        # Determine primary language/framework
        if result.languages_detected:
            code_langs = {k: v for k, v in result.languages_detected.items()
                         if k not in ("json", "yaml", "xml", "markdown", "unknown")}
            if code_langs:
                result.primary_language = max(code_langs, key=code_langs.get)
        
        if result.frameworks_detected:
            result.primary_framework = max(result.frameworks_detected, key=result.frameworks_detected.get)
        
        return result
    
    def get_parser_for_language(self, language: LanguageType) -> Optional[Any]:
        """Get the appropriate parser for a language"""
        parser_type = self.detector.PARSER_MAP.get(language, "generic")
        return self._parsers.get(parser_type)


# Singleton instance
parser_coordinator = ParserCoordinator()
