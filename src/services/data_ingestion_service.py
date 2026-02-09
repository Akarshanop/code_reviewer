"""
Enterprise data ingestion service for 8+ massive repositories.
Handles multi-language codebases with automatic language detection.
No need to specify repository type - auto-detects based on file extensions.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
from collections import defaultdict
import hashlib
import json

from src.data_ingestion.git_service import GitService
from src.data_ingestion.parsers.parser_coordinator import (
    ParserCoordinator, parser_coordinator, ParsedFile, RepositoryParseResult
)
from src.data_ingestion.parsers.language_detector import (
    LanguageDetector, language_detector, LanguageType, FrameworkType
)
from src.data_ingestion.graph_builder.node_factory import NodeFactory
from src.data_ingestion.graph_builder.edge_linker import EdgeLinker
from src.services.search_service import SearchService
from src.services.vector_service import VectorService
from src.database.neo4j_client import neo4j_client
from src.database.supabase_session import supabase_client
from src.core.exceptions import CodeAnalysisException, InternalServerException
from src.core.logger import get_logger

logger = get_logger(__name__)

class DataIngestionService:
    """
    Enterprise-grade codebase ingestion service for massive repositories.
    
    Features:
    - Automatic language and framework detection (no need to specify repo type)
    - Support for 10+ languages (Java, TypeScript, Python, Go, etc.)
    - GitLab, GitHub, Bitbucket, and Azure DevOps support
    - Parallel processing for large codebases
    - Incremental updates for efficient re-indexing
    """
    
    def __init__(self):
        self.git_service = GitService()
        # Use the unified parser coordinator for auto-detection
        self.parser_coordinator = parser_coordinator
        self.language_detector = language_detector
        self.node_factory = NodeFactory()
        self.edge_linker = EdgeLinker()
        self.search_service = SearchService()
        self.vector_service = VectorService()
        
        # Initialize individual parsers from coordinator for direct access
        self.java_parser = self.parser_coordinator.get_parser("java")
        self.react_parser = self.parser_coordinator.get_parser("react")
        self.generic_parser = self.parser_coordinator.get_parser("generic")
        self.python_parser = self.parser_coordinator.get_parser("python")
        
        # File extension patterns for routing
        self.file_patterns = {
            "java": {".java", ".kt", ".kts"},
            "typescript": {".ts", ".tsx"},
            "javascript": {".js", ".jsx", ".mjs", ".cjs"},
            "config": {".json", ".yaml", ".yml", ".xml", ".toml", ".properties"},
            "python": {".py", ".pyx", ".pyi"},
        }
        
        # Repository tracking
        self.repositories = {}
        self.ingestion_stats = defaultdict(lambda: {
            "files_processed": 0,
            "nodes_created": 0,
            "edges_created": 0,
            "last_ingestion": None,
            "ingestion_duration": 0,
            "detected_language": None,
            "detected_framework": None,
            "errors": []
        })
    
    async def initialize(self):
        """Initialize all services"""
        await self.search_service.initialize()
        await self.vector_service.initialize()
        await neo4j_client.initialize()
        await supabase_client.initialize()
        logger.info("Data ingestion service initialized")
    
    async def ingest_repository(self, repo_url: str, branch: str = "main", 
                               incremental: bool = True,
                               repo_type: str = None) -> Dict[str, Any]:
        """
        Ingest a single repository with full analysis.
        
        Language and framework are AUTOMATICALLY DETECTED - no need to specify repo type.
        
        Args:
            repo_url: Repository URL to ingest (GitLab, GitHub, etc.)
            branch: Git branch to analyze
            incremental: Whether to perform incremental update
            repo_type: Type of repository ('backend' or 'frontend') - auto-detected if not provided
        
        Returns:
            Comprehensive ingestion results including auto-detected language/framework
        """
        start_time = datetime.utcnow()
        repo_hash = hashlib.md5(repo_url.encode()).hexdigest()[:12]
        
        # Auto-detect repo_type if not provided
        if not repo_type:
            repo_type = self._detect_repo_type(repo_url)
        
        logger.info(f"Starting repository ingestion: {repo_url}", repo_hash=repo_hash, repo_type=repo_type)
        
        try:
            # 1. Clone or update repository
            repo_path = await self._clone_or_update_repo(repo_url, branch, incremental)
            
            # 2. AUTO-DETECT: Analyze repository and detect languages/frameworks
            parse_result = await self._parse_repository_with_auto_detection(repo_path)
            
            logger.info(
                f"Auto-detected: language={parse_result.primary_language}, "
                f"framework={parse_result.primary_framework}",
                repo_hash=repo_hash
            )
            
            # 3. Convert to internal format for knowledge graph
            parsed_data = self._convert_parse_result(parse_result)
            
            # 4. Build knowledge graph with repo_type metadata
            graph_results = await self._build_knowledge_graph(
                parsed_data, repo_url, repo_hash, 
                repo_type=repo_type
            )
            
            # 5. Generate embeddings and index with repo_type metadata
            vector_results = await self._generate_embeddings(
                parsed_data, repo_url, repo_hash,
                repo_type=repo_type
            )
            
            # 6. Cross-repository linking
            cross_repo_links = await self._link_cross_repository_dependencies(repo_url, repo_hash)
            
            # 7. Store metadata and results
            await self._store_ingestion_metadata(
                repo_url, repo_hash, parse_result.to_dict(), 
                graph_results, vector_results, start_time
            )
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            results = {
                "repository": repo_url,
                "repository_hash": repo_hash,
                "branch": branch,
                "ingestion_status": "success",
                "duration_seconds": round(duration, 2),
                # Auto-detection results
                "auto_detection": {
                    "primary_language": parse_result.primary_language,
                    "primary_framework": parse_result.primary_framework,
                    "languages_found": parse_result.languages_detected,
                    "frameworks_found": parse_result.frameworks_detected,
                },
                "file_statistics": {
                    "total_files": parse_result.total_files,
                    "parsed_files": parse_result.parsed_files,
                    "skipped_files": parse_result.skipped_files,
                    "failed_files": parse_result.failed_files,
                    "by_language": parse_result.languages_detected
                },
                "graph_statistics": graph_results["statistics"],
                "vector_statistics": vector_results["statistics"],
                "cross_repository_links": cross_repo_links,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Update tracking
            self.repositories[repo_url] = {
                "hash": repo_hash,
                "last_ingestion": datetime.utcnow().isoformat(),
                "branch": branch,
                "file_count": parse_result.total_files,
                "primary_language": parse_result.primary_language,
                "primary_framework": parse_result.primary_framework
            }
            
            self.ingestion_stats[repo_url].update({
                "files_processed": parse_result.total_files,
                "last_ingestion": datetime.utcnow().isoformat(),
                "ingestion_duration": duration,
                "detected_language": parse_result.primary_language,
                "detected_framework": parse_result.primary_framework
            })
            
            logger.info(
                "Repository ingestion completed",
                repository=repo_url,
                duration_seconds=round(duration, 2),
                files_processed=parse_result.total_files,
                detected_language=parse_result.primary_language,
                detected_framework=parse_result.primary_framework,
                graph_nodes=graph_results["statistics"]["nodes_created"]
            )
            
            return results
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            error_msg = f"Repository ingestion failed: {str(e)}"
            
            logger.error(
                "Repository ingestion failed",
                repository=repo_url,
                error=str(e),
                duration_seconds=round(duration, 2)
            )
            
            # Record error
            self.ingestion_stats[repo_url]["errors"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "duration": duration
            })
            
            raise CodeAnalysisException(error_msg, file_path=repo_url)
    
    async def _parse_repository_with_auto_detection(self, repo_path: Path) -> RepositoryParseResult:
        """
        Parse repository with automatic language/framework detection.
        
        This is the core auto-detection feature - no need to specify what type of repo it is.
        The system automatically detects based on file extensions and content.
        """
        logger.info(f"Starting auto-detection parsing for: {repo_path}")
        
        # Use async parsing for better performance on large repos
        return await self.parser_coordinator.parse_repository_async(
            repo_path=repo_path,
            batch_size=50
        )
    
    def _detect_repo_type(self, repo_url: str) -> str:
        """
        Auto-detect repository type (backend or frontend) from URL patterns.
        
        Args:
            repo_url: Repository URL
            
        Returns:
            'backend' or 'frontend' or empty string if unknown
        """
        url_lower = repo_url.lower()
        
        # Backend patterns
        backend_patterns = ['backend', 'api', 'server', '-be', '_be', 'service']
        for pattern in backend_patterns:
            if pattern in url_lower:
                return 'backend'
        
        # Frontend patterns
        frontend_patterns = ['frontend', 'ui', 'web', '-fe', '_fe', 'client']
        for pattern in frontend_patterns:
            if pattern in url_lower:
                return 'frontend'
        
        return ''
    
    def _convert_parse_result(self, parse_result: RepositoryParseResult) -> Dict[str, Any]:
        """Convert RepositoryParseResult to the format expected by graph building"""
        parsed_data = {
            "java": [],
            "typescript": [],
            "javascript": [],
            "python": [],
            "config": [],
            "other": [],
            "summary": {
                "parsed_files": parse_result.parsed_files,
                "skipped_files": parse_result.skipped_files,
                "by_language": parse_result.languages_detected,
                "errors": [{"file": e["file"], "error": e["error"]} for e in parse_result.errors]
            }
        }
        
        # Route parsed files to appropriate categories
        for parsed_file in parse_result.files:
            file_data = {
                "file": parsed_file.file_path,
                "language": parsed_file.language,
                "framework": parsed_file.framework,
                "data": {
                    "classes": parsed_file.classes,
                    "functions": parsed_file.functions,
                    "routes": parsed_file.routes,
                    "api_calls": parsed_file.api_calls,
                    "imports": parsed_file.imports,
                    "exports": parsed_file.exports,
                    "dependencies": parsed_file.dependencies,
                    "metadata": parsed_file.metadata
                }
            }
            
            # Route based on detected language
            if parsed_file.language in ("java", "kotlin"):
                parsed_data["java"].append(file_data)
            elif parsed_file.language == "typescript":
                parsed_data["typescript"].append(file_data)
            elif parsed_file.language == "javascript":
                parsed_data["javascript"].append(file_data)
            elif parsed_file.language == "python":
                parsed_data["python"].append(file_data)
            elif parsed_file.language in ("json", "yaml", "xml", "toml"):
                parsed_data["config"].append(file_data)
            else:
                parsed_data["other"].append(file_data)
        
        return parsed_data
    
    async def _clone_or_update_repo(self, repo_url: str, branch: str, 
                                   incremental: bool) -> Path:
        """Clone or update repository with smart caching"""
        # Check if we have existing data
        repo_hash = hashlib.md5(repo_url.encode()).hexdigest()[:12]
        cache_key = f"repo_{repo_hash}"
        
        if incremental and repo_url in self.repositories:
            logger.debug(f"Performing incremental update for: {repo_url}")
            # Get last commit hash and fetch updates
            # Implementation depends on your GitService capabilities
            pass
        
        # Clone fresh copy
        temp_dir = Path(tempfile.mkdtemp(prefix=f"repo_{repo_hash}_"))
        
        try:
            # Use GitService to clone
            clone_result = self.git_service.clone_or_pull(repo_url, branch)
            
            # For simplicity, assuming GitService returns path
            # In production, you'd handle the async cloning properly
            repo_path = Path(clone_result) if isinstance(clone_result, str) else clone_result
            
            logger.debug(f"Repository cloned to: {repo_path}")
            return repo_path
            
        except Exception as e:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise
    
    async def _analyze_repository_structure(self, repo_path: Path) -> Dict[str, Any]:
        """Analyze repository structure and file distribution"""
        structure = {
            "total_files": 0,
            "by_extension": defaultdict(int),
            "by_directory": defaultdict(int),
            "largest_files": [],
            "build_files": [],
            "config_files": [],
            "test_files": []
        }
        
        try:
            # Walk through repository
            for file_path in repo_path.rglob("*"):
                if file_path.is_file():
                    structure["total_files"] += 1
                    
                    # Count by extension
                    ext = file_path.suffix.lower()
                    structure["by_extension"][ext] += 1
                    
                    # Count by directory depth
                    rel_path = file_path.relative_to(repo_path)
                    dir_depth = len(rel_path.parts) - 1
                    structure["by_directory"][dir_depth] += 1
                    
                    # Identify file types
                    file_name = file_path.name.lower()
                    
                    if any(build in file_name for build in ["pom.xml", "build.gradle", "build.sbt", "package.json"]):
                        structure["build_files"].append(str(rel_path))
                    
                    if any(config in file_name for config in [".properties", ".yml", ".yaml", ".conf", ".config"]):
                        structure["config_files"].append(str(rel_path))
                    
                    if any(test in file_name for test in ["test", "spec", "__test__"]):
                        structure["test_files"].append(str(rel_path))
                    
                    # Track large files
                    try:
                        file_size = file_path.stat().st_size
                        if file_size > 1024 * 100:  # > 100KB
                            structure["largest_files"].append({
                                "path": str(rel_path),
                                "size_bytes": file_size,
                                "extension": ext
                            })
                    except:
                        pass
            
            # Sort largest files
            structure["largest_files"].sort(key=lambda x: x["size_bytes"], reverse=True)
            structure["largest_files"] = structure["largest_files"][:10]
            
            logger.debug(
                "Repository structure analyzed",
                total_files=structure["total_files"],
                extensions_count=len(structure["by_extension"])
            )
            
            return structure
            
        except Exception as e:
            logger.error("Failed to analyze repository structure", error=str(e))
            return structure
    
    async def _parse_repository_files(self, repo_path: Path, 
                                     structure: Dict[str, Any]) -> Dict[str, Any]:
        """Parse repository files with appropriate parsers"""
        parsed_data = {
            "java": [],
            "typescript": [],
            "javascript": [],
            "config": [],
            "other": [],
            "summary": {
                "parsed_files": 0,
                "skipped_files": 0,
                "by_language": defaultdict(int),
                "errors": []
            }
        }
        
        try:
            # Process files in parallel batches
            file_batches = []
            current_batch = []
            
            for file_path in repo_path.rglob("*"):
                if file_path.is_file():
                    current_batch.append(file_path)
                    
                    if len(current_batch) >= 50:  # Batch size
                        file_batches.append(current_batch.copy())
                        current_batch.clear()
            
            if current_batch:
                file_batches.append(current_batch)
            
            # Process batches
            tasks = []
            for batch in file_batches:
                task = self._parse_file_batch(batch, repo_path, parsed_data)
                tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update summary
            parsed_data["summary"]["parsed_files"] = sum(
                len(parsed_data[lang]) for lang in ["java", "typescript", "javascript", "config", "other"]
            )
            
            logger.info(
                "Repository files parsed",
                parsed_files=parsed_data["summary"]["parsed_files"],
                skipped_files=parsed_data["summary"]["skipped_files"],
                errors=len(parsed_data["summary"]["errors"])
            )
            
            return parsed_data
            
        except Exception as e:
            logger.error("Failed to parse repository files", error=str(e))
            parsed_data["summary"]["errors"].append(str(e))
            return parsed_data
    
    async def _parse_file_batch(self, file_batch: List[Path], repo_path: Path, 
                               parsed_data: Dict[str, Any]):
        """Parse a batch of files"""
        for file_path in file_batch:
            try:
                rel_path = file_path.relative_to(repo_path)
                file_ext = file_path.suffix.lower()
                
                # Determine parser based on file type
                if file_ext in self.file_patterns["java"]:
                    parsed = self.java_parser.parse_file(file_path)
                    parsed_data["java"].append({
                        "file": str(rel_path),
                        "data": parsed,
                        "language": "java"
                    })
                    parsed_data["summary"]["by_language"]["java"] += 1
                    
                elif file_ext in self.file_patterns["typescript"]:
                    parsed = self.react_parser.parse_file(file_path)
                    parsed_data["typescript"].append({
                        "file": str(rel_path),
                        "data": parsed,
                        "language": "typescript"
                    })
                    parsed_data["summary"]["by_language"]["typescript"] += 1
                    
                elif file_ext in self.file_patterns["javascript"]:
                    # Use TypeScript parser for JS files too
                    parsed = self.react_parser.parse_file(file_path)
                    parsed_data["javascript"].append({
                        "file": str(rel_path),
                        "data": parsed,
                        "language": "javascript"
                    })
                    parsed_data["summary"]["by_language"]["javascript"] += 1
                    
                elif file_ext in self.file_patterns["config"]:
                    parsed = self.generic_parser.parse_file(file_path)
                    parsed_data["config"].append({
                        "file": str(rel_path),
                        "data": parsed,
                        "language": "config"
                    })
                    parsed_data["summary"]["by_language"]["config"] += 1
                    
                else:
                    parsed_data["other"].append({
                        "file": str(rel_path),
                        "extension": file_ext,
                        "language": "unknown"
                    })
                    parsed_data["summary"]["skipped_files"] += 1
                    
            except Exception as e:
                parsed_data["summary"]["errors"].append({
                    "file": str(rel_path) if 'rel_path' in locals() else str(file_path),
                    "error": str(e)
                })
                parsed_data["summary"]["skipped_files"] += 1
    
    async def _build_knowledge_graph(self, parsed_data: Dict[str, Any], 
                                    repo_url: str, repo_hash: str,
                                    repo_type: str = None,
                                    entry_file: str = None) -> Dict[str, Any]:
        """Build Neo4j knowledge graph from parsed data with REQUIRED metadata"""
        statistics = {
            "nodes_created": 0,
            "edges_created": 0,
            "by_node_type": defaultdict(int),
            "by_relationship_type": defaultdict(int)
        }
        
        # Extract repository name for metadata
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        
        try:
            # Start transaction
            async with neo4j_client.session() as session:
                # First, create repository node with repo_type
                repo_query = """
                MERGE (r:Repository {
                    url: $url,
                    hash: $hash,
                    name: $name
                })
                SET r.last_ingested = datetime(),
                    r.ingestion_status = 'in_progress',
                    r.repo_type = $repo_type,
                    r.entry_file = $entry_file
                RETURN r
                """
                
                await session.run(repo_query, {
                    "url": repo_url,
                    "hash": repo_hash,
                    "name": repo_name,
                    "repo_type": repo_type or "",
                    "entry_file": entry_file or ""
                })
                
                # Process Java files
                for java_file in parsed_data["java"]:
                    file_path = java_file["file"]
                    file_name = file_path.split('/')[-1] if file_path else ''
                    
                    nodes_created = await self.node_factory.create_nodes(
                        session,
                        java_file["data"],
                        file_path,
                        repository=repo_url,
                        repository_hash=repo_hash,
                        repo_type=repo_type,
                        repository_name=repo_name,
                        entry_file=entry_file,
                    )
                    statistics["nodes_created"] += nodes_created.get("created", 0)
                    
                    # Link file to repository
                    link_query = """
                    MATCH (r:Repository {hash: $repo_hash})
                    MATCH (f:File {path: $file_path})
                    MERGE (f)-[:BELONGS_TO]->(r)
                    """
                    await session.run(link_query, {
                        "repo_hash": repo_hash,
                        "file_path": java_file["file"]
                    })
                
                # Process TypeScript/React files
                for ts_file in parsed_data["typescript"]:
                    file_path = ts_file["file"]
                    file_name = file_path.split('/')[-1] if file_path else ''
                    
                    nodes_created = await self.node_factory.create_nodes(
                        session,
                        ts_file["data"],
                        file_path,
                        repository=repo_url,
                        repository_hash=repo_hash,
                        repo_type=repo_type,
                        repository_name=repo_name,
                        entry_file=entry_file,
                    )
                    statistics["nodes_created"] += nodes_created.get("created", 0)
                    
                    # Link file to repository
                    link_query = """
                    MATCH (r:Repository {hash: $repo_hash})
                    MATCH (f:File {path: $file_path})
                    MERGE (f)-[:BELONGS_TO]->(r)
                    """
                    await session.run(link_query, {
                        "repo_hash": repo_hash,
                        "file_path": ts_file["file"]
                    })

                # Process JavaScript files (Express.js / Node.js / JSX)
                for js_file in parsed_data.get("javascript", []):
                    file_path = js_file["file"]
                    file_name = file_path.split('/')[-1] if file_path else ''
                    
                    nodes_created = await self.node_factory.create_nodes(
                        session,
                        js_file["data"],
                        file_path,
                        repository=repo_url,
                        repository_hash=repo_hash,
                        repo_type=repo_type,
                        repository_name=repo_name,
                        entry_file=entry_file,
                    )
                    statistics["nodes_created"] += nodes_created.get("created", 0)

                    # Link file to repository
                    link_query = """
                    MATCH (r:Repository {hash: $repo_hash})
                    MATCH (f:File {path: $file_path})
                    MERGE (f)-[:BELONGS_TO]->(r)
                    """
                    await session.run(link_query, {
                        "repo_hash": repo_hash,
                        "file_path": js_file["file"],
                    })
                
                # Link edges between nodes
                edge_results = await self.edge_linker.link_all_edges(session)
                statistics["edges_created"] = edge_results.get("relationships_created", 0)

                # Enhanced graph linking (IMPORTS, CALLS_FUNCTION, CALLS_API, type hierarchy)
                try:
                    from src.data_ingestion.enhanced_graph_linker import EnhancedGraphLinker
                    repo_path = parsed_data.get("repo_path", "")
                    if repo_path:
                        enhanced_linker = EnhancedGraphLinker(repo_path)
                        enhanced_results = await enhanced_linker.run_all_enhancements(session)
                        enhanced_count = enhanced_results.get("total_edges_created", 0)
                        statistics["edges_created"] += enhanced_count
                        statistics["enhanced_edges_created"] = enhanced_count
                        logger.info(f"Enhanced graph linking created {enhanced_count} additional edges")
                except Exception as e:
                    logger.warning(f"Enhanced graph linking skipped: {e}")

                # Update repository status
                update_query = """
                MATCH (r:Repository {hash: $hash})
                SET r.ingestion_status = 'completed',
                    r.nodes_count = $nodes_count,
                    r.edges_count = $edges_count,
                    r.last_ingested = datetime()
                RETURN r
                """
                
                await session.run(update_query, {
                    "hash": repo_hash,
                    "nodes_count": statistics["nodes_created"],
                    "edges_count": statistics["edges_created"]
                })
            
            logger.info(
                "Knowledge graph built",
                repository=repo_url,
                nodes_created=statistics["nodes_created"],
                edges_created=statistics["edges_created"]
            )
            
            return {"statistics": statistics, "status": "success"}
            
        except Exception as e:
            logger.error("Failed to build knowledge graph", error=str(e))
            return {"statistics": statistics, "status": "failed", "error": str(e)}
    
    async def _generate_embeddings(self, parsed_data: Dict[str, Any], 
                                  repo_url: str, repo_hash: str,
                                  repo_type: str = None,
                                  entry_file: str = None) -> Dict[str, Any]:
        """Generate vector embeddings for code artifacts with REQUIRED metadata"""
        statistics = {
            "embeddings_created": 0,
            "by_artifact_type": defaultdict(int),
            "errors": []
        }
        
        # Extract repository_name for metadata
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        
        try:
            artifacts = []
            
            # Extract artifacts from Java files (use .get() to avoid KeyError)
            for java_file in parsed_data.get("java", []):
                file_data = java_file.get("data", {})
                file_path = java_file.get("file", "")
                file_name = file_path.split('/')[-1] if file_path else ""
                
                # Routes
                for route in file_data.get("routes", []):
                    artifacts.append({
                        "type": "route",
                        "name": route.get("handler_method", ""),
                        "language": "java",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": f"Vert.x Route: {route.get('path', '')} Method: {route.get('method', '')} Handler: {route.get('handler_method', '')}",
                        "text": f"Vert.x Route: {route.get('path', '')} "
                               f"Method: {route.get('method', '')} "
                               f"Handler: {route.get('handler_method', '')}",
                        "metadata": {
                            "path": route.get("path"),
                            "method": route.get("method"),
                            "handler": route.get("handler_method")
                        },
                        # REQUIRED fields
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    })
                
                # EventBus
                for eb in file_data.get("eventbus", []):
                    artifacts.append({
                        "type": "eventbus",
                        "name": eb.get("address", ""),
                        "language": "java",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": f"EventBus address: {eb.get('address', '')} type: {eb.get('type', '')} handler: {eb.get('handler_method', '')}",
                        "text": f"EventBus: {eb.get('address', '')} "
                               f"Type: {eb.get('type', '')} "
                               f"Handler: {eb.get('handler_method', '')}",
                        "metadata": {
                            "address": eb.get("address"),
                            "type": eb.get("type"),
                            "handler": eb.get("handler_method")
                        },
                        # REQUIRED fields
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    })
                
                # DTOs
                for dto in file_data.get("dtos", []):
                    artifacts.append({
                        "type": "dto",
                        "name": dto.get("name", ""),
                        "language": "java",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": f"DTO class {dto.get('name', '')} with fields: {', '.join([f.get('name', '') for f in dto.get('fields', [])])}",
                        "text": f"DTO: {dto.get('name', '')} "
                               f"Fields: {', '.join([f.get('name', '') for f in dto.get('fields', [])])}",
                        "metadata": {
                            "name": dto.get("name"),
                            "fields": dto.get("fields", []),
                            "is_data_object": dto.get("is_data_object", False)
                        },
                        # REQUIRED fields
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    })
            
            # Extract artifacts from TypeScript files
            for ts_file in parsed_data.get("typescript", []):
                file_data = ts_file.get("data", {})
                file_metadata = ts_file.get("metadata", {})
                file_path = ts_file.get("file", "")
                file_name = file_path.split('/')[-1] if file_path else ""
                
                # Components - try file_data first, then metadata
                components = file_data.get("components", []) or file_metadata.get("components", [])
                for comp in components:
                    artifacts.append({
                        "type": "react_component",
                        "name": comp.get("name", ""),
                        "language": "typescript",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": f"React component {comp.get('name', '')} with props: {', '.join([p.get('name', '') for p in comp.get('props', [])])} and hooks: {', '.join(comp.get('hooks', []))}",
                        "text": f"React Component: {comp.get('name', '')} "
                               f"Props: {', '.join([p.get('name', '') for p in comp.get('props', [])])} "
                               f"Hooks: {', '.join(comp.get('hooks', []))}",
                        "metadata": {
                            "name": comp.get("name"),
                            "props": comp.get("props", []),
                            "hooks": comp.get("hooks", [])
                        },
                        # REQUIRED fields
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    })
                
                # Functions/Methods from TypeScript (not components)
                functions = file_data.get("functions", [])
                for func in functions:
                    artifacts.append({
                        "type": "function",
                        "name": func.get("name", ""),
                        "language": "typescript",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": f"Function {func.get('name', '')} - {func.get('description', '')}",
                        "text": f"Function: {func.get('name', '')} - {func.get('description', '')}",
                        "metadata": {
                            "name": func.get("name"),
                            "description": func.get("description", "")
                        },
                        # REQUIRED fields
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    })
                
                # API Calls
                for api in file_data.get("api_calls", []):
                    artifacts.append({
                        "type": "api_call",
                        "name": api.get("url", ""),
                        "language": "typescript",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": f"API call to {api.get('url', '')} using {api.get('method', '')} in hook {api.get('hook', '')}",
                        "text": f"API Call: {api.get('url', '')} "
                               f"Method: {api.get('method', '')} "
                               f"Hook: {api.get('hook', '')}",
                        "metadata": {
                            "url": api.get("url"),
                            "method": api.get("method"),
                            "hook": api.get("hook"),
                            "component": api.get("component")
                        },
                        # REQUIRED fields
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    })
            
            # Extract artifacts from JavaScript files (same as TypeScript)
            for js_file in parsed_data.get("javascript", []):
                file_data = js_file.get("data", {})
                file_metadata = js_file.get("metadata", {})
                file_path = js_file.get("file", "")
                file_name = file_path.split('/')[-1] if file_path else ""
                
                # Components
                components = file_data.get("components", []) or file_metadata.get("components", [])
                for comp in components:
                    artifacts.append({
                        "type": "react_component",
                        "name": comp.get("name", ""),
                        "language": "javascript",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": f"React component {comp.get('name', '')} with props: {', '.join([p.get('name', '') for p in comp.get('props', [])])}",
                        "text": f"React Component: {comp.get('name', '')} Props: {', '.join([p.get('name', '') for p in comp.get('props', [])])}",
                        "metadata": {
                            "name": comp.get("name"),
                            "props": comp.get("props", [])
                        },
                        # REQUIRED fields
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    })
                
                # Functions
                functions = file_data.get("functions", [])
                for func in functions:
                    artifacts.append({
                        "type": "function",
                        "name": func.get("name", ""),
                        "language": "javascript",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": f"Function {func.get('name', '')} - {func.get('description', '')}",
                        "text": f"Function: {func.get('name', '')}",
                        "metadata": {
                            "name": func.get("name")
                        },
                        # REQUIRED fields
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    })
                
                # API Calls
                for api in file_data.get("api_calls", []):
                    artifacts.append({
                        "type": "api_call",
                        "name": api.get("url", ""),
                        "language": "javascript",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": f"API call to {api.get('url', '')} using {api.get('method', '')}",
                        "text": f"API Call: {api.get('url', '')} Method: {api.get('method', '')}",
                        "metadata": {
                            "url": api.get("url"),
                            "method": api.get("method")
                        },
                        # REQUIRED fields
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    })
            
            # Generate embeddings from actual code snippets
            if artifacts:
                logger.info(
                    "Starting embedding generation",
                    repository=repo_url,
                    artifact_count=len(artifacts),
                    artifact_types=dict(statistics["by_artifact_type"])
                )
                
                try:
                    result = await self.vector_service.index_code_artifacts(artifacts)
                    statistics["embeddings_created"] = result.get("indexed", 0)
                    statistics["errors"] = result.get("errors", [])
                    
                    # Validate: ensure we actually indexed something
                    if statistics["embeddings_created"] == 0 and len(artifacts) > 0:
                        error_msg = "No artifacts were indexed despite having code to embed"
                        logger.warning(error_msg, artifact_count=len(artifacts))
                        statistics["errors"].append(error_msg)
                    
                    # Log success
                    logger.info(
                        "Embeddings generated successfully",
                        repository=repo_url,
                        embeddings_created=statistics["embeddings_created"],
                        total_artifacts=len(artifacts),
                        artifact_types=dict(statistics["by_artifact_type"]),
                        error_count=len(statistics["errors"])
                    )
                except Exception as embed_error:
                    error_msg = f"Embedding indexing failed: {str(embed_error)}"
                    logger.error(error_msg, exc_info=True)
                    statistics["errors"].append(error_msg)
            else:
                logger.warning(
                    "No artifacts found for embedding",
                    repository=repo_url,
                    parsed_data_keys=list(parsed_data.keys())
                )
            
            return {"statistics": statistics, "status": "success" if not statistics["errors"] else "partial"}
            
        except Exception as e:
            logger.error(
                "Failed to generate embeddings",
                error=str(e),
                repository=repo_url,
                exc_info=True
            )
            statistics["errors"].append(str(e))
            return {"statistics": statistics, "status": "failed"}
    
    async def _link_cross_repository_dependencies(self, repo_url: str, 
                                                 repo_hash: str) -> Dict[str, Any]:
        """Link dependencies across repositories"""
        links = {
            "api_calls": [],
            "eventbus_flows": [],
            "dto_usages": [],
            "component_references": []
        }
        
        try:
            # Find API calls that might target other repositories
            query = """
            MATCH (source:Repository {hash: $repo_hash})
            MATCH (source)-[:CONTAINS]->(file:File)-[:CONTAINS]->(api:ApiCall)
            WHERE api.url CONTAINS 'api/' AND api.url IS NOT NULL
            OPTIONAL MATCH (targetRoute:VertxRoute)
            WHERE targetRoute.path = api.url 
               OR replace(api.url, '/api/v1/', '/') = targetRoute.path
               OR replace(api.url, '/api/', '/') = targetRoute.path
            OPTIONAL MATCH (targetFile:File)-[:CONTAINS]->(targetRoute)
            OPTIONAL MATCH (targetRepo:Repository)<-[:BELONGS_TO]-(targetFile)
            WHERE targetRepo.hash <> $repo_hash
            RETURN api, targetRoute, targetRepo
            LIMIT 100
            """
            
            async with neo4j_client.session() as session:
                results = await session.run(query, {"repo_hash": repo_hash})
                
                for record in results:
                    if record.get("targetRepo"):
                        links["api_calls"].append({
                            "source_repository": repo_url,
                            "target_repository": record["targetRepo"].get("url"),
                            "api_path": record["api"].get("url"),
                            "target_route": record["targetRoute"].get("path")
                        })
            
            # Find EventBus flows across repositories
            eventbus_query = """
            MATCH (source:Repository {hash: $repo_hash})
            MATCH (source)-[:CONTAINS]->(file:File)-[:CONTAINS]->(eventbus:EventBusNode)
            WHERE eventbus.address IS NOT NULL
            OPTIONAL MATCH (targetEventbus:EventBusNode {address: eventbus.address})
            WHERE targetEventbus <> eventbus
            OPTIONAL MATCH (targetFile:File)-[:CONTAINS]->(targetEventbus)
            OPTIONAL MATCH (targetRepo:Repository)<-[:BELONGS_TO]-(targetFile)
            WHERE targetRepo.hash <> $repo_hash
            RETURN eventbus, targetEventbus, targetRepo
            LIMIT 100
            """
            
            async with neo4j_client.session() as session:
                results = await session.run(eventbus_query, {"repo_hash": repo_hash})
                
                for record in results:
                    if record.get("targetRepo"):
                        links["eventbus_flows"].append({
                            "source_repository": repo_url,
                            "target_repository": record["targetRepo"].get("url"),
                            "eventbus_address": record["eventbus"].get("address"),
                            "type": record["eventbus"].get("type")
                        })
            
            logger.debug(
                "Cross-repository dependencies linked",
                repository=repo_url,
                api_calls=len(links["api_calls"]),
                eventbus_flows=len(links["eventbus_flows"])
            )
            
            return links
            
        except Exception as e:
            logger.error("Failed to link cross-repository dependencies", error=str(e))
            return links
    
    async def parse_repositories(
        self,
        repositories: List[Dict[str, Any]],
        file_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Parse source files from cloned repositories.
        
        Args:
            repositories: List of repository info dicts with paths
            file_types: Optional filter for file types (java, typescript, etc.)
        
        Returns:
            Parsed data from all repositories
        """
        all_parsed_data = {
            "files_processed": 0,
            "parsed_data": {},
            "errors": []
        }
        
        for repo_info in repositories:
            repo_name = repo_info.get("name", "unknown")
            repo_path = Path(repo_info.get("path") or repo_info.get("local_path", ""))
            
            if not repo_path.exists():
                all_parsed_data["errors"].append(f"Repository path not found: {repo_path}")
                continue
            
            try:
                structure = await self._analyze_repository_structure(repo_path)
                parsed = await self._parse_repository_files(repo_path, structure)
                
                if file_types:
                    filtered_parsed = {
                        "java": parsed.get("java", []) if "java" in file_types else [],
                        "typescript": parsed.get("typescript", []) if "typescript" in file_types else [],
                        "javascript": parsed.get("javascript", []) if "javascript" in file_types else [],
                        "config": parsed.get("config", []) if "config" in file_types else [],
                        "other": [],
                        "summary": parsed.get("summary", {})
                    }
                    parsed = filtered_parsed
                
                all_parsed_data["parsed_data"][repo_name] = parsed
                all_parsed_data["files_processed"] += parsed.get("summary", {}).get("parsed_files", 0)
                
            except Exception as e:
                logger.error(f"Failed to parse repository {repo_name}", error=str(e))
                all_parsed_data["errors"].append(f"{repo_name}: {str(e)}")
        
        return all_parsed_data
    
    async def build_knowledge_graph(
        self,
        parsed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build knowledge graph from parsed data.
        
        Args:
            parsed_data: Dict mapping repo names to their parsed content
        
        Returns:
            Graph building results with statistics
        """
        total_nodes = 0
        total_edges = 0
        all_nodes = []
        errors = []
        
        for repo_name, repo_parsed in parsed_data.items():
            if not isinstance(repo_parsed, dict):
                continue
            
            try:
                repo_hash = hashlib.md5(repo_name.encode()).hexdigest()[:12]
                graph_results = await self._build_knowledge_graph(
                    repo_parsed,
                    repo_name,
                    repo_hash
                )
                
                total_nodes += graph_results.get("statistics", {}).get("nodes_created", 0)
                total_edges += graph_results.get("statistics", {}).get("edges_created", 0)
                
                for java_file in repo_parsed.get("java", []):
                    file_data = java_file.get("data", {})
                    
                    for route in file_data.get("routes", []):
                        all_nodes.append({
                            "type": "route",
                            "name": route.get("handler_method", ""),
                            "repository": repo_name,
                            "file": java_file.get("file", ""),
                            "data": route
                        })
                    
                    for eb in file_data.get("eventbus", []):
                        all_nodes.append({
                            "type": "eventbus",
                            "name": eb.get("address", ""),
                            "repository": repo_name,
                            "file": java_file.get("file", ""),
                            "data": eb
                        })
                
                for ts_file in repo_parsed.get("typescript", []):
                    file_data = ts_file.get("data", {})
                    
                    for component in file_data.get("components", []):
                        all_nodes.append({
                            "type": "component",
                            "name": component.get("name", ""),
                            "repository": repo_name,
                            "file": ts_file.get("file", ""),
                            "language": "typescript",
                            "data": component
                        })

                for js_file in repo_parsed.get("javascript", []):
                    file_data = js_file.get("data", {})

                    for route in file_data.get("routes", []):
                        all_nodes.append({
                            "type": "route",
                            "name": route.get("path", ""),
                            "repository": repo_name,
                            "file": js_file.get("file", ""),
                            "language": "javascript",
                            "data": route,
                        })

                    for api in file_data.get("api_calls", []):
                        all_nodes.append({
                            "type": "api_call",
                            "name": api.get("url", ""),
                            "repository": repo_name,
                            "file": js_file.get("file", ""),
                            "language": "javascript",
                            "data": api,
                        })

                    for fn in file_data.get("functions", []):
                        if not isinstance(fn, dict):
                            continue
                        all_nodes.append({
                            "type": "function",
                            "name": fn.get("name", ""),
                            "repository": repo_name,
                            "file": js_file.get("file", ""),
                            "language": "javascript",
                            "data": {
                                "name": fn.get("name"),
                                "type": fn.get("type"),
                                "is_async": fn.get("is_async"),
                                "params": fn.get("params"),
                                "start_line": fn.get("start_line"),
                                "end_line": fn.get("end_line"),
                                "calls": fn.get("calls"),
                            },
                        })
                
            except Exception as e:
                logger.error(f"Failed to build graph for {repo_name}", error=str(e))
                errors.append(f"{repo_name}: {str(e)}")
        
        return {
            "nodes_created": total_nodes,
            "edges_created": total_edges,
            "nodes": all_nodes,
            "errors": errors
        }
    
    async def generate_embeddings(
        self,
        nodes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate embeddings for parsed code nodes.
        
        Args:
            nodes: List of node dicts to embed
        
        Returns:
            Embedding generation results
        """
        if not nodes:
            return {"embeddings_generated": 0, "errors": []}
        
        artifacts = []
        
        for node in nodes:
            artifact = {
                "type": node.get("type", "unknown"),
                "name": node.get("name", ""),
                "repository": node.get("repository", ""),
                "file": node.get("file", ""),
                "language": node.get("language")
                or ("java" if node.get("type") in ["route", "eventbus", "dto", "entity"] else "typescript"),
                "metadata": node.get("data", {}),
                "description": self._generate_node_description(node)
            }
            artifacts.append(artifact)
        
        try:
            results = await self.vector_service.index_code_artifacts(artifacts)
            
            return {
                "embeddings_generated": results.get("indexed", 0),
                "errors": results.get("errors", [])
            }
            
        except Exception as e:
            logger.error("Failed to generate embeddings", error=str(e))
            return {
                "embeddings_generated": 0,
                "errors": [str(e)]
            }
    
    def _generate_node_description(self, node: Dict[str, Any]) -> str:
        """Generate a textual description for a node"""
        node_type = node.get("type", "")
        name = node.get("name", "")
        data = node.get("data", {})
        
        if node_type == "route":
            return f"HTTP Route {data.get('method', 'GET')} {data.get('path', '')} handled by {data.get('handler_method', name)}"
        elif node_type == "eventbus":
            return f"EventBus {data.get('type', 'handler')} at address {name}"
        elif node_type == "component":
            props = ", ".join(data.get("props", [])) if data.get("props") else "no props"
            return f"React component {name} with {props}"
        elif node_type == "dto":
            fields = len(data.get("fields", []))
            return f"DTO {name} with {fields} fields"
        else:
            return f"{node_type}: {name}"
    
    async def _store_ingestion_metadata(self, repo_url: str, repo_hash: str,
                                       structure: Dict[str, Any], 
                                       graph_results: Dict[str, Any],
                                       vector_results: Dict[str, Any],
                                       start_time: datetime):
        """Store ingestion metadata in Supabase"""
        try:
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            metadata = {
                "repository_url": repo_url,
                "repository_hash": repo_hash,
                "ingestion_timestamp": datetime.utcnow().isoformat(),
                "duration_seconds": round(duration, 2),
                "file_statistics": json.dumps({
                    "total_files": structure["total_files"],
                    "by_extension": dict(structure["by_extension"]),
                    "build_files": structure["build_files"],
                    "config_files": structure["config_files"],
                    "test_files": structure["test_files"]
                }),
                "graph_statistics": json.dumps(graph_results["statistics"]),
                "vector_statistics": json.dumps(vector_results["statistics"]),
                "status": "completed" if graph_results["status"] == "success" else "failed",
                "errors": json.dumps(vector_results["statistics"].get("errors", []))
            }
            
            await supabase_client.store_analysis_result({
                "analysis_type": "repository_ingestion",
                "repository": repo_url,
                "status": metadata["status"],
                "summary": metadata,
                "duration_seconds": metadata["duration_seconds"]
            })
            
            logger.debug("Ingestion metadata stored", repository=repo_url)
            
        except Exception as e:
            logger.error("Failed to store ingestion metadata", error=str(e))
    
    async def get_ingestion_status(self, repo_url: Optional[str] = None) -> Dict[str, Any]:
        """Get ingestion status for repositories"""
        if repo_url:
            if repo_url in self.ingestion_stats:
                return {
                    "repository": repo_url,
                    "status": self.ingestion_stats[repo_url]
                }
            else:
                return {"repository": repo_url, "status": "not_found"}
        else:
            return {
                "total_repositories": len(self.ingestion_stats),
                "repositories": {
                    repo: {
                        "files_processed": stats["files_processed"],
                        "last_ingestion": stats["last_ingestion"],
                        "ingestion_duration": stats["ingestion_duration"],
                        "error_count": len(stats["errors"])
                    }
                    for repo, stats in self.ingestion_stats.items()
                }
            }
    
    async def cleanup_temporary_data(self):
        """Cleanup temporary data and connections"""
        await self.search_service.close()
        await self.vector_service.close()
        await neo4j_client.close()
        await supabase_client.close()
        
        logger.info("Data ingestion service cleaned up")
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup_temporary_data()