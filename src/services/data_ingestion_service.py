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

# Intelligence modules (post-ingestion passes)
from src.intelligence.cross_file_resolver import CrossFileResolver
from src.intelligence.test_coverage_mapper import TestCoverageMapper
from src.intelligence.config_scanner import ConfigScanner
from src.intelligence.dead_code_detector import DeadCodeDetector
from src.intelligence.performance_profiler import PerformanceProfiler

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
        
        # Intelligence modules (post-ingestion analysis)
        self.cross_file_resolver = CrossFileResolver()
        self.test_coverage_mapper = TestCoverageMapper()
        self.config_scanner = ConfigScanner()
        self.dead_code_detector = DeadCodeDetector()
        self.performance_profiler = PerformanceProfiler()
        
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
            
            # 3.5 FIX 3: Collect git history stats for temporal metadata
            git_file_stats = {}
            try:
                git_file_stats = await self.git_service.get_file_history_stats(repo_url)
                logger.info(f"Git history: collected temporal stats for {len(git_file_stats)} files")
            except Exception as e:
                logger.warning(f"Git history stats skipped: {e}")
            
            # 4. Build knowledge graph with repo_type metadata + git stats
            graph_results = await self._build_knowledge_graph(
                parsed_data, repo_url, repo_hash, 
                repo_type=repo_type,
                git_file_stats=git_file_stats,
                repo_path=str(repo_path),
            )
            
            # 5. Generate embeddings and index with repo_type metadata
            vector_results = await self._generate_embeddings(
                parsed_data, repo_url, repo_hash,
                repo_type=repo_type
            )
            
            # 6. Cross-repository linking
            cross_repo_links = await self._link_cross_repository_dependencies(repo_url, repo_hash)
            
            # 6.5 Intelligence passes (cross-file resolution, test mapping, etc.)
            intelligence_results = await self._run_intelligence_passes(repo_hash, repo_url)
            
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
                "intelligence": intelligence_results,
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
    
    @staticmethod
    def _normalize_path(raw_path: str) -> str:
        """
        FIX 4: Normalize a file path to a canonical, repo-relative form.
        
        Strips leading '/' or './' and collapses duplicate slashes so that
        the same file is always identified by the same string in both
        Neo4j (File.path, File.relative_path) and Qdrant (payload.file,
        payload.relative_path).
        """
        if not raw_path:
            return raw_path
        # Use PurePosixPath to normalise separator / double-dot etc.
        from pathlib import PurePosixPath
        p = str(PurePosixPath(raw_path))
        # Strip leading "./" or "/" so it's purely relative
        while p.startswith("./"):
            p = p[2:]
        if p.startswith("/"):
            p = p.lstrip("/")
        return p

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
            # FIX 4: Normalize the path once here so every downstream consumer
            # (graph, embeddings, cross-repo linking) sees the same value.
            normalized_path = self._normalize_path(parsed_file.file_path)
            data_dict = {
                "classes": parsed_file.classes,
                "functions": parsed_file.functions,
                "routes": parsed_file.routes,
                "api_calls": parsed_file.api_calls,
                "imports": parsed_file.imports,
                "exports": parsed_file.exports,
                "dependencies": parsed_file.dependencies,
                "metadata": parsed_file.metadata,
                "language": parsed_file.language,
            }
            # Promote metadata subfields that NodeFactory / EdgeLinker read directly
            for key in ("from_imports", "eventbus", "dtos", "services",
                        "components", "configs", "error_handling", "control_flow",
                        "raw_content"):
                val = parsed_file.metadata.get(key)
                if val is not None:
                    data_dict[key] = val
            if parsed_file.raw_content:
                data_dict["raw_content"] = parsed_file.raw_content
            
            file_data = {
                "file": normalized_path,
                "language": parsed_file.language,
                "framework": parsed_file.framework,
                "data": data_dict,
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
        """
        Parse repository files using the unified parser_coordinator pipeline.
        
        FIX 3: Removed the duplicate _parse_file_batch path that routed by file
        extension only (bypassing LanguageDetector). Now always uses
        parser_coordinator.parse_repository_async → _convert_parse_result so
        language/framework detection and parsing are consistent everywhere.
        """
        try:
            parse_result = await self.parser_coordinator.parse_repository_async(
                repo_path=repo_path,
                batch_size=50
            )
            parsed_data = self._convert_parse_result(parse_result)
            
            logger.info(
                "Repository files parsed (unified path)",
                parsed_files=parsed_data["summary"]["parsed_files"],
                skipped_files=parsed_data["summary"]["skipped_files"],
                errors=len(parsed_data["summary"]["errors"])
            )
            
            return parsed_data
            
        except Exception as e:
            logger.error("Failed to parse repository files", error=str(e))
            return {
                "java": [], "typescript": [], "javascript": [],
                "python": [], "config": [], "other": [],
                "summary": {
                    "parsed_files": 0, "skipped_files": 0,
                    "by_language": defaultdict(int),
                    "errors": [str(e)]
                }
            }
    
    async def _build_knowledge_graph(self, parsed_data: Dict[str, Any], 
                                    repo_url: str, repo_hash: str,
                                    repo_type: str = None,
                                    entry_file: str = None,
                                    git_file_stats: Dict[str, Any] = None,
                                    repo_path: str = None) -> Dict[str, Any]:
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
                
                # Collect all file paths for batch BELONGS_TO linking later
                all_file_paths: list[str] = []
                
                # Unified file processing across all languages
                language_keys = [
                    ("java", parsed_data.get("java", [])),
                    ("typescript", parsed_data.get("typescript", [])),
                    ("javascript", parsed_data.get("javascript", [])),
                    ("python", parsed_data.get("python", [])),
                ]
                
                for lang_name, lang_files in language_keys:
                    for lang_file in lang_files:
                        file_path = lang_file["file"]
                        
                        nodes_created = await self.node_factory.create_nodes(
                            session,
                            lang_file["data"],
                            file_path,
                            repository=repo_url,
                            repository_hash=repo_hash,
                            repo_type=repo_type,
                            repository_name=repo_name,
                            entry_file=entry_file,
                        )
                        statistics["nodes_created"] += nodes_created.get("created", 0)
                        all_file_paths.append(file_path)
                
                # Batch BELONGS_TO: single UNWIND instead of one query per file
                if all_file_paths:
                    belongs_query = """
                    UNWIND $file_paths AS fp
                    MATCH (r:Repository {hash: $repo_hash})
                    MATCH (f:File {path: fp})
                    MERGE (f)-[:BELONGS_TO]->(r)
                    """
                    await session.run(belongs_query, {
                        "repo_hash": repo_hash,
                        "file_paths": all_file_paths,
                    })
                    logger.info(f"Batch linked {len(all_file_paths)} files to repository")

                # Link edges between nodes
                edge_results = await self.edge_linker.link_all_edges(session)
                statistics["edges_created"] = edge_results.get("relationships_created", 0)

                # Enhanced graph linking (IMPORTS, CALLS_FUNCTION, CALLS_API, type hierarchy)
                if repo_path:
                    try:
                        from src.data_ingestion.enhanced_graph_linker import EnhancedGraphLinker
                        enhanced_linker = EnhancedGraphLinker(repo_path, repo_hash=repo_hash)
                        enhanced_results = await enhanced_linker.run_all_enhancements(session)
                        enhanced_count = enhanced_results.get("total_edges_created", 0)
                        statistics["edges_created"] += enhanced_count
                        statistics["enhanced_edges_created"] = enhanced_count
                        logger.info(f"Enhanced graph linking created {enhanced_count} additional edges")
                    except Exception as e:
                        logger.warning(f"Enhanced graph linking failed: {e}")
                else:
                    logger.warning("Enhanced graph linking skipped: no repo_path provided")
                
                # FIX 5: Create Module/Package nodes and DEPENDS_ON edges
                try:
                    module_count = await self._create_module_nodes(session, repo_hash, repo_name)
                    statistics["modules_created"] = module_count
                    logger.info(f"Created {module_count} Module/Package nodes")
                except Exception as e:
                    logger.warning(f"Module node creation skipped: {e}")

                # FIX 3: Enrich File nodes with git temporal metadata
                if git_file_stats:
                    try:
                        batch_params = []
                        for fpath, fstats in git_file_stats.items():
                            batch_params.append({
                                "path": fpath,
                                "change_count": fstats.get("change_count", 0),
                                "last_modified_at": fstats.get("last_modified_at", ""),
                                "volatility_score": fstats.get("volatility_score", 0.0),
                                "contributors": fstats.get("contributors", []),
                            })
                        # Batch update in chunks
                        for chunk_start in range(0, len(batch_params), 500):
                            chunk = batch_params[chunk_start:chunk_start + 500]
                            await session.run("""
                                UNWIND $stats AS s
                                MATCH (f:File {repository_hash: $repo_hash})
                                WHERE f.path = s.path
                                   OR f.relative_path = s.path
                                SET f.change_count = s.change_count,
                                    f.last_modified_at = s.last_modified_at,
                                    f.volatility_score = s.volatility_score,
                                    f.contributors = s.contributors
                            """, {"stats": chunk, "repo_hash": repo_hash})
                        statistics["git_stats_enriched"] = len(batch_params)
                        logger.info(f"Enriched {len(batch_params)} File nodes with git temporal metadata")
                    except Exception as e:
                        logger.warning(f"Git stats enrichment failed: {e}")
                
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
    
    async def _create_module_nodes(self, session, repo_hash: str, repo_name: str) -> int:
        """
        FIX 5: Create Module/Package nodes by grouping files into their parent
        directories. Creates [:CONTAINS_FILE] from Module→File and [:DEPENDS_ON]
        between modules when one module's files import from another's.
        """
        # 1. Collect all files for this repo and derive modules
        file_result = await session.run("""
            MATCH (r:Repository {hash: $hash})<-[:BELONGS_TO]-(f:File)
            RETURN f.path AS path, f.id AS fid
        """, {"hash": repo_hash})
        
        records = [r async for r in file_result]
        if not records:
            return 0
        
        # Build module set (each directory that contains files becomes a module)
        from pathlib import PurePosixPath
        module_files: Dict[str, list] = defaultdict(list)
        for rec in records:
            fpath = rec["path"] or ""
            parts = PurePosixPath(fpath).parts
            if len(parts) >= 2:
                module_path = "/".join(parts[:-1])
            else:
                module_path = "root"
            module_files[module_path].append(rec["fid"])
        
        # 2. Create Module nodes
        module_params = []
        for mod_path, file_ids in module_files.items():
            mod_id = f"module:{repo_hash}:{mod_path}"
            mod_name = mod_path.split("/")[-1] if "/" in mod_path else mod_path
            module_params.append({
                "id": mod_id,
                "path": mod_path,
                "name": mod_name,
                "file_count": len(file_ids),
                "file_ids": file_ids,
            })
        
        await session.run("""
            UNWIND $modules AS m
            MERGE (mod:Module {id: m.id})
            SET mod.path = m.path,
                mod.name = m.name,
                mod.file_count = m.file_count,
                mod.repository_hash = $repo_hash,
                mod.repository_name = $repo_name
            WITH mod, m
            MATCH (r:Repository {hash: $repo_hash})
            MERGE (r)-[:CONTAINS_MODULE]->(mod)
            WITH mod, m
            UNWIND m.file_ids AS fid
            MATCH (f:File {id: fid})
            MERGE (mod)-[:CONTAINS_FILE]->(f)
        """, {"modules": module_params, "repo_hash": repo_hash, "repo_name": repo_name})
        
        # 3. Create DEPENDS_ON between modules based on imports
        # If File A in Module X imports from File B in Module Y, then X DEPENDS_ON Y
        await session.run("""
            MATCH (mx:Module)-[:CONTAINS_FILE]->(fa:File)-[:IMPORTS]->(fb:File)<-[:CONTAINS_FILE]-(my:Module)
            WHERE mx <> my AND mx.repository_hash = $repo_hash
            MERGE (mx)-[d:DEPENDS_ON]->(my)
            SET d.last_updated = datetime()
        """, {"repo_hash": repo_hash})
        
        return len(module_params)
    
    async def _run_intelligence_passes(self, repo_hash: str, repo_url: str) -> Dict[str, Any]:
        """
        Run all post-ingestion intelligence modules against the knowledge graph.
        
        These passes enrich the graph with cross-file call resolution, type flow,
        test coverage mapping, config/secret scanning, dead code detection,
        and performance profiling hints.
        
        Each pass is fault-tolerant: if one fails the rest still run.
        """
        results: Dict[str, Any] = {}
        
        try:
            async with neo4j_client.session() as session:
                # ── Pass 1: Cross-file call resolution + type flow ──
                try:
                    cross_file_stats = await self.cross_file_resolver.resolve(session, repo_hash)
                    results["cross_file_resolution"] = cross_file_stats
                    logger.info(
                        f"Intelligence Pass 1 (cross-file): "
                        f"{cross_file_stats.get('cross_file_calls_created', 0)} cross-calls, "
                        f"{cross_file_stats.get('type_flow_edges', 0)} type-flow edges"
                    )
                except Exception as e:
                    logger.warning(f"Intelligence Pass 1 (cross-file) failed: {e}")
                    results["cross_file_resolution"] = {"error": str(e)}
                
                # ── Pass 2: Test coverage mapping ──
                try:
                    test_stats = await self.test_coverage_mapper.map_test_coverage(session, repo_hash)
                    results["test_coverage"] = test_stats
                    logger.info(
                        f"Intelligence Pass 2 (tests): "
                        f"{test_stats.get('test_files_found', 0)} test files, "
                        f"{test_stats.get('tests_edges_created', 0)} TESTS edges"
                    )
                except Exception as e:
                    logger.warning(f"Intelligence Pass 2 (tests) failed: {e}")
                    results["test_coverage"] = {"error": str(e)}
                
                # ── Pass 3: Config & secret scanning ──
                try:
                    # Derive repo_path from git service clone cache
                    repo_path_str = ""
                    try:
                        from config.settings import RepositoryConfig
                        rc = RepositoryConfig(url=repo_url, branch="main")
                        rp = self.git_service._get_repo_path(rc)
                        repo_path_str = str(rp) if rp and rp.exists() else ""
                    except Exception:
                        pass
                    config_stats = await self.config_scanner.scan_configs(session, repo_hash, repo_path_str)
                    results["config_scanning"] = config_stats
                    logger.info(
                        f"Intelligence Pass 3 (config): "
                        f"{config_stats.get('config_entries_created', 0)} entries, "
                        f"{config_stats.get('secrets_flagged', 0)} secrets flagged"
                    )
                except Exception as e:
                    logger.warning(f"Intelligence Pass 3 (config) failed: {e}")
                    results["config_scanning"] = {"error": str(e)}
                
                # ── Pass 4: Dead code detection ──
                try:
                    dead_stats = await self.dead_code_detector.detect_dead_code(session, repo_hash)
                    results["dead_code"] = dead_stats
                    logger.info(
                        f"Intelligence Pass 4 (dead code): "
                        f"{dead_stats.get('dead_functions', 0)}/{dead_stats.get('total_functions', 0)} dead functions"
                    )
                except Exception as e:
                    logger.warning(f"Intelligence Pass 4 (dead code) failed: {e}")
                    results["dead_code"] = {"error": str(e)}
                
                # ── Pass 5: Performance profiling ──
                try:
                    perf_stats = await self.performance_profiler.profile(session, repo_hash)
                    results["performance"] = perf_stats
                    logger.info(
                        f"Intelligence Pass 5 (performance): "
                        f"{perf_stats.get('hints_created', 0)} hints across "
                        f"{perf_stats.get('functions_scanned', 0)} functions"
                    )
                except Exception as e:
                    logger.warning(f"Intelligence Pass 5 (performance) failed: {e}")
                    results["performance"] = {"error": str(e)}
        
        except Exception as e:
            logger.error(f"Intelligence passes failed to acquire session: {e}")
            results["session_error"] = str(e)
        
        return results

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
                
                # Routes (FIX 1: embed actual route body code)
                for route in file_data.get("routes", []):
                    route_body = route.get("body", "") or ""
                    route_summary = (
                        f"Route: {route.get('path', '')} "
                        f"Method: {route.get('method', '')} "
                        f"Handler: {route.get('handler_method', route.get('name', ''))}"
                    )
                    artifacts.append({
                        "type": "route",
                        "name": route.get("handler_method", route.get("name", "")),
                        "language": "java",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": route_body[:3000] if route_body else route_summary,
                        "text": route_summary,
                        "metadata": {
                            "path": route.get("path"),
                            "method": route.get("method"),
                            "handler": route.get("handler_method", route.get("name", "")),
                            "error_handling": route.get("error_handling", []),
                        },
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
                
                # Services (FIX 1: embed actual class body)
                for svc in file_data.get("services", []):
                    svc_name = svc.get("name", "") if isinstance(svc, dict) else str(svc)
                    svc_body = svc.get("body", "") if isinstance(svc, dict) else ""
                    artifacts.append({
                        "type": "service",
                        "name": svc_name,
                        "language": "java",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": svc_body[:3000] if svc_body else f"Service class {svc_name} in {file_name}",
                        "text": f"Service: {svc_name} File: {file_name}",
                        "metadata": svc if isinstance(svc, dict) else {"name": svc_name},
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    })
                
                # Classes (FIX 1: embed actual class body)
                for cls in file_data.get("classes", []):
                    cls_name = cls.get("name", "") if isinstance(cls, dict) else str(cls)
                    cls_body = cls.get("body", "") if isinstance(cls, dict) else ""
                    methods_list = cls.get("methods", []) if isinstance(cls, dict) else []
                    method_names = ", ".join([m.get("name", "") if isinstance(m, dict) else str(m) for m in methods_list[:10]])
                    artifacts.append({
                        "type": "class",
                        "name": cls_name,
                        "language": "java",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": cls_body[:3000] if cls_body else (f"Class {cls_name} with methods: {method_names}" if method_names else f"Class {cls_name}"),
                        "text": f"Class: {cls_name} Methods: {method_names}" if method_names else f"Class: {cls_name}",
                        "metadata": cls if isinstance(cls, dict) else {"name": cls_name},
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    })
                
                # Java Functions/Methods (NEW: previously not embedded individually)
                for func in file_data.get("functions", []):
                    if not isinstance(func, dict):
                        continue
                    func_name = func.get("name", "")
                    func_body = func.get("body", "")
                    if not func_name:
                        continue
                    artifacts.append({
                        "type": "function",
                        "name": func_name,
                        "language": "java",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": func_body[:3000] if func_body else f"Java method {func_name}",
                        "text": f"Java Method: {func_name} Returns: {func.get('return_type', 'void')}",
                        "metadata": {
                            "return_type": func.get("return_type"),
                            "parameters": func.get("parameters", []),
                            "calls": func.get("calls", []),
                            "error_handling": func.get("error_handling", []),
                        },
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    })
            
            # FIX 2: Extract artifacts from Python files (previously missing entirely)
            for py_file in parsed_data.get("python", []):
                file_data = py_file.get("data", {})
                file_path = py_file.get("file", "")
                file_name = file_path.split('/')[-1] if file_path else ""
                
                # Python classes (FIX 1: embed actual class body)
                for cls in file_data.get("classes", []):
                    cls_name = cls.get("name", "") if isinstance(cls, dict) else str(cls)
                    cls_body = cls.get("body", "") if isinstance(cls, dict) else ""
                    methods_list = cls.get("methods", []) if isinstance(cls, dict) else []
                    method_names = ", ".join([m.get("name", "") if isinstance(m, dict) else str(m) for m in methods_list[:10]])
                    artifacts.append({
                        "type": "class",
                        "name": cls_name,
                        "language": "python",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": cls_body[:3000] if cls_body else (f"Python class {cls_name} with methods: {method_names}" if method_names else f"Python class {cls_name}"),
                        "text": f"Python Class: {cls_name} Methods: {method_names}" if method_names else f"Python Class: {cls_name}",
                        "metadata": cls if isinstance(cls, dict) else {"name": cls_name},
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    })
                
                # Python functions (FIX 1: embed actual function body)
                for func in file_data.get("functions", []):
                    if not isinstance(func, dict):
                        continue
                    func_name = func.get("name", "")
                    func_body = func.get("body", "")
                    func_desc = func.get("docstring", "") or func.get("description", "")
                    params = func.get("parameters", func.get("params", []))
                    if isinstance(params, list):
                        param_parts = []
                        for p in params:
                            if isinstance(p, dict):
                                pname = p.get("name", "")
                                ptype = p.get("type", "")
                                param_parts.append(f"{pname}: {ptype}" if ptype else pname)
                            else:
                                param_parts.append(str(p))
                        param_str = ", ".join(param_parts)
                    else:
                        param_str = str(params)
                    artifacts.append({
                        "type": "function",
                        "name": func_name,
                        "language": "python",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": func_body[:3000] if func_body else f"def {func_name}({param_str})",
                        "text": f"Python Function: {func_name}({param_str}) {func_desc[:200]}",
                        "metadata": {
                            "parameters": params,
                            "return_type": func.get("return_type"),
                            "calls": func.get("calls", []),
                            "error_handling": func.get("error_handling", []),
                            "is_async": func.get("is_async", False),
                        },
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    })
                
                # Python routes (FastAPI / Flask / Django)
                for route in file_data.get("routes", []):
                    route_path = route.get("path", "") if isinstance(route, dict) else str(route)
                    method = route.get("method", "GET") if isinstance(route, dict) else "GET"
                    handler = route.get("handler", "") if isinstance(route, dict) else ""
                    artifacts.append({
                        "type": "route",
                        "name": handler or route_path,
                        "language": "python",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": f"Python Route: {method} {route_path} Handler: {handler}",
                        "text": f"Python Route: {method} {route_path} Handler: {handler}",
                        "metadata": route if isinstance(route, dict) else {"path": route_path},
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
                
                # Components (FIX 1: embed actual component body)
                components = file_data.get("components", []) or file_metadata.get("components", [])
                for comp in components:
                    comp_body = comp.get("body", "") or ""
                    artifacts.append({
                        "type": "react_component",
                        "name": comp.get("name", ""),
                        "language": "typescript",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": comp_body[:3000] if comp_body else f"React component {comp.get('name', '')}",
                        "text": f"React Component: {comp.get('name', '')} "
                               f"Props: {', '.join([p.get('name', '') for p in comp.get('props', [])])} "
                               f"Hooks: {', '.join(comp.get('hooks', []))}",
                        "metadata": {
                            "name": comp.get("name"),
                            "props": comp.get("props", []),
                            "hooks": comp.get("hooks", [])
                        },
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    })
                
                # Functions/Methods from TypeScript (FIX 1: embed actual function body)
                functions = file_data.get("functions", [])
                for func in functions:
                    func_body = func.get("body", "") or ""
                    artifacts.append({
                        "type": "function",
                        "name": func.get("name", ""),
                        "language": "typescript",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": func_body[:3000] if func_body else f"Function {func.get('name', '')}",
                        "text": f"TS Function: {func.get('name', '')} calls: {', '.join(func.get('calls', [])[:5])}",
                        "metadata": {
                            "name": func.get("name"),
                            "calls": func.get("calls", []),
                            "is_async": func.get("is_async", False),
                        },
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
                
                # Components (FIX 1: embed actual component body)
                components = file_data.get("components", []) or file_metadata.get("components", [])
                for comp in components:
                    comp_body = comp.get("body", "") or ""
                    artifacts.append({
                        "type": "react_component",
                        "name": comp.get("name", ""),
                        "language": "javascript",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": comp_body[:3000] if comp_body else f"React component {comp.get('name', '')}",
                        "text": f"React Component: {comp.get('name', '')} Props: {', '.join([p.get('name', '') for p in comp.get('props', [])])}",
                        "metadata": {
                            "name": comp.get("name"),
                            "props": comp.get("props", [])
                        },
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    })
                
                # Functions (FIX 1: embed actual function body)
                functions = file_data.get("functions", [])
                for func in functions:
                    func_body = func.get("body", "") or ""
                    artifacts.append({
                        "type": "function",
                        "name": func.get("name", ""),
                        "language": "javascript",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": func_body[:3000] if func_body else f"Function {func.get('name', '')}",
                        "text": f"JS Function: {func.get('name', '')} calls: {', '.join(func.get('calls', [])[:5])}",
                        "metadata": {
                            "name": func.get("name"),
                            "calls": func.get("calls", []),
                            "is_async": func.get("is_async", False),
                        },
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
            # Traverse File-[:BELONGS_TO]->Repository (correct direction)
            query = """
            MATCH (file:File)-[:BELONGS_TO]->(source:Repository {hash: $repo_hash})
            MATCH (file)-[:CONTAINS]->(api:ApiCall)
            WHERE api.url IS NOT NULL AND api.url CONTAINS 'api/'
            OPTIONAL MATCH (targetRoute:VertxRoute)
            WHERE targetRoute.path = api.url
               OR replace(api.url, '/api/v1/', '/') = targetRoute.path
               OR replace(api.url, '/api/', '/') = targetRoute.path
            OPTIONAL MATCH (targetFile:File)-[:CONTAINS]->(targetRoute)
            OPTIONAL MATCH (targetFile)-[:BELONGS_TO]->(targetRepo:Repository)
            WHERE targetRepo.hash <> $repo_hash
            RETURN api, targetRoute, targetRepo
            LIMIT 100
            """

            async with neo4j_client.session() as session:
                results = await session.run(query, {"repo_hash": repo_hash})
                async for record in results:
                    if record.get("targetRepo"):
                        links["api_calls"].append({
                            "source_repository": repo_url,
                            "target_repository": record["targetRepo"].get("url"),
                            "api_path": record["api"].get("url"),
                            "target_route": record["targetRoute"].get("path"),
                        })

            eventbus_query = """
            MATCH (file:File)-[:BELONGS_TO]->(source:Repository {hash: $repo_hash})
            MATCH (file)-[:CONTAINS]->(eventbus:EventBusNode)
            WHERE eventbus.address IS NOT NULL
            OPTIONAL MATCH (targetEventbus:EventBusNode {address: eventbus.address})
            WHERE targetEventbus <> eventbus
            OPTIONAL MATCH (targetFile:File)-[:CONTAINS]->(targetEventbus)
            OPTIONAL MATCH (targetFile)-[:BELONGS_TO]->(targetRepo:Repository)
            WHERE targetRepo.hash <> $repo_hash
            RETURN eventbus, targetEventbus, targetRepo
            LIMIT 100
            """

            async with neo4j_client.session() as session:
                results = await session.run(eventbus_query, {"repo_hash": repo_hash})
                async for record in results:
                    if record.get("targetRepo"):
                        links["eventbus_flows"].append({
                            "source_repository": repo_url,
                            "target_repository": record["targetRepo"].get("url"),
                            "eventbus_address": record["eventbus"].get("address"),
                            "type": record["eventbus"].get("type"),
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
                        "python": parsed.get("python", []) if "python" in file_types else [],
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
    
    @staticmethod
    def _extract_all_nodes(repo_parsed: Dict[str, Any], repo_name: str) -> List[Dict[str, Any]]:
        """
        FIX 5 helper: Single place to extract a flat node list from per-repo
        parsed data.  Used by both the public build_knowledge_graph (for
        returning nodes to the caller) and generate_embeddings (for embedding).
        Covers all languages: java, typescript, javascript, python.
        """
        nodes: List[Dict[str, Any]] = []

        # Mapping of (language_key, artifact_key, type_label, name_key)
        _ARTIFACT_MAP = [
            # Java
            ("java", "routes",   "route",     "handler_method"),
            ("java", "eventbus", "eventbus",  "address"),
            ("java", "dtos",     "dto",       "name"),
            ("java", "services", "service",   "name"),
            ("java", "classes",  "class",     "name"),
            ("java", "functions","function",  "name"),
            # TypeScript
            ("typescript", "components", "component",    "name"),
            ("typescript", "functions",  "function",     "name"),
            ("typescript", "api_calls",  "api_call",     "url"),
            ("typescript", "routes",     "route",        "path"),
            # JavaScript
            ("javascript", "components", "component",    "name"),
            ("javascript", "functions",  "function",     "name"),
            ("javascript", "api_calls",  "api_call",     "url"),
            ("javascript", "routes",     "route",        "path"),
            # Python
            ("python", "classes",   "class",    "name"),
            ("python", "functions", "function", "name"),
            ("python", "routes",    "route",    "path"),
        ]

        for lang_key, artifact_key, type_label, name_key in _ARTIFACT_MAP:
            for lang_file in repo_parsed.get(lang_key, []):
                file_data = lang_file.get("data", {})
                file_path = lang_file.get("file", "")
                for item in file_data.get(artifact_key, []):
                    if not isinstance(item, dict):
                        continue
                    nodes.append({
                        "type": type_label,
                        "name": item.get(name_key, ""),
                        "repository": repo_name,
                        "file": file_path,
                        "language": lang_key,
                        "data": item,
                    })

        return nodes

    async def build_knowledge_graph(
        self,
        parsed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build knowledge graph from parsed data.
        
        FIX 5: Now a thin wrapper around _build_knowledge_graph.
        Previously this duplicated node-extraction logic and only covered
        a subset of artifact types.  Now delegates graph writing entirely
        to _build_knowledge_graph and uses _extract_all_nodes for the
        returned node list.
        
        Args:
            parsed_data: Dict mapping repo names to their parsed content
        
        Returns:
            Graph building results with statistics
        """
        total_nodes = 0
        total_edges = 0
        all_nodes: List[Dict[str, Any]] = []
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
                
                # Use the shared helper so the node list always matches
                # what _build_knowledge_graph wrote to Neo4j.
                all_nodes.extend(self._extract_all_nodes(repo_parsed, repo_name))
                
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