"""
Enterprise Git Service for Multi-Repository Code Ingestion
Supports 10+ repositories with parallel cloning and incremental updates
"""
import logging
import os
import shutil
import asyncio
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import tempfile

from git import Repo, GitCommandError, InvalidGitRepositoryError
from git.remote import RemoteProgress

from config.settings import settings, RepositoryConfig
from src.core.logger import get_logger
from src.core.exceptions import CodeAnalysisException

logger = get_logger(__name__)


class CloneProgress(RemoteProgress):
    """Progress handler for git clone operations"""
    
    def __init__(self, repo_name: str):
        super().__init__()
        self.repo_name = repo_name
        self.last_percent = 0
    
    def update(self, op_code, cur_count, max_count=None, message=''):
        if max_count:
            percent = int((cur_count / max_count) * 100)
            if percent >= self.last_percent + 10:  # Log every 10%
                logger.debug(
                    f"Cloning {self.repo_name}: {percent}%",
                    repo=self.repo_name,
                    progress=percent
                )
                self.last_percent = percent

# dataclass is used to create a class that has default values for its attributes
@dataclass
class RepositoryState:
    """State tracking for a repository"""
    url: str
    name: str
    branch: str
    local_path: Optional[Path] = None
    last_commit_hash: Optional[str] = None
    last_sync: Optional[datetime] = None
    file_count: int = 0
    status: str = "pending"  # pending, cloning, cloned, syncing, synced, error
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def hash(self) -> str:
        return hashlib.md5(self.url.encode()).hexdigest()[:12]


class GitService:
    """
    Enterprise-grade Git service for managing multiple repositories.
    Supports parallel cloning, incremental updates, and caching.
    """
    
    def __init__(self):
        self.git_token = settings.GIT_TOKEN
        self.git_username = settings.GIT_USERNAME
        self.work_dir = settings.GIT_WORK_DIR
        self.clone_depth = settings.GIT_CLONE_DEPTH
        self.timeout = settings.GIT_TIMEOUT
        
        # Repository state tracking
        self._repositories: Dict[str, RepositoryState] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._lock = asyncio.Lock()
        
        # Ensure work directory exists
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "GitService initialized",
            work_dir=str(self.work_dir),
            max_workers=4
        )
    
    def _get_auth_url(self, url: str) -> str:
        """Add authentication to repository URL"""
        if not url:
            return url
        
        if self.git_token:
            if 'github.com' in url:
                # GitHub: https://token@github.com/...
                return url.replace('https://', f'https://{self.git_token}@')
            elif 'gitlab.com' in url or 'gitlab' in url.lower():
                # GitLab: https://oauth2:token@gitlab.com/...
                return url.replace('https://', f'https://oauth2:{self.git_token}@')
            elif 'bitbucket.org' in url:
                # Bitbucket: https://username:app_password@bitbucket.org/...
                username = self.git_username or 'x-token-auth'
                return url.replace('https://', f'https://{username}:{self.git_token}@')
            elif 'dev.azure.com' in url or 'visualstudio.com' in url:
                # Azure DevOps
                return url.replace('https://', f'https://{self.git_token}@')
            else:
                # Generic: try token auth
                return url.replace('https://', f'https://{self.git_token}@')
        
        return url
    
    def _get_repo_path(self, repo_config: RepositoryConfig) -> Path:
        """Get local path for repository"""
        repo_hash = hashlib.md5(repo_config.url.encode()).hexdigest()[:12]
        return self.work_dir / f"{repo_config.name}_{repo_hash}"
    
    def _clone_repository_sync(self, repo_config: RepositoryConfig, 
                               force_fresh: bool = False) -> Tuple[Path, Dict[str, Any]]:
        """Synchronously clone or update a repository (runs in thread pool)"""
        repo_path = self._get_repo_path(repo_config)
        auth_url = self._get_auth_url(repo_config.url)
        result = {
            "status": "pending",
            "commit_hash": None,
            "file_count": 0,
            "error": None
        }
        
        try:
            # Check if repo exists and is valid
            if repo_path.exists() and not force_fresh:
                try:
                    existing_repo = Repo(repo_path)
                    
                    # Fetch and update
                    logger.debug(f"Updating existing repository: {repo_config.name}")
                    
                    # Reset any local changes
                    existing_repo.head.reset(index=True, working_tree=True)
                    
                    # Fetch latest
                    origin = existing_repo.remotes.origin
                    origin.fetch()
                    
                    # Checkout branch
                    if repo_config.branch in existing_repo.heads:
                        existing_repo.heads[repo_config.branch].checkout()
                    else:
                        existing_repo.git.checkout(f'origin/{repo_config.branch}', b=repo_config.branch)
                    
                    # Pull latest
                    origin.pull(repo_config.branch)
                    
                    result["status"] = "updated"
                    result["commit_hash"] = str(existing_repo.head.commit.hexsha)
                    result["file_count"] = sum(1 for _ in repo_path.rglob("*") if _.is_file())
                    
                    logger.info(
                        f"Repository updated: {repo_config.name}",
                        commit=result["commit_hash"][:8]
                    )
                    
                    return repo_path, result
                    
                except InvalidGitRepositoryError:
                    logger.warning(f"Invalid repository at {repo_path}, removing...")
                    shutil.rmtree(repo_path, ignore_errors=True)
            
            if repo_path.exists() and force_fresh:
                shutil.rmtree(repo_path, ignore_errors=True)
            
            logger.info(
                f"Cloning repository: {repo_config.name}",
                url=repo_config.url,
                branch=repo_config.branch
            )
            
            clone_kwargs = {
                "branch": repo_config.branch,
                "progress": CloneProgress(repo_config.name)
            }
            
            if self.clone_depth > 0:
                clone_kwargs["depth"] = self.clone_depth
            
            repo = Repo.clone_from(auth_url, repo_path, **clone_kwargs)
            
            result["status"] = "cloned"
            result["commit_hash"] = str(repo.head.commit.hexsha)
            result["file_count"] = sum(1 for _ in repo_path.rglob("*") if _.is_file())
            
            logger.info(
                f"Repository cloned successfully: {repo_config.name}",
                commit=result["commit_hash"][:8],
                files=result["file_count"]
            )
            
            return repo_path, result
            
        except GitCommandError as e:
            error_msg = f"Git command failed: {str(e)}"
            logger.error(error_msg, repo=repo_config.name)
            result["status"] = "error"
            result["error"] = error_msg
            
            # Cleanup on error
            if repo_path.exists():
                shutil.rmtree(repo_path, ignore_errors=True)
            
            raise CodeAnalysisException(error_msg, file_path=repo_config.url)
            
        except Exception as e:
            error_msg = f"Clone failed: {str(e)}"
            logger.error(error_msg, repo=repo_config.name)
            result["status"] = "error"
            result["error"] = error_msg
            
            raise CodeAnalysisException(error_msg, file_path=repo_config.url)
    
    async def clone_or_update(self, repo_config: RepositoryConfig, 
                             force_fresh: bool = False) -> Tuple[Path, Dict[str, Any]]:
        """Asynchronously clone or update a repository"""
        loop = asyncio.get_event_loop()
        
        result = await loop.run_in_executor(
            self._executor,
            self._clone_repository_sync,
            repo_config,
            force_fresh
        )
        
        async with self._lock:
            state = RepositoryState(
                url=repo_config.url,
                name=repo_config.name,
                branch=repo_config.branch,
                local_path=result[0],
                last_commit_hash=result[1].get("commit_hash"),
                last_sync=datetime.utcnow(),
                file_count=result[1].get("file_count", 0),
                status=result[1].get("status", "unknown")
            )
            self._repositories[repo_config.url] = state
        
        return result
    
    def clone_or_pull(self, repo_url: str, branch: str = 'main') -> str:
        """
        Legacy synchronous method for backward compatibility.
        Returns the path to the cloned repository.
        """
        repo_config = RepositoryConfig(url=repo_url, branch=branch)
        path, _ = self._clone_repository_sync(repo_config)
        return str(path)
    
    async def clone_all_repositories(
        self, 
        force_fresh: bool = False,
        force_refresh: bool = False,  # Alias for force_fresh
        repository_filter: List[str] = None  # Optional filter for specific repos
    ) -> Dict[str, Any]:
        """Clone or update all configured repositories in parallel"""
        # Support both parameter names
        force = force_fresh or force_refresh
        
        repos = settings.get_repositories()
        
        # Filter repositories if specified
        if repository_filter:
            repos = [r for r in repos if r.name in repository_filter or r.url in repository_filter]
        
        if not repos:
            logger.warning("No repositories configured")
            return {
                "status": "warning",
                "message": "No repositories configured",
                "repositories": []
            }
        
        logger.info(f"Starting parallel clone/update for {len(repos)} repositories")
        
        results = {
            "status": "success",
            "total_repositories": len(repos),
            "successful": 0,
            "failed": 0,
            "repositories": []
        }
        
        # Clone repositories in parallel
        tasks = [
            self.clone_or_update(repo, force)
            for repo in repos
        ]
        
        completed = await asyncio.gather(*tasks, return_exceptions=True)
        
        for repo, result in zip(repos, completed):
            if isinstance(result, Exception):
                results["failed"] += 1
                results["repositories"].append({
                    "name": repo.name,
                    "url": repo.url,
                    "status": "error",
                    "error": str(result)
                })
            else:
                path, info = result
                results["successful"] += 1
                results["repositories"].append({
                    "name": repo.name,
                    "url": repo.url,
                    "status": info.get("status"),
                    "commit_hash": info.get("commit_hash"),
                    "file_count": info.get("file_count"),
                    "local_path": str(path)
                })
        
        if results["failed"] > 0:
            results["status"] = "partial" if results["successful"] > 0 else "failed"
        
        logger.info(
            "Repository clone completed",
            total=results["total_repositories"],
            successful=results["successful"],
            failed=results["failed"]
        )
        
        return results
    
    async def get_repository_info(self, repo_url: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific repository"""
        if repo_url in self._repositories:
            state = self._repositories[repo_url]
            return {
                "url": state.url,
                "name": state.name,
                "branch": state.branch,
                "local_path": str(state.local_path) if state.local_path else None,
                "last_commit": state.last_commit_hash,
                "last_sync": state.last_sync.isoformat() if state.last_sync else None,
                "file_count": state.file_count,
                "status": state.status
            }

        # Fallback: after app restarts, in-memory state is empty; infer from disk.
        repo_config = next((r for r in settings.get_repositories() if r.url == repo_url), None)
        if not repo_config:
            return None

        repo_path = self._get_repo_path(repo_config)
        if not repo_path.exists():
            return {
                "url": repo_config.url,
                "name": repo_config.name,
                "branch": repo_config.branch,
                "local_path": None,
                "last_commit": None,
                "last_sync": None,
                "file_count": 0,
                "status": "not_cloned",
            }

        try:
            repo = Repo(repo_path)
            commit_hash = str(repo.head.commit.hexsha) if repo.head and repo.head.commit else None
            file_count = sum(1 for _ in repo_path.rglob("*") if _.is_file())
            return {
                "url": repo_config.url,
                "name": repo_config.name,
                "branch": repo_config.branch,
                "local_path": str(repo_path),
                "last_commit": commit_hash,
                "last_sync": None,
                "file_count": file_count,
                "status": "cloned",
            }
        except Exception as e:
            logger.debug(f"Failed to read repository state from disk: {e}", repo=repo_config.name)
            return {
                "url": repo_config.url,
                "name": repo_config.name,
                "branch": repo_config.branch,
                "local_path": str(repo_path),
                "last_commit": None,
                "last_sync": None,
                "file_count": 0,
                "status": "error",
            }
    
    async def get_all_repository_states(self) -> List[Dict[str, Any]]:
        """Get state of all managed repositories"""
        return [
            {
                "url": state.url,
                "name": state.name,
                "branch": state.branch,
                "last_commit": state.last_commit_hash,
                "last_sync": state.last_sync.isoformat() if state.last_sync else None,
                "file_count": state.file_count,
                "status": state.status,
                "error": state.error_message
            }
            for state in self._repositories.values()
        ]
    
    async def get_changed_files(self, repo_url: str, 
                               since_commit: str = None) -> List[Dict[str, Any]]:
        """Get list of changed files since a specific commit"""
        if repo_url not in self._repositories:
            return []
        
        state = self._repositories[repo_url]
        if not state.local_path or not state.local_path.exists():
            return []
        
        try:
            repo = Repo(state.local_path)
            
            if since_commit:
                # Get diff between commits
                diff = repo.git.diff(since_commit, state.last_commit_hash, name_status=True)
            else:
                # Get all tracked files
                diff = repo.git.ls_files()
            
            changed_files = []
            for line in diff.split('\n'):
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        changed_files.append({
                            "status": parts[0],  # A=added, M=modified, D=deleted
                            "path": parts[1]
                        })
                    else:
                        changed_files.append({
                            "status": "tracked",
                            "path": parts[0]
                        })
            
            return changed_files
            
        except Exception as e:
            logger.error(f"Failed to get changed files: {e}", repo=repo_url)
            return []
    
    # =========================================================================
    # FIX 3: Git history integration — per-file temporal metadata
    # =========================================================================
    
    async def get_file_history_stats(
        self, repo_url: str, file_paths: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get per-file temporal metadata from git log:
          - change_count: total commits touching the file
          - last_modified_at: timestamp of most recent commit
          - volatility_score: normalized 0-1 (high = changed often)
          - contributors: unique author emails
        
        Returns dict keyed by repo-relative file path.
        """
        if repo_url not in self._repositories:
            return {}
        
        state = self._repositories[repo_url]
        if not state.local_path or not state.local_path.exists():
            return {}
        
        def _compute_stats() -> Dict[str, Dict[str, Any]]:
            try:
                repo = Repo(state.local_path)
                stats: Dict[str, Dict[str, Any]] = {}
                max_changes = 1
                
                # Use --numstat to get per-file change counts in one pass
                log_output = repo.git.log(
                    '--pretty=format:%H|%ae|%aI',
                    '--name-only',
                    '--diff-filter=AMRC',
                    '--no-merges',
                )
                
                current_commit = None
                current_author = None
                current_date = None
                
                for line in log_output.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    if '|' in line and line.count('|') == 2:
                        parts = line.split('|', 2)
                        current_commit = parts[0]
                        current_author = parts[1]
                        current_date = parts[2]
                    else:
                        fpath = line
                        if file_paths and fpath not in file_paths:
                            continue
                        if fpath not in stats:
                            stats[fpath] = {
                                'change_count': 0,
                                'last_modified_at': current_date,
                                'contributors': set(),
                            }
                        stats[fpath]['change_count'] += 1
                        if current_author:
                            stats[fpath]['contributors'].add(current_author)
                        # Keep the most recent date
                        if current_date and current_date > stats[fpath]['last_modified_at']:
                            stats[fpath]['last_modified_at'] = current_date
                        if stats[fpath]['change_count'] > max_changes:
                            max_changes = stats[fpath]['change_count']
                
                # Normalize volatility
                for fpath, info in stats.items():
                    info['volatility_score'] = round(info['change_count'] / max_changes, 3)
                    info['contributors'] = list(info['contributors'])
                
                return stats
                
            except Exception as e:
                logger.error(f"Failed to compute file history stats: {e}", repo=repo_url)
                return {}
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _compute_stats)
    
    async def cleanup_repository(self, repo_url: str) -> bool:
        """Remove a cloned repository"""
        if repo_url in self._repositories:
            state = self._repositories[repo_url]
            if state.local_path and state.local_path.exists():
                try:
                    shutil.rmtree(state.local_path, ignore_errors=True)
                    del self._repositories[repo_url]
                    logger.info(f"Cleaned up repository: {state.name}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to cleanup repository: {e}")
                    return False
        return False
    
    async def cleanup_all(self):
        """Clean up all cloned repositories"""
        for repo_url in list(self._repositories.keys()):
            await self.cleanup_repository(repo_url)
        
        logger.info("All repositories cleaned up")
    
    async def ingest_repos(self) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility.
        Triggers full ingestion of all configured repositories.
        """
        # Import here to avoid circular imports
        from src.services.data_ingestion_service import DataIngestionService
        
        results = await self.clone_all_repositories()
        
        if results["status"] == "failed":
            return {
                "status": "failed",
                "error": "All repository clones failed",
                "details": results
            }
        
        # Trigger ingestion for each successful clone
        ingestion_service = DataIngestionService()
        await ingestion_service.initialize()
        
        ingestion_results = []
        for repo_info in results["repositories"]:
            if repo_info["status"] in ["cloned", "updated"]:
                try:
                    repo_result = await ingestion_service.ingest_repository(
                        repo_url=repo_info["url"],
                        branch=repo_info.get("branch", "main"),
                        incremental=(repo_info["status"] == "updated")
                    )
                    ingestion_results.append(repo_result)
                except Exception as e:
                    logger.error(f"Ingestion failed for {repo_info['name']}: {e}")
                    ingestion_results.append({
                        "repository": repo_info["url"],
                        "status": "failed",
                        "error": str(e)
                    })
        
        return {
            "status": "completed",
            "clone_results": results,
            "ingestion_results": ingestion_results
        }
    
    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


# Singleton instance for import
git_service = GitService()