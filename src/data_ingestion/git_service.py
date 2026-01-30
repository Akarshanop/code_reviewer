import logging
import os
import shutil
from pathlib import Path
import tempfile
from git import Repo
from config.settings import settings
from src.data_ingestion.parsers.java_parser import JavaParser
from src.data_ingestion.parsers.react_parser import TsReactParser
from src.data_ingestion.graph_builder.node_factory import NodeFactory
from src.data_ingestion.graph_builder.edge_linker import EdgeLinker
from src.database.neo4j_client import get_neo4j_driver

logger = logging.getLogger(__name__)

class GitService:
    def __init__(self):
        self.vertx_repo_url = settings.GIT_REPO_VERTX
        self.react_repo_url = settings.GIT_REPO_REACT
        self.git_token = settings.GIT_TOKEN

    def _get_auth_url(self, url):
        if self.git_token and 'github.com' in url:
            return url.replace('https://', f'https://{self.git_token}@')
        return url

    def clone_or_pull(self, repo_url, branch='main'):
        tmp_dir = tempfile.mkdtemp()
        try:
            auth_url = self._get_auth_url(repo_url)
            repo = Repo.clone_from(auth_url, tmp_dir, branch=branch)
            logger.info(f"Cloned/pulled repo to {tmp_dir}")
            return tmp_dir
        except Exception as e:
            logger.error(f"Git operation failed for {repo_url}: {str(e)}")
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            raise

    async def ingest_repos(self):
        vertx_path = self.clone_or_pull(self.vertx_repo_url)
        react_path = self.clone_or_pull(self.react_repo_url)
        try:
            # Parse (using improved parsers)
            java_parser = JavaParser()
            ts_react_parser = TsReactParser()
            vertx_files = [f for f in os.listdir(vertx_path) if f.endswith('.java')]
            react_files = [f for f in os.listdir(react_path) if f.endswith(('.tsx', '.ts'))]
            vertx_parsed = {file: java_parser.parse_file(Path(vertx_path) / file) for file in vertx_files}
            react_parsed = {file: ts_react_parser.parse_file(Path(react_path) / file) for file in react_files}
            
            # Build graph with improved data
            node_factory = NodeFactory()
            edge_linker = EdgeLinker()
            driver = await anext(get_neo4j_driver())
            try:
                for file, parsed in vertx_parsed.items():
                    await node_factory.create_nodes(driver, parsed, str(Path(vertx_path) / file))
                for file, parsed in react_parsed.items():
                    await node_factory.create_nodes(driver, parsed, str(Path(react_path) / file))
                await edge_linker.link_all_edges()  # Use new method
            finally:
                await driver.close()
            
            logger.info("Ingestion completed")
            return {"status": "ingested", "parsed_vertx": vertx_parsed, "parsed_react": react_parsed}  # Return for debugging
        finally:
            shutil.rmtree(vertx_path, ignore_errors=True)
            shutil.rmtree(react_path, ignore_errors=True)