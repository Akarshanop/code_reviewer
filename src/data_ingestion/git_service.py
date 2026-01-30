import logging
import os
import shutil
import tempfile
from git import Repo
from config.settings import settings

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
        with tempfile.TemporaryDirectory() as tmp_dir:
            auth_url = self._get_auth_url(repo_url)
            if os.path.exists(tmp_dir):
                repo = Repo(tmp_dir)
                repo.remotes.origin.pull(branch)
            else:
                repo = Repo.clone_from(auth_url, tmp_dir, branch=branch)
            logger.info(f"Cloned/pulled repo to {tmp_dir}")
            return tmp_dir  

    def ingest_repos(self):
        vertx_path = self.clone_or_pull(self.vertx_repo_url)
        react_path = self.clone_or_pull(self.react_repo_url)
        # Call parsers here in future steps
        return {"vertx": vertx_path, "react": react_path}
    
    def ingest_repos(self):
    vertx_path = self.clone_or_pull(self.vertx_repo_url)
    react_path = self.clone_or_pull(self.react_repo_url)
    
    # Parse
    java_parser = JavaParser()
    ts_parser = TsReactParser()
    vertx_parsed = {file: java_parser.parse_file(Path(vertx_path) / file) for file in os.listdir(vertx_path) if file.endswith('.java')}
    react_parsed = {file: ts_parser.parse_file(Path(react_path) / file) for file in os.listdir(react_path) if file.endswith('.tsx') or file.endswith('.ts')}
    
    # Build graph
    node_factory = NodeFactory()
    edge_linker = EdgeLinker()
    async with get_neo4j_driver() as driver:
        for parsed in vertx_parsed.values():
            await node_factory.create_nodes(driver, parsed, 'vertx')
        for parsed in react_parsed.values():
            await node_factory.create_nodes(driver, parsed, 'react')
        await edge_linker.link_edges(driver, vertx_parsed, react_parsed)

    return {"status": "ingested"}

