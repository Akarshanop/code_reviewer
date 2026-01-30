import logging
from git import Repo
from src.services.graph_service import GraphService
import tempfile
import os

logger = logging.getLogger(__name__)

class PRAnalyzer:
    """Tool for analyzing a GitHub pull request and assessing its impact and risks."""

    async def analyze_pr(self, pr_url: str) -> dict:
        try:
            parts = pr_url.split('/')
            if len(parts) < 7:
                return {"error": "Invalid PR URL"}
            owner, repo_name = parts[-4], parts[-3]
            pr_number = int(parts[-1])

            with tempfile.TemporaryDirectory() as tmp_dir:
                repo_url = f"https://github.com/{owner}/{repo_name}.git"
                repo = Repo.clone_from(repo_url, tmp_dir, depth=1, branch='main')
                
                pr_ref = f"pull/{pr_number}/head:pr-{pr_number}"
                repo.remotes.origin.fetch(pr_ref)
                repo.git.checkout(f"pr-{pr_number}")

                diff = repo.git.diff('main', 'pr-{pr_number}')

                # Analyze each changed file for impact
                graph_service = GraphService()
                impacts = []
                changed_files = []
                for line in diff.splitlines():
                    if line.startswith('+++ b/'):
                        file = line[6:].strip()
                        changed_files.append(file)
                        impact = await graph_service.query_impact(file)
                        impacts.append({"file": file, "impact": impact})

                # Look for risks
                risks = []
                if 'UserDTO' in diff:
                    risks.append("Potential breaking change in UserDTO")

                # Check for EventBus changes without tests
                if 'EventBus' in diff and 'test' not in diff.lower():
                    risks.append("EventBus changes without corresponding tests")

                return {
                    "pr": pr_url,
                    "changed_files": changed_files,
                    "high_impact_changes": [i for i in impacts if len(i['impact']) > 5],
                    "risks": risks,
                    "recommendation": "Requires QA review" if risks else "Looks safe"
                }
        except Exception as e:
            logger.error(f"PR analysis failed for {pr_url}: {str(e)}")
            return {"error": str(e)}