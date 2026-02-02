"""
Enterprise PR Analyzer Tool
Analyzes pull requests for code quality, impact assessment, and risk identification
"""
import tempfile
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from src.services.graph_service import GraphService
from src.core.logger import get_logger

logger = get_logger(__name__)


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PRChange:
    """Represents a change in a PR"""
    file_path: str
    change_type: str  # added, modified, deleted
    additions: int
    deletions: int
    language: str
    is_test: bool = False


@dataclass
class PRImpact:
    """Impact assessment for a file change"""
    file_path: str
    affected_components: List[str] = field(default_factory=list)
    affected_apis: List[str] = field(default_factory=list)
    affected_eventbus: List[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    reasons: List[str] = field(default_factory=list)


@dataclass
class SecurityIssue:
    """Security issue found in PR"""
    severity: str
    type: str
    description: str
    file_path: str
    line_number: Optional[int] = None
    recommendation: str = ""


class PRAnalyzer:
    """
    Enterprise tool for analyzing pull requests.
    
    Features:
    - Impact analysis using knowledge graph
    - Breaking change detection
    - Security scanning
    - Test coverage analysis
    """
    
    def __init__(self):
        self.graph_service = GraphService()
        self._security_patterns = [
            {"pattern": "password", "severity": "high", "type": "credential_exposure"},
            {"pattern": "api_key", "severity": "high", "type": "credential_exposure"},
            {"pattern": "secret", "severity": "high", "type": "credential_exposure"},
            {"pattern": "eval(", "severity": "high", "type": "code_injection"},
            {"pattern": "exec(", "severity": "high", "type": "code_injection"},
            {"pattern": "dangerouslySetInnerHTML", "severity": "medium", "type": "xss"},
            {"pattern": "innerHTML", "severity": "medium", "type": "xss"},
            {"pattern": "SQL", "severity": "medium", "type": "sql_injection"},
        ]
    
    async def analyze_pr(
        self,
        pr_number: int,
        repository: str,
        include_tests: bool = True,
        include_security: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a pull request for quality, impact, and risks.
        
        Args:
            pr_number: PR number
            repository: Repository name (owner/repo format)
            include_tests: Include test analysis
            include_security: Include security scan
        
        Returns:
            Comprehensive PR analysis
        """
        try:
            logger.info(f"Analyzing PR #{pr_number} in {repository}")
            
            analysis_result = {
                "pr_number": pr_number,
                "repository": repository,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "quality_score": 0,
                "risk_level": "unknown",
                "files_changed": 0,
                "additions": 0,
                "deletions": 0,
                "breaking_changes": [],
                "test_analysis": {},
                "security_issues": [],
                "recommendations": [],
                "impacted_components": []
            }
            
            # Fetch PR data (simulated - in production would use GitHub API)
            pr_data = await self._fetch_pr_data(pr_number, repository)
            
            if not pr_data:
                analysis_result["error"] = "Could not fetch PR data"
                return analysis_result
            
            # Analyze changes
            changes = await self._analyze_changes(pr_data)
            analysis_result["files_changed"] = len(changes)
            analysis_result["additions"] = sum(c.additions for c in changes)
            analysis_result["deletions"] = sum(c.deletions for c in changes)
            
            # Impact analysis using graph
            impacts = await self._analyze_impact(changes)
            analysis_result["impacted_components"] = self._extract_impacted_components(impacts)
            
            # Breaking change detection
            breaking_changes = await self._detect_breaking_changes(changes, impacts)
            analysis_result["breaking_changes"] = breaking_changes
            
            # Test analysis
            if include_tests:
                test_analysis = await self._analyze_tests(changes, pr_data)
                analysis_result["test_analysis"] = test_analysis
            
            # Security scan
            if include_security:
                security_issues = await self._security_scan(pr_data)
                analysis_result["security_issues"] = [
                    {
                        "severity": issue.severity,
                        "type": issue.type,
                        "description": issue.description,
                        "file": issue.file_path,
                        "recommendation": issue.recommendation
                    }
                    for issue in security_issues
                ]
            
            # Calculate quality score and risk level
            analysis_result["quality_score"] = self._calculate_quality_score(
                changes, breaking_changes, security_issues if include_security else []
            )
            analysis_result["risk_level"] = self._determine_risk_level(
                breaking_changes, impacts, security_issues if include_security else []
            )
            
            # Generate recommendations
            analysis_result["recommendations"] = self._generate_recommendations(
                changes, breaking_changes, impacts, 
                security_issues if include_security else [],
                test_analysis if include_tests else {}
            )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"PR analysis failed for {repository}#{pr_number}: {e}", exc_info=True)
            return {
                "pr_number": pr_number,
                "repository": repository,
                "error": str(e),
                "status": "failed"
            }
    
    async def _fetch_pr_data(self, pr_number: int, repository: str) -> Optional[Dict]:
        """
        Fetch PR data from GitLab or GitHub API.
        
        Supports:
        - GitLab (via GITLAB_URL and GIT_TOKEN)
        - GitHub (via GIT_TOKEN)
        
        Args:
            pr_number: PR/MR number
            repository: Repository path (owner/repo)
        
        Returns:
            PR/MR data dict or None if fetch fails
        """
        import aiohttp
        from config.settings import settings
        
        token = settings.GIT_TOKEN
        if not token:
            logger.warning("No GIT_TOKEN configured, using mock PR data")
            return self._get_mock_pr_data(pr_number, repository)
        
        # Detect provider from repository URL or settings
        gitlab_url = getattr(settings, 'GITLAB_URL', 'https://gitlab.com')
        git_provider = getattr(settings, 'GIT_PROVIDER', 'gitlab')
        
        try:
            async with aiohttp.ClientSession() as session:
                if git_provider == 'gitlab' or 'gitlab' in repository.lower():
                    # GitLab API - MR endpoint
                    # URL format: /api/v4/projects/:id/merge_requests/:merge_request_iid
                    project_path = repository.replace('/', '%2F')
                    api_url = f"{gitlab_url}/api/v4/projects/{project_path}/merge_requests/{pr_number}"
                    
                    headers = {"PRIVATE-TOKEN": token}
                    
                    async with session.get(api_url, headers=headers) as resp:
                        if resp.status == 200:
                            mr_data = await resp.json()
                            
                            # Fetch diff/changes
                            changes_url = f"{api_url}/changes"
                            async with session.get(changes_url, headers=headers) as changes_resp:
                                changes_data = await changes_resp.json() if changes_resp.status == 200 else {}
                            
                            return {
                                "number": mr_data.get("iid"),
                                "repository": repository,
                                "title": mr_data.get("title"),
                                "description": mr_data.get("description", ""),
                                "state": mr_data.get("state"),
                                "source_branch": mr_data.get("source_branch"),
                                "target_branch": mr_data.get("target_branch"),
                                "author": mr_data.get("author", {}).get("username"),
                                "files": [
                                    {
                                        "path": change.get("new_path"),
                                        "old_path": change.get("old_path"),
                                        "status": "added" if change.get("new_file") else "deleted" if change.get("deleted_file") else "modified",
                                        "additions": change.get("diff", "").count("\n+"),
                                        "deletions": change.get("diff", "").count("\n-"),
                                        "diff": change.get("diff", "")
                                    }
                                    for change in changes_data.get("changes", [])
                                ],
                                "commits": [],
                                "labels": [l.get("name") for l in mr_data.get("labels", [])],
                                "web_url": mr_data.get("web_url")
                            }
                        else:
                            logger.warning(f"GitLab API returned {resp.status} for MR {pr_number}")
                            return self._get_mock_pr_data(pr_number, repository)
                
                else:
                    # GitHub API
                    api_url = f"https://api.github.com/repos/{repository}/pulls/{pr_number}"
                    headers = {
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/vnd.github.v3+json"
                    }
                    
                    async with session.get(api_url, headers=headers) as resp:
                        if resp.status == 200:
                            pr_data = await resp.json()
                            
                            # Fetch files
                            files_url = f"{api_url}/files"
                            async with session.get(files_url, headers=headers) as files_resp:
                                files_data = await files_resp.json() if files_resp.status == 200 else []
                            
                            return {
                                "number": pr_data.get("number"),
                                "repository": repository,
                                "title": pr_data.get("title"),
                                "description": pr_data.get("body", ""),
                                "state": pr_data.get("state"),
                                "source_branch": pr_data.get("head", {}).get("ref"),
                                "target_branch": pr_data.get("base", {}).get("ref"),
                                "author": pr_data.get("user", {}).get("login"),
                                "files": [
                                    {
                                        "path": f.get("filename"),
                                        "status": f.get("status"),
                                        "additions": f.get("additions", 0),
                                        "deletions": f.get("deletions", 0),
                                        "patch": f.get("patch", "")
                                    }
                                    for f in files_data
                                ],
                                "commits": [],
                                "labels": [l.get("name") for l in pr_data.get("labels", [])],
                                "web_url": pr_data.get("html_url")
                            }
                        else:
                            logger.warning(f"GitHub API returned {resp.status} for PR {pr_number}")
                            return self._get_mock_pr_data(pr_number, repository)
                            
        except Exception as e:
            logger.error(f"Failed to fetch PR data: {e}")
            return self._get_mock_pr_data(pr_number, repository)
    
    def _get_mock_pr_data(self, pr_number: int, repository: str) -> Dict:
        """Return mock PR data for testing or when API is unavailable"""
        return {
            "number": pr_number,
            "repository": repository,
            "title": f"PR #{pr_number}",
            "description": "Mock PR data - configure GIT_TOKEN for real data",
            "state": "open",
            "source_branch": "feature-branch",
            "target_branch": "main",
            "files": [],
            "commits": [],
            "is_mock": True
        }
    
    async def _analyze_changes(self, pr_data: Dict) -> List[PRChange]:
        """Analyze file changes in PR"""
        changes = []
        
        for file_info in pr_data.get("files", []):
            file_path = file_info.get("path", "")
            
            # Determine language
            if file_path.endswith(".java"):
                language = "java"
            elif file_path.endswith((".ts", ".tsx")):
                language = "typescript"
            elif file_path.endswith((".js", ".jsx")):
                language = "javascript"
            elif file_path.endswith(".py"):
                language = "python"
            else:
                language = "other"
            
            # Check if test file
            is_test = any(t in file_path.lower() for t in ["test", "spec", "__test__"])
            
            change = PRChange(
                file_path=file_path,
                change_type=file_info.get("status", "modified"),
                additions=file_info.get("additions", 0),
                deletions=file_info.get("deletions", 0),
                language=language,
                is_test=is_test
            )
            changes.append(change)
        
        return changes
    
    async def _analyze_impact(self, changes: List[PRChange]) -> List[PRImpact]:
        """Analyze impact of changes using knowledge graph"""
        impacts = []
        
        for change in changes:
            if change.is_test:
                continue
            
            try:
                # Query graph for related entities
                result = await self.graph_service.query_impact(change.file_path)
                
                impact = PRImpact(
                    file_path=change.file_path,
                    affected_components=result.get("components", []),
                    affected_apis=result.get("apis", []),
                    affected_eventbus=result.get("eventbus", []),
                    risk_level=self._assess_change_risk(change, result),
                    reasons=result.get("reasons", [])
                )
                impacts.append(impact)
                
            except Exception as e:
                logger.warning(f"Impact analysis failed for {change.file_path}: {e}")
                impacts.append(PRImpact(file_path=change.file_path))
        
        return impacts
    
    def _assess_change_risk(self, change: PRChange, graph_result: Dict) -> RiskLevel:
        """Assess risk level of a single change"""
        risk_score = 0
        
        # High deletions = higher risk
        if change.deletions > 100:
            risk_score += 2
        elif change.deletions > 50:
            risk_score += 1
        
        # Many affected components = higher risk
        affected_count = len(graph_result.get("components", []))
        if affected_count > 10:
            risk_score += 3
        elif affected_count > 5:
            risk_score += 2
        elif affected_count > 2:
            risk_score += 1
        
        # API changes = higher risk
        if graph_result.get("apis"):
            risk_score += 2
        
        # EventBus changes = higher risk
        if graph_result.get("eventbus"):
            risk_score += 2
        
        # Map score to risk level
        if risk_score >= 5:
            return RiskLevel.CRITICAL
        elif risk_score >= 3:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    async def _detect_breaking_changes(
        self,
        changes: List[PRChange],
        impacts: List[PRImpact]
    ) -> List[Dict]:
        """Detect potential breaking changes"""
        breaking_changes = []
        
        for change in changes:
            # Check for DTO/model changes
            if any(pattern in change.file_path for pattern in ["DTO", "Model", "Entity", "Schema"]):
                if change.deletions > 0:
                    breaking_changes.append({
                        "type": "model_change",
                        "file": change.file_path,
                        "severity": "high",
                        "description": "Changes to data model may break API consumers",
                        "recommendation": "Ensure backward compatibility or version the API"
                    })
            
            # Check for API route changes
            if any(pattern in change.file_path for pattern in ["Router", "Controller", "Handler"]):
                if change.deletions > 5:
                    breaking_changes.append({
                        "type": "api_change",
                        "file": change.file_path,
                        "severity": "medium",
                        "description": "API route changes may affect clients",
                        "recommendation": "Document API changes and notify consumers"
                    })
            
            # Check for EventBus address changes
            impact = next((i for i in impacts if i.file_path == change.file_path), None)
            if impact and impact.affected_eventbus:
                breaking_changes.append({
                    "type": "eventbus_change",
                    "file": change.file_path,
                    "severity": "high",
                    "description": f"EventBus addresses affected: {', '.join(impact.affected_eventbus[:3])}",
                    "recommendation": "Coordinate with dependent services"
                })
        
        return breaking_changes
    
    async def _analyze_tests(self, changes: List[PRChange], pr_data: Dict) -> Dict:
        """Analyze test coverage for changed files"""
        test_files = [c for c in changes if c.is_test]
        non_test_files = [c for c in changes if not c.is_test]
        
        # Check if test files exist for changed source files
        missing_tests = []
        for source in non_test_files:
            has_test = any(
                source.file_path.replace(".java", "Test.java") in t.file_path or
                source.file_path.replace(".ts", ".test.ts") in t.file_path or
                source.file_path.replace(".tsx", ".test.tsx") in t.file_path
                for t in test_files
            )
            if not has_test and source.additions > 10:
                missing_tests.append(source.file_path)
        
        return {
            "test_files_changed": len(test_files),
            "source_files_changed": len(non_test_files),
            "test_coverage_ratio": len(test_files) / max(len(non_test_files), 1),
            "missing_tests_for": missing_tests[:5],  # Top 5
            "status": "good" if not missing_tests else "needs_tests"
        }
    
    async def _security_scan(self, pr_data: Dict) -> List[SecurityIssue]:
        """Scan PR for security issues"""
        issues = []
        diff = pr_data.get("diff", "")
        
        for pattern_info in self._security_patterns:
            if pattern_info["pattern"].lower() in diff.lower():
                issues.append(SecurityIssue(
                    severity=pattern_info["severity"],
                    type=pattern_info["type"],
                    description=f"Potential {pattern_info['type'].replace('_', ' ')} detected",
                    file_path="unknown",  # Would need proper diff parsing
                    recommendation=f"Review usage of {pattern_info['pattern']}"
                ))
        
        return issues
    
    def _extract_impacted_components(self, impacts: List[PRImpact]) -> List[Dict]:
        """Extract unique impacted components from impacts"""
        components = {}
        
        for impact in impacts:
            for comp in impact.affected_components:
                if comp not in components:
                    components[comp] = {
                        "name": comp,
                        "affected_by": [],
                        "risk_level": impact.risk_level.value
                    }
                components[comp]["affected_by"].append(impact.file_path)
        
        return list(components.values())
    
    def _calculate_quality_score(
        self,
        changes: List[PRChange],
        breaking_changes: List[Dict],
        security_issues: List[SecurityIssue]
    ) -> float:
        """Calculate overall quality score (0-100)"""
        score = 100.0
        
        # Deduct for breaking changes
        score -= len(breaking_changes) * 10
        
        # Deduct for security issues
        for issue in security_issues:
            if issue.severity == "high":
                score -= 15
            elif issue.severity == "medium":
                score -= 10
            else:
                score -= 5
        
        # Deduct for very large changes
        total_changes = sum(c.additions + c.deletions for c in changes)
        if total_changes > 1000:
            score -= 15
        elif total_changes > 500:
            score -= 10
        elif total_changes > 200:
            score -= 5
        
        return max(0, min(100, score))
    
    def _determine_risk_level(
        self,
        breaking_changes: List[Dict],
        impacts: List[PRImpact],
        security_issues: List[SecurityIssue]
    ) -> str:
        """Determine overall risk level"""
        # Check for critical security issues
        if any(i.severity == "high" for i in security_issues):
            return "critical"
        
        # Check for many breaking changes
        if len(breaking_changes) >= 3:
            return "critical"
        elif len(breaking_changes) >= 2:
            return "high"
        
        # Check impact risk levels
        critical_impacts = sum(1 for i in impacts if i.risk_level == RiskLevel.CRITICAL)
        high_impacts = sum(1 for i in impacts if i.risk_level == RiskLevel.HIGH)
        
        if critical_impacts > 0:
            return "critical"
        elif high_impacts >= 2 or len(breaking_changes) >= 1:
            return "high"
        elif high_impacts >= 1:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendations(
        self,
        changes: List[PRChange],
        breaking_changes: List[Dict],
        impacts: List[PRImpact],
        security_issues: List[SecurityIssue],
        test_analysis: Dict
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Breaking change recommendations
        if breaking_changes:
            recommendations.append(
                f"Review {len(breaking_changes)} potential breaking change(s) before merging"
            )
        
        # Security recommendations
        if security_issues:
            recommendations.append(
                f"Address {len(security_issues)} security concern(s) flagged in scan"
            )
        
        # Test recommendations
        if test_analysis.get("status") == "needs_tests":
            missing = test_analysis.get("missing_tests_for", [])
            recommendations.append(
                f"Add tests for {len(missing)} file(s) without test coverage"
            )
        
        # Large change recommendations
        total_changes = sum(c.additions + c.deletions for c in changes)
        if total_changes > 500:
            recommendations.append(
                "Consider splitting this PR into smaller, focused changes"
            )
        
        # Impact recommendations
        high_risk = [i for i in impacts if i.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        if high_risk:
            recommendations.append(
                f"High-impact changes detected in {len(high_risk)} file(s) - manual review recommended"
            )
        
        # Default recommendation
        if not recommendations:
            recommendations.append("PR looks good for merge after code review")
        
        return recommendations