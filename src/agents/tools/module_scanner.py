# module_scanner.py - ENTERPRISE VERSION
import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from src.services.search_service import SearchService
from src.services.graph_service import GraphService
from src.database.neo4j_client import Neo4jClient
from src.core.logger import get_logger

logger = get_logger(__name__)

class IssueSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class CodeIssue:
    type: str
    severity: IssueSeverity
    description: str
    location: str  # file:line or file:line-range
    suggestion: str
    impact: str
    confidence: float  # 0.0 to 1.0

@dataclass
class ModuleAnalysis:
    module_name: str
    repository: str
    file_count: int
    lines_of_code: int
    issues: List[CodeIssue]
    dependencies: Dict[str, List[str]]
    complexity_score: float  # 0-100
    test_coverage: Optional[float]  # 0-100
    recommendations: List[str]
    health_score: float  # 0-100

class ModuleScanner:
    """Enterprise-grade module analysis tool for Vert.x/React codebases"""
    
    def __init__(self):
        self.search_service = SearchService()
        self.graph_service = GraphService()
        self.neo4j_client = Neo4jClient()
    
    async def scan_module(self, module_name: str, repository: str = None, 
                         deep_scan: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive module analysis
        
        Args:
            module_name: Module to scan (e.g., 'payment', 'user-management')
            repository: Specific repository to scan (optional)
            deep_scan: Perform deeper analysis including code metrics
        
        Returns:
            Complete module analysis
        """
        try:
            logger.info(f"Starting module scan: {module_name}, repo={repository}, deep={deep_scan}")
            
            # Step 1: Retrieve module context
            context = await self._retrieve_module_context(module_name, repository)
            
            # Step 2: Analyze dependencies
            dependencies = await self._analyze_dependencies(module_name, repository)
            
            # Step 3: Identify issues
            issues = await self._identify_issues(context, dependencies)
            
            # Step 4: Calculate metrics (if deep scan)
            metrics = {}
            if deep_scan:
                metrics = await self._calculate_metrics(module_name, repository)
            
            # Step 5: Generate health score
            health_score = self._calculate_health_score(issues, metrics)
            
            # Step 6: Generate recommendations
            recommendations = await self._generate_recommendations(issues, dependencies, metrics)
            
            analysis = ModuleAnalysis(
                module_name=module_name,
                repository=repository or "all",
                file_count=context.get('file_count', 0),
                lines_of_code=metrics.get('lines_of_code', 0),
                issues=issues,
                dependencies=dependencies,
                complexity_score=metrics.get('complexity', 0),
                test_coverage=metrics.get('test_coverage'),
                recommendations=recommendations,
                health_score=health_score
            )
            
            return self._format_analysis(analysis)
            
        except Exception as e:
            logger.error(f"Module scan failed for {module_name}: {e}")
            return {
                "error": str(e),
                "module": module_name,
                "status": "failed"
            }
    
    async def _retrieve_module_context(self, module_name: str, repository: Optional[str]) -> Dict:
        """Retrieve comprehensive context for the module"""
        # Use multiple search strategies
        contexts = {}
        
        # 1. Semantic search for module files
        contexts['semantic'] = await self.search_service.hybrid_search(
            query=f"{module_name} module component service",
            top_k=30,
            filter_repos=[repository] if repository else None
        )
        
        # 2. Graph search for module structure
        contexts['graph'] = await self.graph_service.find_module_structure(
            module_name=module_name,
            repo_filter=repository
        )
        
        # 3. Search for EventBus addresses related to module
        contexts['eventbus'] = await self._find_eventbus_flows(module_name)
        
        # 4. Find API routes in module
        contexts['api_routes'] = await self._find_api_routes(module_name, repository)
        
        return contexts
    
    async def _find_eventbus_flows(self, module_name: str) -> List[Dict]:
        """Find EventBus producers and consumers in module"""
        query = """
        MATCH (m:Module {name: $module_name})
        OPTIONAL MATCH (m)-[:CONTAINS]->(producer:VertxMethod {type: 'eventbus_publish'})
        OPTIONAL MATCH (m)-[:CONTAINS]->(consumer:VertxMethod {type: 'eventbus_consume'})
        RETURN producer, consumer
        """
        
        results = await self.neo4j_client.execute_query(query, {"module_name": module_name})
        
        flows = []
        for record in results:
            if record.get("producer"):
                flows.append({
                    "type": "producer",
                    "address": record["producer"].get("address"),
                    "method": record["producer"].get("name"),
                    "file": record["producer"].get("file")
                })
            if record.get("consumer"):
                flows.append({
                    "type": "consumer",
                    "address": record["consumer"].get("address"),
                    "method": record["consumer"].get("name"),
                    "file": record["consumer"].get("file")
                })
        
        return flows
    
    async def _find_api_routes(self, module_name: str, repository: Optional[str]) -> List[Dict]:
        """Find API routes in module"""
        query = """
        MATCH (m:Module {name: $module_name})
        OPTIONAL MATCH (m)-[:CONTAINS]->(route:VertxRoute)
        RETURN route
        """
        
        results = await self.neo4j_client.execute_query(query, {"module_name": module_name})
        
        routes = []
        for record in results:
            if record.get("route"):
                routes.append({
                    "path": record["route"].get("path"),
                    "method": record["route"].get("method"),
                    "file": record["route"].get("file"),
                    "repository": repository
                })
        
        return routes
    
    async def _analyze_dependencies(self, module_name: str, repository: Optional[str]) -> Dict[str, List[str]]:
        """Analyze module dependencies"""
        dependencies = {
            "internal": [],  # Within same repository
            "external": [],  # Other repositories
            "third_party": [],  # External libraries
            "database": [],  # Database tables/schemas
            "eventbus": []  # EventBus addresses
        }
        
        # Find cross-module dependencies
        query = """
        MATCH (m:Module {name: $module_name})
        OPTIONAL MATCH (m)-[:DEPENDS_ON]->(dep:Module)
        RETURN dep.name as dependency
        """
        
        results = await self.neo4j_client.execute_query(query, {"module_name": module_name})
        
        for record in results:
            if record.get("dependency"):
                dependencies["internal"].append(record["dependency"])
        
        # Find external API dependencies
        query = """
        MATCH (m:Module {name: $module_name})
        OPTIONAL MATCH (m)-[:CALLS_API]->(api:ExternalAPI)
        RETURN api.url as api_endpoint
        """
        
        results = await self.neo4j_client.execute_query(query, {"module_name": module_name})
        
        for record in results:
            if record.get("api_endpoint"):
                dependencies["external"].append(record["api_endpoint"])
        
        return dependencies
    
    async def _identify_issues(self, context: Dict, dependencies: Dict) -> List[CodeIssue]:
        """Identify potential issues in the module"""
        issues = []
        
        # Check for EventBus issues
        for flow in context.get('eventbus', []):
            if flow["type"] == "producer":
                # Check if producer has matching consumer
                has_consumer = any(
                    c["address"] == flow["address"] and c["type"] == "consumer"
                    for c in context.get('eventbus', [])
                )
                
                if not has_consumer:
                    issues.append(CodeIssue(
                        type="orphaned_producer",
                        severity=IssueSeverity.MEDIUM,
                        description=f"EventBus producer has no consumer: {flow['address']}",
                        location=f"{flow['file']}",
                        suggestion="Add consumer or document intentional orphan",
                        impact="Messages may be lost",
                        confidence=0.9
                    ))
        
        # Check for API route issues
        for route in context.get('api_routes', []):
            # Check for missing authentication
            if not any(auth_key in str(route).lower() for auth_key in ['@authenticated', '@secured', 'auth']):
                issues.append(CodeIssue(
                    type="missing_auth",
                    severity=IssueSeverity.HIGH,
                    description=f"API route missing authentication: {route['method']} {route['path']}",
                    location=f"{route['file']}",
                    suggestion="Add @Authenticated annotation or security check",
                    impact="Security vulnerability",
                    confidence=0.8
                ))
        
        # Check for dependency issues
        if len(dependencies.get("external", [])) > 10:
            issues.append(CodeIssue(
                type="high_external_dependencies",
                severity=IssueSeverity.MEDIUM,
                description=f"Module has {len(dependencies['external'])} external dependencies",
                location="module-level",
                suggestion="Consider caching or circuit breakers for external calls",
                impact="Increased failure probability",
                confidence=0.7
            ))
        
        return issues
    
    async def _calculate_metrics(self, module_name: str, repository: Optional[str]) -> Dict[str, Any]:
        """Calculate code metrics for the module"""
        metrics = {
            "lines_of_code": 0,
            "complexity": 0,
            "test_coverage": None,
            "method_count": 0,
            "class_count": 0,
            "comment_density": 0
        }
        
        # This would integrate with tools like SonarQube, Jacoco, etc.
        # For now, return placeholder values
        
        return metrics
    
    def _calculate_health_score(self, issues: List[CodeIssue], metrics: Dict) -> float:
        """Calculate module health score (0-100)"""
        base_score = 100.0
        
        # Deduct for issues
        severity_weights = {
            IssueSeverity.CRITICAL: 20,
            IssueSeverity.HIGH: 10,
            IssueSeverity.MEDIUM: 5,
            IssueSeverity.LOW: 2,
            IssueSeverity.INFO: 0
        }
        
        for issue in issues:
            base_score -= severity_weights.get(issue.severity, 0)
        
        # Adjust for test coverage
        test_coverage = metrics.get('test_coverage')
        if test_coverage is not None:
            if test_coverage < 50:
                base_score -= 20
            elif test_coverage < 80:
                base_score -= 10
        
        # Adjust for complexity
        complexity = metrics.get('complexity', 0)
        if complexity > 80:
            base_score -= 15
        elif complexity > 60:
            base_score -= 8
        
        return max(0, min(100, base_score))
    
    async def _generate_recommendations(self, issues: List[CodeIssue], 
                                       dependencies: Dict, 
                                       metrics: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Recommendations based on issues
        critical_issues = [i for i in issues if i.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH]]
        if critical_issues:
            recommendations.append(f"Address {len(critical_issues)} high/critical issues before making changes")
        
        # Recommendations based on dependencies
        if len(dependencies.get("external", [])) > 5:
            recommendations.append("Implement circuit breakers for external API calls")
        
        if len(dependencies.get("eventbus", [])) > 10:
            recommendations.append("Consider adding dead letter queue for EventBus messages")
        
        # Recommendations based on metrics
        test_coverage = metrics.get('test_coverage')
        if test_coverage is not None and test_coverage < 80:
            recommendations.append(f"Increase test coverage from {test_coverage}% to at least 80%")
        
        return recommendations
    
    def _format_analysis(self, analysis: ModuleAnalysis) -> Dict[str, Any]:
        """Format analysis for API response"""
        return {
            "module": analysis.module_name,
            "repository": analysis.repository,
            "health_score": analysis.health_score,
            "file_count": analysis.file_count,
            "lines_of_code": analysis.lines_of_code,
            "complexity_score": analysis.complexity_score,
            "test_coverage": analysis.test_coverage,
            "issues": [
                {
                    "type": issue.type,
                    "severity": issue.severity.value,
                    "description": issue.description,
                    "location": issue.location,
                    "suggestion": issue.suggestion,
                    "impact": issue.impact,
                    "confidence": issue.confidence
                }
                for issue in analysis.issues
            ],
            "dependencies": analysis.dependencies,
            "recommendations": analysis.recommendations,
            "summary": self._generate_summary(analysis)
        }
    
    def _generate_summary(self, analysis: ModuleAnalysis) -> str:
        """Generate human-readable summary"""
        health_status = "healthy" if analysis.health_score > 80 else \
                       "needs attention" if analysis.health_score > 60 else \
                       "critical"
        
        critical_issues = len([i for i in analysis.issues if i.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH]])
        
        return f"""
        Module: {analysis.module_name}
        Health: {health_status} ({analysis.health_score}/100)
        
        Statistics:
        - Files: {analysis.file_count}
        - Lines of Code: {analysis.lines_of_code:,}
        - Issues: {len(analysis.issues)} total, {critical_issues} critical/high
        - Test Coverage: {analysis.test_coverage or 'Unknown'}%
        - Dependencies: {sum(len(v) for v in analysis.dependencies.values())}
        
        Recommendations:
        {chr(10).join(f'- {r}' for r in analysis.recommendations)}
        """