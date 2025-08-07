"""
Code Review and Analysis Suite for Cyber-LLM
Advanced static analysis, security review, and optimization identification

Author: Muzan Sano <sanosensei36@gmail.com>
"""

import ast
import re
import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import subprocess
from collections import defaultdict, Counter

# Security analysis imports
import bandit
from bandit.core.config import BanditConfig
from bandit.core.manager import BanditManager

# Code quality imports
try:
    import pylint.lint
    import flake8.api.legacy as flake8
    from mypy import api as mypy_api
except ImportError:
    print("Install code quality tools: pip install pylint flake8 mypy")

@dataclass
class CodeIssue:
    """Represents a code issue found during analysis"""
    file_path: str
    line_number: int
    severity: str  # critical, high, medium, low, info
    issue_type: str  # security, performance, maintainability, style, bug
    description: str
    recommendation: str
    confidence: float = 1.0
    cwe_id: Optional[str] = None  # Common Weakness Enumeration ID
    
@dataclass 
class ReviewResults:
    """Complete code review results"""
    total_files_analyzed: int
    total_lines_analyzed: int
    issues: List[CodeIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    security_score: float = 0.0
    maintainability_score: float = 0.0
    performance_score: float = 0.0
    overall_score: float = 0.0

class SecurityAnalyzer:
    """Advanced security analysis for cybersecurity applications"""
    
    def __init__(self):
        self.logger = logging.getLogger("security_analyzer")
        
        # Custom security patterns for cybersecurity tools
        self.security_patterns = {
            "hardcoded_credentials": [
                r"password\s*=\s*['\"][^'\"]{3,}['\"]",
                r"api_key\s*=\s*['\"][^'\"]{10,}['\"]",
                r"secret\s*=\s*['\"][^'\"]{8,}['\"]",
                r"token\s*=\s*['\"][^'\"]{16,}['\"]"
            ],
            "command_injection": [
                r"os\.system\s*\(",
                r"subprocess\.call\s*\(",
                r"subprocess\.run\s*\(",
                r"eval\s*\(",
                r"exec\s*\("
            ],
            "sql_injection": [
                r"execute\s*\(\s*['\"].*%s.*['\"]",
                r"cursor\.execute\s*\(\s*[f]?['\"].*\{.*\}.*['\"]"
            ],
            "path_traversal": [
                r"open\s*\(\s*.*\+.*\)",
                r"file\s*\(\s*.*\+.*\)",
                r"\.\./"
            ],
            "insecure_random": [
                r"random\.random\(\)",
                r"random\.choice\(",
                r"random\.randint\("
            ]
        }
    
    async def analyze_security(self, file_paths: List[str]) -> List[CodeIssue]:
        """Comprehensive security analysis"""
        
        security_issues = []
        
        for file_path in file_paths:
            if not file_path.endswith('.py'):
                continue
                
            self.logger.info(f"Security analysis: {file_path}")
            
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Pattern-based security analysis
                pattern_issues = await self._analyze_security_patterns(file_path, content)
                security_issues.extend(pattern_issues)
                
                # AST-based security analysis
                ast_issues = await self._analyze_ast_security(file_path, content)
                security_issues.extend(ast_issues)
                
                # Bandit integration for comprehensive security scanning
                bandit_issues = await self._run_bandit_analysis(file_path)
                security_issues.extend(bandit_issues)
                
            except Exception as e:
                self.logger.error(f"Error analyzing {file_path}: {str(e)}")
                security_issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=0,
                    severity="medium",
                    issue_type="security",
                    description=f"Analysis error: {str(e)}",
                    recommendation="Manual review required"
                ))
        
        return security_issues
    
    async def _analyze_security_patterns(self, file_path: str, content: str) -> List[CodeIssue]:
        """Pattern-based security vulnerability detection"""
        
        issues = []
        lines = content.split('\n')
        
        for category, patterns in self.security_patterns.items():
            for pattern in patterns:
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        severity, recommendation = self._get_security_severity(category, line)
                        
                        issues.append(CodeIssue(
                            file_path=file_path,
                            line_number=line_num,
                            severity=severity,
                            issue_type="security",
                            description=f"Potential {category.replace('_', ' ')}: {line.strip()}",
                            recommendation=recommendation,
                            confidence=0.8
                        ))
        
        return issues
    
    async def _analyze_ast_security(self, file_path: str, content: str) -> List[CodeIssue]:
        """AST-based security analysis for complex patterns"""
        
        issues = []
        
        try:
            tree = ast.parse(content)
            
            class SecurityVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.issues = []
                
                def visit_Call(self, node):
                    # Check for dangerous function calls
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        
                        if func_name in ['eval', 'exec']:
                            self.issues.append(CodeIssue(
                                file_path=file_path,
                                line_number=node.lineno,
                                severity="critical",
                                issue_type="security",
                                description=f"Dangerous function call: {func_name}",
                                recommendation="Avoid using eval/exec, use safer alternatives",
                                cwe_id="CWE-94"
                            ))
                    
                    elif isinstance(node.func, ast.Attribute):
                        if (isinstance(node.func.value, ast.Name) and 
                            node.func.value.id == 'os' and 
                            node.func.attr == 'system'):
                            
                            self.issues.append(CodeIssue(
                                file_path=file_path,
                                line_number=node.lineno,
                                severity="high",
                                issue_type="security", 
                                description="Command injection risk: os.system()",
                                recommendation="Use subprocess with shell=False",
                                cwe_id="CWE-78"
                            ))
                    
                    self.generic_visit(node)
                
                def visit_Import(self, node):
                    for alias in node.names:
                        if alias.name in ['pickle', 'cPickle']:
                            self.issues.append(CodeIssue(
                                file_path=file_path,
                                line_number=node.lineno,
                                severity="medium",
                                issue_type="security",
                                description="Insecure deserialization: pickle import",
                                recommendation="Use json or safer serialization methods",
                                cwe_id="CWE-502"
                            ))
                    
                    self.generic_visit(node)
            
            visitor = SecurityVisitor()
            visitor.visit(tree)
            issues.extend(visitor.issues)
            
        except SyntaxError as e:
            issues.append(CodeIssue(
                file_path=file_path,
                line_number=e.lineno or 0,
                severity="high",
                issue_type="security",
                description=f"Syntax error prevents security analysis: {str(e)}",
                recommendation="Fix syntax errors before security analysis"
            ))
        
        return issues
    
    async def _run_bandit_analysis(self, file_path: str) -> List[CodeIssue]:
        """Run Bandit security scanner"""
        
        issues = []
        
        try:
            # Configure Bandit
            config = BanditConfig()
            manager = BanditManager(config, 'file')
            manager.discover_files([file_path])
            manager.run_tests()
            
            # Convert Bandit results to CodeIssue format
            for result in manager.get_issue_list():
                issues.append(CodeIssue(
                    file_path=result.filename,
                    line_number=result.lineno,
                    severity=result.severity,
                    issue_type="security",
                    description=result.text,
                    recommendation=f"Bandit {result.test_id}: {result.text}",
                    confidence=self._convert_bandit_confidence(result.confidence),
                    cwe_id=getattr(result, 'cwe_id', None)
                ))
        
        except Exception as e:
            self.logger.warning(f"Bandit analysis failed for {file_path}: {str(e)}")
        
        return issues
    
    def _get_security_severity(self, category: str, line: str) -> Tuple[str, str]:
        """Get severity and recommendation for security issue"""
        
        severity_map = {
            "hardcoded_credentials": ("critical", "Use environment variables or secure vaults"),
            "command_injection": ("critical", "Use parameterized commands and input validation"),
            "sql_injection": ("critical", "Use parameterized queries and prepared statements"),
            "path_traversal": ("high", "Validate and sanitize file paths"),
            "insecure_random": ("medium", "Use cryptographically secure random functions")
        }
        
        return severity_map.get(category, ("medium", "Review for security implications"))
    
    def _convert_bandit_confidence(self, confidence: str) -> float:
        """Convert Bandit confidence to numeric value"""
        
        confidence_map = {
            "HIGH": 0.9,
            "MEDIUM": 0.7,
            "LOW": 0.5
        }
        
        return confidence_map.get(confidence, 0.6)

class PerformanceAnalyzer:
    """Performance analysis and optimization identification"""
    
    def __init__(self):
        self.logger = logging.getLogger("performance_analyzer")
        
        self.performance_patterns = {
            "inefficient_loops": [
                r"for.*in.*range\(len\(",
                r"while.*len\("
            ],
            "string_concatenation": [
                r"\+\s*['\"].*['\"]",
                r".*\+=.*['\"]"
            ],
            "global_variables": [
                r"^global\s+\w+"
            ],
            "nested_loops": [],  # Detected via AST
            "database_queries_in_loops": [],  # Detected via AST
        }
    
    async def analyze_performance(self, file_paths: List[str]) -> List[CodeIssue]:
        """Comprehensive performance analysis"""
        
        performance_issues = []
        
        for file_path in file_paths:
            if not file_path.endswith('.py'):
                continue
                
            self.logger.info(f"Performance analysis: {file_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Pattern-based analysis
                pattern_issues = await self._analyze_performance_patterns(file_path, content)
                performance_issues.extend(pattern_issues)
                
                # AST-based analysis for complex patterns
                ast_issues = await self._analyze_ast_performance(file_path, content)
                performance_issues.extend(ast_issues)
                
            except Exception as e:
                self.logger.error(f"Error analyzing {file_path}: {str(e)}")
        
        return performance_issues
    
    async def _analyze_performance_patterns(self, file_path: str, content: str) -> List[CodeIssue]:
        """Pattern-based performance issue detection"""
        
        issues = []
        lines = content.split('\n')
        
        for category, patterns in self.performance_patterns.items():
            if not patterns:  # Skip empty pattern lists
                continue
                
            for pattern in patterns:
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line):
                        severity, recommendation = self._get_performance_severity(category)
                        
                        issues.append(CodeIssue(
                            file_path=file_path,
                            line_number=line_num,
                            severity=severity,
                            issue_type="performance",
                            description=f"Performance issue - {category.replace('_', ' ')}: {line.strip()}",
                            recommendation=recommendation
                        ))
        
        return issues
    
    async def _analyze_ast_performance(self, file_path: str, content: str) -> List[CodeIssue]:
        """AST-based performance analysis"""
        
        issues = []
        
        try:
            tree = ast.parse(content)
            
            class PerformanceVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.issues = []
                    self.loop_depth = 0
                    self.in_loop = False
                
                def visit_For(self, node):
                    self.loop_depth += 1
                    old_in_loop = self.in_loop
                    self.in_loop = True
                    
                    # Check for nested loops
                    if self.loop_depth > 2:
                        self.issues.append(CodeIssue(
                            file_path=file_path,
                            line_number=node.lineno,
                            severity="medium",
                            issue_type="performance",
                            description="Deeply nested loops detected",
                            recommendation="Consider algorithm optimization or breaking into functions"
                        ))
                    
                    self.generic_visit(node)
                    self.loop_depth -= 1
                    self.in_loop = old_in_loop
                
                def visit_While(self, node):
                    self.loop_depth += 1
                    old_in_loop = self.in_loop
                    self.in_loop = True
                    
                    if self.loop_depth > 2:
                        self.issues.append(CodeIssue(
                            file_path=file_path,
                            line_number=node.lineno,
                            severity="medium",
                            issue_type="performance",
                            description="Deeply nested while loops detected",
                            recommendation="Consider algorithm optimization"
                        ))
                    
                    self.generic_visit(node)
                    self.loop_depth -= 1
                    self.in_loop = old_in_loop
                
                def visit_Call(self, node):
                    # Check for database calls in loops
                    if self.in_loop and isinstance(node.func, ast.Attribute):
                        method_name = node.func.attr
                        if method_name in ['execute', 'query', 'find', 'get']:
                            self.issues.append(CodeIssue(
                                file_path=file_path,
                                line_number=node.lineno,
                                severity="high",
                                issue_type="performance",
                                description="Potential database query in loop",
                                recommendation="Move query outside loop or use batch operations"
                            ))
                    
                    self.generic_visit(node)
            
            visitor = PerformanceVisitor()
            visitor.visit(tree)
            issues.extend(visitor.issues)
            
        except SyntaxError:
            pass  # Skip files with syntax errors
        
        return issues
    
    def _get_performance_severity(self, category: str) -> Tuple[str, str]:
        """Get severity and recommendation for performance issue"""
        
        severity_map = {
            "inefficient_loops": ("medium", "Use enumerate() or direct iteration"),
            "string_concatenation": ("low", "Use string formatting or join() for multiple concatenations"),
            "global_variables": ("low", "Consider using class attributes or function parameters"),
            "nested_loops": ("medium", "Optimize algorithm complexity"),
            "database_queries_in_loops": ("high", "Use batch operations or optimize query placement")
        }
        
        return severity_map.get(category, ("low", "Review for performance implications"))

class MaintainabilityAnalyzer:
    """Code maintainability and quality analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger("maintainability_analyzer")
    
    async def analyze_maintainability(self, file_paths: List[str]) -> Tuple[List[CodeIssue], Dict[str, Any]]:
        """Comprehensive maintainability analysis"""
        
        maintainability_issues = []
        metrics = {
            "complexity_metrics": {},
            "documentation_coverage": 0.0,
            "code_duplication": {},
            "naming_conventions": {}
        }
        
        for file_path in file_paths:
            if not file_path.endswith('.py'):
                continue
                
            self.logger.info(f"Maintainability analysis: {file_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Complexity analysis
                complexity_issues, complexity_metrics = await self._analyze_complexity(file_path, content)
                maintainability_issues.extend(complexity_issues)
                metrics["complexity_metrics"][file_path] = complexity_metrics
                
                # Documentation analysis
                doc_issues, doc_metrics = await self._analyze_documentation(file_path, content)
                maintainability_issues.extend(doc_issues)
                
                # Code duplication detection
                duplication_issues = await self._detect_code_duplication(file_path, content)
                maintainability_issues.extend(duplication_issues)
                
            except Exception as e:
                self.logger.error(f"Error analyzing {file_path}: {str(e)}")
        
        return maintainability_issues, metrics
    
    async def _analyze_complexity(self, file_path: str, content: str) -> Tuple[List[CodeIssue], Dict[str, Any]]:
        """Analyze cyclomatic complexity and other complexity metrics"""
        
        issues = []
        metrics = {
            "cyclomatic_complexity": 0,
            "lines_of_code": 0,
            "function_count": 0,
            "class_count": 0,
            "max_function_complexity": 0
        }
        
        try:
            tree = ast.parse(content)
            
            class ComplexityVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.complexity = 1  # Base complexity
                    self.function_complexities = []
                    self.function_count = 0
                    self.class_count = 0
                    self.current_function = None
                    self.current_complexity = 1
                
                def visit_FunctionDef(self, node):
                    self.function_count += 1
                    old_complexity = self.current_complexity
                    old_function = self.current_function
                    
                    self.current_function = node.name
                    self.current_complexity = 1
                    
                    self.generic_visit(node)
                    
                    # Check if function complexity is too high
                    if self.current_complexity > 10:
                        issues.append(CodeIssue(
                            file_path=file_path,
                            line_number=node.lineno,
                            severity="medium",
                            issue_type="maintainability",
                            description=f"High cyclomatic complexity in function '{node.name}': {self.current_complexity}",
                            recommendation="Consider breaking down function into smaller functions"
                        ))
                    
                    self.function_complexities.append(self.current_complexity)
                    self.current_complexity = old_complexity
                    self.current_function = old_function
                
                def visit_ClassDef(self, node):
                    self.class_count += 1
                    self.generic_visit(node)
                
                def visit_If(self, node):
                    self.current_complexity += 1
                    self.generic_visit(node)
                
                def visit_For(self, node):
                    self.current_complexity += 1
                    self.generic_visit(node)
                
                def visit_While(self, node):
                    self.current_complexity += 1
                    self.generic_visit(node)
                
                def visit_Try(self, node):
                    self.current_complexity += len(node.handlers)
                    self.generic_visit(node)
            
            visitor = ComplexityVisitor()
            visitor.visit(tree)
            
            lines = content.split('\n')
            metrics["lines_of_code"] = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            metrics["function_count"] = visitor.function_count
            metrics["class_count"] = visitor.class_count
            metrics["cyclomatic_complexity"] = sum(visitor.function_complexities) if visitor.function_complexities else 1
            metrics["max_function_complexity"] = max(visitor.function_complexities) if visitor.function_complexities else 0
            
        except SyntaxError:
            pass  # Skip files with syntax errors
        
        return issues, metrics
    
    async def _analyze_documentation(self, file_path: str, content: str) -> Tuple[List[CodeIssue], Dict[str, Any]]:
        """Analyze documentation coverage and quality"""
        
        issues = []
        metrics = {"documented_functions": 0, "total_functions": 0}
        
        try:
            tree = ast.parse(content)
            
            class DocVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.total_functions = 0
                    self.documented_functions = 0
                
                def visit_FunctionDef(self, node):
                    self.total_functions += 1
                    
                    # Check if function has docstring
                    if (node.body and 
                        isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, ast.Str)):
                        self.documented_functions += 1
                    else:
                        # Only report missing docstrings for non-private functions
                        if not node.name.startswith('_'):
                            issues.append(CodeIssue(
                                file_path=file_path,
                                line_number=node.lineno,
                                severity="low",
                                issue_type="maintainability",
                                description=f"Missing docstring for function '{node.name}'",
                                recommendation="Add descriptive docstring"
                            ))
                    
                    self.generic_visit(node)
                
                def visit_ClassDef(self, node):
                    # Check if class has docstring
                    if not (node.body and 
                            isinstance(node.body[0], ast.Expr) and
                            isinstance(node.body[0].value, ast.Str)):
                        issues.append(CodeIssue(
                            file_path=file_path,
                            line_number=node.lineno,
                            severity="low", 
                            issue_type="maintainability",
                            description=f"Missing docstring for class '{node.name}'",
                            recommendation="Add descriptive class docstring"
                        ))
                    
                    self.generic_visit(node)
            
            visitor = DocVisitor()
            visitor.visit(tree)
            
            metrics["documented_functions"] = visitor.documented_functions
            metrics["total_functions"] = visitor.total_functions
            
        except SyntaxError:
            pass
        
        return issues, metrics
    
    async def _detect_code_duplication(self, file_path: str, content: str) -> List[CodeIssue]:
        """Detect code duplication patterns"""
        
        issues = []
        lines = content.split('\n')
        
        # Simple line-based duplication detection
        line_counts = Counter()
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if len(stripped) > 20 and not stripped.startswith('#'):  # Ignore short lines and comments
                line_counts[stripped] += 1
                
                if line_counts[stripped] == 3:  # Report after 3 occurrences
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=line_num,
                        severity="low",
                        issue_type="maintainability",
                        description=f"Potential code duplication: {stripped[:50]}...",
                        recommendation="Consider extracting common code into functions"
                    ))
        
        return issues

class ComprehensiveCodeReviewer:
    """Main code review orchestrator"""
    
    def __init__(self):
        self.logger = logging.getLogger("code_reviewer")
        self.security_analyzer = SecurityAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.maintainability_analyzer = MaintainabilityAnalyzer()
    
    async def conduct_comprehensive_review(self, project_path: str, 
                                         include_patterns: Optional[List[str]] = None,
                                         exclude_patterns: Optional[List[str]] = None) -> ReviewResults:
        """Conduct comprehensive code review"""
        
        self.logger.info(f"Starting comprehensive code review of {project_path}")
        start_time = datetime.now()
        
        # Discover files to analyze
        file_paths = await self._discover_files(project_path, include_patterns, exclude_patterns)
        
        if not file_paths:
            self.logger.warning("No files found for analysis")
            return ReviewResults(0, 0)
        
        self.logger.info(f"Analyzing {len(file_paths)} files")
        
        # Calculate total lines
        total_lines = 0
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    total_lines += len(f.readlines())
            except:
                pass
        
        # Run all analyzers concurrently
        security_task = asyncio.create_task(self.security_analyzer.analyze_security(file_paths))
        performance_task = asyncio.create_task(self.performance_analyzer.analyze_performance(file_paths))
        maintainability_task = asyncio.create_task(self.maintainability_analyzer.analyze_maintainability(file_paths))
        
        # Wait for all analyses to complete
        security_issues = await security_task
        performance_issues = await performance_task
        maintainability_issues, maintainability_metrics = await maintainability_task
        
        # Combine all issues
        all_issues = security_issues + performance_issues + maintainability_issues
        
        # Calculate scores
        security_score = await self._calculate_security_score(security_issues)
        maintainability_score = await self._calculate_maintainability_score(maintainability_issues)
        performance_score = await self._calculate_performance_score(performance_issues)
        
        overall_score = (security_score + maintainability_score + performance_score) / 3
        
        # Create comprehensive results
        results = ReviewResults(
            total_files_analyzed=len(file_paths),
            total_lines_analyzed=total_lines,
            issues=all_issues,
            metrics=maintainability_metrics,
            security_score=security_score,
            maintainability_score=maintainability_score,
            performance_score=performance_score,
            overall_score=overall_score
        )
        
        # Generate review report
        await self._generate_review_report(results, project_path)
        
        duration = datetime.now() - start_time
        self.logger.info(f"Code review completed in {duration.total_seconds():.2f}s")
        self.logger.info(f"Overall score: {overall_score:.1f}/100")
        
        return results
    
    async def _discover_files(self, project_path: str, 
                            include_patterns: Optional[List[str]] = None,
                            exclude_patterns: Optional[List[str]] = None) -> List[str]:
        """Discover files to analyze"""
        
        file_paths = []
        project_path = Path(project_path)
        
        include_patterns = include_patterns or ['*.py']
        exclude_patterns = exclude_patterns or [
            '*/venv/*', '*/env/*', '*/__pycache__/*', 
            '*/node_modules/*', '*/.*/*', '*/.git/*'
        ]
        
        def should_include(file_path: Path) -> bool:
            path_str = str(file_path)
            
            # Check exclude patterns
            for exclude in exclude_patterns:
                if exclude.replace('*', '.*') in path_str:
                    return False
            
            # Check include patterns
            for include in include_patterns:
                if file_path.match(include):
                    return True
            
            return False
        
        # Walk through project directory
        for root, dirs, files in os.walk(project_path):
            # Skip hidden and excluded directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'env']]
            
            for file in files:
                file_path = Path(root) / file
                if should_include(file_path):
                    file_paths.append(str(file_path))
        
        return file_paths
    
    async def _calculate_security_score(self, security_issues: List[CodeIssue]) -> float:
        """Calculate security score based on issues found"""
        
        if not security_issues:
            return 100.0
        
        severity_weights = {
            "critical": -20,
            "high": -10,
            "medium": -5,
            "low": -2,
            "info": -1
        }
        
        total_deduction = sum(severity_weights.get(issue.severity, -1) for issue in security_issues)
        return max(0, 100 + total_deduction)
    
    async def _calculate_maintainability_score(self, maintainability_issues: List[CodeIssue]) -> float:
        """Calculate maintainability score"""
        
        base_score = 100.0
        
        for issue in maintainability_issues:
            if issue.severity == "high":
                base_score -= 5
            elif issue.severity == "medium":
                base_score -= 3
            else:
                base_score -= 1
        
        return max(0, base_score)
    
    async def _calculate_performance_score(self, performance_issues: List[CodeIssue]) -> float:
        """Calculate performance score"""
        
        base_score = 100.0
        
        for issue in performance_issues:
            if issue.severity == "high":
                base_score -= 8
            elif issue.severity == "medium":
                base_score -= 4
            else:
                base_score -= 2
        
        return max(0, base_score)
    
    async def _generate_review_report(self, results: ReviewResults, project_path: str):
        """Generate comprehensive review report"""
        
        report = {
            "review_summary": {
                "project_path": project_path,
                "review_date": datetime.now().isoformat(),
                "files_analyzed": results.total_files_analyzed,
                "lines_analyzed": results.total_lines_analyzed,
                "total_issues": len(results.issues),
                "scores": {
                    "security": results.security_score,
                    "maintainability": results.maintainability_score,
                    "performance": results.performance_score,
                    "overall": results.overall_score
                }
            },
            "issue_breakdown": {
                "by_severity": {},
                "by_type": {},
                "by_file": {}
            },
            "recommendations": [],
            "detailed_issues": []
        }
        
        # Analyze issue breakdown
        severity_counts = Counter(issue.severity for issue in results.issues)
        type_counts = Counter(issue.issue_type for issue in results.issues)
        file_counts = Counter(issue.file_path for issue in results.issues)
        
        report["issue_breakdown"]["by_severity"] = dict(severity_counts)
        report["issue_breakdown"]["by_type"] = dict(type_counts)
        report["issue_breakdown"]["by_file"] = dict(file_counts.most_common(10))  # Top 10 files
        
        # Generate high-level recommendations
        if severity_counts.get("critical", 0) > 0:
            report["recommendations"].append("Address critical security vulnerabilities immediately")
        
        if severity_counts.get("high", 0) > 5:
            report["recommendations"].append("Focus on high-severity issues for immediate improvement")
        
        if results.security_score < 70:
            report["recommendations"].append("Conduct security training and implement secure coding practices")
        
        if results.maintainability_score < 70:
            report["recommendations"].append("Improve code documentation and reduce complexity")
        
        if results.performance_score < 70:
            report["recommendations"].append("Optimize performance bottlenecks and algorithmic efficiency")
        
        # Add detailed issues (top 50 most severe)
        sorted_issues = sorted(results.issues, 
                             key=lambda x: {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}.get(x.severity, 0),
                             reverse=True)
        
        for issue in sorted_issues[:50]:
            report["detailed_issues"].append({
                "file": issue.file_path,
                "line": issue.line_number,
                "severity": issue.severity,
                "type": issue.issue_type,
                "description": issue.description,
                "recommendation": issue.recommendation,
                "confidence": issue.confidence,
                "cwe_id": issue.cwe_id
            })
        
        # Save report
        report_path = Path(project_path) / "code_review_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Review report saved to {report_path}")
        
        # Also create a summary markdown report
        await self._generate_markdown_summary(report, project_path)
    
    async def _generate_markdown_summary(self, report: Dict[str, Any], project_path: str):
        """Generate markdown summary report"""
        
        summary_path = Path(project_path) / "CODE_REVIEW_SUMMARY.md"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# Code Review Summary\n\n")
            
            # Overview
            f.write("## Overview\n\n")
            f.write(f"- **Files Analyzed**: {report['review_summary']['files_analyzed']}\n")
            f.write(f"- **Lines Analyzed**: {report['review_summary']['lines_analyzed']}\n")
            f.write(f"- **Total Issues**: {report['review_summary']['total_issues']}\n\n")
            
            # Scores
            f.write("## Scores\n\n")
            scores = report['review_summary']['scores']
            f.write(f"- **Overall Score**: {scores['overall']:.1f}/100\n")
            f.write(f"- **Security Score**: {scores['security']:.1f}/100\n")
            f.write(f"- **Maintainability Score**: {scores['maintainability']:.1f}/100\n")
            f.write(f"- **Performance Score**: {scores['performance']:.1f}/100\n\n")
            
            # Issue breakdown
            f.write("## Issue Breakdown\n\n")
            
            f.write("### By Severity\n\n")
            for severity, count in report['issue_breakdown']['by_severity'].items():
                f.write(f"- **{severity.title()}**: {count}\n")
            f.write("\n")
            
            f.write("### By Type\n\n")
            for issue_type, count in report['issue_breakdown']['by_type'].items():
                f.write(f"- **{issue_type.title()}**: {count}\n")
            f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            for i, recommendation in enumerate(report['recommendations'], 1):
                f.write(f"{i}. {recommendation}\n")
            f.write("\n")
            
            # Top issues
            f.write("## Top Critical Issues\n\n")
            critical_issues = [issue for issue in report['detailed_issues'] 
                             if issue['severity'] == 'critical'][:10]
            
            for issue in critical_issues:
                f.write(f"### {issue['file']}:{issue['line']}\n\n")
                f.write(f"**Type**: {issue['type']}\n\n")
                f.write(f"**Description**: {issue['description']}\n\n")
                f.write(f"**Recommendation**: {issue['recommendation']}\n\n")
                f.write("---\n\n")
        
        self.logger.info(f"Markdown summary saved to {summary_path}")

# Main execution interface
async def run_code_review(project_path: str, config: Optional[Dict[str, Any]] = None) -> ReviewResults:
    """Run comprehensive code review"""
    
    config = config or {}
    reviewer = ComprehensiveCodeReviewer()
    
    return await reviewer.conduct_comprehensive_review(
        project_path=project_path,
        include_patterns=config.get('include_patterns'),
        exclude_patterns=config.get('exclude_patterns')
    )

# CLI interface
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) < 2:
        print("Usage: python code_reviewer.py <project_path> [config.json]")
        sys.exit(1)
    
    project_path = sys.argv[1]
    config = {}
    
    if len(sys.argv) > 2:
        with open(sys.argv[2], 'r') as f:
            config = json.load(f)
    
    # Run code review
    async def main():
        results = await run_code_review(project_path, config)
        print(f"\nCode review completed!")
        print(f"Overall score: {results.overall_score:.1f}/100")
        print(f"Total issues found: {len(results.issues)}")
        
        # Show issue breakdown
        severity_counts = Counter(issue.severity for issue in results.issues)
        print("\nIssue breakdown:")
        for severity in ["critical", "high", "medium", "low", "info"]:
            count = severity_counts.get(severity, 0)
            if count > 0:
                print(f"  {severity.title()}: {count}")
    
    asyncio.run(main())
