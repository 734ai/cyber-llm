"""
Security Audit Automation System for Cyber-LLM
Integrates Trivy, Bandit, Safety, and other security tools for comprehensive security scanning
"""

import os
import json
import subprocess
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
import xml.etree.ElementTree as ET

from .logging_system import CyberLLMLogger, CyberLLMError, ErrorCategory, retry_with_backoff

class SeverityLevel(Enum):
    """Security vulnerability severity levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

@dataclass
class SecurityVulnerability:
    """Represents a security vulnerability"""
    id: str
    title: str
    description: str
    severity: SeverityLevel
    confidence: str
    file_path: str
    line_number: Optional[int]
    tool: str
    category: str
    cvss_score: Optional[float] = None
    cve_id: Optional[str] = None
    references: List[str] = None
    remediation: Optional[str] = None
    
    def __post_init__(self):
        if self.references is None:
            self.references = []

@dataclass
class SecurityScanResult:
    """Results from a security scan"""
    tool: str
    scan_time: datetime
    target: str
    vulnerabilities: List[SecurityVulnerability]
    scan_duration: float
    exit_code: int
    raw_output: str
    
    @property
    def vulnerability_count_by_severity(self) -> Dict[str, int]:
        """Count vulnerabilities by severity"""
        counts = {level.value: 0 for level in SeverityLevel}
        for vuln in self.vulnerabilities:
            counts[vuln.severity.value] += 1
        return counts
    
    @property
    def critical_and_high_count(self) -> int:
        """Count of critical and high severity vulnerabilities"""
        return (self.vulnerability_count_by_severity[SeverityLevel.CRITICAL.value] +
                self.vulnerability_count_by_severity[SeverityLevel.HIGH.value])

class SecurityScanner:
    """Base class for security scanners"""
    
    def __init__(self, name: str, logger: Optional[CyberLLMLogger] = None):
        self.name = name
        self.logger = logger or CyberLLMLogger(name=f"security_scanner_{name}")
    
    async def scan(self, target: str) -> SecurityScanResult:
        """Perform security scan"""
        raise NotImplementedError
    
    def is_available(self) -> bool:
        """Check if scanner tool is available"""
        raise NotImplementedError

class TrivyScanner(SecurityScanner):
    """Trivy vulnerability scanner for containers and filesystems"""
    
    def __init__(self, logger: Optional[CyberLLMLogger] = None):
        super().__init__("trivy", logger)
    
    def is_available(self) -> bool:
        """Check if Trivy is installed"""
        try:
            result = subprocess.run(['trivy', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    @retry_with_backoff(max_retries=3)
    async def scan(self, target: str) -> SecurityScanResult:
        """Scan with Trivy"""
        start_time = datetime.now()
        
        # Determine scan type based on target
        if os.path.isdir(target):
            scan_type = "fs"
        elif target.startswith(("http://", "https://")):
            scan_type = "repo"
        else:
            scan_type = "image"
        
        cmd = [
            "trivy",
            scan_type,
            "--format", "json",
            "--severity", "CRITICAL,HIGH,MEDIUM,LOW",
            target
        ]
        
        self.logger.info(f"Starting Trivy scan", target=target, scan_type=scan_type)
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            scan_duration = (datetime.now() - start_time).total_seconds()
            
            raw_output = stdout.decode() if stdout else stderr.decode()
            
            # Parse Trivy JSON output
            vulnerabilities = []
            if process.returncode == 0 and stdout:
                try:
                    trivy_data = json.loads(stdout.decode())
                    vulnerabilities = self._parse_trivy_output(trivy_data)
                except json.JSONDecodeError as e:
                    self.logger.error("Failed to parse Trivy output", error=str(e))
            
            result = SecurityScanResult(
                tool="trivy",
                scan_time=start_time,
                target=target,
                vulnerabilities=vulnerabilities,
                scan_duration=scan_duration,
                exit_code=process.returncode,
                raw_output=raw_output
            )
            
            self.logger.info(f"Trivy scan completed", 
                           target=target,
                           vulnerabilities_found=len(vulnerabilities),
                           duration=scan_duration)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Trivy scan failed", target=target, error=str(e))
            raise CyberLLMError(f"Trivy scan failed: {str(e)}", ErrorCategory.SYSTEM)
    
    def _parse_trivy_output(self, data: Dict) -> List[SecurityVulnerability]:
        """Parse Trivy JSON output into SecurityVulnerability objects"""
        vulnerabilities = []
        
        results = data.get("Results", [])
        for result in results:
            target = result.get("Target", "")
            vulns = result.get("Vulnerabilities", [])
            
            for vuln in vulns:
                vulnerability = SecurityVulnerability(
                    id=vuln.get("VulnerabilityID", ""),
                    title=vuln.get("Title", ""),
                    description=vuln.get("Description", "")[:500],  # Truncate
                    severity=SeverityLevel(vuln.get("Severity", "LOW")),
                    confidence="High",  # Trivy is generally high confidence
                    file_path=target,
                    line_number=None,
                    tool="trivy",
                    category="vulnerability",
                    cvss_score=vuln.get("CVSS", {}).get("nvd", {}).get("V3Score"),
                    cve_id=vuln.get("VulnerabilityID") if vuln.get("VulnerabilityID", "").startswith("CVE") else None,
                    references=vuln.get("References", [])[:10],  # Limit references
                    remediation=vuln.get("FixedVersion", "")
                )
                vulnerabilities.append(vulnerability)
        
        return vulnerabilities

class BanditScanner(SecurityScanner):
    """Bandit security scanner for Python code"""
    
    def __init__(self, logger: Optional[CyberLLMLogger] = None):
        super().__init__("bandit", logger)
    
    def is_available(self) -> bool:
        """Check if Bandit is installed"""
        try:
            result = subprocess.run(['bandit', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    @retry_with_backoff(max_retries=3)
    async def scan(self, target: str) -> SecurityScanResult:
        """Scan with Bandit"""
        start_time = datetime.now()
        
        cmd = [
            "bandit",
            "-r", target,
            "-f", "json",
            "-ll"  # Low level and above
        ]
        
        self.logger.info(f"Starting Bandit scan", target=target)
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            scan_duration = (datetime.now() - start_time).total_seconds()
            
            raw_output = stdout.decode() if stdout else stderr.decode()
            
            # Parse Bandit JSON output
            vulnerabilities = []
            if stdout:
                try:
                    bandit_data = json.loads(stdout.decode())
                    vulnerabilities = self._parse_bandit_output(bandit_data)
                except json.JSONDecodeError as e:
                    self.logger.error("Failed to parse Bandit output", error=str(e))
            
            result = SecurityScanResult(
                tool="bandit",
                scan_time=start_time,
                target=target,
                vulnerabilities=vulnerabilities,
                scan_duration=scan_duration,
                exit_code=process.returncode,
                raw_output=raw_output
            )
            
            self.logger.info(f"Bandit scan completed", 
                           target=target,
                           vulnerabilities_found=len(vulnerabilities),
                           duration=scan_duration)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Bandit scan failed", target=target, error=str(e))
            raise CyberLLMError(f"Bandit scan failed: {str(e)}", ErrorCategory.SYSTEM)
    
    def _parse_bandit_output(self, data: Dict) -> List[SecurityVulnerability]:
        """Parse Bandit JSON output into SecurityVulnerability objects"""
        vulnerabilities = []
        
        results = data.get("results", [])
        for result in results:
            # Map Bandit severity to our enum
            severity_map = {
                "HIGH": SeverityLevel.HIGH,
                "MEDIUM": SeverityLevel.MEDIUM,
                "LOW": SeverityLevel.LOW
            }
            
            vulnerability = SecurityVulnerability(
                id=result.get("test_id", ""),
                title=result.get("test_name", ""),
                description=result.get("issue_text", ""),
                severity=severity_map.get(result.get("issue_severity", "LOW"), SeverityLevel.LOW),
                confidence=result.get("issue_confidence", "Medium"),
                file_path=result.get("filename", ""),
                line_number=result.get("line_number"),
                tool="bandit",
                category="code_security",
                references=[f"https://bandit.readthedocs.io/en/latest/plugins/{result.get('test_id', '').lower()}.html"]
            )
            vulnerabilities.append(vulnerability)
        
        return vulnerabilities

class SafetyScanner(SecurityScanner):
    """Safety scanner for Python dependency vulnerabilities"""
    
    def __init__(self, logger: Optional[CyberLLMLogger] = None):
        super().__init__("safety", logger)
    
    def is_available(self) -> bool:
        """Check if Safety is installed"""
        try:
            result = subprocess.run(['safety', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    @retry_with_backoff(max_retries=3)
    async def scan(self, target: str) -> SecurityScanResult:
        """Scan with Safety"""
        start_time = datetime.now()
        
        cmd = ["safety", "check", "--json"]
        
        # If target is a requirements file, use it
        if os.path.isfile(target) and target.endswith(('.txt', '.in')):
            cmd.extend(["-r", target])
        
        self.logger.info(f"Starting Safety scan", target=target)
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.path.dirname(target) if os.path.isfile(target) else target
            )
            
            stdout, stderr = await process.communicate()
            scan_duration = (datetime.now() - start_time).total_seconds()
            
            raw_output = stdout.decode() if stdout else stderr.decode()
            
            # Parse Safety JSON output
            vulnerabilities = []
            if stdout:
                try:
                    safety_data = json.loads(stdout.decode())
                    vulnerabilities = self._parse_safety_output(safety_data)
                except json.JSONDecodeError as e:
                    self.logger.error("Failed to parse Safety output", error=str(e))
            
            result = SecurityScanResult(
                tool="safety",
                scan_time=start_time,
                target=target,
                vulnerabilities=vulnerabilities,
                scan_duration=scan_duration,
                exit_code=process.returncode,
                raw_output=raw_output
            )
            
            self.logger.info(f"Safety scan completed", 
                           target=target,
                           vulnerabilities_found=len(vulnerabilities),
                           duration=scan_duration)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Safety scan failed", target=target, error=str(e))
            raise CyberLLMError(f"Safety scan failed: {str(e)}", ErrorCategory.SYSTEM)
    
    def _parse_safety_output(self, data: List) -> List[SecurityVulnerability]:
        """Parse Safety JSON output into SecurityVulnerability objects"""
        vulnerabilities = []
        
        for vuln_data in data:
            vulnerability = SecurityVulnerability(
                id=vuln_data.get("id", ""),
                title=f"Vulnerable dependency: {vuln_data.get('package_name', '')}",
                description=vuln_data.get("advisory", ""),
                severity=SeverityLevel.HIGH,  # Safety considers all vulnerabilities high
                confidence="High",
                file_path="requirements",
                line_number=None,
                tool="safety",
                category="dependency_vulnerability",
                references=[f"https://pyup.io/vulnerabilities/{vuln_data.get('id', '')}/"]
            )
            vulnerabilities.append(vulnerability)
        
        return vulnerabilities

class SecurityAuditSystem:
    """Central security audit system orchestrating multiple scanners"""
    
    def __init__(self, 
                 scanners: Optional[List[SecurityScanner]] = None,
                 logger: Optional[CyberLLMLogger] = None):
        
        self.logger = logger or CyberLLMLogger(name="security_audit")
        
        # Initialize scanners
        if scanners:
            self.scanners = scanners
        else:
            self.scanners = [
                TrivyScanner(logger=self.logger),
                BanditScanner(logger=self.logger),
                SafetyScanner(logger=self.logger)
            ]
        
        # Filter to only available scanners
        self.available_scanners = [s for s in self.scanners if s.is_available()]
        
        if not self.available_scanners:
            self.logger.warning("No security scanners are available")
        else:
            scanner_names = [s.name for s in self.available_scanners]
            self.logger.info(f"Available scanners: {', '.join(scanner_names)}")
    
    async def full_security_audit(self, 
                                target: str,
                                skip_scanners: Optional[List[str]] = None) -> Dict[str, SecurityScanResult]:
        """Perform full security audit with all available scanners"""
        
        skip_scanners = skip_scanners or []
        results = {}
        
        for scanner in self.available_scanners:
            if scanner.name in skip_scanners:
                self.logger.info(f"Skipping scanner: {scanner.name}")
                continue
            
            try:
                self.logger.info(f"Running security scan with {scanner.name}")
                result = await scanner.scan(target)
                results[scanner.name] = result
                
                # Log summary
                vuln_counts = result.vulnerability_count_by_severity
                self.logger.info(f"{scanner.name} scan summary",
                               critical=vuln_counts[SeverityLevel.CRITICAL.value],
                               high=vuln_counts[SeverityLevel.HIGH.value],
                               medium=vuln_counts[SeverityLevel.MEDIUM.value],
                               low=vuln_counts[SeverityLevel.LOW.value])
                
            except Exception as e:
                self.logger.error(f"Scanner {scanner.name} failed", error=str(e))
                continue
        
        return results
    
    def generate_security_report(self, 
                               results: Dict[str, SecurityScanResult],
                               output_format: str = "json") -> str:
        """Generate security audit report"""
        
        # Aggregate statistics
        total_vulnerabilities = sum(len(result.vulnerabilities) for result in results.values())
        
        severity_totals = {level.value: 0 for level in SeverityLevel}
        for result in results.values():
            counts = result.vulnerability_count_by_severity
            for severity, count in counts.items():
                severity_totals[severity] += count
        
        # Build report
        report_data = {
            "audit_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_vulnerabilities": total_vulnerabilities,
                "critical_vulnerabilities": severity_totals[SeverityLevel.CRITICAL.value],
                "high_vulnerabilities": severity_totals[SeverityLevel.HIGH.value],
                "medium_vulnerabilities": severity_totals[SeverityLevel.MEDIUM.value],
                "low_vulnerabilities": severity_totals[SeverityLevel.LOW.value],
                "scanners_used": list(results.keys())
            },
            "scan_results": {}
        }
        
        # Add detailed results
        for scanner_name, result in results.items():
            report_data["scan_results"][scanner_name] = {
                "scan_time": result.scan_time.isoformat(),
                "target": result.target,
                "duration": result.scan_duration,
                "exit_code": result.exit_code,
                "vulnerability_count": len(result.vulnerabilities),
                "vulnerabilities": [asdict(vuln) for vuln in result.vulnerabilities]
            }
        
        if output_format.lower() == "json":
            return json.dumps(report_data, indent=2, default=str)
        elif output_format.lower() == "yaml":
            return yaml.dump(report_data, default_flow_style=False)
        else:
            # Generate markdown report
            return self._generate_markdown_report(report_data)
    
    def _generate_markdown_report(self, report_data: Dict) -> str:
        """Generate markdown security report"""
        
        md_lines = [
            "# Security Audit Report",
            f"\n**Generated:** {report_data['audit_timestamp']}",
            "\n## Summary",
            f"- **Total Vulnerabilities:** {report_data['summary']['total_vulnerabilities']}",
            f"- **Critical:** {report_data['summary']['critical_vulnerabilities']}",
            f"- **High:** {report_data['summary']['high_vulnerabilities']}",
            f"- **Medium:** {report_data['summary']['medium_vulnerabilities']}",
            f"- **Low:** {report_data['summary']['low_vulnerabilities']}",
            f"- **Scanners Used:** {', '.join(report_data['summary']['scanners_used'])}",
            "\n## Detailed Results"
        ]
        
        for scanner_name, result in report_data["scan_results"].items():
            md_lines.extend([
                f"\n### {scanner_name.title()} Scanner",
                f"- **Target:** {result['target']}",
                f"- **Duration:** {result['duration']:.2f}s",
                f"- **Vulnerabilities Found:** {result['vulnerability_count']}"
            ])
            
            if result['vulnerabilities']:
                md_lines.append("\n#### Vulnerabilities")
                for vuln in result['vulnerabilities'][:10]:  # Limit to top 10
                    md_lines.extend([
                        f"\n**{vuln['title']}** ({vuln['severity']})",
                        f"- **File:** {vuln['file_path']}",
                        f"- **Description:** {vuln['description'][:200]}..."
                    ])
                
                if len(result['vulnerabilities']) > 10:
                    md_lines.append(f"\n*... and {len(result['vulnerabilities']) - 10} more vulnerabilities*")
        
        return "\n".join(md_lines)
    
    async def install_scanners(self) -> Dict[str, bool]:
        """Install missing security scanners"""
        installation_results = {}
        
        # Try to install missing scanners
        scanner_installs = {
            "trivy": [
                "curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin"
            ],
            "bandit": ["pip install bandit"],
            "safety": ["pip install safety"]
        }
        
        for scanner_name, install_cmds in scanner_installs.items():
            # Check if already available
            scanner_class = {
                "trivy": TrivyScanner,
                "bandit": BanditScanner,
                "safety": SafetyScanner
            }[scanner_name]
            
            scanner = scanner_class(logger=self.logger)
            if scanner.is_available():
                installation_results[scanner_name] = True
                continue
            
            # Try to install
            self.logger.info(f"Installing {scanner_name}")
            
            for cmd in install_cmds:
                try:
                    process = await asyncio.create_subprocess_shell(
                        cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode == 0:
                        self.logger.info(f"Successfully installed {scanner_name}")
                        installation_results[scanner_name] = True
                        break
                    else:
                        self.logger.error(f"Failed to install {scanner_name}", 
                                        error=stderr.decode())
                        
                except Exception as e:
                    self.logger.error(f"Installation error for {scanner_name}", error=str(e))
            
            if scanner_name not in installation_results:
                installation_results[scanner_name] = False
        
        return installation_results

# Convenience functions
async def run_security_audit(target: str, 
                           output_file: Optional[str] = None,
                           output_format: str = "json") -> Dict[str, SecurityScanResult]:
    """Run security audit and optionally save report"""
    
    audit_system = SecurityAuditSystem()
    results = await audit_system.full_security_audit(target)
    
    if output_file:
        report = audit_system.generate_security_report(results, output_format)
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"Security report saved to: {output_file}")
    
    return results

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize security audit system
        audit_system = SecurityAuditSystem()
        
        # Install missing scanners
        install_results = await audit_system.install_scanners()
        print("Scanner installation results:", install_results)
        
        # Run security audit on current directory
        results = await audit_system.full_security_audit(".")
        
        # Generate report
        report = audit_system.generate_security_report(results, "markdown")
        print("\n" + "="*80)
        print(report)
    
    asyncio.run(main())
