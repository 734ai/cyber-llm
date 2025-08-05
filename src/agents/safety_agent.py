"""
SafetyAgent: OPSEC Compliance and Safety Validation Agent
Validates operations for OPSEC compliance and safety considerations.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SafetyCheck:
    """Safety check result."""
    check_name: str
    risk_level: RiskLevel
    description: str
    violations: List[str]
    recommendations: List[str]

@dataclass
class SafetyAssessment:
    """Complete safety assessment result."""
    overall_risk: RiskLevel
    checks: List[SafetyCheck]
    approved: bool
    summary: str
    safe_alternatives: List[str]

class SafetyAgent:
    """Advanced safety and OPSEC compliance validation agent."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.opsec_rules = self._load_opsec_rules()
        self.risk_patterns = self._load_risk_patterns()
        
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load safety agent configuration."""
        default_config = {
            'strict_mode': True,
            'auto_approve_low_risk': False,
            'require_human_approval': ['high', 'critical'],
            'logging_level': 'INFO',
            'detection_threshold': 0.7
        }
        
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
                
        return default_config
    
    def _load_opsec_rules(self) -> Dict:
        """Load OPSEC rules and best practices."""
        return {
            'timing_rules': {
                'max_requests_per_minute': 10,
                'min_delay_between_scans': 1000,  # milliseconds
                'avoid_business_hours': True,
                'spread_over_days': ['high', 'maximum']
            },
            'stealth_rules': {
                'use_decoy_ips': ['high', 'maximum'],
                'fragment_packets': ['medium', 'high', 'maximum'],
                'randomize_source_ports': True,
                'avoid_default_timing': ['medium', 'high', 'maximum']
            },
            'target_rules': {
                'avoid_honeypots': True,
                'check_threat_intelligence': True,
                'respect_robots_txt': True,
                'avoid_government_domains': True
            },
            'operational_rules': {
                'log_all_activities': True,
                'use_vpn_tor': ['high', 'maximum'],
                'rotate_infrastructure': ['high', 'maximum'],
                'monitor_defensive_responses': True
            }
        }
    
    def _load_risk_patterns(self) -> Dict:
        """Load patterns that indicate high-risk activities."""
        return {
            'high_detection_commands': [
                r'-A\b',  # Aggressive scan
                r'--script.*vuln',  # Vulnerability scripts
                r'-sU.*-sS',  # UDP + SYN scan combination
                r'--top-ports\s+(\d+)',  # High port count
                r'-T[45]',  # Aggressive timing
                r'--min-rate\s+\d{3,}',  # High rate scanning
                r'nikto',  # Web vulnerability scanner
                r'sqlmap',  # SQL injection tool
                r'hydra',  # Brute force tool
                r'john',  # Password cracker
                r'hashcat'  # Password cracker
            ],
            'opsec_violations': [
                r'--reason',  # Custom scan reason (logging risk)
                r'-v{2,}',  # High verbosity
                r'--packet-trace',  # Packet tracing
                r'--traceroute',  # Network path disclosure
                r'-sn.*--traceroute',  # Ping sweep with traceroute
                r'--source-port\s+53',  # DNS source port spoofing
                r'--data-string',  # Custom data (potential signature)
            ],
            'infrastructure_risks': [
                r'shodan.*api',  # Shodan API usage
                r'censys.*search',  # Censys API usage
                r'virustotal',  # VirusTotal queries
                r'threatcrowd',  # Threat intelligence queries
                r'passivetotal'  # PassiveTotal queries
            ],
            'time_sensitive': [
                r'while.*true',  # Infinite loops
                r'for.*in.*range\(\s*\d{3,}',  # Large iterations
                r'sleep\s+[0-9]*[.][0-9]+',  # Very short delays
                r'--max-rate',  # Rate limiting bypass
            ]
        }
    
    def validate_commands(self, commands: Dict[str, List[str]], opsec_level: str = 'medium') -> SafetyAssessment:
        """
        Validate a set of commands for OPSEC compliance and safety.
        
        Args:
            commands: Dictionary of command categories and command lists
            opsec_level: Required OPSEC level ('low', 'medium', 'high', 'maximum')
        
        Returns:
            SafetyAssessment with validation results
        """
        self.logger.info(f"Validating commands for OPSEC level: {opsec_level}")
        
        checks = []
        
        # Perform individual safety checks
        checks.append(self._check_detection_risk(commands))
        checks.append(self._check_opsec_compliance(commands, opsec_level))
        checks.append(self._check_timing_compliance(commands, opsec_level))
        checks.append(self._check_infrastructure_safety(commands))
        checks.append(self._check_target_appropriateness(commands))
        
        # Calculate overall risk
        overall_risk = self._calculate_overall_risk(checks)
        
        # Determine approval status
        approved = self._determine_approval(overall_risk, opsec_level)
        
        # Generate summary
        summary = self._generate_summary(checks, overall_risk, approved)
        
        # Generate safe alternatives if not approved
        safe_alternatives = []
        if not approved:
            safe_alternatives = self._generate_safe_alternatives(commands, opsec_level)
        
        return SafetyAssessment(
            overall_risk=overall_risk,
            checks=checks,
            approved=approved,
            summary=summary,
            safe_alternatives=safe_alternatives
        )
    
    def _check_detection_risk(self, commands: Dict[str, List[str]]) -> SafetyCheck:
        """Check for commands with high detection risk."""
        violations = []
        recommendations = []
        
        all_commands = []
        for cmd_list in commands.values():
            all_commands.extend(cmd_list)
        
        command_text = ' '.join(all_commands)
        
        for pattern in self.risk_patterns['high_detection_commands']:
            matches = re.findall(pattern, command_text, re.IGNORECASE)
            if matches:
                violations.append(f"High-detection pattern found: {pattern}")
        
        # Check for aggressive scanning combinations
        if '-sS' in command_text and '-sV' in command_text:
            violations.append("Aggressive scanning combination: SYN scan + service detection")
        
        if len(violations) == 0:
            risk_level = RiskLevel.LOW
            description = "No high-detection risk patterns found"
        elif len(violations) <= 2:
            risk_level = RiskLevel.MEDIUM
            description = "Some detection risk patterns identified"
            recommendations.extend([
                "Consider using stealth timing (-T1 or -T2)",
                "Add packet fragmentation (-f)",
                "Implement delays between scans"
            ])
        else:
            risk_level = RiskLevel.HIGH
            description = "Multiple high-detection risk patterns found"
            recommendations.extend([
                "Significantly reduce scanning aggressiveness",
                "Use passive techniques where possible",
                "Implement substantial delays",
                "Consider using decoy IPs"
            ])
        
        return SafetyCheck(
            check_name="Detection Risk Analysis",
            risk_level=risk_level,
            description=description,
            violations=violations,
            recommendations=recommendations
        )
    
    def _check_opsec_compliance(self, commands: Dict[str, List[str]], opsec_level: str) -> SafetyCheck:
        """Check OPSEC compliance based on required level."""
        violations = []
        recommendations = []
        
        all_commands = ' '.join([' '.join(cmd_list) for cmd_list in commands.values()])
        
        # Check stealth requirements
        stealth_rules = self.opsec_rules['stealth_rules']
        
        if opsec_level in ['medium', 'high', 'maximum'] and '-T4' in all_commands:
            violations.append("Aggressive timing (-T4) conflicts with stealth requirements")
        
        if opsec_level in ['high', 'maximum'] and not any('-f' in cmd for cmd_list in commands.values() for cmd in cmd_list):
            violations.append("Packet fragmentation (-f) recommended for high stealth")
        
        if opsec_level == 'maximum' and any('nmap' in cmd for cmd_list in commands.values() for cmd in cmd_list):
            violations.append("Active scanning not recommended for maximum stealth")
        
        # Check for OPSEC violation patterns
        for pattern in self.risk_patterns['opsec_violations']:
            if re.search(pattern, all_commands, re.IGNORECASE):
                violations.append(f"OPSEC violation pattern: {pattern}")
        
        # Determine risk level
        if len(violations) == 0:
            risk_level = RiskLevel.LOW
            description = f"Commands comply with {opsec_level} OPSEC requirements"
        elif len(violations) <= 2:
            risk_level = RiskLevel.MEDIUM
            description = f"Minor OPSEC compliance issues for {opsec_level} level"
        else:
            risk_level = RiskLevel.HIGH
            description = f"Significant OPSEC violations for {opsec_level} level"
        
        # Generate recommendations based on OPSEC level
        if opsec_level == 'maximum':
            recommendations.extend([
                "Use only passive reconnaissance techniques",
                "Employ Tor or VPN for all queries",
                "Spread activities over multiple days"
            ])
        elif opsec_level == 'high':
            recommendations.extend([
                "Use stealth timing (-T1 or -T2)",
                "Implement packet fragmentation",
                "Add significant delays between operations"
            ])
        
        return SafetyCheck(
            check_name="OPSEC Compliance",
            risk_level=risk_level,
            description=description,
            violations=violations,
            recommendations=recommendations
        )
    
    def _check_timing_compliance(self, commands: Dict[str, List[str]], opsec_level: str) -> SafetyCheck:
        """Check timing and rate limiting compliance."""
        violations = []
        recommendations = []
        
        all_commands = ' '.join([' '.join(cmd_list) for cmd_list in commands.values()])
        
        # Check for timing violations
        timing_rules = self.opsec_rules['timing_rules']
        
        # Check for aggressive timing
        aggressive_timing = re.findall(r'-T([45])', all_commands)
        if aggressive_timing and opsec_level in ['medium', 'high', 'maximum']:
            violations.append(f"Aggressive timing (-T{'/'.join(aggressive_timing)}) violates {opsec_level} OPSEC")
        
        # Check for high rate scanning
        rate_matches = re.findall(r'--min-rate\s+(\d+)', all_commands)
        if rate_matches:
            for rate in rate_matches:
                if int(rate) > 100 and opsec_level in ['high', 'maximum']:
                    violations.append(f"High scan rate ({rate}) not suitable for {opsec_level} OPSEC")
        
        # Check for insufficient delays
        delay_matches = re.findall(r'--scan-delay\s+(\d+)', all_commands)
        if opsec_level in ['high', 'maximum'] and not delay_matches:
            violations.append("Scan delays not specified for high stealth requirement")
        
        risk_level = RiskLevel.LOW if len(violations) == 0 else (
            RiskLevel.MEDIUM if len(violations) <= 2 else RiskLevel.HIGH
        )
        
        if risk_level != RiskLevel.LOW:
            recommendations.extend([
                "Implement appropriate scan timing for OPSEC level",
                "Add delays between scan phases",
                "Consider spreading scans over longer time periods"
            ])
        
        return SafetyCheck(
            check_name="Timing Compliance",
            risk_level=risk_level,
            description=f"Timing analysis for {opsec_level} OPSEC level",
            violations=violations,
            recommendations=recommendations
        )
    
    def _check_infrastructure_safety(self, commands: Dict[str, List[str]]) -> SafetyCheck:
        """Check for infrastructure and API safety."""
        violations = []
        recommendations = []
        
        all_commands = ' '.join([' '.join(cmd_list) for cmd_list in commands.values()])
        
        # Check for infrastructure risks
        for pattern in self.risk_patterns['infrastructure_risks']:
            if re.search(pattern, all_commands, re.IGNORECASE):
                violations.append(f"Infrastructure risk: {pattern}")
        
        # Check for API key exposure
        if 'api' in all_commands.lower() and 'key' in all_commands.lower():
            violations.append("Potential API key exposure in commands")
        
        risk_level = RiskLevel.LOW if len(violations) == 0 else RiskLevel.MEDIUM
        
        if violations:
            recommendations.extend([
                "Secure API keys using environment variables",
                "Use VPN/proxy for external API queries",
                "Monitor API usage quotas"
            ])
        
        return SafetyCheck(
            check_name="Infrastructure Safety",
            risk_level=risk_level,
            description="Infrastructure and API safety analysis",
            violations=violations,
            recommendations=recommendations
        )
    
    def _check_target_appropriateness(self, commands: Dict[str, List[str]]) -> SafetyCheck:
        """Check target appropriateness and legal considerations."""
        violations = []
        recommendations = []
        
        # Extract targets from commands
        targets = self._extract_targets_from_commands(commands)
        
        for target in targets:
            # Check for government domains
            if any(gov_tld in target.lower() for gov_tld in ['.gov', '.mil', '.fed']):
                violations.append(f"Government domain detected: {target}")
            
            # Check for known honeypot indicators
            if any(honeypot in target.lower() for honeypot in ['honeypot', 'canary', 'trap']):
                violations.append(f"Potential honeypot detected: {target}")
        
        risk_level = RiskLevel.CRITICAL if any('.gov' in v or '.mil' in v for v in violations) else (
            RiskLevel.HIGH if violations else RiskLevel.LOW
        )
        
        if risk_level != RiskLevel.LOW:
            recommendations.extend([
                "Verify authorization for all targets",
                "Review legal implications",
                "Consider using test environments"
            ])
        
        return SafetyCheck(
            check_name="Target Appropriateness",
            risk_level=risk_level,
            description="Target selection and legal compliance",
            violations=violations,
            recommendations=recommendations
        )
    
    def _extract_targets_from_commands(self, commands: Dict[str, List[str]]) -> List[str]:
        """Extract target IPs/domains from commands."""
        targets = []
        
        for cmd_list in commands.values():
            for cmd in cmd_list:
                # Simple regex to find IP addresses and domains
                ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
                domain_pattern = r'\b[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*\b'
                
                targets.extend(re.findall(ip_pattern, cmd))
                targets.extend(re.findall(domain_pattern, cmd))
        
        return list(set(targets))  # Remove duplicates
    
    def _calculate_overall_risk(self, checks: List[SafetyCheck]) -> RiskLevel:
        """Calculate overall risk level from individual checks."""
        risk_scores = {
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4
        }
        
        max_risk = max(check.risk_level for check in checks)
        avg_risk = sum(risk_scores[check.risk_level] for check in checks) / len(checks)
        
        # If any check is critical, overall is critical
        if max_risk == RiskLevel.CRITICAL:
            return RiskLevel.CRITICAL
        
        # If average risk is high, overall is high
        if avg_risk >= 3.0:
            return RiskLevel.HIGH
        elif avg_risk >= 2.0:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _determine_approval(self, overall_risk: RiskLevel, opsec_level: str) -> bool:
        """Determine if commands are approved based on risk and configuration."""
        if overall_risk == RiskLevel.CRITICAL:
            return False
        
        if overall_risk == RiskLevel.HIGH and self.config['strict_mode']:
            return False
        
        if overall_risk.value in self.config['require_human_approval']:
            self.logger.warning(f"HUMAN APPROVAL REQUIRED for {overall_risk.value} risk level")
            return False  # Requires manual approval
        
        if overall_risk == RiskLevel.LOW and self.config['auto_approve_low_risk']:
            return True
        
        return overall_risk in [RiskLevel.LOW, RiskLevel.MEDIUM]
    
    def _generate_summary(self, checks: List[SafetyCheck], overall_risk: RiskLevel, approved: bool) -> str:
        """Generate a summary of the safety assessment."""
        violation_count = sum(len(check.violations) for check in checks)
        
        status = "APPROVED" if approved else "REJECTED"
        
        summary = f"Safety Assessment: {status}\n"
        summary += f"Overall Risk Level: {overall_risk.value.upper()}\n"
        summary += f"Total Violations: {violation_count}\n"
        
        if not approved:
            summary += "\nREASONS FOR REJECTION:\n"
            for check in checks:
                if check.violations:
                    summary += f"- {check.check_name}: {len(check.violations)} violations\n"
        
        return summary
    
    def _generate_safe_alternatives(self, commands: Dict[str, List[str]], opsec_level: str) -> List[str]:
        """Generate safer alternative commands."""
        alternatives = []
        
        # General safer alternatives
        alternatives.extend([
            "Use passive reconnaissance techniques first",
            "Implement longer delays between scans (--scan-delay 2000ms)",
            "Use stealth timing (-T1 or -T2)",
            "Add packet fragmentation (-f)",
            "Reduce port scan range (--top-ports 100)",
            "Use decoy IPs (-D RND:10)"
        ])
        
        if opsec_level in ['high', 'maximum']:
            alternatives.extend([
                "Consider using only passive techniques",
                "Employ Tor/VPN for all reconnaissance",
                "Spread activities over multiple days",
                "Use different source IPs for different phases"
            ])
        
        return alternatives

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize safety agent
    agent = SafetyAgent()
    
    # Example commands to validate
    test_commands = {
        'nmap': [
            'nmap -sS -T4 --top-ports 1000 example.com',
            'nmap -A -v example.com'
        ],
        'passive': [
            'dig example.com ANY',
            'whois example.com'
        ]
    }
    
    # Validate commands
    assessment = agent.validate_commands(test_commands, opsec_level='high')
    
    print(f"Assessment: {assessment.approved}")
    print(f"Overall Risk: {assessment.overall_risk.value}")
    print(f"Summary:\n{assessment.summary}")
    
    if assessment.safe_alternatives:
        print(f"\nSafe Alternatives:")
        for alt in assessment.safe_alternatives:
            print(f"- {alt}")
