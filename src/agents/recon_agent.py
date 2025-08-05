"""
ReconAgent: Cybersecurity Reconnaissance Agent
Performs stealth reconnaissance and information gathering operations.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# HUMAN_APPROVAL_REQUIRED: Review reconnaissance strategies before execution

@dataclass
class ReconTarget:
    """Target information for reconnaissance operations."""
    target: str
    target_type: str  # 'domain', 'ip', 'network', 'organization'
    constraints: Dict[str, Any]
    opsec_level: str = 'medium'  # 'low', 'medium', 'high', 'maximum'

@dataclass
class ReconResult:
    """Results from reconnaissance operations."""
    target: str
    commands: Dict[str, List[str]]
    passive_techniques: List[str]
    opsec_notes: List[str]
    risk_assessment: str
    next_steps: List[str]

class ReconAgent:
    """Advanced reconnaissance agent with OPSEC awareness."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.opsec_profiles = self._load_opsec_profiles()
        
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load agent configuration."""
        default_config = {
            'max_scan_ports': 1000,
            'scan_timing': 'T3',  # Normal timing
            'stealth_mode': True,
            'passive_only': False,
            'shodan_api_key': None,
            'censys_api_key': None
        }
        
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
                
        return default_config
    
    def _load_opsec_profiles(self) -> Dict:
        """Load OPSEC profiles for different stealth levels."""
        return {
            'low': {
                'timing': 'T4',
                'port_limit': 65535,
                'techniques': ['tcp_connect', 'udp_scan', 'service_detection'],
                'delay_between_scans': 0
            },
            'medium': {
                'timing': 'T3',
                'port_limit': 1000,
                'techniques': ['syn_scan', 'service_detection'],
                'delay_between_scans': 1
            },
            'high': {
                'timing': 'T2',
                'port_limit': 100,
                'techniques': ['syn_scan'],
                'delay_between_scans': 5
            },
            'maximum': {
                'timing': 'T1',
                'port_limit': 22,  # Common ports only
                'techniques': ['passive_only'],
                'delay_between_scans': 30
            }
        }
    
    def analyze_target(self, target_info: ReconTarget) -> ReconResult:
        """
        Analyze target and generate reconnaissance strategy.
        
        HUMAN_APPROVAL_REQUIRED: Review target analysis before proceeding
        """
        self.logger.info(f"Analyzing target: {target_info.target}")
        
        # Get OPSEC profile
        opsec_profile = self.opsec_profiles.get(target_info.opsec_level, self.opsec_profiles['medium'])
        
        # Generate reconnaissance commands
        commands = {
            'nmap': self._generate_nmap_commands(target_info, opsec_profile),
            'passive_dns': self._generate_passive_dns_commands(target_info),
            'osint': self._generate_osint_commands(target_info),
            'shodan': self._generate_shodan_queries(target_info)
        }
        
        # Generate passive techniques
        passive_techniques = self._generate_passive_techniques(target_info)
        
        # OPSEC considerations
        opsec_notes = self._generate_opsec_notes(target_info, opsec_profile)
        
        # Risk assessment
        risk_assessment = self._assess_reconnaissance_risk(target_info, commands)
        
        # Next steps
        next_steps = self._suggest_next_steps(target_info, commands)
        
        return ReconResult(
            target=target_info.target,
            commands=commands,
            passive_techniques=passive_techniques,
            opsec_notes=opsec_notes,
            risk_assessment=risk_assessment,
            next_steps=next_steps
        )
    
    def _generate_nmap_commands(self, target: ReconTarget, opsec_profile: Dict) -> List[str]:
        """Generate OPSEC-aware Nmap commands."""
        commands = []
        timing = opsec_profile['timing']
        port_limit = min(opsec_profile['port_limit'], self.config['max_scan_ports'])
        
        if 'passive_only' in opsec_profile['techniques']:
            return []  # No active scanning for maximum stealth
        
        # Host discovery
        if target.opsec_level in ['low', 'medium']:
            commands.append(f"nmap -sn {target.target}")
        
        # Port scanning
        if 'syn_scan' in opsec_profile['techniques']:
            commands.append(f"nmap -sS -{timing} --top-ports {port_limit} {target.target}")
        elif 'tcp_connect' in opsec_profile['techniques']:
            commands.append(f"nmap -sT -{timing} --top-ports {port_limit} {target.target}")
        
        # Service detection (careful with stealth)
        if 'service_detection' in opsec_profile['techniques'] and target.opsec_level != 'high':
            commands.append(f"nmap -sV -{timing} --version-intensity 2 {target.target}")
        
        # OS detection (only for low OPSEC)
        if target.opsec_level == 'low':
            commands.append(f"nmap -O -{timing} {target.target}")
        
        # Add stealth flags
        for i, cmd in enumerate(commands):
            if target.opsec_level in ['high', 'maximum']:
                commands[i] += " -f --scan-delay 1000ms"  # Fragment packets, add delay
        
        return commands
    
    def _generate_passive_dns_commands(self, target: ReconTarget) -> List[str]:
        """Generate passive DNS reconnaissance commands."""
        commands = []
        
        if target.target_type == 'domain':
            commands.extend([
                f"dig {target.target} ANY",
                f"dig {target.target} TXT",
                f"dig {target.target} MX",
                f"dig {target.target} NS",
                f"whois {target.target}",
                f"curl -s 'https://crt.sh/?q={target.target}&output=json'"
            ])
        
        return commands
    
    def _generate_osint_commands(self, target: ReconTarget) -> List[str]:
        """Generate OSINT gathering commands."""
        commands = []
        
        if target.target_type in ['domain', 'organization']:
            commands.extend([
                f"theharvester -d {target.target} -b google,bing,linkedin",
                f"amass enum -d {target.target}",
                f"subfinder -d {target.target}",
                f"curl -s 'https://api.github.com/search/code?q={target.target}'"
            ])
        
        return commands
    
    def _generate_shodan_queries(self, target: ReconTarget) -> List[str]:
        """Generate Shodan search queries."""
        if not self.config.get('shodan_api_key'):
            return ["# Shodan API key not configured"]
        
        queries = []
        
        if target.target_type == 'ip':
            queries.append(f"host:{target.target}")
        elif target.target_type == 'domain':
            queries.extend([
                f"hostname:{target.target}",
                f"ssl:{target.target}",
                f"org:\"{target.target}\""
            ])
        elif target.target_type == 'organization':
            queries.append(f"org:\"{target.target}\"")
        
        return queries
    
    def _generate_passive_techniques(self, target: ReconTarget) -> List[str]:
        """Generate list of passive reconnaissance techniques."""
        techniques = [
            "Certificate Transparency log analysis",
            "DNS cache snooping",
            "BGP route analysis",
            "Social media reconnaissance",
            "Job posting analysis",
            "Public document metadata extraction",
            "Wayback Machine analysis",
            "GitHub/GitLab repository search"
        ]
        
        if target.target_type == 'organization':
            techniques.extend([
                "LinkedIn employee enumeration",
                "SEC filing analysis",
                "Press release analysis",
                "Conference presentation search"
            ])
        
        return techniques
    
    def _generate_opsec_notes(self, target: ReconTarget, opsec_profile: Dict) -> List[str]:
        """Generate OPSEC considerations and warnings."""
        notes = []
        
        if target.opsec_level == 'maximum':
            notes.extend([
                "MAXIMUM STEALTH: Use only passive techniques",
                "Consider using Tor or VPN for all queries",
                "Spread reconnaissance over multiple days",
                "Use different source IPs for different queries"
            ])
        elif target.opsec_level == 'high':
            notes.extend([
                "HIGH STEALTH: Minimize active scanning",
                "Use packet fragmentation and timing delays",
                "Consider using decoy IPs",
                "Monitor for defensive responses"
            ])
        elif target.opsec_level == 'medium':
            notes.extend([
                "MEDIUM STEALTH: Balance speed and stealth",
                "Use moderate timing delays",
                "Avoid aggressive service detection"
            ])
        else:  # low
            notes.extend([
                "LOW STEALTH: Speed prioritized over stealth",
                "Full port ranges and service detection enabled",
                "Monitor logs for potential detection"
            ])
        
        # General OPSEC notes
        notes.extend([
            "Log all reconnaissance activities",
            "Use legitimate-looking User-Agent strings",
            "Vary timing between different techniques",
            "Document any anomalous responses"
        ])
        
        return notes
    
    def _assess_reconnaissance_risk(self, target: ReconTarget, commands: Dict) -> str:
        """Assess the risk level of the reconnaissance plan."""
        risk_factors = []
        
        # Count active scanning commands
        active_commands = len(commands.get('nmap', []))
        if active_commands > 5:
            risk_factors.append("High number of active scans")
        
        # Check OPSEC level vs techniques
        if target.opsec_level == 'maximum' and active_commands > 0:
            risk_factors.append("Active scanning conflicts with maximum stealth requirement")
        
        # Check for aggressive techniques
        nmap_commands = ' '.join(commands.get('nmap', []))
        if '-A' in nmap_commands or '--script' in nmap_commands:
            risk_factors.append("Aggressive scanning techniques detected")
        
        if not risk_factors:
            return "LOW: Reconnaissance plan follows OPSEC guidelines"
        elif len(risk_factors) <= 2:
            return f"MEDIUM: Consider addressing: {'; '.join(risk_factors)}"
        else:
            return f"HIGH: Multiple risk factors identified: {'; '.join(risk_factors)}"
    
    def _suggest_next_steps(self, target: ReconTarget, commands: Dict) -> List[str]:
        """Suggest next steps based on reconnaissance results."""
        steps = [
            "Execute passive reconnaissance first",
            "Analyze results for interesting services/ports",
            "Proceed with active scanning if OPSEC allows",
            "Document all findings in structured format",
            "Identify potential attack vectors",
            "Plan next phase based on discovered services"
        ]
        
        if target.opsec_level in ['high', 'maximum']:
            steps.insert(1, "Wait 24-48 hours between reconnaissance phases")
        
        return steps
    
    def execute_reconnaissance(self, target_info: ReconTarget) -> Dict:
        """
        Execute reconnaissance plan (simulation/planning mode).
        
        HUMAN_APPROVAL_REQUIRED: Manual execution required for actual scanning
        """
        self.logger.warning("SIMULATION MODE: Actual command execution disabled for safety")
        
        recon_result = self.analyze_target(target_info)
        
        # Return structured results for logging/analysis
        return {
            'target': target_info.target,
            'opsec_level': target_info.opsec_level,
            'plan': recon_result.__dict__,
            'execution_status': 'SIMULATION_ONLY',
            'timestamp': str(Path().cwd())
        }

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize agent
    agent = ReconAgent()
    
    # Example target
    target = ReconTarget(
        target="example.com",
        target_type="domain",
        constraints={"time_limit": "2h", "stealth": True},
        opsec_level="high"
    )
    
    # Analyze target
    result = agent.execute_reconnaissance(target)
    print(json.dumps(result, indent=2))
