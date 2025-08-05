"""
Unit tests for SafetyAgent
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from agents.safety_agent import SafetyAgent, RiskLevel, SafetyCheck, SafetyAssessment

class TestSafetyAgent:
    """Test suite for SafetyAgent."""
    
    @pytest.fixture
    def agent(self):
        """Create a SafetyAgent instance for testing."""
        return SafetyAgent()
    
    @pytest.fixture
    def safe_commands(self):
        """Sample safe commands for testing."""
        return {
            'nmap': ['nmap -sS -T2 --top-ports 100 example.com'],
            'passive': ['dig example.com ANY', 'whois example.com']
        }
    
    @pytest.fixture
    def risky_commands(self):
        """Sample risky commands for testing."""
        return {
            'nmap': [
                'nmap -A -T4 --script vuln example.com',
                'nmap -sS -sV --top-ports 65535 example.com'
            ],
            'tools': ['nikto -h example.com', 'sqlmap -u http://example.com']
        }
    
    @pytest.fixture
    def government_commands(self):
        """Commands targeting government domains."""
        return {
            'nmap': ['nmap -sS whitehouse.gov'],
            'passive': ['dig defense.gov ANY']
        }
    
    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent is not None
        assert hasattr(agent, 'config')
        assert hasattr(agent, 'opsec_rules')
        assert hasattr(agent, 'risk_patterns')
    
    def test_opsec_rules_loaded(self, agent):
        """Test OPSEC rules are properly loaded."""
        expected_categories = ['timing_rules', 'stealth_rules', 'target_rules', 'operational_rules']
        for category in expected_categories:
            assert category in agent.opsec_rules
    
    def test_risk_patterns_loaded(self, agent):
        """Test risk patterns are properly loaded."""
        expected_patterns = ['high_detection_commands', 'opsec_violations', 'infrastructure_risks']
        for pattern in expected_patterns:
            assert pattern in agent.risk_patterns
            assert isinstance(agent.risk_patterns[pattern], list)
    
    def test_validate_safe_commands(self, agent, safe_commands):
        """Test validation of safe commands."""
        assessment = agent.validate_commands(safe_commands, opsec_level='medium')
        
        assert isinstance(assessment, SafetyAssessment)
        assert assessment.overall_risk in [RiskLevel.LOW, RiskLevel.MEDIUM]
        assert len(assessment.checks) > 0
    
    def test_validate_risky_commands(self, agent, risky_commands):
        """Test validation of risky commands."""
        assessment = agent.validate_commands(risky_commands, opsec_level='high')
        
        assert isinstance(assessment, SafetyAssessment)
        assert assessment.overall_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert not assessment.approved  # Should not be approved for high OPSEC
    
    def test_government_domain_detection(self, agent, government_commands):
        """Test detection of government domains."""
        assessment = agent.validate_commands(government_commands, opsec_level='medium')
        
        # Should detect government domains as critical risk
        target_check = None
        for check in assessment.checks:
            if check.check_name == "Target Appropriateness":
                target_check = check
                break
        
        assert target_check is not None
        assert target_check.risk_level == RiskLevel.CRITICAL
        assert len(target_check.violations) > 0
    
    def test_detection_risk_analysis(self, agent):
        """Test detection risk analysis."""
        high_risk_commands = {
            'nmap': ['nmap -A -T4 --script vuln target.com']
        }
        
        assessment = agent.validate_commands(high_risk_commands, opsec_level='medium')
        
        detection_check = None
        for check in assessment.checks:
            if check.check_name == "Detection Risk Analysis":
                detection_check = check
                break
        
        assert detection_check is not None
        assert detection_check.risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH]
        assert len(detection_check.violations) > 0
    
    def test_opsec_compliance_check(self, agent):
        """Test OPSEC compliance checking."""
        # Commands that violate high OPSEC requirements
        commands = {
            'nmap': ['nmap -T4 target.com']  # Aggressive timing
        }
        
        assessment = agent.validate_commands(commands, opsec_level='high')
        
        opsec_check = None
        for check in assessment.checks:
            if check.check_name == "OPSEC Compliance":
                opsec_check = check
                break
        
        assert opsec_check is not None
        assert len(opsec_check.violations) > 0
    
    def test_timing_compliance_check(self, agent):
        """Test timing compliance checking."""
        fast_commands = {
            'nmap': ['nmap --min-rate 1000 target.com']
        }
        
        assessment = agent.validate_commands(fast_commands, opsec_level='maximum')
        
        timing_check = None
        for check in assessment.checks:
            if check.check_name == "Timing Compliance":
                timing_check = check
                break
        
        assert timing_check is not None
        assert timing_check.risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH]
    
    def test_infrastructure_safety_check(self, agent):
        """Test infrastructure safety checking."""
        api_commands = {
            'shodan': ['shodan search apache'],
            'api': ['curl https://api.shodan.io/shodan/host/1.1.1.1?key=API_KEY']
        }
        
        assessment = agent.validate_commands(api_commands, opsec_level='medium')
        
        infra_check = None
        for check in assessment.checks:
            if check.check_name == "Infrastructure Safety":
                infra_check = check
                break
        
        assert infra_check is not None
    
    def test_overall_risk_calculation(self, agent):
        """Test overall risk level calculation."""
        # Mix of different risk levels
        mixed_commands = {
            'safe': ['dig example.com'],
            'risky': ['nmap -A example.com']
        }
        
        assessment = agent.validate_commands(mixed_commands, opsec_level='medium')
        
        # Should calculate appropriate overall risk
        assert assessment.overall_risk in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
        
        # Check that it's reasonable based on individual checks
        max_individual_risk = max(check.risk_level for check in assessment.checks)
        assert assessment.overall_risk.value in [risk.value for risk in RiskLevel]
    
    def test_safe_alternatives_generation(self, agent, risky_commands):
        """Test generation of safe alternatives."""
        assessment = agent.validate_commands(risky_commands, opsec_level='high')
        
        if not assessment.approved:
            assert len(assessment.safe_alternatives) > 0
            
            # Check that alternatives are actually safer
            alternatives_text = ' '.join(assessment.safe_alternatives).lower()
            safe_keywords = ['passive', 'stealth', 'delay', 'fragment']
            
            assert any(keyword in alternatives_text for keyword in safe_keywords)
    
    def test_approval_logic(self, agent):
        """Test approval logic for different scenarios."""
        # Very safe commands should be approved
        safe_commands = {
            'passive': ['dig example.com', 'whois example.com']
        }
        
        assessment = agent.validate_commands(safe_commands, opsec_level='low')
        # Should likely be approved for low OPSEC requirements
        
        # Critical risk should never be approved
        critical_commands = {
            'nmap': ['nmap -sS whitehouse.gov']
        }
        
        critical_assessment = agent.validate_commands(critical_commands, opsec_level='low')
        assert not critical_assessment.approved
    
    def test_target_extraction(self, agent):
        """Test target extraction from commands."""
        commands = {
            'nmap': ['nmap -sS 192.168.1.1', 'nmap example.com'],
            'dig': ['dig google.com ANY']
        }
        
        targets = agent._extract_targets_from_commands(commands)
        
        assert '192.168.1.1' in targets
        assert 'example.com' in targets
        assert 'google.com' in targets
    
    def test_different_opsec_levels(self, agent, safe_commands):
        """Test behavior with different OPSEC levels."""
        opsec_levels = ['low', 'medium', 'high', 'maximum']
        
        for level in opsec_levels:
            assessment = agent.validate_commands(safe_commands, opsec_level=level)
            
            assert isinstance(assessment, SafetyAssessment)
            assert assessment.overall_risk in [risk for risk in RiskLevel]
            
            # Higher OPSEC levels should be more restrictive
            if level in ['high', 'maximum']:
                # Should have more stringent checks
                assert len(assessment.checks) > 0

if __name__ == '__main__':
    pytest.main([__file__])
