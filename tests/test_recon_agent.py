"""
Unit tests for ReconAgent
"""

import pytest
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from agents.recon_agent import ReconAgent, ReconTarget, ReconResult

class TestReconAgent:
    """Test suite for ReconAgent."""
    
    @pytest.fixture
    def agent(self):
        """Create a ReconAgent instance for testing."""
        return ReconAgent()
    
    @pytest.fixture
    def sample_target(self):
        """Create a sample target for testing."""
        return ReconTarget(
            target="example.com",
            target_type="domain",
            constraints={"time_limit": "1h"},
            opsec_level="medium"
        )
    
    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent is not None
        assert hasattr(agent, 'config')
        assert hasattr(agent, 'opsec_profiles')
        assert 'medium' in agent.opsec_profiles
    
    def test_opsec_profiles_loaded(self, agent):
        """Test OPSEC profiles are properly loaded."""
        expected_levels = ['low', 'medium', 'high', 'maximum']
        for level in expected_levels:
            assert level in agent.opsec_profiles
            profile = agent.opsec_profiles[level]
            assert 'timing' in profile
            assert 'port_limit' in profile
            assert 'techniques' in profile
    
    def test_analyze_target_basic(self, agent, sample_target):
        """Test basic target analysis."""
        result = agent.analyze_target(sample_target)
        
        assert isinstance(result, ReconResult)
        assert result.target == sample_target.target
        assert isinstance(result.commands, dict)
        assert isinstance(result.opsec_notes, list)
        assert isinstance(result.next_steps, list)
    
    def test_nmap_command_generation(self, agent, sample_target):
        """Test Nmap command generation."""
        result = agent.analyze_target(sample_target)
        nmap_commands = result.commands.get('nmap', [])
        
        # Should generate some nmap commands for medium OPSEC
        assert len(nmap_commands) > 0
        
        # Commands should contain the target
        for cmd in nmap_commands:
            assert sample_target.target in cmd
    
    def test_high_opsec_restrictions(self, agent):
        """Test that high OPSEC level applies proper restrictions."""
        high_opsec_target = ReconTarget(
            target="sensitive.gov",
            target_type="domain",
            constraints={},
            opsec_level="high"
        )
        
        result = agent.analyze_target(high_opsec_target)
        nmap_commands = result.commands.get('nmap', [])
        
        # Should have fewer or more restricted commands
        for cmd in nmap_commands:
            # Should not use aggressive timing
            assert '-T4' not in cmd
            assert '-T5' not in cmd
    
    def test_maximum_opsec_passive_only(self, agent):
        """Test that maximum OPSEC uses passive techniques only."""
        max_opsec_target = ReconTarget(
            target="critical.mil",
            target_type="domain",
            constraints={},
            opsec_level="maximum"
        )
        
        result = agent.analyze_target(max_opsec_target)
        nmap_commands = result.commands.get('nmap', [])
        
        # Should have no active nmap commands for maximum stealth
        assert len(nmap_commands) == 0
    
    def test_passive_dns_commands(self, agent, sample_target):
        """Test passive DNS command generation."""
        result = agent.analyze_target(sample_target)
        passive_dns_commands = result.commands.get('passive_dns', [])
        
        # Should generate passive DNS commands for domain targets
        assert len(passive_dns_commands) > 0
        
        # Should include common DNS tools
        dns_tools = ['dig', 'whois']
        commands_text = ' '.join(passive_dns_commands)
        
        for tool in dns_tools:
            assert tool in commands_text
    
    def test_risk_assessment(self, agent, sample_target):
        """Test risk assessment functionality."""
        result = agent.analyze_target(sample_target)
        
        assert result.risk_assessment is not None
        assert isinstance(result.risk_assessment, str)
        assert len(result.risk_assessment) > 0
    
    def test_execute_reconnaissance_simulation(self, agent, sample_target):
        """Test reconnaissance execution in simulation mode."""
        result = agent.execute_reconnaissance(sample_target)
        
        assert isinstance(result, dict)
        assert 'target' in result
        assert 'execution_status' in result
        assert result['execution_status'] == 'SIMULATION_ONLY'
        assert 'plan' in result
    
    def test_opsec_notes_generation(self, agent):
        """Test OPSEC notes generation for different levels."""
        opsec_levels = ['low', 'medium', 'high', 'maximum']
        
        for level in opsec_levels:
            target = ReconTarget(
                target="test.com",
                target_type="domain",
                constraints={},
                opsec_level=level
            )
            
            result = agent.analyze_target(target)
            assert len(result.opsec_notes) > 0
            
            # Check that OPSEC notes are relevant to the level
            notes_text = ' '.join(result.opsec_notes).lower()
            assert level in notes_text or level.upper() in notes_text
    
    def test_target_type_handling(self, agent):
        """Test handling of different target types."""
        target_types = [
            ("192.168.1.1", "ip"),
            ("example.com", "domain"),
            ("192.168.1.0/24", "network"),
            ("Example Corp", "organization")
        ]
        
        for target_value, target_type in target_types:
            target = ReconTarget(
                target=target_value,
                target_type=target_type,
                constraints={},
                opsec_level="medium"
            )
            
            result = agent.analyze_target(target)
            assert result.target == target_value
            assert isinstance(result.commands, dict)

if __name__ == '__main__':
    pytest.main([__file__])
