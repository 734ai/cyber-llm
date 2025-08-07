"""
Comprehensive Testing Suite for Cyber-LLM
Integration tests, performance tests, and validation framework

Author: Muzan Sano <sanosensei36@gmail.com>
"""

import asyncio
import pytest
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import aiohttp
import websockets
import docker
import subprocess
from pathlib import Path

from ..src.agents.orchestrator import CyberLLMOrchestrator
from ..src.agents.recon_agent import ReconAgent
from ..src.agents.safety_agent import SafetyAgent
from ..src.memory.persistent_memory import PersistentMemoryManager
from ..src.cognitive.meta_cognitive import MetaCognitiveEngine
from ..src.collaboration.multi_agent_framework import AgentCommunicationProtocol
from ..src.integration.universal_tool_framework import UniversalToolRegistry
from ..src.certification.enterprise_certification import EnterpriseCertificationManager

class TestSuiteRunner:
    """Comprehensive test suite execution and reporting"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        self.logger = logging.getLogger("test_suite")
        
        # Test configurations
        self.test_configs = {
            "unit_tests": {
                "timeout": 300,
                "parallel": True,
                "coverage_threshold": 80
            },
            "integration_tests": {
                "timeout": 900,
                "parallel": False,
                "requires_services": ["postgresql", "redis", "neo4j"]
            },
            "performance_tests": {
                "timeout": 1800,
                "load_levels": [10, 50, 100, 500],
                "metrics": ["response_time", "throughput", "resource_usage"]
            }
        }
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run complete test suite with all categories"""
        
        self.logger.info("Starting comprehensive test suite")
        start_time = time.time()
        
        # Run test categories in order
        test_categories = [
            ("unit_tests", self.run_unit_tests),
            ("integration_tests", self.run_integration_tests),
            ("performance_tests", self.run_performance_tests),
            ("security_tests", self.run_security_tests),
            ("compliance_tests", self.run_compliance_tests),
            ("agent_tests", self.run_agent_tests),
            ("memory_system_tests", self.run_memory_system_tests),
            ("cognitive_tests", self.run_cognitive_tests)
        ]
        
        overall_results = {
            "start_time": datetime.now().isoformat(),
            "test_results": {},
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "skipped_tests": 0
            }
        }
        
        for category_name, test_function in test_categories:
            try:
                self.logger.info(f"Running {category_name}")
                category_results = await test_function()
                overall_results["test_results"][category_name] = category_results
                
                # Update summary
                overall_results["summary"]["total_tests"] += category_results.get("total", 0)
                overall_results["summary"]["passed_tests"] += category_results.get("passed", 0)
                overall_results["summary"]["failed_tests"] += category_results.get("failed", 0)
                overall_results["summary"]["skipped_tests"] += category_results.get("skipped", 0)
                
            except Exception as e:
                self.logger.error(f"Failed to run {category_name}: {str(e)}")
                overall_results["test_results"][category_name] = {
                    "status": "error",
                    "error": str(e),
                    "total": 0,
                    "passed": 0,
                    "failed": 1
                }
        
        overall_results["duration_seconds"] = time.time() - start_time
        overall_results["end_time"] = datetime.now().isoformat()
        overall_results["success_rate"] = (
            overall_results["summary"]["passed_tests"] / 
            max(overall_results["summary"]["total_tests"], 1) * 100
        )
        
        # Generate test report
        await self._generate_test_report(overall_results)
        
        self.logger.info(f"Test suite completed in {overall_results['duration_seconds']:.2f}s")
        self.logger.info(f"Success rate: {overall_results['success_rate']:.1f}%")
        
        return overall_results

@pytest.mark.asyncio
class TestAgentOperations:
    """Test AI agent operations and interactions"""
    
    @pytest.fixture
    async def orchestrator(self):
        """Setup orchestrator for testing"""
        orchestrator = CyberLLMOrchestrator(
            config_path="tests/config/test_config.yaml"
        )
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.cleanup()
    
    @pytest.fixture
    async def recon_agent(self):
        """Setup reconnaissance agent for testing"""
        agent = ReconAgent(
            agent_id="test_recon_001",
            config={"test_mode": True}
        )
        await agent.initialize()
        yield agent
        await agent.cleanup()
    
    async def test_agent_initialization(self, orchestrator):
        """Test agent initialization and registration"""
        
        # Test orchestrator initialization
        assert orchestrator.is_initialized
        assert len(orchestrator.registered_agents) > 0
        
        # Test individual agent registration
        agent_ids = [agent.agent_id for agent in orchestrator.registered_agents]
        assert "recon_agent" in agent_ids
        assert "safety_agent" in agent_ids
        assert "explainability_agent" in agent_ids
    
    async def test_reconnaissance_workflow(self, recon_agent):
        """Test reconnaissance agent workflow"""
        
        # Test network discovery
        discovery_results = await recon_agent.discover_network({
            "target_range": "127.0.0.1/32",  # Localhost only for testing
            "scan_type": "quick",
            "stealth_mode": True
        })
        
        assert discovery_results["status"] == "success"
        assert "discovered_hosts" in discovery_results
        assert isinstance(discovery_results["discovered_hosts"], list)
        
        # Test port scanning
        port_scan_results = await recon_agent.scan_ports({
            "target": "127.0.0.1",
            "ports": "22,80,443",
            "scan_type": "syn"
        })
        
        assert port_scan_results["status"] == "success"
        assert "open_ports" in port_scan_results
    
    async def test_safety_agent_intervention(self, orchestrator):
        """Test safety agent intervention capabilities"""
        
        safety_agent = orchestrator.get_agent("safety_agent")
        
        # Test ethical violation detection
        violation_test = await safety_agent.assess_action({
            "action_type": "file_deletion",
            "target": "/etc/passwd",
            "context": "unauthorized_access",
            "authorization_level": "none"
        })
        
        assert violation_test["allowed"] is False
        assert violation_test["risk_level"] == "critical"
        assert "intervention_reason" in violation_test
    
    async def test_multi_agent_collaboration(self, orchestrator):
        """Test multi-agent collaboration and communication"""
        
        # Start collaborative assessment
        collaboration_task = await orchestrator.start_collaborative_assessment({
            "target": "test-target.local",
            "agents": ["recon_agent", "c2_agent"],
            "coordination_mode": "sequential",
            "information_sharing": True
        })
        
        assert collaboration_task["status"] == "started"
        assert len(collaboration_task["participating_agents"]) == 2
        
        # Wait for task completion
        await orchestrator.wait_for_task_completion(collaboration_task["task_id"])
        
        # Verify information sharing occurred
        shared_information = await orchestrator.get_shared_information(
            collaboration_task["task_id"]
        )
        assert len(shared_information) > 0

@pytest.mark.asyncio 
class TestMemorySystem:
    """Test persistent memory and strategic planning"""
    
    @pytest.fixture
    async def memory_manager(self):
        """Setup memory manager for testing"""
        manager = PersistentMemoryManager(
            db_path=":memory:",  # In-memory database for testing
            config={"test_mode": True}
        )
        await manager.initialize()
        yield manager
        await manager.cleanup()
    
    async def test_memory_storage_retrieval(self, memory_manager):
        """Test memory storage and retrieval operations"""
        
        # Store episodic memory
        episode = {
            "timestamp": datetime.now(),
            "event_type": "vulnerability_discovered",
            "details": {
                "cve_id": "CVE-2024-TEST",
                "severity": "high",
                "target": "test.example.com"
            }
        }
        
        memory_id = await memory_manager.store_episodic_memory(episode)
        assert memory_id is not None
        
        # Retrieve memory
        retrieved = await memory_manager.retrieve_memory(memory_id)
        assert retrieved is not None
        assert retrieved["event_type"] == "vulnerability_discovered"
    
    async def test_reasoning_chain_storage(self, memory_manager):
        """Test reasoning chain storage and retrieval"""
        
        reasoning_chain = {
            "chain_id": "test_reasoning_001",
            "steps": [
                "Identified open port 22 on target",
                "Attempted SSH connection",
                "Discovered weak authentication",
                "Recommended security hardening"
            ],
            "conclusion": "Target vulnerable to SSH attacks",
            "confidence": 0.85
        }
        
        stored = await memory_manager.store_reasoning_chain(
            chain_id=reasoning_chain["chain_id"],
            steps=reasoning_chain["steps"],
            conclusion=reasoning_chain["conclusion"],
            confidence=reasoning_chain["confidence"]
        )
        
        assert stored is True
        
        # Retrieve and verify
        retrieved = await memory_manager.get_reasoning_chain(
            reasoning_chain["chain_id"]
        )
        assert retrieved is not None
        assert len(retrieved["steps"]) == 4
        assert retrieved["confidence"] == 0.85

@pytest.mark.asyncio
class TestCognitiveCapabilities:
    """Test meta-cognitive and strategic planning capabilities"""
    
    @pytest.fixture
    async def cognitive_engine(self, memory_manager):
        """Setup cognitive engine for testing"""
        from ..src.memory.strategic_planning import StrategicPlanningEngine
        
        strategic_planner = StrategicPlanningEngine(memory_manager)
        await strategic_planner.initialize()
        
        engine = MetaCognitiveEngine(memory_manager, strategic_planner)
        yield engine
        await engine.cleanup()
    
    async def test_self_reflection(self, cognitive_engine):
        """Test self-reflection capabilities"""
        
        # Simulate performance data
        cognitive_engine.cognitive_metrics.extend([
            CognitiveMetrics(
                timestamp=datetime.now() - timedelta(minutes=i),
                task_completion_rate=0.85 + (i * 0.01),
                accuracy_score=0.92 - (i * 0.005),
                response_time=200 + (i * 10),
                resource_utilization=0.6 + (i * 0.02),
                attention_fragmentation=0.3,
                working_memory_usage=0.7,
                processing_complexity=0.5,
                learning_rate=0.001,
                confidence_level=0.8,
                adaptation_success_rate=0.75,
                error_count=i,
                critical_errors=0
            )
            for i in range(10)
        ])
        
        # Conduct self-reflection
        reflection = await cognitive_engine.conduct_self_reflection(
            time_period=timedelta(hours=1)
        )
        
        assert reflection is not None
        assert len(reflection.strengths) > 0
        assert len(reflection.immediate_adjustments) >= 0
        assert reflection.cognitive_patterns is not None
    
    async def test_learning_rate_optimization(self, cognitive_engine):
        """Test learning rate optimization"""
        
        # Test with improving performance
        performance_data = [0.7, 0.75, 0.8, 0.85, 0.9]
        new_lr = await cognitive_engine.optimize_learning_rate(
            recent_performance=performance_data,
            task_complexity=0.6
        )
        
        assert new_lr > 0
        assert new_lr <= 0.1  # Within reasonable bounds
        
        # Test with declining performance
        declining_performance = [0.9, 0.85, 0.8, 0.75, 0.7]
        declining_lr = await cognitive_engine.optimize_learning_rate(
            recent_performance=declining_performance,
            task_complexity=0.8
        )
        
        assert declining_lr > 0
        assert declining_lr < new_lr  # Should be lower for declining performance

@pytest.mark.asyncio
class TestIntegrationFramework:
    """Test universal tool integration framework"""
    
    @pytest.fixture
    async def tool_registry(self):
        """Setup tool registry for testing"""
        registry = UniversalToolRegistry()
        yield registry
    
    async def test_tool_registration(self, tool_registry):
        """Test tool registration and discovery"""
        
        from ..src.integration.universal_tool_framework import ToolMetadata, ToolType, IntegrationMethod
        
        # Create test tool metadata
        test_tool = ToolMetadata(
            tool_id="test_nmap",
            name="Network Mapper Test",
            version="7.94",
            vendor="Nmap Project",
            tool_type=ToolType.SCANNER,
            integration_method=IntegrationMethod.CLI_WRAPPER,
            capabilities=[],
            supported_formats=["xml", "json"],
            executable_path="/usr/bin/nmap",
            command_template="nmap {target} -oX -"
        )
        
        # Register tool
        registered = await tool_registry.register_tool(test_tool)
        assert registered is True
        
        # Verify registration
        retrieved_tool = tool_registry.registered_tools.get("test_nmap")
        assert retrieved_tool is not None
        assert retrieved_tool.name == "Network Mapper Test"
    
    async def test_tool_execution(self, tool_registry):
        """Test tool execution through framework"""
        
        # This would require actual tool execution in real environment
        # For testing, we'll mock the execution
        pass

@pytest.mark.asyncio
class TestComplianceFramework:
    """Test enterprise compliance and certification"""
    
    @pytest.fixture
    async def compliance_manager(self):
        """Setup compliance manager for testing"""
        from ..src.governance.enterprise_governance import EnterpriseGovernanceManager
        
        governance_manager = EnterpriseGovernanceManager()
        await governance_manager.initialize()
        
        manager = EnterpriseCertificationManager(governance_manager)
        yield manager
        await manager.cleanup()
    
    async def test_compliance_assessment(self, compliance_manager):
        """Test compliance assessment execution"""
        
        from ..src.certification.enterprise_certification import CertificationStandard
        
        # Run SOC2 compliance assessment
        assessment = await compliance_manager.conduct_comprehensive_compliance_assessment([
            CertificationStandard.SOC2_TYPE_II
        ])
        
        assert CertificationStandard.SOC2_TYPE_II.value in assessment
        soc2_result = assessment[CertificationStandard.SOC2_TYPE_II.value]
        assert soc2_result.score >= 0
        assert soc2_result.score <= 100
        assert soc2_result.requirements_met >= 0
    
    async def test_security_audit(self, compliance_manager):
        """Test comprehensive security audit"""
        
        audit_result = await compliance_manager.conduct_comprehensive_security_audit()
        
        assert audit_result is not None
        assert audit_result.overall_score >= 0
        assert audit_result.overall_score <= 100
        assert len(audit_result.immediate_actions) >= 0

class PerformanceTestSuite:
    """Performance and load testing suite"""
    
    def __init__(self):
        self.results = {}
        self.logger = logging.getLogger("performance_tests")
    
    async def run_load_test(self, endpoint: str, concurrent_users: int, duration_seconds: int) -> Dict[str, Any]:
        """Run load test against API endpoint"""
        
        self.logger.info(f"Starting load test: {concurrent_users} users for {duration_seconds}s")
        
        start_time = time.time()
        response_times = []
        error_count = 0
        success_count = 0
        
        async def make_request(session: aiohttp.ClientSession):
            try:
                async with session.get(endpoint) as response:
                    response_time = time.time() - request_start_time
                    response_times.append(response_time)
                    
                    if response.status == 200:
                        return "success"
                    else:
                        return "error"
            except Exception:
                return "error"
        
        # Run load test
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for i in range(concurrent_users):
                for j in range(duration_seconds):
                    request_start_time = time.time()
                    task = asyncio.create_task(make_request(session))
                    tasks.append(task)
                    await asyncio.sleep(1)  # 1 request per second per user
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if result == "success":
                    success_count += 1
                else:
                    error_count += 1
        
        total_time = time.time() - start_time
        
        return {
            "concurrent_users": concurrent_users,
            "duration_seconds": duration_seconds,
            "total_requests": len(results),
            "successful_requests": success_count,
            "failed_requests": error_count,
            "average_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "requests_per_second": len(results) / total_time,
            "error_rate": error_count / len(results) * 100 if results else 0
        }
    
    async def test_agent_performance(self) -> Dict[str, Any]:
        """Test AI agent performance under load"""
        
        results = {}
        
        # Test different load levels
        load_levels = [1, 5, 10, 20]
        
        for load_level in load_levels:
            self.logger.info(f"Testing agent performance with {load_level} concurrent requests")
            
            start_time = time.time()
            
            # Simulate concurrent agent requests
            async def simulate_agent_request():
                orchestrator = CyberLLMOrchestrator()
                await orchestrator.initialize()
                
                try:
                    result = await orchestrator.run_assessment({
                        "target": "test.example.com",
                        "agents": ["recon"],
                        "quick_mode": True
                    })
                    return "success"
                except Exception:
                    return "error"
                finally:
                    await orchestrator.cleanup()
            
            # Run concurrent requests
            tasks = [simulate_agent_request() for _ in range(load_level)]
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_time = time.time() - start_time
            success_count = sum(1 for r in task_results if r == "success")
            
            results[f"load_level_{load_level}"] = {
                "concurrent_requests": load_level,
                "successful_requests": success_count,
                "failed_requests": load_level - success_count,
                "total_time": total_time,
                "average_time_per_request": total_time / load_level,
                "success_rate": success_count / load_level * 100
            }
        
        return results

# Factory function for test suite
def create_test_suite() -> TestSuiteRunner:
    """Create comprehensive test suite runner"""
    return TestSuiteRunner()

# Main test execution
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests based on command line arguments
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        
        if test_type == "unit":
            # Run unit tests with pytest
            exit_code = pytest.main(["-v", "tests/unit/"])
            sys.exit(exit_code)
            
        elif test_type == "integration":
            # Run integration tests
            exit_code = pytest.main(["-v", "tests/integration/"])
            sys.exit(exit_code)
            
        elif test_type == "performance":
            # Run performance tests
            async def run_perf_tests():
                perf_suite = PerformanceTestSuite()
                results = await perf_suite.test_agent_performance()
                print(json.dumps(results, indent=2))
            
            asyncio.run(run_perf_tests())
            
        elif test_type == "all":
            # Run comprehensive test suite
            async def run_all_tests():
                suite = TestSuiteRunner()
                results = await suite.run_comprehensive_test_suite()
                print(json.dumps(results, indent=2))
            
            asyncio.run(run_all_tests())
    
    else:
        print("Usage: python test_suite.py [unit|integration|performance|all]")
        sys.exit(1)
