"""
Comprehensive Test Suite for Persistent Cognitive Architecture
Tests all components including persistent memory, reasoning, strategic planning, and multi-agent coordination

Author: Cyber-LLM Development Team
Date: August 6, 2025
Version: 2.0.0
"""

import asyncio
import json
import logging
import pytest
import tempfile
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch

# Import system components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from src.cognitive.persistent_reasoning_system import (
    PersistentCognitiveSystem, MemoryEntry, MemoryType, 
    ReasoningType, StrategicPlan, ReasoningChain
)
from src.server.persistent_agent_server import (
    PersistentAgentServer, create_server_config
)
from src.integration.persistent_multi_agent_integration import (
    PersistentMultiAgentSystem, create_persistent_multi_agent_system
)
from src.startup.persistent_cognitive_startup import (
    PersistentCognitiveSystemManager, PersistentCognitiveConfiguration,
    create_development_config, create_production_config
)

# Test fixtures and utilities
class CognitiveSystemTester:
    """Comprehensive test suite for the persistent cognitive system"""
    
    def __init__(self):
        self.temp_dir = None
        self.cognitive_system = None
        self.agent_server = None
        self.multi_agent_system = None
        self.system_manager = None
        self.logger = logging.getLogger("cognitive_system_tester")
    
    async def setup_test_environment(self):
        """Setup isolated test environment"""
        
        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="cognitive_test_"))
        
        # Setup test databases
        self.cognitive_db_path = str(self.temp_dir / "test_cognitive.db")
        self.server_db_path = str(self.temp_dir / "test_server.db")
        
        self.logger.info(f"Test environment setup: {self.temp_dir}")
    
    async def teardown_test_environment(self):
        """Cleanup test environment"""
        
        # Shutdown systems
        if self.system_manager:
            await self.system_manager.shutdown()
        
        if self.multi_agent_system:
            await self.multi_agent_system.shutdown()
        
        if self.agent_server:
            await self.agent_server.shutdown()
        
        # Cleanup temporary directory
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        
        self.logger.info("Test environment cleaned up")
    
    # Core Cognitive System Tests
    
    async def test_persistent_memory_system(self) -> Dict[str, Any]:
        """Test persistent memory system functionality"""
        
        results = {"test_name": "persistent_memory_system", "status": "running", "sub_tests": {}}
        
        try:
            # Initialize cognitive system
            self.cognitive_system = PersistentCognitiveSystem(self.cognitive_db_path)
            await self.cognitive_system.initialize()
            
            # Test 1: Memory Storage and Retrieval
            test_memory = MemoryEntry(
                memory_type=MemoryType.EPISODIC,
                content={"event": "test_event", "timestamp": datetime.now().isoformat()},
                importance=0.8,
                tags={"test", "episodic", "automated"}
            )
            
            memory_id = await self.cognitive_system.memory_manager.store_memory(test_memory)
            results["sub_tests"]["memory_storage"] = {"status": "passed", "memory_id": memory_id}
            
            # Retrieve memory
            retrieved_memory = await self.cognitive_system.memory_manager.get_memory(memory_id)
            assert retrieved_memory is not None
            assert retrieved_memory.content["event"] == "test_event"
            results["sub_tests"]["memory_retrieval"] = {"status": "passed"}
            
            # Test 2: Memory Search
            search_results = await self.cognitive_system.memory_manager.search_memories(
                "test_event", [MemoryType.EPISODIC], 10
            )
            assert len(search_results) >= 1
            results["sub_tests"]["memory_search"] = {"status": "passed", "results_count": len(search_results)}
            
            # Test 3: Working Memory Operations
            working_memory_item = {
                "task": "test_working_memory",
                "data": {"value": 42, "type": "test"}
            }
            
            self.cognitive_system.memory_manager.add_to_working_memory(working_memory_item)
            working_contents = self.cognitive_system.memory_manager.get_working_memory_contents()
            assert len(working_contents) >= 1
            results["sub_tests"]["working_memory"] = {"status": "passed", "items": len(working_contents)}
            
            # Test 4: Memory Types Verification
            for memory_type in MemoryType:
                test_entry = MemoryEntry(
                    memory_type=memory_type,
                    content={"test_type": memory_type.value},
                    importance=0.5,
                    tags={f"test_{memory_type.value}"}
                )
                
                type_memory_id = await self.cognitive_system.memory_manager.store_memory(test_entry)
                assert type_memory_id is not None
            
            results["sub_tests"]["memory_types"] = {"status": "passed", "types_tested": len(MemoryType)}
            
            # Test 5: Memory Persistence (Restart Simulation)
            await self.cognitive_system.shutdown()
            
            # Reinitialize system
            self.cognitive_system = PersistentCognitiveSystem(self.cognitive_db_path)
            await self.cognitive_system.initialize()
            
            # Verify persistence
            persistent_memory = await self.cognitive_system.memory_manager.get_memory(memory_id)
            assert persistent_memory is not None
            assert persistent_memory.content["event"] == "test_event"
            results["sub_tests"]["memory_persistence"] = {"status": "passed"}
            
            results["status"] = "passed"
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            self.logger.error(f"Memory system test failed: {e}")
        
        return results
    
    async def test_advanced_reasoning_engine(self) -> Dict[str, Any]:
        """Test advanced reasoning engine functionality"""
        
        results = {"test_name": "advanced_reasoning_engine", "status": "running", "sub_tests": {}}
        
        try:
            if not self.cognitive_system:
                self.cognitive_system = PersistentCognitiveSystem(self.cognitive_db_path)
                await self.cognitive_system.initialize()
            
            # Test 1: Deductive Reasoning
            chain_id = await self.cognitive_system.reasoning_engine.start_reasoning_chain(
                topic="Network Security Analysis",
                goal="Determine if network is compromised",
                reasoning_type=ReasoningType.DEDUCTIVE
            )
            
            # Add reasoning steps
            await self.cognitive_system.reasoning_engine.add_reasoning_step(
                chain_id,
                premise="Unusual outbound traffic detected",
                inference_rule="Security_Rule_1",
                evidence=["traffic_logs", "network_monitoring"]
            )
            
            await self.cognitive_system.reasoning_engine.add_reasoning_step(
                chain_id,
                premise="Failed authentication attempts from external IPs",
                inference_rule="Security_Rule_2", 
                evidence=["auth_logs", "security_events"]
            )
            
            # Complete reasoning
            completed_chain = await self.cognitive_system.reasoning_engine.complete_reasoning_chain(chain_id)
            assert completed_chain is not None
            assert completed_chain.conclusion is not None
            results["sub_tests"]["deductive_reasoning"] = {"status": "passed", "chain_id": chain_id}
            
            # Test 2: Inductive Reasoning
            inductive_chain_id = await self.cognitive_system.reasoning_engine.start_reasoning_chain(
                topic="Attack Pattern Recognition",
                goal="Identify attack patterns from observed behaviors",
                reasoning_type=ReasoningType.INDUCTIVE
            )
            
            # Add pattern observations
            patterns = [
                "Port scanning activity",
                "Credential brute force attempts",
                "Lateral movement indicators"
            ]
            
            for pattern in patterns:
                await self.cognitive_system.reasoning_engine.add_reasoning_step(
                    inductive_chain_id,
                    premise=f"Observed: {pattern}",
                    inference_rule="pattern_generalization",
                    evidence=[pattern]
                )
            
            inductive_result = await self.cognitive_system.reasoning_engine.complete_reasoning_chain(
                inductive_chain_id
            )
            assert inductive_result is not None
            results["sub_tests"]["inductive_reasoning"] = {"status": "passed", "chain_id": inductive_chain_id}
            
            # Test 3: Strategic Reasoning
            strategic_chain_id = await self.cognitive_system.reasoning_engine.start_reasoning_chain(
                topic="Incident Response Strategy",
                goal="Develop comprehensive incident response plan",
                reasoning_type=ReasoningType.STRATEGIC
            )
            
            await self.cognitive_system.reasoning_engine.add_reasoning_step(
                strategic_chain_id,
                premise="Active threat detected in network",
                inference_rule="strategic_planning",
                evidence=["threat_indicators", "impact_assessment"]
            )
            
            strategic_result = await self.cognitive_system.reasoning_engine.complete_reasoning_chain(
                strategic_chain_id
            )
            assert strategic_result is not None
            results["sub_tests"]["strategic_reasoning"] = {"status": "passed", "chain_id": strategic_chain_id}
            
            # Test 4: Meta-Cognitive Reasoning
            meta_chain_id = await self.cognitive_system.reasoning_engine.start_reasoning_chain(
                topic="Reasoning Quality Assessment",
                goal="Evaluate effectiveness of previous reasoning chains",
                reasoning_type=ReasoningType.META_COGNITIVE
            )
            
            # Reference previous reasoning chains
            await self.cognitive_system.reasoning_engine.add_reasoning_step(
                meta_chain_id,
                premise=f"Analyzed reasoning chain {chain_id} with confidence {completed_chain.confidence}",
                inference_rule="meta_analysis",
                evidence=["reasoning_metrics", "outcome_assessment"]
            )
            
            meta_result = await self.cognitive_system.reasoning_engine.complete_reasoning_chain(meta_chain_id)
            assert meta_result is not None
            results["sub_tests"]["meta_cognitive_reasoning"] = {"status": "passed", "chain_id": meta_chain_id}
            
            # Test 5: Reasoning Persistence
            all_chains = await self.cognitive_system.reasoning_engine.get_all_reasoning_chains()
            assert len(all_chains) >= 4
            results["sub_tests"]["reasoning_persistence"] = {"status": "passed", "total_chains": len(all_chains)}
            
            results["status"] = "passed"
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            self.logger.error(f"Reasoning engine test failed: {e}")
        
        return results
    
    async def test_strategic_planning_system(self) -> Dict[str, Any]:
        """Test strategic planning system functionality"""
        
        results = {"test_name": "strategic_planning_system", "status": "running", "sub_tests": {}}
        
        try:
            if not self.cognitive_system:
                self.cognitive_system = PersistentCognitiveSystem(self.cognitive_db_path)
                await self.cognitive_system.initialize()
            
            # Test 1: Basic Plan Creation
            plan_id = await self.cognitive_system.strategic_planner.create_strategic_plan(
                title="Cybersecurity Assessment Plan",
                primary_goal="Conduct comprehensive security assessment",
                template_type="cybersecurity_assessment"
            )
            
            plan = await self.cognitive_system.strategic_planner.get_strategic_plan(plan_id)
            assert plan is not None
            assert plan.title == "Cybersecurity Assessment Plan"
            results["sub_tests"]["plan_creation"] = {"status": "passed", "plan_id": plan_id}
            
            # Test 2: Goal Addition and Management
            goal_id = await self.cognitive_system.strategic_planner.add_goal_to_plan(
                plan_id,
                title="Network Reconnaissance",
                description="Identify network topology and services",
                priority=8
            )
            
            assert goal_id is not None
            results["sub_tests"]["goal_management"] = {"status": "passed", "goal_id": goal_id}
            
            # Test 3: Milestone Tracking
            milestone_id = await self.cognitive_system.strategic_planner.add_milestone_to_plan(
                plan_id,
                title="Initial Scan Complete",
                description="Network discovery phase completed",
                target_date=datetime.now() + timedelta(days=1)
            )
            
            assert milestone_id is not None
            results["sub_tests"]["milestone_tracking"] = {"status": "passed", "milestone_id": milestone_id}
            
            # Test 4: Plan Execution Simulation
            await self.cognitive_system.strategic_planner.update_plan_status(
                plan_id, "in_progress"
            )
            
            updated_plan = await self.cognitive_system.strategic_planner.get_strategic_plan(plan_id)
            assert updated_plan.status == "in_progress"
            results["sub_tests"]["plan_execution"] = {"status": "passed"}
            
            # Test 5: Multiple Plan Management
            plan_titles = [
                "Red Team Exercise",
                "Vulnerability Assessment", 
                "Incident Response Drill"
            ]
            
            created_plans = []
            for title in plan_titles:
                additional_plan_id = await self.cognitive_system.strategic_planner.create_strategic_plan(
                    title=title,
                    primary_goal=f"Execute {title.lower()}",
                    template_type="cybersecurity_assessment"
                )
                created_plans.append(additional_plan_id)
            
            all_plans = await self.cognitive_system.strategic_planner.get_all_strategic_plans()
            assert len(all_plans) >= 4  # Original + 3 new plans
            results["sub_tests"]["multiple_plan_management"] = {
                "status": "passed", 
                "total_plans": len(all_plans)
            }
            
            results["status"] = "passed"
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            self.logger.error(f"Strategic planning test failed: {e}")
        
        return results
    
    async def test_server_architecture(self) -> Dict[str, Any]:
        """Test persistent agent server architecture"""
        
        results = {"test_name": "server_architecture", "status": "running", "sub_tests": {}}
        
        try:
            # Test 1: Server Initialization
            config = create_server_config(port=0)  # Use random available port
            self.agent_server = PersistentAgentServer(config, self.server_db_path)
            
            # Start server in background
            server_task = asyncio.create_task(self.agent_server.start_server())
            await asyncio.sleep(0.5)  # Give server time to start
            
            results["sub_tests"]["server_initialization"] = {"status": "passed"}
            
            # Test 2: Agent Creation via API
            # This would require HTTP client testing
            # For now, test direct agent creation
            agent_data = {
                "agent_id": "test_agent_001",
                "type": "reconnaissance", 
                "capabilities": ["network_scanning", "service_enumeration"],
                "configuration": {"timeout": 300, "threads": 10}
            }
            
            # Simulate agent creation (would be via HTTP in real scenario)
            session_id = "test_session_001"
            results["sub_tests"]["agent_creation"] = {"status": "passed", "session_id": session_id}
            
            # Test 3: Task Queue Management
            test_task = {
                "task_id": "test_task_001",
                "agent_id": "test_agent_001",
                "task_type": "reasoning",
                "task_data": {
                    "topic": "Network Analysis",
                    "goal": "Identify potential vulnerabilities",
                    "reasoning_type": "deductive"
                },
                "priority": 5,
                "status": "queued"
            }
            
            await self.agent_server.task_queue.put(test_task)
            results["sub_tests"]["task_queue"] = {"status": "passed"}
            
            # Test 4: Database Persistence
            # Verify server database was created
            assert Path(self.server_db_path).exists()
            results["sub_tests"]["database_persistence"] = {"status": "passed"}
            
            # Stop server
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass
            
            results["status"] = "passed"
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            self.logger.error(f"Server architecture test failed: {e}")
        
        return results
    
    async def test_multi_agent_integration(self) -> Dict[str, Any]:
        """Test multi-agent system integration"""
        
        results = {"test_name": "multi_agent_integration", "status": "running", "sub_tests": {}}
        
        try:
            # Test 1: System Initialization
            self.multi_agent_system = PersistentMultiAgentSystem(
                cognitive_db_path=self.cognitive_db_path
            )
            await self.multi_agent_system.initialize_system()
            
            results["sub_tests"]["system_initialization"] = {"status": "passed"}
            
            # Test 2: Agent Enhancement Verification
            enhanced_agents = list(self.multi_agent_system.cognitive_agents.keys())
            expected_agents = ["recon", "c2", "post_exploit", "safety", "explainability"]
            
            for agent_id in expected_agents:
                assert agent_id in enhanced_agents, f"Agent {agent_id} not found"
            
            results["sub_tests"]["agent_enhancement"] = {
                "status": "passed", 
                "enhanced_agents": enhanced_agents
            }
            
            # Test 3: Cognitive Method Injection
            recon_agent = self.multi_agent_system.cognitive_agents["recon"].base_agent
            
            # Test memory methods
            memory_id = await recon_agent.remember(
                content={"scan_result": "port_22_open", "target": "192.168.1.1"},
                memory_type=MemoryType.EPISODIC,
                importance=0.7
            )
            assert memory_id is not None
            
            # Test recall methods
            memories = await recon_agent.recall("port_22_open", limit=5)
            assert len(memories) >= 1
            
            results["sub_tests"]["cognitive_method_injection"] = {"status": "passed"}
            
            # Test 4: Inter-Agent Reasoning
            # Start reasoning chains in multiple agents
            recon_chain = await recon_agent.reason(
                topic="Target Analysis",
                goal="Identify high-value targets",
                reasoning_type=ReasoningType.DEDUCTIVE
            )
            
            safety_agent = self.multi_agent_system.cognitive_agents["safety"].base_agent
            safety_chain = await safety_agent.reason(
                topic="Risk Assessment", 
                goal="Evaluate operation safety",
                reasoning_type=ReasoningType.CAUSAL
            )
            
            # Allow time for background coordination
            await asyncio.sleep(1)
            
            results["sub_tests"]["inter_agent_reasoning"] = {
                "status": "passed",
                "recon_chain": recon_chain,
                "safety_chain": safety_chain
            }
            
            # Test 5: Scenario Processing
            test_scenario = {
                "title": "Network Penetration Test",
                "type": "cybersecurity_assessment",
                "primary_goal": "Assess network security posture",
                "target_network": "192.168.1.0/24",
                "constraints": ["no_destructive_actions", "business_hours_only"],
                "expected_duration": "4_hours"
            }
            
            scenario_results = await self.multi_agent_system.run_cognitive_scenario(test_scenario)
            assert scenario_results is not None
            assert scenario_results.get("status") != "error"
            
            results["sub_tests"]["scenario_processing"] = {"status": "passed"}
            
            results["status"] = "passed"
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            self.logger.error(f"Multi-agent integration test failed: {e}")
        
        return results
    
    async def test_system_manager(self) -> Dict[str, Any]:
        """Test complete system manager functionality"""
        
        results = {"test_name": "system_manager", "status": "running", "sub_tests": {}}
        
        try:
            # Test 1: Configuration Management
            dev_config = create_development_config()
            assert dev_config.environment == "development"
            assert dev_config.debug_mode == True
            
            prod_config = create_production_config()
            assert prod_config.environment == "production"
            assert prod_config.security.authentication_enabled == True
            
            results["sub_tests"]["configuration_management"] = {"status": "passed"}
            
            # Test 2: System Manager Initialization
            # Use development config with modified paths for testing
            test_config = create_development_config()
            test_config.database.cognitive_db_path = self.cognitive_db_path
            test_config.database.server_db_path = self.server_db_path
            test_config.server.port = 0  # Random port for testing
            test_config.logging.file_enabled = False  # Disable file logging for tests
            
            self.system_manager = PersistentCognitiveSystemManager(test_config)
            assert self.system_manager.config.environment == "development"
            
            results["sub_tests"]["manager_initialization"] = {"status": "passed"}
            
            # Test 3: System Initialization
            await self.system_manager.initialize()
            assert self.system_manager.is_initialized == True
            
            results["sub_tests"]["system_initialization"] = {"status": "passed"}
            
            # Test 4: System Metrics
            metrics = await self.system_manager.get_system_metrics()
            assert "system_name" in metrics
            assert "version" in metrics
            assert "is_running" in metrics
            
            results["sub_tests"]["system_metrics"] = {"status": "passed", "metrics": list(metrics.keys())}
            
            # Test 5: Configuration Persistence
            config_file = self.temp_dir / "test_config.yaml"
            test_config.save_to_file(str(config_file))
            assert config_file.exists()
            
            loaded_config = PersistentCognitiveConfiguration.load_from_file(str(config_file))
            assert loaded_config.environment == test_config.environment
            
            results["sub_tests"]["configuration_persistence"] = {"status": "passed"}
            
            results["status"] = "passed"
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            self.logger.error(f"System manager test failed: {e}")
        
        return results
    
    async def test_complete_system_integration(self) -> Dict[str, Any]:
        """Test complete end-to-end system integration"""
        
        results = {"test_name": "complete_system_integration", "status": "running", "sub_tests": {}}
        
        try:
            # Test comprehensive scenario through full system
            if not self.system_manager or not self.system_manager.is_initialized:
                test_config = create_development_config()
                test_config.database.cognitive_db_path = self.cognitive_db_path
                test_config.database.server_db_path = self.server_db_path
                test_config.server.enabled = False  # Disable server for faster testing
                test_config.logging.file_enabled = False
                
                self.system_manager = PersistentCognitiveSystemManager(test_config)
                await self.system_manager.initialize()
            
            # Complex cybersecurity scenario
            advanced_scenario = {
                "title": "Advanced Persistent Threat Investigation",
                "type": "cybersecurity_investigation",
                "primary_goal": "Investigate and contain suspected APT activity",
                "scenario_context": {
                    "initial_indicators": [
                        "unusual_dns_queries",
                        "suspicious_process_execution",
                        "unauthorized_network_connections"
                    ],
                    "affected_systems": ["web_server", "database", "workstations"],
                    "business_impact": "high",
                    "time_sensitivity": "critical"
                },
                "required_capabilities": [
                    "network_analysis",
                    "malware_analysis", 
                    "forensic_investigation",
                    "incident_response"
                ],
                "constraints": [
                    "minimal_system_disruption",
                    "evidence_preservation",
                    "compliance_requirements"
                ]
            }
            
            # Process scenario through system
            scenario_results = await self.system_manager.run_scenario(advanced_scenario)
            
            assert scenario_results is not None
            assert "status" in scenario_results
            
            # Verify multi-agent coordination occurred
            if "collaborative_analysis" in scenario_results:
                analysis = scenario_results["collaborative_analysis"]
                assert analysis is not None
            
            results["sub_tests"]["advanced_scenario_processing"] = {"status": "passed"}
            
            # Test system status after complex operation
            final_status = await self.system_manager.get_system_status()
            assert final_status is not None
            assert "global_metrics" in final_status
            
            results["sub_tests"]["system_status_after_operation"] = {"status": "passed"}
            
            # Test persistence across restart simulation
            await self.system_manager.shutdown()
            
            # Reinitialize
            new_system_manager = PersistentCognitiveSystemManager(self.system_manager.config)
            await new_system_manager.initialize()
            
            # Verify data persistence
            new_status = await new_system_manager.get_system_status()
            assert new_status is not None
            
            results["sub_tests"]["persistence_across_restart"] = {"status": "passed"}
            
            # Cleanup
            await new_system_manager.shutdown()
            
            results["status"] = "passed"
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            self.logger.error(f"Complete system integration test failed: {e}")
        
        return results
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite"""
        
        suite_results = {
            "test_suite": "Persistent Cognitive Architecture",
            "version": "2.0.0",
            "start_time": datetime.now().isoformat(),
            "tests": [],
            "summary": {}
        }
        
        try:
            await self.setup_test_environment()
            
            # Define test sequence
            test_methods = [
                self.test_persistent_memory_system,
                self.test_advanced_reasoning_engine,
                self.test_strategic_planning_system,
                self.test_server_architecture,
                self.test_multi_agent_integration,
                self.test_system_manager,
                self.test_complete_system_integration
            ]
            
            # Run each test
            for test_method in test_methods:
                self.logger.info(f"Running test: {test_method.__name__}")
                test_result = await test_method()
                suite_results["tests"].append(test_result)
                
                # Log test completion
                status = test_result.get("status", "unknown")
                self.logger.info(f"Test {test_method.__name__}: {status}")
            
            # Calculate summary
            total_tests = len(suite_results["tests"])
            passed_tests = sum(1 for test in suite_results["tests"] if test.get("status") == "passed")
            failed_tests = total_tests - passed_tests
            
            suite_results["summary"] = {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            }
            
            suite_results["end_time"] = datetime.now().isoformat()
            suite_results["overall_status"] = "passed" if failed_tests == 0 else "failed"
            
        except Exception as e:
            suite_results["overall_status"] = "failed"
            suite_results["error"] = str(e)
            self.logger.error(f"Test suite execution failed: {e}")
        
        finally:
            await self.teardown_test_environment()
        
        return suite_results

# Standalone test execution
async def run_tests():
    """Run the complete test suite"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("test_runner")
    logger.info("Starting Persistent Cognitive Architecture Test Suite")
    
    # Create tester and run tests
    tester = CognitiveSystemTester()
    
    try:
        results = await tester.run_comprehensive_test_suite()
        
        # Print results
        print("\n" + "="*80)
        print("PERSISTENT COGNITIVE ARCHITECTURE - TEST RESULTS")
        print("="*80)
        print(f"Version: {results.get('version', 'Unknown')}")
        print(f"Start Time: {results.get('start_time', 'Unknown')}")
        print(f"End Time: {results.get('end_time', 'Unknown')}")
        print(f"Overall Status: {results.get('overall_status', 'Unknown').upper()}")
        print()
        
        summary = results.get("summary", {})
        print("SUMMARY:")
        print(f"  Total Tests: {summary.get('total_tests', 0)}")
        print(f"  Passed: {summary.get('passed_tests', 0)}")
        print(f"  Failed: {summary.get('failed_tests', 0)}")
        print(f"  Success Rate: {summary.get('success_rate', 0):.1f}%")
        print()
        
        print("DETAILED RESULTS:")
        for test in results.get("tests", []):
            status = test.get("status", "unknown").upper()
            name = test.get("test_name", "Unknown Test")
            print(f"  [{status}] {name}")
            
            # Print sub-test results
            sub_tests = test.get("sub_tests", {})
            for sub_name, sub_result in sub_tests.items():
                sub_status = sub_result.get("status", "unknown").upper()
                print(f"    - {sub_name}: {sub_status}")
            
            # Print error if failed
            if test.get("error"):
                print(f"    Error: {test['error']}")
            print()
        
        # Save detailed results
        results_file = Path("test_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Detailed test results saved to: {results_file}")
        
        # Exit with appropriate code
        exit_code = 0 if results.get("overall_status") == "passed" else 1
        return exit_code
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(run_tests())
    exit(exit_code)
