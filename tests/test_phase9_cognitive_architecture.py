"""
Comprehensive Test Suite for Phase 9: Advanced Cognitive Architecture
Tests all cognitive components and their integration
"""
import asyncio
import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import json

# Import all Phase 9 components
from src.cognitive.long_term_memory import LongTermMemoryManager, MemoryRecord
from src.cognitive.episodic_memory import EpisodicMemorySystem, Episode
from src.cognitive.semantic_memory import SemanticMemoryNetwork, SemanticConcept
from src.cognitive.working_memory import WorkingMemoryManager, WorkingMemoryItem
from src.cognitive.chain_of_thought import ChainOfThoughtReasoner, ReasoningStep
from src.cognitive.advanced_integration import AdvancedCognitiveSystem

class TestPhase9CognitiveComponents:
    """Test suite for all Phase 9 cognitive components"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test databases"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def long_term_memory(self, temp_dir):
        """Initialize long-term memory for testing"""
        return LongTermMemoryManager(f"{temp_dir}/ltm_test.db")
    
    @pytest.fixture
    def episodic_memory(self, temp_dir):
        """Initialize episodic memory for testing"""
        return EpisodicMemorySystem(f"{temp_dir}/em_test.db")
    
    @pytest.fixture
    def semantic_memory(self, temp_dir):
        """Initialize semantic memory for testing"""
        return SemanticMemoryNetwork(f"{temp_dir}/sm_test.db")
    
    @pytest.fixture
    def working_memory(self, temp_dir):
        """Initialize working memory for testing"""
        return WorkingMemoryManager(f"{temp_dir}/wm_test.db")
    
    @pytest.fixture
    def reasoning_engine(self, temp_dir):
        """Initialize reasoning engine for testing"""
        return ChainOfThoughtReasoner(f"{temp_dir}/reasoning_test.db")
    
    @pytest.fixture
    def cognitive_architecture(self, temp_dir):
        """Initialize integrated cognitive architecture for testing"""
        return AdvancedCognitiveSystem(temp_dir)
    
    # Long-Term Memory Tests
    def test_long_term_memory_storage_and_retrieval(self, long_term_memory):
        """Test basic storage and retrieval in long-term memory"""
        # Store a memory
        memory_id = long_term_memory.store_memory(
            content="Test security incident detected",
            memory_type="incident",
            importance=0.8,
            agent_id="test_agent",
            tags=["security", "incident"]
        )
        
        assert memory_id, "Memory should be stored successfully"
        
        # Retrieve memories
        memories = long_term_memory.retrieve_memories(
            query="security incident",
            agent_id="test_agent"
        )
        
        assert len(memories) == 1, "Should retrieve one memory"
        assert memories[0].content == "Test security incident detected"
        assert memories[0].importance == 0.8
    
    def test_long_term_memory_consolidation(self, long_term_memory):
        """Test memory consolidation process"""
        # Store multiple related memories
        for i in range(5):
            long_term_memory.store_memory(
                content=f"Similar security event {i}",
                memory_type="incident",
                importance=0.6,
                agent_id="test_agent"
            )
        
        # Run consolidation
        stats = long_term_memory.consolidate_memories()
        
        assert 'memories_processed' in stats
        assert stats['memories_processed'] >= 5
        assert 'patterns_discovered' in stats
    
    def test_long_term_memory_cross_session_context(self, long_term_memory):
        """Test cross-session context retrieval"""
        # Store memories from different sessions
        long_term_memory.store_memory(
            content="Session 1 activity",
            memory_type="activity",
            importance=0.7,
            agent_id="test_agent",
            session_id="session_1"
        )
        
        long_term_memory.store_memory(
            content="Session 2 activity", 
            memory_type="activity",
            importance=0.8,
            agent_id="test_agent",
            session_id="session_2"
        )
        
        # Get cross-session context
        context = long_term_memory.get_cross_session_context("test_agent", limit=10)
        
        assert len(context) == 2, "Should retrieve memories from both sessions"
        
        # Should be ordered by importance
        assert context[0].importance >= context[1].importance
    
    # Episodic Memory Tests
    def test_episodic_memory_episode_lifecycle(self, episodic_memory):
        """Test complete episode lifecycle"""
        # Start episode
        episode_id = episodic_memory.start_episode(
            agent_id="test_agent",
            session_id="test_session",
            episode_type="reconnaissance",
            context={"target": "test_system"}
        )
        
        assert episode_id, "Episode should start successfully"
        
        # Record actions
        episodic_memory.record_action(episode_id, {
            "type": "scan",
            "target": "192.168.1.1",
            "result": "open_ports_found"
        })
        
        episodic_memory.record_action(episode_id, {
            "type": "analyze",
            "target": "port_22",
            "result": "ssh_service_identified"
        })
        
        # Record observations
        episodic_memory.record_observation(episode_id, {
            "type": "vulnerability",
            "description": "Outdated SSH version",
            "severity": "medium"
        })
        
        # Record rewards
        episodic_memory.record_reward(episode_id, 0.7)
        episodic_memory.record_reward(episode_id, 0.8)
        
        # End episode
        episodic_memory.end_episode(
            episode_id,
            success=True,
            outcome="Vulnerability identified successfully"
        )
        
        # Verify episode completion
        episodes = episodic_memory.get_episodes_for_replay(
            agent_id="test_agent",
            limit=1
        )
        
        assert len(episodes) == 1
        episode = episodes[0]
        assert episode.success == True
        assert len(episode.actions) == 2
        assert len(episode.observations) == 1
        assert len(episode.rewards) == 2
    
    def test_episodic_memory_experience_replay(self, episodic_memory):
        """Test experience replay functionality"""
        # Create and complete an episode
        episode_id = episodic_memory.start_episode(
            agent_id="test_agent",
            session_id="test_session", 
            episode_type="training"
        )
        
        episodic_memory.record_action(episode_id, {"type": "test_action"})
        episodic_memory.record_reward(episode_id, 0.9)
        episodic_memory.end_episode(episode_id, success=True)
        
        # Replay the episode
        replay_result = episodic_memory.replay_experience(episode_id)
        
        assert 'episode' in replay_result
        assert 'insights' in replay_result
        assert len(replay_result['insights']) > 0
    
    def test_episodic_memory_pattern_discovery(self, episodic_memory):
        """Test pattern discovery across episodes"""
        # Create multiple episodes with similar patterns
        for i in range(3):
            episode_id = episodic_memory.start_episode(
                agent_id="test_agent",
                session_id=f"session_{i}",
                episode_type="pattern_test"
            )
            
            # Similar action sequence
            episodic_memory.record_action(episode_id, {"type": "scan"})
            episodic_memory.record_action(episode_id, {"type": "analyze"})
            episodic_memory.record_action(episode_id, {"type": "exploit"})
            
            episodic_memory.end_episode(episode_id, success=True)
        
        # Discover patterns
        patterns = episodic_memory.discover_patterns()
        
        assert 'action_patterns' in patterns
        assert 'success_patterns' in patterns
        assert len(patterns['action_patterns']) > 0
    
    # Semantic Memory Tests
    def test_semantic_memory_concept_management(self, semantic_memory):
        """Test semantic concept creation and relationships"""
        # Add concepts
        vuln_id = semantic_memory.add_concept(
            name="SQL Injection",
            concept_type="vulnerability",
            description="Code injection attack targeting SQL databases",
            confidence=0.9
        )
        
        technique_id = semantic_memory.add_concept(
            name="Input Validation Bypass",
            concept_type="technique", 
            description="Method to bypass input validation controls",
            confidence=0.8
        )
        
        mitigation_id = semantic_memory.add_concept(
            name="Parameterized Queries",
            concept_type="mitigation",
            description="SQL defense using parameterized statements",
            confidence=0.95
        )
        
        assert vuln_id and technique_id and mitigation_id
        
        # Add relationships
        relation1 = semantic_memory.add_relation(
            vuln_id, technique_id, "uses", strength=0.8
        )
        
        relation2 = semantic_memory.add_relation(
            mitigation_id, vuln_id, "mitigates", strength=0.9
        )
        
        assert relation1 and relation2
        
        # Test concept search
        concepts = semantic_memory.find_concept(name="SQL", concept_type="vulnerability")
        assert len(concepts) == 1
        assert concepts[0].name == "SQL Injection"
    
    def test_semantic_memory_threat_reasoning(self, semantic_memory):
        """Test threat reasoning capabilities"""
        # Add threat-related concepts
        semantic_memory.add_concept(
            name="malware", 
            concept_type="malware",
            confidence=0.9
        )
        
        semantic_memory.add_concept(
            name="phishing",
            concept_type="technique", 
            confidence=0.8
        )
        
        # Test threat reasoning
        threat_indicators = ["malware", "suspicious_email", "phishing"]
        reasoning_result = semantic_memory.reason_about_threat(threat_indicators)
        
        assert 'threat_assessment' in reasoning_result
        assert 'confidence' in reasoning_result
        assert reasoning_result['confidence'] > 0.0
        assert reasoning_result['threat_assessment']['risk_level'] in ['LOW', 'MEDIUM', 'HIGH']
    
    # Working Memory Tests
    def test_working_memory_attention_management(self, working_memory):
        """Test attention-based working memory management"""
        # Add items with different priorities
        high_priority_id = working_memory.add_item(
            content="Critical security alert",
            item_type="alert",
            priority=0.9,
            source_agent="test_agent"
        )
        
        low_priority_id = working_memory.add_item(
            content="Routine log entry",
            item_type="log",
            priority=0.2,
            source_agent="test_agent"
        )
        
        assert high_priority_id and low_priority_id
        
        # Test attention focusing
        focus_id = working_memory.focus_attention(
            "critical_alert",
            [high_priority_id],
            attention_weight=0.9,
            agent_id="test_agent"
        )
        
        assert focus_id
        
        # Get focused items
        focused_items = working_memory.get_focused_items()
        assert len(focused_items) > 0
        assert any(item.id == high_priority_id for item in focused_items)
    
    def test_working_memory_context_switching(self, working_memory):
        """Test context switching functionality"""
        # Create initial focus
        item1_id = working_memory.add_item(
            content="Initial focus item",
            item_type="focus",
            priority=0.8,
            source_agent="test_agent"
        )
        
        working_memory.focus_attention("initial_focus", [item1_id])
        
        # Create new items for context switch
        item2_id = working_memory.add_item(
            content="New focus item",
            item_type="urgent",
            priority=0.9,
            source_agent="test_agent"
        )
        
        # Switch context
        switch_result = working_memory.switch_context(
            "urgent_response",
            [item2_id],
            switch_reason="High priority alert",
            agent_id="test_agent"
        )
        
        assert switch_result['success']
        assert 'switch_cost' in switch_result
        assert switch_result['switch_cost'] >= 0.0
    
    def test_working_memory_decay_and_eviction(self, working_memory):
        """Test memory decay and capacity management"""
        # Fill working memory near capacity
        item_ids = []
        for i in range(working_memory.capacity - 2):
            item_id = working_memory.add_item(
                content=f"Test item {i}",
                item_type="test",
                priority=0.3,
                source_agent="test_agent"
            )
            item_ids.append(item_id)
        
        # Add high-priority item
        high_priority_id = working_memory.add_item(
            content="Important item",
            item_type="important",
            priority=0.9,
            source_agent="test_agent"
        )
        
        # Trigger decay manually
        working_memory.decay_memory()
        
        # High priority item should still be accessible
        high_priority_item = working_memory.get_item(high_priority_id)
        assert high_priority_item is not None
        assert high_priority_item.activation_level > 0.1
    
    # Chain-of-Thought Reasoning Tests
    def test_reasoning_chain_basic_functionality(self, reasoning_engine):
        """Test basic reasoning chain functionality"""
        # Start reasoning chain
        chain_id = reasoning_engine.start_reasoning_chain(
            problem_statement="Analyze suspicious network activity",
            chain_type="threat_analysis",
            agent_id="test_agent"
        )
        
        assert chain_id, "Reasoning chain should start successfully"
        
        # Add reasoning steps
        obs_id = reasoning_engine.add_reasoning_step(
            chain_id, ReasoningStep.OBSERVATION,
            "Unusual traffic patterns detected on port 443",
            confidence=0.8
        )
        
        hyp_id = reasoning_engine.add_reasoning_step(
            chain_id, ReasoningStep.HYPOTHESIS,
            "Possible data exfiltration attempt",
            premises=[obs_id],
            confidence=0.6
        )
        
        inf_id = reasoning_engine.add_reasoning_step(
            chain_id, ReasoningStep.INFERENCE,
            "Traffic analysis suggests encrypted file transfer",
            premises=[hyp_id],
            confidence=0.7
        )
        
        assert obs_id and hyp_id and inf_id
        
        # Complete reasoning chain
        completion_result = reasoning_engine.complete_reasoning_chain(
            chain_id, 
            "Potential data exfiltration detected - requires immediate investigation"
        )
        
        assert completion_result['chain_id'] == chain_id
        assert 'confidence' in completion_result
        assert completion_result['step_count'] == 3
    
    def test_reasoning_threat_analysis(self, reasoning_engine):
        """Test integrated threat analysis reasoning"""
        threat_description = "Multiple failed login attempts from external IP, followed by successful login and unusual file access patterns"
        
        analysis_result = reasoning_engine.analyze_threat_scenario(
            threat_description,
            context={"system": "file_server", "time_window": "30_minutes"},
            agent_id="test_agent"
        )
        
        assert 'chain_id' in analysis_result
        assert 'final_conclusion' in analysis_result
        assert 'threat_analysis' in analysis_result
        assert 'indicators' in analysis_result['threat_analysis']
        assert 'severity' in analysis_result['threat_analysis']
        assert 'countermeasures' in analysis_result['threat_analysis']
    
    def test_reasoning_step_by_step_problem_solving(self, reasoning_engine):
        """Test step-by-step problem solving"""
        problem = "Network performance has degraded significantly, users report slow response times"
        
        solution_result = reasoning_engine.solve_problem_step_by_step(
            problem,
            domain="cybersecurity", 
            agent_id="test_agent"
        )
        
        assert 'chain_id' in solution_result
        assert 'final_conclusion' in solution_result
        assert 'solution' in solution_result
        assert 'sub_problems' in solution_result['solution']
        assert 'overall_solution' in solution_result['solution']
    
    # Integration Tests
    @pytest.mark.asyncio
    async def test_cognitive_architecture_initialization(self, cognitive_architecture):
        """Test cognitive architecture initialization"""
        await cognitive_architecture.start_cognitive_system()
        
        # Verify all components are initialized
        assert cognitive_architecture.long_term_memory is not None
        assert cognitive_architecture.episodic_memory is not None  
        assert cognitive_architecture.semantic_memory is not None
        assert cognitive_architecture.working_memory is not None
        assert cognitive_architecture.reasoning_engine is not None
        
        # Get system statistics
        stats = cognitive_architecture.get_system_statistics()
        assert 'component_stats' in stats
        assert 'integration_status' in stats
        
        await cognitive_architecture.shutdown()
    
    @pytest.mark.asyncio
    async def test_integrated_experience_processing(self, cognitive_architecture):
        """Test integrated experience processing across all systems"""
        await cognitive_architecture.start_cognitive_system()
        
        # Process a complex experience
        experience_data = {
            'id': 'test_experience_001',
            'type': 'episode',
            'episode_type': 'security_analysis',
            'session_id': 'test_session',
            'priority': 0.8,
            'importance': 0.7,
            'complete': True,
            'success': True,
            'outcome': 'Vulnerability identified and mitigated',
            'summary': 'Comprehensive security analysis revealed and addressed critical vulnerability',
            'context': {
                'target_system': 'web_server',
                'tools_used': ['nmap', 'burp_suite'],
                'duration_minutes': 45
            },
            'actions': [
                {'type': 'scan', 'description': 'Network port scan'},
                {'type': 'analyze', 'description': 'Web application analysis'},
                {'type': 'exploit', 'description': 'Vulnerability exploitation test'}
            ],
            'observations': [
                {'type': 'finding', 'description': 'SQL injection vulnerability found'}
            ],
            'concepts': [
                {'name': 'SQL Injection', 'type': 'vulnerability', 'confidence': 0.9},
                {'name': 'Web Security', 'type': 'domain', 'confidence': 0.8}
            ],
            'tags': ['security', 'vulnerability', 'web_app']
        }
        
        # Process the experience
        processing_result = await cognitive_architecture.process_experience(
            agent_id="test_agent",
            experience_data=experience_data
        )
        
        # Verify processing across all systems
        assert 'cognitive_updates' in processing_result
        updates = processing_result['cognitive_updates']
        
        # Should have updates in multiple memory systems
        assert 'episodic_memory' in updates
        assert 'semantic_memory' in updates  
        assert 'working_memory' in updates
        assert 'long_term_memory' in updates
        
        await cognitive_architecture.shutdown()
    
    @pytest.mark.asyncio
    async def test_integrated_reasoning_with_memory_context(self, cognitive_architecture):
        """Test integrated reasoning using memory context from all systems"""
        await cognitive_architecture.start_cognitive_system()
        
        # First, populate memory systems with relevant context
        experience_data = {
            'id': 'context_experience',
            'type': 'episode',
            'episode_type': 'threat_detection',
            'complete': True,
            'success': True,
            'priority': 0.7,
            'importance': 0.8,
            'summary': 'Previous threat detection experience',
            'actions': [{'type': 'detect', 'description': 'Anomaly detection'}],
            'concepts': [
                {'name': 'Network Anomaly', 'type': 'indicator', 'confidence': 0.8}
            ]
        }
        
        await cognitive_architecture.process_experience("test_agent", experience_data)
        
        # Now perform reasoning that should leverage this context
        scenario = "Unusual network traffic patterns detected on internal network"
        
        reasoning_result = await cognitive_architecture.reason_about_scenario(
            scenario,
            agent_id="test_agent",
            reasoning_type="threat_analysis"
        )
        
        # Verify integrated reasoning result
        assert 'final_conclusion' in reasoning_result
        assert 'confidence' in reasoning_result
        assert 'integrated_context' in reasoning_result
        
        context = reasoning_result['integrated_context']
        assert 'working_memory_context' in context
        assert 'long_term_memory_context' in context
        assert 'semantic_knowledge' in context
        
        await cognitive_architecture.shutdown()
    
    @pytest.mark.asyncio
    async def test_experience_replay_learning(self, cognitive_architecture):
        """Test learning from experience replay"""
        await cognitive_architecture.start_cognitive_system()
        
        # Create multiple learning experiences
        for i in range(3):
            experience_data = {
                'id': f'learning_experience_{i}',
                'type': 'episode',
                'episode_type': 'learning_scenario',
                'complete': True,
                'success': i % 2 == 0,  # Alternate success/failure
                'priority': 0.6,
                'importance': 0.5,
                'summary': f'Learning experience {i}',
                'actions': [
                    {'type': 'learn', 'description': f'Learning action {i}'}
                ]
            }
            
            await cognitive_architecture.process_experience("test_agent", experience_data)
        
        # Perform experience replay learning
        learning_result = await cognitive_architecture.learn_from_experience_replay(
            agent_id="test_agent",
            max_episodes=5
        )
        
        assert 'episodes_replayed' in learning_result
        assert 'insights_discovered' in learning_result
        assert learning_result['episodes_replayed'] > 0
        assert len(learning_result['insights_discovered']) > 0
        
        await cognitive_architecture.shutdown()
    
    @pytest.mark.asyncio
    async def test_cognitive_profile_generation(self, cognitive_architecture):
        """Test comprehensive cognitive profile generation"""
        await cognitive_architecture.start_cognitive_system()
        
        # Create varied agent activity
        activities = [
            {
                'id': 'profile_activity_1',
                'type': 'episode', 
                'episode_type': 'analysis',
                'complete': True,
                'success': True,
                'priority': 0.8,
                'importance': 0.7
            },
            {
                'id': 'profile_activity_2',
                'type': 'episode',
                'episode_type': 'investigation', 
                'complete': True,
                'success': False,
                'priority': 0.6,
                'importance': 0.5
            }
        ]
        
        for activity in activities:
            await cognitive_architecture.process_experience("profile_agent", activity)
        
        # Generate cognitive profile
        profile = await cognitive_architecture.get_agent_cognitive_profile("profile_agent")
        
        assert 'agent_id' in profile
        assert profile['agent_id'] == "profile_agent"
        assert 'memory_systems' in profile
        assert 'reasoning_performance' in profile
        assert 'cognitive_patterns' in profile
        
        # Verify memory system profiles
        memory_systems = profile['memory_systems']
        assert 'long_term' in memory_systems
        assert 'episodic' in memory_systems
        assert 'working' in memory_systems
        
        await cognitive_architecture.shutdown()
    
    @pytest.mark.asyncio
    async def test_system_performance_monitoring(self, cognitive_architecture):
        """Test system performance monitoring and optimization"""
        await cognitive_architecture.start_cognitive_system()
        
        # Let background processes run briefly
        await asyncio.sleep(2)
        
        # Get initial statistics
        initial_stats = cognitive_architecture.get_system_statistics()
        assert 'cognitive_state' in initial_stats
        assert 'component_stats' in initial_stats
        assert 'integration_status' in initial_stats
        
        # Add some load to test performance monitoring
        for i in range(10):
            await cognitive_architecture.process_experience(
                "load_test_agent",
                {
                    'id': f'load_test_{i}',
                    'type': 'episode',
                    'complete': True,
                    'priority': 0.5
                }
            )
        
        # Get updated statistics
        updated_stats = cognitive_architecture.get_system_statistics()
        
        # Verify system is tracking changes
        assert updated_stats['component_stats']['long_term_memory']['total_memories'] >= \
               initial_stats['component_stats']['long_term_memory']['total_memories']
        
        await cognitive_architecture.shutdown()

    # Performance and Stress Tests
    def test_long_term_memory_performance(self, long_term_memory):
        """Test long-term memory performance with large dataset"""
        import time
        
        # Store many memories
        start_time = time.time()
        memory_ids = []
        
        for i in range(100):
            memory_id = long_term_memory.store_memory(
                content=f"Performance test memory {i}",
                memory_type="performance_test",
                importance=0.5,
                agent_id="perf_agent"
            )
            memory_ids.append(memory_id)
        
        storage_time = time.time() - start_time
        
        # Retrieve memories
        start_time = time.time()
        memories = long_term_memory.retrieve_memories(
            memory_type="performance_test",
            limit=100
        )
        retrieval_time = time.time() - start_time
        
        assert len(memories) == 100
        assert storage_time < 10.0  # Should store 100 memories in under 10 seconds
        assert retrieval_time < 5.0   # Should retrieve 100 memories in under 5 seconds
        
        print(f"Storage time: {storage_time:.3f}s, Retrieval time: {retrieval_time:.3f}s")
    
    def test_working_memory_capacity_management(self, working_memory):
        """Test working memory capacity management under load"""
        # Fill beyond capacity
        item_ids = []
        for i in range(working_memory.capacity + 20):
            item_id = working_memory.add_item(
                content=f"Capacity test item {i}",
                item_type="capacity_test",
                priority=0.1 + (i % 10) * 0.1,  # Varying priorities
                source_agent="capacity_agent"
            )
            if item_id:
                item_ids.append(item_id)
        
        # Should not exceed capacity
        active_items = working_memory.get_active_items(limit=200)
        assert len(active_items) <= working_memory.capacity
        
        # Higher priority items should be retained
        high_priority_items = [item for item in active_items if item.priority > 0.7]
        assert len(high_priority_items) > 0
    
    def test_reasoning_chain_complexity(self, reasoning_engine):
        """Test reasoning with complex multi-step chains"""
        chain_id = reasoning_engine.start_reasoning_chain(
            "Complex multi-stage cyber attack analysis",
            "advanced_threat_analysis",
            "complexity_agent"
        )
        
        # Create a complex reasoning chain with multiple branches
        step_ids = []
        
        # Initial observation
        obs_id = reasoning_engine.add_reasoning_step(
            chain_id, ReasoningStep.OBSERVATION,
            "Multiple attack vectors detected simultaneously",
            confidence=0.9
        )
        step_ids.append(obs_id)
        
        # Multiple hypotheses
        hyp1_id = reasoning_engine.add_reasoning_step(
            chain_id, ReasoningStep.HYPOTHESIS,
            "Coordinated APT campaign",
            premises=[obs_id],
            confidence=0.7
        )
        
        hyp2_id = reasoning_engine.add_reasoning_step(
            chain_id, ReasoningStep.HYPOTHESIS,
            "Automated attack tool deployment",
            premises=[obs_id],
            confidence=0.6
        )
        
        # Cross-referencing inferences
        inf1_id = reasoning_engine.add_reasoning_step(
            chain_id, ReasoningStep.INFERENCE,
            "Attack timing suggests human coordination",
            premises=[hyp1_id, hyp2_id],
            confidence=0.8
        )
        
        inf2_id = reasoning_engine.add_reasoning_step(
            chain_id, ReasoningStep.INFERENCE,
            "Tool signatures match known APT group",
            premises=[hyp1_id],
            confidence=0.75
        )
        
        # Final conclusion
        reasoning_engine.add_reasoning_step(
            chain_id, ReasoningStep.CONCLUSION,
            "High-confidence APT attack requiring immediate response",
            premises=[inf1_id, inf2_id],
            confidence=0.85
        )
        
        # Complete complex chain
        result = reasoning_engine.complete_reasoning_chain(
            chain_id,
            "Advanced persistent threat confirmed - initiate incident response protocol"
        )
        
        assert result['step_count'] == 6
        assert result['confidence'] > 0.7
        assert 'validation' in result
    
    # Error Handling and Edge Cases
    def test_error_handling_invalid_operations(self, long_term_memory, working_memory):
        """Test error handling for invalid operations"""
        # Test invalid memory retrieval
        invalid_memories = long_term_memory.retrieve_memories(
            agent_id="nonexistent_agent",
            importance_threshold=2.0  # Invalid threshold > 1.0
        )
        assert isinstance(invalid_memories, list)  # Should return empty list, not crash
        
        # Test working memory with invalid item ID
        invalid_item = working_memory.get_item("nonexistent_id")
        assert invalid_item is None
        
        # Test focus on nonexistent items
        focus_result = working_memory.focus_attention(
            "invalid_focus",
            ["nonexistent_id_1", "nonexistent_id_2"],
            agent_id="test_agent"
        )
        # Should handle gracefully without crashing
        assert focus_result == "" or isinstance(focus_result, str)
    
    def test_concurrent_access_safety(self, temp_dir):
        """Test thread safety of memory systems"""
        import threading
        import time
        
        ltm = LongTermMemoryManager(f"{temp_dir}/concurrent_test.db")
        results = []
        errors = []
        
        def worker_thread(worker_id):
            try:
                for i in range(10):
                    memory_id = ltm.store_memory(
                        content=f"Worker {worker_id} memory {i}",
                        memory_type="concurrent_test",
                        importance=0.5,
                        agent_id=f"worker_{worker_id}"
                    )
                    results.append(memory_id)
                    time.sleep(0.01)  # Small delay
            except Exception as e:
                errors.append(str(e))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 50, f"Expected 50 results, got {len(results)}"
        assert len(set(results)) == 50, "All memory IDs should be unique"

def run_comprehensive_test_suite():
    """Run the complete Phase 9 test suite"""
    print("🧠 Starting Phase 9: Advanced Cognitive Architecture Test Suite")
    print("=" * 70)
    
    # Run pytest with verbose output
    import pytest
    import sys
    
    test_args = [
        __file__,
        "-v",  # Verbose output
        "-s",  # Show print statements
        "--tb=short",  # Shorter traceback format
        "-x",  # Stop on first failure
    ]
    
    # Add performance markers if available
    try:
        import pytest_benchmark
        test_args.extend(["--benchmark-only", "--benchmark-sort=mean"])
    except ImportError:
        print("Note: pytest-benchmark not available, skipping performance benchmarks")
    
    result = pytest.main(test_args)
    
    print("\n" + "=" * 70)
    if result == 0:
        print("✅ All Phase 9 tests passed successfully!")
        print("🚀 Advanced Cognitive Architecture is ready for deployment")
    else:
        print("❌ Some tests failed. Please review and fix issues before deployment.")
        print("📋 Check test output above for detailed failure information.")
    
    return result == 0

if __name__ == "__main__":
    success = run_comprehensive_test_suite()
    exit(0 if success else 1)
