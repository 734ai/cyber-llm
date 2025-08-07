#!/usr/bin/env python3
"""
Phase 9 Validation Script
Validates all components of the Advanced Cognitive Architecture
"""
import os
import sys
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class Phase9Validator:
    """Validates Phase 9 Advanced Cognitive Architecture"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.results = {
            'component_checks': {},
            'integration_checks': {},
            'performance_checks': {},
            'errors': [],
            'warnings': []
        }
    
    def log_result(self, component, test_name, passed, details=""):
        """Log validation result"""
        if component not in self.results['component_checks']:
            self.results['component_checks'][component] = {}
        
        self.results['component_checks'][component][test_name] = {
            'passed': passed,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {test_name}: {details}")
    
    def validate_imports(self):
        """Validate all Phase 9 imports"""
        print("🔍 Validating Phase 9 Component Imports...")
        
        components = [
            ('Long-term Memory', 'src.cognitive.long_term_memory', 'LongTermMemoryManager'),
            ('Episodic Memory', 'src.cognitive.episodic_memory', 'EpisodicMemorySystem'),
            ('Semantic Memory', 'src.cognitive.semantic_memory', 'SemanticMemoryNetwork'),
            ('Working Memory', 'src.cognitive.working_memory', 'WorkingMemoryManager'),
            ('Chain of Thought', 'src.cognitive.chain_of_thought', 'ChainOfThoughtReasoning'),
            ('Advanced Integration', 'src.cognitive.advanced_integration', 'AdvancedCognitiveSystem')
        ]
        
        all_passed = True
        
        for name, module_path, class_name in components:
            try:
                module = __import__(module_path, fromlist=[class_name])
                cls = getattr(module, class_name)
                self.log_result('imports', name, True, f"Successfully imported {class_name}")
            except ImportError as e:
                self.log_result('imports', name, False, f"Import failed: {e}")
                self.results['errors'].append(f"Cannot import {name}: {e}")
                all_passed = False
            except AttributeError as e:
                self.log_result('imports', name, False, f"Class not found: {e}")
                self.results['errors'].append(f"Class {class_name} not found in {name}: {e}")
                all_passed = False
        
        return all_passed
    
    def validate_long_term_memory(self):
        """Validate long-term memory system"""
        print("\n🧠 Validating Long-term Memory System...")
        
        try:
            from src.cognitive.long_term_memory import LongTermMemoryManager
            
            ltm = LongTermMemoryManager(f"{self.temp_dir}/ltm_validation.db")
            
            # Test basic storage
            memory_id = ltm.store_memory(
                content="Validation test memory",
                memory_type="test",
                importance=0.8,
                agent_id="validator"
            )
            
            self.log_result('long_term_memory', 'basic_storage', 
                          memory_id is not None, "Memory storage successful")
            
            # Test retrieval
            memories = ltm.retrieve_memories(agent_id="validator")
            self.log_result('long_term_memory', 'basic_retrieval',
                          len(memories) > 0, f"Retrieved {len(memories)} memories")
            
            # Test consolidation
            stats = ltm.consolidate_memories()
            self.log_result('long_term_memory', 'consolidation',
                          'memories_processed' in stats, "Memory consolidation works")
            
            return True
            
        except Exception as e:
            self.log_result('long_term_memory', 'system_test', False, str(e))
            self.results['errors'].append(f"Long-term memory validation failed: {e}")
            return False
    
    def validate_episodic_memory(self):
        """Validate episodic memory system"""
        print("\n📚 Validating Episodic Memory System...")
        
        try:
            from src.cognitive.episodic_memory import EpisodicMemorySystem
            
            em = EpisodicMemorySystem(f"{self.temp_dir}/em_validation.db")
            
            # Test episode lifecycle
            episode_id = em.start_episode(
                agent_id="validator",
                session_id="validation_session",
                episode_type="validation"
            )
            
            self.log_result('episodic_memory', 'episode_creation',
                          episode_id is not None, "Episode creation successful")
            
            # Test action recording
            em.record_action(episode_id, {"type": "validate", "target": "system"})
            em.record_observation(episode_id, {"type": "test", "result": "passed"})
            em.record_reward(episode_id, 0.9)
            
            # End episode
            em.end_episode(episode_id, success=True, outcome="Validation completed")
            
            self.log_result('episodic_memory', 'episode_lifecycle',
                          True, "Complete episode lifecycle works")
            
            # Test replay
            replay_result = em.replay_experience(episode_id)
            self.log_result('episodic_memory', 'experience_replay',
                          'episode' in replay_result, "Experience replay successful")
            
            return True
            
        except Exception as e:
            self.log_result('episodic_memory', 'system_test', False, str(e))
            self.results['errors'].append(f"Episodic memory validation failed: {e}")
            return False
    
    def validate_semantic_memory(self):
        """Validate semantic memory system"""
        print("\n🕸️ Validating Semantic Memory System...")
        
        try:
            from src.cognitive.semantic_memory import SemanticMemoryNetwork
            
            sm = SemanticMemoryNetwork(f"{self.temp_dir}/sm_validation.db")
            
            # Test concept management
            concept_id = sm.add_concept(
                name="Validation Test",
                concept_type="test_concept",
                description="Test concept for validation",
                confidence=0.9
            )
            
            self.log_result('semantic_memory', 'concept_creation',
                          concept_id is not None, "Concept creation successful")
            
            # Test concept search
            concepts = sm.find_concept(name="Validation")
            self.log_result('semantic_memory', 'concept_search',
                          len(concepts) > 0, f"Found {len(concepts)} matching concepts")
            
            # Test threat reasoning
            threat_result = sm.reason_about_threat(["test", "validation"])
            self.log_result('semantic_memory', 'threat_reasoning',
                          'threat_assessment' in threat_result, "Threat reasoning works")
            
            return True
            
        except Exception as e:
            self.log_result('semantic_memory', 'system_test', False, str(e))
            self.results['errors'].append(f"Semantic memory validation failed: {e}")
            return False
    
    def validate_working_memory(self):
        """Validate working memory system"""
        print("\n⚡ Validating Working Memory System...")
        
        try:
            from src.cognitive.working_memory import WorkingMemoryManager
            
            wm = WorkingMemoryManager(f"{self.temp_dir}/wm_validation.db")
            
            # Test item management
            item_id = wm.add_item(
                content="Validation test item",
                item_type="test",
                priority=0.8,
                source_agent="validator"
            )
            
            self.log_result('working_memory', 'item_creation',
                          item_id is not None, "Working memory item creation successful")
            
            # Test attention focusing
            focus_id = wm.focus_attention(
                "validation_focus",
                [item_id],
                attention_weight=0.9,
                agent_id="validator"
            )
            
            self.log_result('working_memory', 'attention_focus',
                          focus_id is not None, "Attention focusing successful")
            
            # Test focused item retrieval
            active_items = wm.get_active_items()
            self.log_result('working_memory', 'focus_retrieval',
                          len(active_items) > 0, f"Retrieved {len(active_items)} active items")
            
            return True
            
        except Exception as e:
            self.log_result('working_memory', 'system_test', False, str(e))
            self.results['errors'].append(f"Working memory validation failed: {e}")
            return False
    
    def validate_reasoning_engine(self):
        """Validate chain-of-thought reasoning"""
        print("\n🔗 Validating Chain-of-Thought Reasoning...")
        
        try:
            from src.cognitive.chain_of_thought import ChainOfThoughtReasoning, ReasoningType
            
            reasoner = ChainOfThoughtReasoning(f"{self.temp_dir}/reasoning_validation.db")
            
            # Test reasoning chain
            chain_id = reasoner.start_reasoning_chain(
                "Validation reasoning test",
                "validation",
                "validator"
            )
            
            self.log_result('reasoning_engine', 'chain_creation',
                          chain_id is not None, "Reasoning chain creation successful")
            
            # Add reasoning steps
            obs_id = reasoner.add_reasoning_step(
                chain_id, ReasoningType.DEDUCTIVE,
                "System validation in progress",
                "validation_rule",
                evidence=["test_run"]
            )
            
            hyp_id = reasoner.add_reasoning_step(
                chain_id, ReasoningType.INDUCTIVE,
                "All components should validate successfully",
                "generalization_rule",
                evidence=[obs_id] if obs_id else []
            )
            
            self.log_result('reasoning_engine', 'step_addition',
                          obs_id and hyp_id, "Reasoning step addition successful")
            
            # Complete chain
            result = reasoner.complete_reasoning_chain(
                chain_id, "Validation reasoning completed successfully"
            )
            
            self.log_result('reasoning_engine', 'chain_completion',
                          'chain_id' in result, "Reasoning chain completion successful")
            
            return True
            
        except Exception as e:
            self.log_result('reasoning_engine', 'system_test', False, str(e))
            self.results['errors'].append(f"Reasoning engine validation failed: {e}")
            return False
    
    async def validate_integration_system(self):
        """Validate integrated cognitive architecture"""
        print("\n🔄 Validating Integrated Cognitive Architecture...")
        
        try:
            from src.cognitive.advanced_integration import AdvancedCognitiveSystem
            
            architecture = AdvancedCognitiveSystem(self.temp_dir)
            
            # Test system startup
            await architecture.start_cognitive_system()
            self.log_result('integration', 'system_startup', True, "System startup successful")
            
            # Test statistics retrieval
            stats = architecture.get_system_statistics()
            self.log_result('integration', 'statistics_retrieval',
                          'component_stats' in stats, "System statistics available")
            
            # Test experience processing
            experience_data = {
                'id': 'validation_experience',
                'type': 'episode',
                'episode_type': 'validation',
                'complete': True,
                'success': True,
                'priority': 0.8,
                'importance': 0.7,
                'summary': 'Validation experience test'
            }
            
            result = await architecture.process_experience("validator", experience_data)
            self.log_result('integration', 'experience_processing',
                          'cognitive_updates' in result, "Experience processing successful")
            
            # Test integrated reasoning
            reasoning_result = await architecture.reason_about_scenario(
                "Validation scenario test",
                agent_id="validator",
                reasoning_type="validation"
            )
            
            self.log_result('integration', 'integrated_reasoning',
                          'final_conclusion' in reasoning_result, "Integrated reasoning successful")
            
            # Test agent profile generation
            profile = await architecture.get_agent_cognitive_profile("validator")
            self.log_result('integration', 'profile_generation',
                          'agent_id' in profile, "Agent profile generation successful")
            
            # Clean shutdown
            await architecture.shutdown()
            self.log_result('integration', 'system_shutdown', True, "System shutdown successful")
            
            return True
            
        except Exception as e:
            self.log_result('integration', 'system_test', False, str(e))
            self.results['errors'].append(f"Integration system validation failed: {e}")
            return False
    
    def validate_performance(self):
        """Validate system performance characteristics"""
        print("\n⚡ Validating System Performance...")
        
        try:
            import time
            from src.cognitive.long_term_memory import LongTermMemoryManager
            
            ltm = LongTermMemoryManager(f"{self.temp_dir}/perf_validation.db")
            
            # Performance test: Storage speed
            start_time = time.time()
            for i in range(100):
                ltm.store_memory(
                    content=f"Performance test memory {i}",
                    memory_type="performance",
                    importance=0.5,
                    agent_id="perf_validator"
                )
            storage_time = time.time() - start_time
            
            self.log_result('performance', 'storage_speed',
                          storage_time < 10.0,
                          f"Stored 100 memories in {storage_time:.2f}s")
            
            # Performance test: Retrieval speed
            start_time = time.time()
            memories = ltm.retrieve_memories(memory_type="performance", limit=100)
            retrieval_time = time.time() - start_time
            
            self.log_result('performance', 'retrieval_speed',
                          retrieval_time < 5.0 and len(memories) == 100,
                          f"Retrieved 100 memories in {retrieval_time:.2f}s")
            
            return True
            
        except Exception as e:
            self.log_result('performance', 'performance_test', False, str(e))
            self.results['errors'].append(f"Performance validation failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up temporary resources"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except:
            pass
    
    def generate_report(self):
        """Generate validation report"""
        print("\n" + "="*70)
        print("📋 PHASE 9 VALIDATION REPORT")
        print("="*70)
        
        total_tests = 0
        passed_tests = 0
        
        for component, tests in self.results['component_checks'].items():
            print(f"\n📁 {component.upper()}")
            for test_name, test_result in tests.items():
                total_tests += 1
                if test_result['passed']:
                    passed_tests += 1
                    status = "✅"
                else:
                    status = "❌"
                print(f"  {status} {test_name}: {test_result['details']}")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n📊 SUMMARY")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {total_tests - passed_tests}")
        print(f"  Success Rate: {success_rate:.1f}%")
        
        if self.results['errors']:
            print(f"\n❌ ERRORS ({len(self.results['errors'])})")
            for error in self.results['errors']:
                print(f"  • {error}")
        
        if self.results['warnings']:
            print(f"\n⚠️  WARNINGS ({len(self.results['warnings'])})")
            for warning in self.results['warnings']:
                print(f"  • {warning}")
        
        print("\n" + "="*70)
        
        if success_rate >= 90:
            print("🎉 PHASE 9 VALIDATION SUCCESSFUL!")
            print("✅ Advanced Cognitive Architecture is ready for deployment")
        elif success_rate >= 70:
            print("⚠️  PHASE 9 VALIDATION PARTIALLY SUCCESSFUL")
            print("🔧 Some issues found - review and fix before deployment")
        else:
            print("❌ PHASE 9 VALIDATION FAILED")
            print("🚨 Critical issues found - requires attention before deployment")
        
        return success_rate >= 90
    
    async def run_full_validation(self):
        """Run complete Phase 9 validation"""
        print("🚀 Starting Phase 9: Advanced Cognitive Architecture Validation")
        print("="*70)
        
        validation_steps = [
            ("Import Validation", self.validate_imports),
            ("Long-term Memory", self.validate_long_term_memory),
            ("Episodic Memory", self.validate_episodic_memory), 
            ("Semantic Memory", self.validate_semantic_memory),
            ("Working Memory", self.validate_working_memory),
            ("Reasoning Engine", self.validate_reasoning_engine),
            ("Performance Testing", self.validate_performance),
            ("Integration System", self.validate_integration_system)
        ]
        
        all_passed = True
        
        for step_name, validation_func in validation_steps:
            print(f"\n🔍 Running {step_name}...")
            try:
                if asyncio.iscoroutinefunction(validation_func):
                    result = await validation_func()
                else:
                    result = validation_func()
                
                if not result:
                    all_passed = False
                    print(f"❌ {step_name} validation failed")
                else:
                    print(f"✅ {step_name} validation passed")
                    
            except Exception as e:
                all_passed = False
                print(f"❌ {step_name} validation error: {e}")
                self.results['errors'].append(f"{step_name}: {e}")
        
        # Generate final report
        success = self.generate_report()
        
        # Cleanup
        self.cleanup()
        
        return success

async def main():
    """Main validation function"""
    validator = Phase9Validator()
    
    try:
        success = await validator.run_full_validation()
        return success
    except KeyboardInterrupt:
        print("\n⚠️ Validation interrupted by user")
        validator.cleanup()
        return False
    except Exception as e:
        print(f"\n💥 Validation failed with error: {e}")
        validator.cleanup()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
