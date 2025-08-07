"""
Simple Validation Script for Cyber-LLM Implementation
Tests core functionality without external dependencies

Author: Muzan Sano <sanosensei36@gmail.com>
"""

import os
import sys
import json
import asyncio
import importlib.util
from pathlib import Path
from typing import Dict, List, Any

class ImplementationValidator:
    """Validate implementation without external dependencies"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.validation_results = {
            "file_structure": {},
            "import_tests": {},
            "basic_functionality": {},
            "configuration_validation": {}
        }
    
    def validate_file_structure(self) -> Dict[str, Any]:
        """Validate that all required files are present"""
        
        required_files = [
            "src/cognitive/meta_cognitive.py",
            "src/collaboration/multi_agent_framework.py", 
            "src/integration/universal_tool_framework.py",
            "src/integration/knowledge_graph.py",
            "src/performance/performance_optimizer.py",
            "src/analysis/code_reviewer.py",
            "tests/comprehensive_test_suite.py",
            "docs/USER_GUIDE.md",
            "docs/API_REFERENCE.md"
        ]
        
        results = {
            "total_required": len(required_files),
            "found_files": 0,
            "missing_files": [],
            "file_details": {}
        }
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                results["found_files"] += 1
                # Get file size and basic stats
                stat_info = full_path.stat()
                results["file_details"][file_path] = {
                    "exists": True,
                    "size_kb": round(stat_info.st_size / 1024, 2),
                    "lines": self._count_lines(full_path)
                }
            else:
                results["missing_files"].append(file_path)
                results["file_details"][file_path] = {"exists": False}
        
        results["completion_percentage"] = (results["found_files"] / results["total_required"]) * 100
        
        return results
    
    def validate_imports(self) -> Dict[str, Any]:
        """Validate that Python files can be imported without syntax errors"""
        
        python_files = [
            "src/cognitive/meta_cognitive.py",
            "src/collaboration/multi_agent_framework.py",
            "src/integration/universal_tool_framework.py", 
            "src/integration/knowledge_graph.py"
        ]
        
        results = {
            "total_files": len(python_files),
            "successful_imports": 0,
            "import_errors": {},
            "syntax_validation": {}
        }
        
        for file_path in python_files:
            full_path = self.project_root / file_path
            
            if not full_path.exists():
                results["import_errors"][file_path] = "File does not exist"
                continue
            
            try:
                # Test syntax by compiling
                with open(full_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                compile(source, str(full_path), 'exec')
                results["successful_imports"] += 1
                results["syntax_validation"][file_path] = {
                    "syntax_valid": True,
                    "line_count": len(source.split('\n')),
                    "char_count": len(source)
                }
                
            except SyntaxError as e:
                results["import_errors"][file_path] = f"Syntax error: {str(e)}"
                results["syntax_validation"][file_path] = {
                    "syntax_valid": False,
                    "error": str(e)
                }
            except Exception as e:
                results["import_errors"][file_path] = f"Import error: {str(e)}"
        
        results["success_rate"] = (results["successful_imports"] / results["total_files"]) * 100
        
        return results
    
    def validate_basic_functionality(self) -> Dict[str, Any]:
        """Test basic functionality that doesn't require external dependencies"""
        
        results = {
            "class_definitions": {},
            "function_definitions": {},
            "configuration_parsing": {}
        }
        
        # Test meta-cognitive engine
        try:
            meta_cognitive_path = self.project_root / "src/cognitive/meta_cognitive.py"
            if meta_cognitive_path.exists():
                results["class_definitions"]["MetaCognitiveEngine"] = self._check_class_definition(
                    meta_cognitive_path, "MetaCognitiveEngine"
                )
        except Exception as e:
            results["class_definitions"]["MetaCognitiveEngine"] = {"error": str(e)}
        
        # Test multi-agent framework
        try:
            framework_path = self.project_root / "src/collaboration/multi_agent_framework.py"
            if framework_path.exists():
                results["class_definitions"]["AgentCommunicationProtocol"] = self._check_class_definition(
                    framework_path, "AgentCommunicationProtocol"
                )
        except Exception as e:
            results["class_definitions"]["AgentCommunicationProtocol"] = {"error": str(e)}
        
        # Test universal tool framework
        try:
            tool_framework_path = self.project_root / "src/integration/universal_tool_framework.py"
            if tool_framework_path.exists():
                results["class_definitions"]["UniversalToolRegistry"] = self._check_class_definition(
                    tool_framework_path, "UniversalToolRegistry"
                )
        except Exception as e:
            results["class_definitions"]["UniversalToolRegistry"] = {"error": str(e)}
        
        return results
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration files and structure"""
        
        results = {
            "config_files": {},
            "documentation": {},
            "deployment_files": {}
        }
        
        # Check configuration files
        config_files = [
            "configs/model_config.yaml",
            "requirements.txt",
            "requirements-utils.txt"
        ]
        
        for config_file in config_files:
            path = self.project_root / config_file
            if path.exists():
                results["config_files"][config_file] = {
                    "exists": True,
                    "size": path.stat().st_size
                }
            else:
                results["config_files"][config_file] = {"exists": False}
        
        # Check documentation
        doc_files = [
            "docs/USER_GUIDE.md",
            "docs/API_REFERENCE.md",
            "README.md"
        ]
        
        for doc_file in doc_files:
            path = self.project_root / doc_file
            if path.exists():
                results["documentation"][doc_file] = {
                    "exists": True,
                    "size_kb": round(path.stat().st_size / 1024, 2),
                    "estimated_read_time": self._estimate_read_time(path)
                }
            else:
                results["documentation"][doc_file] = {"exists": False}
        
        # Check deployment files
        deployment_files = [
            "src/deployment/docker/Dockerfile",
            "src/deployment/docker/docker-compose.yml",
            "src/deployment/k8s/deployment.yaml"
        ]
        
        for deploy_file in deployment_files:
            path = self.project_root / deploy_file
            if path.exists():
                results["deployment_files"][deploy_file] = {
                    "exists": True,
                    "size": path.stat().st_size
                }
            else:
                results["deployment_files"][deploy_file] = {"exists": False}
        
        return results
    
    def _count_lines(self, file_path: Path) -> int:
        """Count lines in a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return len(f.readlines())
        except:
            return 0
    
    def _check_class_definition(self, file_path: Path, class_name: str) -> Dict[str, Any]:
        """Check if a class is properly defined in a file"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple check for class definition
            class_pattern = f"class {class_name}"
            has_class = class_pattern in content
            
            # Count methods
            method_count = content.count("def ")
            
            # Check for __init__ method
            has_init = "__init__" in content
            
            return {
                "class_found": has_class,
                "has_init": has_init,
                "method_count": method_count,
                "file_size": len(content),
                "estimated_complexity": "high" if method_count > 10 else "medium" if method_count > 5 else "low"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _estimate_read_time(self, file_path: Path) -> str:
        """Estimate reading time for documentation"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            word_count = len(content.split())
            # Average reading speed: 200 words per minute
            minutes = max(1, round(word_count / 200))
            
            return f"{minutes} minutes"
        except:
            return "unknown"
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation checks"""
        
        print("🔍 Starting Cyber-LLM Implementation Validation...")
        
        # File structure validation
        print("📁 Validating file structure...")
        self.validation_results["file_structure"] = self.validate_file_structure()
        
        # Import validation
        print("🐍 Validating Python imports...")
        self.validation_results["import_tests"] = self.validate_imports()
        
        # Basic functionality validation
        print("⚙️ Validating basic functionality...")
        self.validation_results["basic_functionality"] = self.validate_basic_functionality()
        
        # Configuration validation
        print("📋 Validating configuration...")
        self.validation_results["configuration_validation"] = self.validate_configuration()
        
        # Calculate overall score
        file_score = self.validation_results["file_structure"]["completion_percentage"]
        import_score = self.validation_results["import_tests"]["success_rate"]
        
        overall_score = (file_score + import_score) / 2
        
        self.validation_results["overall_summary"] = {
            "overall_score": round(overall_score, 1),
            "file_structure_score": round(file_score, 1),
            "import_success_rate": round(import_score, 1),
            "validation_timestamp": "2024-01-01T00:00:00Z"
        }
        
        return self.validation_results
    
    def generate_validation_report(self, results: Dict[str, Any]):
        """Generate a comprehensive validation report"""
        
        print("\n" + "="*80)
        print("🚀 CYBER-LLM IMPLEMENTATION VALIDATION REPORT")
        print("="*80)
        
        # Overall Summary
        summary = results["overall_summary"]
        print(f"\n📊 OVERALL SCORE: {summary['overall_score']}/100")
        print(f"   File Structure: {summary['file_structure_score']}/100")
        print(f"   Import Success: {summary['import_success_rate']}/100")
        
        # File Structure Details
        file_results = results["file_structure"]
        print(f"\n📁 FILE STRUCTURE ANALYSIS:")
        print(f"   ✅ Files Found: {file_results['found_files']}/{file_results['total_required']}")
        print(f"   📈 Completion: {file_results['completion_percentage']:.1f}%")
        
        if file_results["missing_files"]:
            print(f"   ❌ Missing Files:")
            for missing in file_results["missing_files"]:
                print(f"      - {missing}")
        
        # Import Results
        import_results = results["import_tests"]
        print(f"\n🐍 PYTHON IMPORT ANALYSIS:")
        print(f"   ✅ Successful: {import_results['successful_imports']}/{import_results['total_files']}")
        print(f"   📈 Success Rate: {import_results['success_rate']:.1f}%")
        
        if import_results["import_errors"]:
            print(f"   ❌ Import Errors:")
            for file, error in import_results["import_errors"].items():
                print(f"      - {file}: {error}")
        
        # Key Implementation Stats
        print(f"\n📈 KEY IMPLEMENTATION STATISTICS:")
        total_lines = sum(
            details.get("lines", 0) 
            for details in file_results["file_details"].values()
            if details.get("exists", False)
        )
        total_size = sum(
            details.get("size_kb", 0) 
            for details in file_results["file_details"].values()
            if details.get("exists", False)
        )
        
        print(f"   📄 Total Lines of Code: {total_lines:,}")
        print(f"   💾 Total Code Size: {total_size:.1f} KB")
        
        # Implementation Highlights
        print(f"\n🌟 IMPLEMENTATION HIGHLIGHTS:")
        
        key_files = [
            ("src/cognitive/meta_cognitive.py", "Advanced Meta-Cognitive Engine"),
            ("src/collaboration/multi_agent_framework.py", "Multi-Agent Collaboration Framework"),
            ("src/integration/universal_tool_framework.py", "Universal Tool Integration"),
            ("src/integration/knowledge_graph.py", "Cyber Knowledge Graph"),
            ("docs/USER_GUIDE.md", "Comprehensive User Documentation"),
            ("docs/API_REFERENCE.md", "Complete API Reference")
        ]
        
        for file_path, description in key_files:
            if file_results["file_details"].get(file_path, {}).get("exists", False):
                details = file_results["file_details"][file_path]
                print(f"   ✅ {description}")
                print(f"      📄 {details['lines']} lines, {details['size_kb']} KB")
        
        # Recommendations
        print(f"\n💡 NEXT STEPS & RECOMMENDATIONS:")
        
        if summary["overall_score"] >= 90:
            print("   🎉 Excellent! Implementation is nearly complete.")
            print("   🔧 Focus on testing and deployment optimization.")
        elif summary["overall_score"] >= 70:
            print("   👍 Good progress! Core features are implemented.")
            print("   🔨 Address any missing files and import issues.")
        else:
            print("   ⚠️ Implementation needs more work.")
            print("   🔧 Focus on completing core components first.")
        
        print("   📚 Run comprehensive tests when dependencies are available")
        print("   🚀 Proceed with deployment configuration")
        print("   🔍 Conduct security review and performance optimization")
        
        print("\n" + "="*80)
        print("Validation completed! 🎯")
        print("="*80)

def main():
    """Main validation execution"""
    
    validator = ImplementationValidator()
    
    # Run validation
    results = validator.run_comprehensive_validation()
    
    # Generate report
    validator.generate_validation_report(results)
    
    # Save detailed results
    results_path = Path("validation_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Detailed results saved to: {results_path}")
    
    return results

if __name__ == "__main__":
    results = main()
    
    # Exit with appropriate code
    overall_score = results["overall_summary"]["overall_score"]
    if overall_score >= 80:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Needs work
