#!/usr/bin/env python3
"""
Final Quality Assurance script for LLMBuilder CLI integration.

This script performs comprehensive checks to ensure all components
are properly integrated and working together.
"""

import sys
import subprocess
import importlib
from pathlib import Path
from typing import List, Dict, Any
import json

def run_command(cmd: List[str]) -> Dict[str, Any]:
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Command timed out",
            "returncode": -1
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1
        }

def check_imports() -> bool:
    """Check that all core modules can be imported."""
    print("🔍 Checking module imports...")
    
    modules_to_check = [
        "llmbuilder",
        "llmbuilder.cli.main",
        "llmbuilder.cli.pipeline",
        "llmbuilder.utils.workflow",
        "llmbuilder.utils.config",
        "llmbuilder.core.data",
        "llmbuilder.core.training",
        "llmbuilder.core.model",
    ]
    
    failed_imports = []
    
    for module in modules_to_check:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import {len(failed_imports)} modules")
        return False
    
    print("✅ All modules imported successfully")
    return True

def check_cli_commands() -> bool:
    """Check that CLI commands are working."""
    print("\n🔍 Checking CLI commands...")
    
    commands_to_check = [
        ["python", "-m", "llmbuilder", "--help"],
        ["python", "-m", "llmbuilder", "--version"],
        ["python", "-m", "llmbuilder", "config", "--help"],
        ["python", "-m", "llmbuilder", "data", "--help"],
        ["python", "-m", "llmbuilder", "model", "--help"],
        ["python", "-m", "llmbuilder", "train", "--help"],
        ["python", "-m", "llmbuilder", "pipeline", "--help"],
    ]
    
    failed_commands = []
    
    for cmd in commands_to_check:
        result = run_command(cmd)
        cmd_str = " ".join(cmd[2:])  # Skip "python -m"
        
        if result["success"]:
            print(f"  ✅ {cmd_str}")
        else:
            print(f"  ❌ {cmd_str}: {result['stderr']}")
            failed_commands.append(cmd_str)
    
    if failed_commands:
        print(f"\n❌ {len(failed_commands)} CLI commands failed")
        return False
    
    print("✅ All CLI commands working")
    return True

def check_package_structure() -> bool:
    """Check that package structure is correct."""
    print("\n🔍 Checking package structure...")
    
    required_files = [
        "llmbuilder/__init__.py",
        "llmbuilder/cli/__init__.py",
        "llmbuilder/cli/main.py",
        "llmbuilder/cli/pipeline.py",
        "llmbuilder/utils/__init__.py",
        "llmbuilder/utils/workflow.py",
        "llmbuilder/utils/config.py",
        "llmbuilder/core/__init__.py",
        "pyproject.toml",
        "README.md",
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ {len(missing_files)} required files missing")
        return False
    
    print("✅ Package structure is correct")
    return True

def check_configuration() -> bool:
    """Check configuration system."""
    print("\n🔍 Checking configuration system...")
    
    try:
        from llmbuilder.utils.config import ConfigManager
        
        config_manager = ConfigManager()
        default_config = config_manager.get_default_config()
        
        required_sections = ["model", "training", "data", "deployment"]
        missing_sections = []
        
        for section in required_sections:
            if section in default_config:
                print(f"  ✅ Config section: {section}")
            else:
                print(f"  ❌ Config section: {section}")
                missing_sections.append(section)
        
        if missing_sections:
            print(f"\n❌ {len(missing_sections)} config sections missing")
            return False
        
        print("✅ Configuration system working")
        return True
        
    except Exception as e:
        print(f"❌ Configuration system error: {e}")
        return False

def check_workflow_system() -> bool:
    """Check workflow system."""
    print("\n🔍 Checking workflow system...")
    
    try:
        from llmbuilder.utils.workflow import WorkflowManager, PipelineBuilder
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            workflow_manager = WorkflowManager(Path(temp_dir))
            
            # Test workflow creation
            steps = [{"command": "test", "args": {"param": "value"}}]
            workflow_id = workflow_manager.create_workflow("test", steps)
            
            if workflow_id:
                print("  ✅ Workflow creation")
            else:
                print("  ❌ Workflow creation")
                return False
            
            # Test workflow loading
            workflow_data = workflow_manager.load_workflow(workflow_id)
            if workflow_data and workflow_data["name"] == "test":
                print("  ✅ Workflow loading")
            else:
                print("  ❌ Workflow loading")
                return False
            
            # Test pipeline builder
            pipeline_steps = PipelineBuilder.full_training_pipeline(
                data_path="./data",
                model_name="test-model",
                output_dir="./output"
            )
            
            if pipeline_steps and len(pipeline_steps) > 0:
                print("  ✅ Pipeline builder")
            else:
                print("  ❌ Pipeline builder")
                return False
        
        print("✅ Workflow system working")
        return True
        
    except Exception as e:
        print(f"❌ Workflow system error: {e}")
        return False

def check_documentation() -> bool:
    """Check documentation files."""
    print("\n🔍 Checking documentation...")
    
    doc_files = [
        "docs/README.md",
        "examples/README.md",
        "README.md",
        "USAGE.md",
        "CONTRIBUTING.md",
    ]
    
    missing_docs = []
    
    for doc_file in doc_files:
        if Path(doc_file).exists():
            print(f"  ✅ {doc_file}")
        else:
            print(f"  ❌ {doc_file}")
            missing_docs.append(doc_file)
    
    if missing_docs:
        print(f"\n⚠️  {len(missing_docs)} documentation files missing")
        # Don't fail for missing docs, just warn
    
    print("✅ Documentation check complete")
    return True

def check_tests() -> bool:
    """Check that tests can run."""
    print("\n🔍 Checking tests...")
    
    # Run a simple test to verify test infrastructure
    result = run_command([
        "python", "-m", "pytest", 
        "tests/integration/test_cli_integration.py", 
        "-v", "--tb=short"
    ])
    
    if result["success"]:
        print("  ✅ Integration tests pass")
        return True
    else:
        print(f"  ❌ Integration tests failed: {result['stderr']}")
        return False

def generate_qa_report(results: Dict[str, bool]) -> None:
    """Generate a QA report."""
    print("\n" + "="*60)
    print("📊 FINAL QUALITY ASSURANCE REPORT")
    print("="*60)
    
    total_checks = len(results)
    passed_checks = sum(results.values())
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:<30} {status}")
    
    print("-"*60)
    print(f"Total Checks: {total_checks}")
    print(f"Passed: {passed_checks}")
    print(f"Failed: {total_checks - passed_checks}")
    print(f"Success Rate: {(passed_checks/total_checks)*100:.1f}%")
    
    if passed_checks == total_checks:
        print("\n🎉 ALL CHECKS PASSED! LLMBuilder is ready for use.")
        return True
    else:
        print(f"\n⚠️  {total_checks - passed_checks} checks failed. Please review and fix issues.")
        return False

def main():
    """Run all quality assurance checks."""
    print("🚀 Starting LLMBuilder Final Quality Assurance")
    print("="*60)
    
    checks = {
        "Module Imports": check_imports,
        "CLI Commands": check_cli_commands,
        "Package Structure": check_package_structure,
        "Configuration System": check_configuration,
        "Workflow System": check_workflow_system,
        "Documentation": check_documentation,
        "Tests": check_tests,
    }
    
    results = {}
    
    for check_name, check_func in checks.items():
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ {check_name} failed with exception: {e}")
            results[check_name] = False
    
    success = generate_qa_report(results)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()