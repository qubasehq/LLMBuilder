#!/usr/bin/env python3
"""
Validate the comprehensive test suite setup.

This script checks that all test components are properly configured
and can be discovered by pytest.
"""

import sys
import subprocess
from pathlib import Path
import importlib.util


def check_pytest_installation():
    """Check if pytest is properly installed."""
    try:
        import pytest
        print(f"✅ pytest installed: {pytest.__version__}")
        return True
    except ImportError:
        print("❌ pytest not installed")
        return False


def check_test_discovery():
    """Check if pytest can discover all tests."""
    project_root = Path(__file__).parent.parent
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest',
            '--collect-only', '-q'
        ], cwd=project_root, capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            test_count = 0
            for line in lines:
                if '<Function' in line or '<Method' in line:
                    test_count += 1
            
            print(f"✅ Test discovery successful: {test_count} tests found")
            return True
        else:
            print(f"❌ Test discovery failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error during test discovery: {e}")
        return False


def check_test_structure():
    """Check if test directory structure is correct."""
    project_root = Path(__file__).parent.parent
    tests_dir = project_root / 'tests'
    
    required_dirs = [
        'unit',
        'integration', 
        'performance',
        'fixtures'
    ]
    
    required_files = [
        'conftest.py',
        'run_comprehensive_tests.py',
        'validate_test_suite.py'
    ]
    
    all_good = True
    
    # Check directories
    for dir_name in required_dirs:
        dir_path = tests_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"✅ Directory exists: tests/{dir_name}/")
            
            # Check for __init__.py
            init_file = dir_path / '__init__.py'
            if init_file.exists():
                print(f"✅ Init file exists: tests/{dir_name}/__init__.py")
            else:
                print(f"⚠️  Init file missing: tests/{dir_name}/__init__.py")
        else:
            print(f"❌ Directory missing: tests/{dir_name}/")
            all_good = False
    
    # Check files
    for file_name in required_files:
        file_path = tests_dir / file_name
        if file_path.exists() and file_path.is_file():
            print(f"✅ File exists: tests/{file_name}")
        else:
            print(f"❌ File missing: tests/{file_name}")
            all_good = False
    
    return all_good


def check_test_markers():
    """Check if test markers are properly configured."""
    project_root = Path(__file__).parent.parent
    pytest_ini = project_root / 'pytest.ini'
    
    if not pytest_ini.exists():
        print("❌ pytest.ini not found")
        return False
    
    content = pytest_ini.read_text()
    
    required_markers = [
        'slow', 'integration', 'performance', 'unit', 
        'gpu', 'ocr', 'embedding', 'cli', 'core', 'regression'
    ]
    
    all_markers_found = True
    for marker in required_markers:
        if marker in content:
            print(f"✅ Marker configured: {marker}")
        else:
            print(f"❌ Marker missing: {marker}")
            all_markers_found = False
    
    return all_markers_found


def check_dependencies():
    """Check if required test dependencies are available."""
    required_packages = [
        'pytest',
        'pytest-cov',
        'click',
        'pathlib'
    ]
    
    optional_packages = [
        'pytest-xdist',
        'pytest-timeout',
        'pytest-benchmark',
        'psutil'
    ]
    
    all_required = True
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"✅ Required package available: {package}")
        except ImportError:
            print(f"❌ Required package missing: {package}")
            all_required = False
    
    for package in optional_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"✅ Optional package available: {package}")
        except ImportError:
            print(f"⚠️  Optional package missing: {package}")
    
    return all_required


def check_cli_availability():
    """Check if CLI commands are available for testing."""
    project_root = Path(__file__).parent.parent
    
    try:
        # Try to import the main CLI
        sys.path.insert(0, str(project_root))
        from llmbuilder.cli.main import cli
        print("✅ CLI module importable")
        
        # Try to run CLI help
        result = subprocess.run([
            sys.executable, '-c', 
            'from llmbuilder.cli.main import cli; cli(["--help"])'
        ], cwd=project_root, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ CLI help command works")
            return True
        else:
            print(f"⚠️  CLI help command failed: {result.stderr}")
            return False
            
    except ImportError as e:
        print(f"❌ CLI module not importable: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing CLI: {e}")
        return False


def run_sample_test():
    """Run a simple sample test to verify everything works."""
    project_root = Path(__file__).parent.parent
    
    try:
        # Run a simple test
        result = subprocess.run([
            sys.executable, '-m', 'pytest',
            'tests/unit/test_cli_main.py::TestMainCLI::test_cli_help',
            '-v'
        ], cwd=project_root, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Sample test passed")
            return True
        else:
            print(f"❌ Sample test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running sample test: {e}")
        return False


def main():
    """Main validation function."""
    print("🔍 Validating LLMBuilder Test Suite")
    print("=" * 50)
    
    checks = [
        ("Pytest Installation", check_pytest_installation),
        ("Test Structure", check_test_structure),
        ("Test Markers", check_test_markers),
        ("Dependencies", check_dependencies),
        ("CLI Availability", check_cli_availability),
        ("Test Discovery", check_test_discovery),
        ("Sample Test", run_sample_test),
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}:")
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"💥 Error in {check_name}: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All validation checks passed! Test suite is ready.")
        return 0
    else:
        print("⚠️  Some validation checks failed. Please fix the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())