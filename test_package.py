#!/usr/bin/env python3
"""
Test script to verify llmbuilder package functionality before publishing.
"""

import sys
import os

# Add the package to the path so we can import it
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_import():
    """Test that we can import the package and check its version."""
    try:
        import llmbuilder
        print(f"✓ Successfully imported llmbuilder version {llmbuilder.__version__}")
        return True
    except Exception as e:
        print(f"✗ Failed to import llmbuilder: {e}")
        return False

def test_cli_import():
    """Test that we can import the CLI module."""
    try:
        from llmbuilder import cli
        print("✓ Successfully imported CLI module")
        return True
    except Exception as e:
        print(f"✗ Failed to import CLI module: {e}")
        return False

def test_config_import():
    """Test that we can import the config module."""
    try:
        from llmbuilder import config
        print("✓ Successfully imported config module")
        return True
    except Exception as e:
        print(f"✗ Failed to import config module: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of the package."""
    try:
        import llmbuilder
        
        # Test that we can access the main functions
        assert hasattr(llmbuilder, 'train')
        assert hasattr(llmbuilder, 'generate_text')
        assert hasattr(llmbuilder, 'load_config')
        assert hasattr(llmbuilder, 'build_model')
        
        print("✓ Basic functionality test passed")
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running llmbuilder package tests...")
    print("=" * 50)
    
    tests = [
        test_import,
        test_cli_import,
        test_config_import,
        test_basic_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Package is ready for publishing.")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues before publishing.")
        return 1

if __name__ == "__main__":
    sys.exit(main())