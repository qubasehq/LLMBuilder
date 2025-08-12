#!/usr/bin/env python3
"""
Comprehensive test runner for LLMBuilder.
Provides different test execution modes and reporting options.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import json
import time


class TestRunner:
    """Comprehensive test runner with multiple execution modes."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_dir = self.project_root / "tests"
        self.results = {}
        
    def run_command(self, cmd: List[str], description: str = "") -> Dict[str, Any]:
        """Run a command and capture results."""
        print(f"\n{'='*60}")
        print(f"Running: {description or ' '.join(cmd)}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            duration = time.time() - start_time
            
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            return {
                'success': result.returncode == 0,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'duration': duration,
                'command': ' '.join(cmd)
            }
            
        except subprocess.TimeoutExpired:
            print(f"❌ Command timed out after 30 minutes")
            return {
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': 'Command timed out',
                'duration': time.time() - start_time,
                'command': ' '.join(cmd)
            }
        except Exception as e:
            print(f"❌ Command failed with exception: {e}")
            return {
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'duration': time.time() - start_time,
                'command': ' '.join(cmd)
            }
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests."""
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/",
            "-v",
            "--tb=short",
            "-m", "not slow and not integration"
        ]
        
        return self.run_command(cmd, "Unit Tests")
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        cmd = [
            sys.executable, "-m", "pytest", 
            "tests/test_integration.py",
            "-v",
            "--tb=short",
            "-m", "not slow"
        ]
        
        return self.run_command(cmd, "Integration Tests (Fast)")
    
    def run_slow_integration_tests(self) -> Dict[str, Any]:
        """Run slow integration tests."""
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/test_integration.py",
            "-v", 
            "--tb=short",
            "-m", "slow"
        ]
        
        return self.run_command(cmd, "Integration Tests (Slow)")
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/test_performance.py",
            "-v",
            "--tb=short"
        ]
        
        return self.run_command(cmd, "Performance Tests")
    
    def run_coverage_tests(self) -> Dict[str, Any]:
        """Run tests with coverage reporting."""
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/",
            "--cov=.",
            "--cov-report=html",
            "--cov-report=xml",
            "--cov-report=term",
            "-v",
            "--tb=short",
            "-m", "not slow"
        ]
        
        return self.run_command(cmd, "Coverage Tests")
    
    def run_specific_tests(self, test_pattern: str) -> Dict[str, Any]:
        """Run specific tests matching a pattern."""
        cmd = [
            sys.executable, "-m", "pytest",
            "-k", test_pattern,
            "-v",
            "--tb=short"
        ]
        
        return self.run_command(cmd, f"Specific Tests ({test_pattern})")
    
    def run_component_tests(self, component: str) -> Dict[str, Any]:
        """Run tests for a specific component."""
        component_map = {
            'ingestion': 'tests/test_ingestion.py tests/test_*_extractor.py',
            'deduplication': 'tests/test_deduplication*.py tests/test_*_deduplicator.py',
            'tokenizer': 'tests/test_tokenizer*.py tests/test_sentencepiece*.py',
            'conversion': 'tests/test_conversion*.py tests/test_gguf*.py tests/test_quantization*.py',
            'training': 'tests/test_training*.py tests/test_dataset.py',
            'pipeline': 'tests/test_*pipeline*.py'
        }
        
        if component not in component_map:
            print(f"❌ Unknown component: {component}")
            print(f"Available components: {', '.join(component_map.keys())}")
            return {'success': False, 'returncode': 1}
        
        test_files = component_map[component].split()
        
        cmd = [sys.executable, "-m", "pytest"] + test_files + [
            "-v", "--tb=short"
        ]
        
        return self.run_command(cmd, f"Component Tests ({component})")
    
    def install_dev_dependencies(self) -> bool:
        """Install development dependencies if available."""
        dev_requirements = self.project_root / "requirements-dev.txt"
        if dev_requirements.exists():
            print("Installing development dependencies...")
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(dev_requirements)
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print("✅ Development dependencies installed successfully")
                    return True
                else:
                    print(f"⚠️ Warning: Could not install development dependencies: {result.stderr}")
                    return False
            except Exception as e:
                print(f"⚠️ Warning: Could not install development dependencies: {e}")
                return False
        return False

    def check_dependencies(self) -> Dict[str, Any]:
        """Check if all dependencies are available."""
        print("\n" + "="*60)
        print("Checking Dependencies")
        print("="*60)
        
        dependencies = {
            'required': [
                'torch', 'transformers', 'numpy', 'pytest', 'loguru'
            ],
            'optional': [
                'sentence_transformers', 'ebooklib', 'pytesseract', 
                'beautifulsoup4', 'sklearn', 'memory_profiler'
            ],
            'dev': [
                'pytest_cov', 'pytest_timeout', 'black', 'isort', 'flake8'
            ],
            'system': [
                'tesseract'
            ]
        }
        
        results = {'required': {}, 'optional': {}, 'dev': {}, 'system': {}}
        all_available = True
        
        # Check Python packages
        for category in ['required', 'optional', 'dev']:
            for package in dependencies[category]:
                try:
                    __import__(package)
                    results[category][package] = True
                    print(f"✅ {package}")
                except ImportError:
                    results[category][package] = False
                    if category == 'required':
                        status = "❌"
                        all_available = False
                    elif category == 'dev':
                        status = "⚠️"
                    else:
                        status = "⚠️"
                    category_name = category.upper() if category == 'required' else category.upper()
                    print(f"{status} {package} - {category_name}")
                    if category == 'required':
                        all_available = False
        
        # Check system dependencies
        import shutil
        for package in dependencies['system']:
            available = shutil.which(package) is not None
            results['system'][package] = available
            status = "✅" if available else "⚠️"
            print(f"{status} {package} - SYSTEM")
        
        return {
            'success': all_available,
            'results': results,
            'returncode': 0 if all_available else 1
        }
    
    def create_test_report(self, results: Dict[str, Any]) -> None:
        """Create a comprehensive test report."""
        report_file = self.project_root / "test_report.json"
        
        # Add summary statistics
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        total_duration = 0
        
        for test_type, result in results.items():
            if isinstance(result, dict) and 'success' in result:
                total_duration += result.get('duration', 0)
                if result['success']:
                    passed_tests += 1
                else:
                    failed_tests += 1
                total_tests += 1
        
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'total_duration': total_duration,
            'results': results
        }
        
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Test report saved to: {report_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total test suites: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        print(f"Total duration: {total_duration:.1f}s")
        
        if failed_tests > 0:
            print(f"\n❌ Failed test suites:")
            for test_type, result in results.items():
                if isinstance(result, dict) and not result.get('success', True):
                    print(f"  - {test_type}")
    
    def run_all_tests(self, include_slow: bool = False) -> Dict[str, Any]:
        """Run all test suites."""
        results = {}
        
        # Try to install development dependencies
        self.install_dev_dependencies()
        
        # Check dependencies first
        results['dependencies'] = self.check_dependencies()
        if not results['dependencies']['success']:
            print("❌ Missing required dependencies. Please install them first.")
            return results
        
        # Run test suites
        results['unit_tests'] = self.run_unit_tests()
        results['integration_tests'] = self.run_integration_tests()
        
        if include_slow:
            results['slow_integration_tests'] = self.run_slow_integration_tests()
        
        results['performance_tests'] = self.run_performance_tests()
        
        return results
    
    def run_ci_tests(self) -> Dict[str, Any]:
        """Run tests suitable for CI/CD."""
        results = {}
        
        # Check dependencies
        results['dependencies'] = self.check_dependencies()
        
        # Run fast tests only
        results['unit_tests'] = self.run_unit_tests()
        results['integration_tests'] = self.run_integration_tests()
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for LLMBuilder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --all                    # Run all tests
  python run_tests.py --unit                   # Run unit tests only
  python run_tests.py --integration            # Run integration tests
  python run_tests.py --performance            # Run performance tests
  python run_tests.py --coverage               # Run with coverage
  python run_tests.py --component ingestion    # Test specific component
  python run_tests.py --pattern "test_dedup"   # Run specific test pattern
  python run_tests.py --ci                     # Run CI-suitable tests
  python run_tests.py --deps                   # Check dependencies only
        """
    )
    
    parser.add_argument('--all', action='store_true', 
                       help='Run all test suites')
    parser.add_argument('--unit', action='store_true',
                       help='Run unit tests only')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration tests')
    parser.add_argument('--slow', action='store_true',
                       help='Include slow tests')
    parser.add_argument('--performance', action='store_true',
                       help='Run performance tests')
    parser.add_argument('--coverage', action='store_true',
                       help='Run tests with coverage reporting')
    parser.add_argument('--component', type=str,
                       help='Run tests for specific component')
    parser.add_argument('--pattern', type=str,
                       help='Run tests matching pattern')
    parser.add_argument('--ci', action='store_true',
                       help='Run CI-suitable tests (fast only)')
    parser.add_argument('--deps', action='store_true',
                       help='Check dependencies only')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed test report')
    
    args = parser.parse_args()
    
    runner = TestRunner()
    results = {}
    
    try:
        if args.deps:
            results['dependencies'] = runner.check_dependencies()
        elif args.unit:
            results['unit_tests'] = runner.run_unit_tests()
        elif args.integration:
            results['integration_tests'] = runner.run_integration_tests()
            if args.slow:
                results['slow_integration_tests'] = runner.run_slow_integration_tests()
        elif args.performance:
            results['performance_tests'] = runner.run_performance_tests()
        elif args.coverage:
            results['coverage_tests'] = runner.run_coverage_tests()
        elif args.component:
            results[f'{args.component}_tests'] = runner.run_component_tests(args.component)
        elif args.pattern:
            results['pattern_tests'] = runner.run_specific_tests(args.pattern)
        elif args.ci:
            results = runner.run_ci_tests()
        elif args.all:
            results = runner.run_all_tests(include_slow=args.slow)
        else:
            # Default: run basic test suite
            results = runner.run_ci_tests()
        
        # Generate report if requested or if running all tests
        if args.report or args.all:
            runner.create_test_report(results)
        
        # Determine exit code
        failed_suites = [
            name for name, result in results.items() 
            if isinstance(result, dict) and not result.get('success', True)
        ]
        
        if failed_suites:
            print(f"\n❌ Test execution completed with failures in: {', '.join(failed_suites)}")
            sys.exit(1)
        else:
            print(f"\n✅ All test suites completed successfully!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Test execution failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()