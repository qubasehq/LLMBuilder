#!/usr/bin/env python3
"""
Comprehensive test runner for LLMBuilder.

This script runs all test suites with proper categorization and reporting.
"""

import sys
import subprocess
import argparse
from pathlib import Path
import time
import json
from typing import Dict, List, Any


class TestRunner:
    """Comprehensive test runner with categorization and reporting."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_results = {}
        self.start_time = None
        self.end_time = None
    
    def run_unit_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run unit tests."""
        print("🧪 Running Unit Tests...")
        
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/unit/',
            '-v' if verbose else '-q',
            '--tb=short',
            '--durations=10',
            '-m', 'unit or not slow'
        ]
        
        result = self._run_pytest_command(cmd, 'unit_tests')
        return result
    
    def run_integration_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run integration tests."""
        print("🔗 Running Integration Tests...")
        
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/integration/',
            '-v' if verbose else '-q',
            '--tb=short',
            '--durations=10',
            '-m', 'integration'
        ]
        
        result = self._run_pytest_command(cmd, 'integration_tests')
        return result
    
    def run_performance_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run performance tests."""
        print("⚡ Running Performance Tests...")
        
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/performance/',
            '-v' if verbose else '-q',
            '--tb=short',
            '--durations=10',
            '-m', 'performance'
        ]
        
        result = self._run_pytest_command(cmd, 'performance_tests')
        return result
    
    def run_cli_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run CLI-specific tests."""
        print("💻 Running CLI Tests...")
        
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/unit/test_cli_*.py',
            'tests/integration/test_cli_*.py',
            '-v' if verbose else '-q',
            '--tb=short',
            '--durations=10'
        ]
        
        result = self._run_pytest_command(cmd, 'cli_tests')
        return result
    
    def run_core_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run core functionality tests."""
        print("🏗️ Running Core Tests...")
        
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/',
            '-v' if verbose else '-q',
            '--tb=short',
            '--durations=10',
            '-k', 'not cli and not performance',
            '-m', 'not slow'
        ]
        
        result = self._run_pytest_command(cmd, 'core_tests')
        return result
    
    def run_regression_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run regression tests."""
        print("🔄 Running Regression Tests...")
        
        # Run a subset of critical tests to catch regressions
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/integration/test_full_pipeline.py',
            'tests/unit/test_cli_main.py',
            '-v' if verbose else '-q',
            '--tb=short',
            '--durations=10'
        ]
        
        result = self._run_pytest_command(cmd, 'regression_tests')
        return result
    
    def run_coverage_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run tests with coverage reporting."""
        print("📊 Running Coverage Tests...")
        
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/',
            '--cov=llmbuilder',
            '--cov-report=term-missing',
            '--cov-report=html:htmlcov',
            '--cov-report=json:coverage.json',
            '-v' if verbose else '-q',
            '--tb=short',
            '-m', 'not slow and not performance'
        ]
        
        result = self._run_pytest_command(cmd, 'coverage_tests')
        
        # Try to read coverage data
        try:
            coverage_file = self.project_root / 'coverage.json'
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                    result['coverage_percent'] = coverage_data.get('totals', {}).get('percent_covered', 0)
        except Exception as e:
            print(f"Warning: Could not read coverage data: {e}")
        
        return result
    
    def _run_pytest_command(self, cmd: List[str], test_type: str) -> Dict[str, Any]:
        """Run pytest command and capture results."""
        start_time = time.time()
        
        try:
            # Change to project root directory
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Parse pytest output for test counts
            output_lines = result.stdout.split('\n')
            test_summary = self._parse_pytest_output(output_lines)
            
            test_result = {
                'success': result.returncode == 0,
                'duration': duration,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                **test_summary
            }
            
            self.test_results[test_type] = test_result
            
            # Print summary
            if test_result['success']:
                print(f"✅ {test_type}: {test_summary.get('passed', 0)} passed in {duration:.2f}s")
            else:
                print(f"❌ {test_type}: {test_summary.get('failed', 0)} failed, {test_summary.get('passed', 0)} passed in {duration:.2f}s")
            
            return test_result
            
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_type}: Tests timed out after 5 minutes")
            return {
                'success': False,
                'duration': 300,
                'return_code': -1,
                'error': 'timeout',
                'stdout': '',
                'stderr': 'Test execution timed out'
            }
        except Exception as e:
            print(f"💥 {test_type}: Error running tests: {e}")
            return {
                'success': False,
                'duration': 0,
                'return_code': -1,
                'error': str(e),
                'stdout': '',
                'stderr': str(e)
            }
    
    def _parse_pytest_output(self, output_lines: List[str]) -> Dict[str, int]:
        """Parse pytest output to extract test counts."""
        summary = {
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': 0,
            'warnings': 0
        }
        
        for line in output_lines:
            line = line.strip()
            
            # Look for summary line like "5 passed, 2 failed, 1 skipped in 10.5s"
            if ' passed' in line or ' failed' in line or ' skipped' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'passed' and i > 0:
                        try:
                            summary['passed'] = int(parts[i-1])
                        except (ValueError, IndexError):
                            pass
                    elif part == 'failed' and i > 0:
                        try:
                            summary['failed'] = int(parts[i-1])
                        except (ValueError, IndexError):
                            pass
                    elif part == 'skipped' and i > 0:
                        try:
                            summary['skipped'] = int(parts[i-1])
                        except (ValueError, IndexError):
                            pass
                    elif part == 'error' and i > 0:
                        try:
                            summary['errors'] = int(parts[i-1])
                        except (ValueError, IndexError):
                            pass
        
        return summary
    
    def run_all_tests(self, verbose: bool = False, include_slow: bool = False) -> Dict[str, Any]:
        """Run all test suites."""
        print("🚀 Running Comprehensive Test Suite...")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Run test suites in order
        test_suites = [
            ('unit', self.run_unit_tests),
            ('cli', self.run_cli_tests),
            ('core', self.run_core_tests),
            ('integration', self.run_integration_tests),
            ('regression', self.run_regression_tests),
        ]
        
        if include_slow:
            test_suites.append(('performance', self.run_performance_tests))
        
        # Always run coverage last
        test_suites.append(('coverage', self.run_coverage_tests))
        
        for suite_name, suite_func in test_suites:
            try:
                suite_func(verbose)
                print()  # Add spacing between suites
            except KeyboardInterrupt:
                print(f"\n⚠️ Test suite '{suite_name}' interrupted by user")
                break
            except Exception as e:
                print(f"\n💥 Error in test suite '{suite_name}': {e}")
                continue
        
        self.end_time = time.time()
        
        # Generate final report
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        if not self.start_time or not self.end_time:
            return {}
        
        total_duration = self.end_time - self.start_time
        
        # Calculate totals
        total_passed = sum(result.get('passed', 0) for result in self.test_results.values())
        total_failed = sum(result.get('failed', 0) for result in self.test_results.values())
        total_skipped = sum(result.get('skipped', 0) for result in self.test_results.values())
        total_errors = sum(result.get('errors', 0) for result in self.test_results.values())
        
        successful_suites = sum(1 for result in self.test_results.values() if result.get('success', False))
        total_suites = len(self.test_results)
        
        # Get coverage if available
        coverage_percent = None
        if 'coverage_tests' in self.test_results:
            coverage_percent = self.test_results['coverage_tests'].get('coverage_percent')
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_duration': total_duration,
            'summary': {
                'total_tests': total_passed + total_failed + total_errors,
                'passed': total_passed,
                'failed': total_failed,
                'skipped': total_skipped,
                'errors': total_errors,
                'success_rate': (total_passed / (total_passed + total_failed + total_errors)) * 100 if (total_passed + total_failed + total_errors) > 0 else 0
            },
            'suites': {
                'successful': successful_suites,
                'total': total_suites,
                'success_rate': (successful_suites / total_suites) * 100 if total_suites > 0 else 0
            },
            'coverage': coverage_percent,
            'details': self.test_results
        }
        
        # Print report
        self.print_report(report)
        
        # Save report to file
        report_file = self.project_root / 'test_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted test report."""
        print("=" * 60)
        print("📋 COMPREHENSIVE TEST REPORT")
        print("=" * 60)
        
        summary = report['summary']
        suites = report['suites']
        
        print(f"⏱️  Total Duration: {report['total_duration']:.2f}s")
        print(f"📊 Test Summary:")
        print(f"   • Total Tests: {summary['total_tests']}")
        print(f"   • Passed: {summary['passed']} ✅")
        print(f"   • Failed: {summary['failed']} ❌")
        print(f"   • Skipped: {summary['skipped']} ⏭️")
        print(f"   • Errors: {summary['errors']} 💥")
        print(f"   • Success Rate: {summary['success_rate']:.1f}%")
        
        print(f"\n🧪 Suite Summary:")
        print(f"   • Successful Suites: {suites['successful']}/{suites['total']}")
        print(f"   • Suite Success Rate: {suites['success_rate']:.1f}%")
        
        if report['coverage'] is not None:
            print(f"\n📈 Code Coverage: {report['coverage']:.1f}%")
        
        print(f"\n📝 Detailed Results:")
        for suite_name, result in report['details'].items():
            status = "✅" if result.get('success', False) else "❌"
            duration = result.get('duration', 0)
            passed = result.get('passed', 0)
            failed = result.get('failed', 0)
            print(f"   {status} {suite_name}: {passed} passed, {failed} failed ({duration:.2f}s)")
        
        # Overall result
        overall_success = summary['failed'] == 0 and summary['errors'] == 0
        print(f"\n🎯 Overall Result: {'SUCCESS' if overall_success else 'FAILURE'} {'✅' if overall_success else '❌'}")
        
        if not overall_success:
            print("\n⚠️  Some tests failed. Check the detailed output above for more information.")
        
        print("=" * 60)


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description='Run comprehensive LLMBuilder tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--include-slow', action='store_true', help='Include slow tests (performance)')
    parser.add_argument('--suite', choices=['unit', 'integration', 'performance', 'cli', 'core', 'regression', 'coverage', 'all'], 
                       default='all', help='Test suite to run')
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.suite == 'all':
        report = runner.run_all_tests(verbose=args.verbose, include_slow=args.include_slow)
        # Exit with error code if tests failed
        if report and report['summary']['failed'] > 0:
            sys.exit(1)
    else:
        # Run specific suite
        suite_methods = {
            'unit': runner.run_unit_tests,
            'integration': runner.run_integration_tests,
            'performance': runner.run_performance_tests,
            'cli': runner.run_cli_tests,
            'core': runner.run_core_tests,
            'regression': runner.run_regression_tests,
            'coverage': runner.run_coverage_tests
        }
        
        if args.suite in suite_methods:
            result = suite_methods[args.suite](verbose=args.verbose)
            if not result.get('success', False):
                sys.exit(1)
        else:
            print(f"Unknown test suite: {args.suite}")
            sys.exit(1)


if __name__ == '__main__':
    main()