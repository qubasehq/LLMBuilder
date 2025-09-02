#!/usr/bin/env python3
"""
Test package distribution in isolated environments.
"""

import sys
import subprocess
import tempfile
import shutil
import venv
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import platform


class DistributionTester:
    """Test package distribution in clean environments."""
    
    def __init__(self, package_path: Optional[Path] = None):
        self.project_root = Path(__file__).parent.parent
        self.package_path = package_path
        self.test_results = {}
        self.temp_dirs = []
    
    def _log(self, message: str, level: str = "INFO"):
        """Log test message."""
        log_entry = f"[{level}] {message}"
        print(log_entry)
    
    def cleanup(self):
        """Clean up temporary directories."""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def find_package_files(self) -> Dict[str, Path]:
        """Find built package files."""
        dist_dir = self.project_root / 'dist'
        
        if not dist_dir.exists():
            raise FileNotFoundError("No dist directory found. Run build first.")
        
        # Find latest files
        wheel_files = list(dist_dir.glob('*.whl'))
        sdist_files = list(dist_dir.glob('*.tar.gz'))
        
        if not wheel_files or not sdist_files:
            raise FileNotFoundError("Package files not found in dist directory")
        
        # Get the latest files
        wheel_file = max(wheel_files, key=lambda p: p.stat().st_mtime)
        sdist_file = max(sdist_files, key=lambda p: p.stat().st_mtime)
        
        return {
            'wheel': wheel_file,
            'sdist': sdist_file
        }
    
    def create_virtual_environment(self, name: str) -> Path:
        """Create a clean virtual environment."""
        temp_dir = Path(tempfile.mkdtemp(prefix=f'llmbuilder_test_{name}_'))
        self.temp_dirs.append(temp_dir)
        
        venv_dir = temp_dir / 'venv'
        
        self._log(f"Creating virtual environment: {venv_dir}")
        
        # Create virtual environment
        venv.create(venv_dir, with_pip=True, clear=True)
        
        return venv_dir
    
    def get_venv_python(self, venv_dir: Path) -> Path:
        """Get Python executable from virtual environment."""
        if platform.system() == 'Windows':
            return venv_dir / 'Scripts' / 'python.exe'
        else:
            return venv_dir / 'bin' / 'python'
    
    def test_wheel_installation(self, wheel_file: Path) -> bool:
        """Test wheel installation in clean environment."""
        self._log("Testing wheel installation...")
        
        venv_dir = self.create_virtual_environment('wheel')
        python_exe = self.get_venv_python(venv_dir)
        
        try:
            # Install wheel
            result = subprocess.run([
                str(python_exe), '-m', 'pip', 'install', str(wheel_file)
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                self._log(f"Wheel installation failed: {result.stderr}", "ERROR")
                return False
            
            # Test import
            result = subprocess.run([
                str(python_exe), '-c', 'import llmbuilder; print(llmbuilder.__version__)'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                version = result.stdout.strip()
                self._log(f"Wheel installation successful: version {version}")
                
                # Test CLI
                result = subprocess.run([
                    str(python_exe), '-m', 'llmbuilder', '--help'
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    self._log("CLI working from wheel installation")
                    return True
                else:
                    self._log("CLI not working from wheel installation", "ERROR")
                    return False
            else:
                self._log(f"Import failed: {result.stderr}", "ERROR")
                return False
                
        except Exception as e:
            self._log(f"Wheel test error: {e}", "ERROR")
            return False
    
    def test_sdist_installation(self, sdist_file: Path) -> bool:
        """Test source distribution installation."""
        self._log("Testing source distribution installation...")
        
        venv_dir = self.create_virtual_environment('sdist')
        python_exe = self.get_venv_python(venv_dir)
        
        try:
            # Upgrade pip and install build tools
            subprocess.run([
                str(python_exe), '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'
            ], capture_output=True, text=True, timeout=120)
            
            # Install source distribution
            result = subprocess.run([
                str(python_exe), '-m', 'pip', 'install', str(sdist_file)
            ], capture_output=True, text=True, timeout=600)  # Longer timeout for compilation
            
            if result.returncode != 0:
                self._log(f"Source installation failed: {result.stderr}", "ERROR")
                return False
            
            # Test import
            result = subprocess.run([
                str(python_exe), '-c', 'import llmbuilder; print(llmbuilder.__version__)'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                version = result.stdout.strip()
                self._log(f"Source installation successful: version {version}")
                return True
            else:
                self._log(f"Import failed: {result.stderr}", "ERROR")
                return False
                
        except Exception as e:
            self._log(f"Source test error: {e}", "ERROR")
            return False
    
    def test_dependency_resolution(self, package_file: Path) -> bool:
        """Test dependency resolution."""
        self._log("Testing dependency resolution...")
        
        venv_dir = self.create_virtual_environment('deps')
        python_exe = self.get_venv_python(venv_dir)
        
        try:
            # Install with dependencies
            result = subprocess.run([
                str(python_exe), '-m', 'pip', 'install', str(package_file)
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                self._log(f"Dependency installation failed: {result.stderr}", "ERROR")
                return False
            
            # Check core dependencies
            core_deps = ['torch', 'transformers', 'click', 'rich', 'fastapi']
            missing_deps = []
            
            for dep in core_deps:
                result = subprocess.run([
                    str(python_exe), '-c', f'import {dep}'
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode != 0:
                    missing_deps.append(dep)
            
            if missing_deps:
                self._log(f"Missing dependencies: {', '.join(missing_deps)}", "ERROR")
                return False
            
            self._log("All core dependencies resolved successfully")
            return True
            
        except Exception as e:
            self._log(f"Dependency test error: {e}", "ERROR")
            return False
    
    def test_optional_dependencies(self, package_file: Path) -> Dict[str, bool]:
        """Test optional dependency groups."""
        self._log("Testing optional dependencies...")
        
        optional_groups = ['gpu', 'dev', 'docs', 'monitoring']
        results = {}
        
        for group in optional_groups:
            self._log(f"Testing optional group: {group}")
            
            venv_dir = self.create_virtual_environment(f'opt_{group}')
            python_exe = self.get_venv_python(venv_dir)
            
            try:
                # Install with optional group
                result = subprocess.run([
                    str(python_exe), '-m', 'pip', 'install', f'{package_file}[{group}]'
                ], capture_output=True, text=True, timeout=600)
                
                if result.returncode == 0:
                    # Test import
                    result = subprocess.run([
                        str(python_exe), '-c', 'import llmbuilder; print("OK")'
                    ], capture_output=True, text=True, timeout=30)
                    
                    results[group] = result.returncode == 0
                    if results[group]:
                        self._log(f"Optional group '{group}' installed successfully")
                    else:
                        self._log(f"Optional group '{group}' import failed", "ERROR")
                else:
                    self._log(f"Optional group '{group}' installation failed", "ERROR")
                    results[group] = False
                    
            except Exception as e:
                self._log(f"Optional group '{group}' test error: {e}", "ERROR")
                results[group] = False
        
        return results
    
    def test_cli_functionality(self, package_file: Path) -> bool:
        """Test CLI functionality after installation."""
        self._log("Testing CLI functionality...")
        
        venv_dir = self.create_virtual_environment('cli')
        python_exe = self.get_venv_python(venv_dir)
        
        try:
            # Install package
            result = subprocess.run([
                str(python_exe), '-m', 'pip', 'install', str(package_file)
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                self._log(f"CLI test installation failed: {result.stderr}", "ERROR")
                return False
            
            # Test CLI commands
            cli_tests = [
                ['--help'],
                ['--version'],
                ['init', '--help'],
                ['data', '--help'],
                ['config', '--help']
            ]
            
            for cmd in cli_tests:
                result = subprocess.run([
                    str(python_exe), '-m', 'llmbuilder'
                ] + cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode != 0:
                    self._log(f"CLI command failed: {' '.join(cmd)}", "ERROR")
                    return False
            
            self._log("All CLI commands working")
            return True
            
        except Exception as e:
            self._log(f"CLI test error: {e}", "ERROR")
            return False
    
    def test_cross_platform_compatibility(self, package_file: Path) -> bool:
        """Test cross-platform compatibility."""
        self._log("Testing cross-platform compatibility...")
        
        # This is a basic test - full cross-platform testing requires multiple systems
        current_platform = platform.system()
        self._log(f"Current platform: {current_platform}")
        
        venv_dir = self.create_virtual_environment('platform')
        python_exe = self.get_venv_python(venv_dir)
        
        try:
            # Install and test basic functionality
            result = subprocess.run([
                str(python_exe), '-m', 'pip', 'install', str(package_file)
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                return False
            
            # Test platform-specific features
            test_script = '''
import llmbuilder
import platform
import sys

print(f"Platform: {platform.system()}")
print(f"Python: {sys.version}")
print(f"LLMBuilder: {llmbuilder.__version__}")

# Test basic imports
from llmbuilder.cli.main import cli
from llmbuilder.utils.config import ConfigManager

print("Basic imports successful")
'''
            
            result = subprocess.run([
                str(python_exe), '-c', test_script
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self._log("Cross-platform compatibility test passed")
                return True
            else:
                self._log(f"Cross-platform test failed: {result.stderr}", "ERROR")
                return False
                
        except Exception as e:
            self._log(f"Cross-platform test error: {e}", "ERROR")
            return False
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all distribution tests."""
        self._log("🧪 Running Comprehensive Distribution Tests")
        self._log("=" * 60)
        
        try:
            # Find package files
            package_files = self.find_package_files()
            wheel_file = package_files['wheel']
            sdist_file = package_files['sdist']
            
            self._log(f"Testing wheel: {wheel_file.name}")
            self._log(f"Testing source: {sdist_file.name}")
            
            # Run tests
            test_results = {}
            
            # Basic installation tests
            test_results['wheel_installation'] = self.test_wheel_installation(wheel_file)
            test_results['sdist_installation'] = self.test_sdist_installation(sdist_file)
            
            # Dependency tests
            test_results['dependency_resolution'] = self.test_dependency_resolution(wheel_file)
            test_results['optional_dependencies'] = self.test_optional_dependencies(wheel_file)
            
            # Functionality tests
            test_results['cli_functionality'] = self.test_cli_functionality(wheel_file)
            test_results['cross_platform'] = self.test_cross_platform_compatibility(wheel_file)
            
            # Summary
            self._log("\n" + "=" * 60)
            self._log("📊 DISTRIBUTION TEST SUMMARY")
            self._log("=" * 60)
            
            passed_tests = 0
            total_tests = 0
            
            for test_name, result in test_results.items():
                if isinstance(result, bool):
                    total_tests += 1
                    if result:
                        passed_tests += 1
                    status = "✓" if result else "✗"
                    self._log(f"{status} {test_name}")
                elif isinstance(result, dict):
                    # Handle optional dependencies
                    for sub_test, sub_result in result.items():
                        total_tests += 1
                        if sub_result:
                            passed_tests += 1
                        status = "✓" if sub_result else "✗"
                        self._log(f"{status} {test_name}.{sub_test}")
            
            success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            self._log(f"\nPassed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
            
            overall_success = passed_tests == total_tests
            status_text = "✅ PASS" if overall_success else "❌ FAIL"
            self._log(f"Overall Status: {status_text}")
            
            # Create test report
            report = {
                'timestamp': subprocess.run(['date'], capture_output=True, text=True).stdout.strip(),
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'wheel_file': str(wheel_file),
                'sdist_file': str(sdist_file),
                'test_results': test_results,
                'passed_tests': passed_tests,
                'total_tests': total_tests,
                'success_rate': success_rate,
                'overall_success': overall_success
            }
            
            # Save report
            report_file = self.project_root / 'distribution_test_report.json'
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            self._log(f"\n📄 Test report saved to: {report_file}")
            
            return report
            
        except Exception as e:
            self._log(f"Test suite error: {e}", "ERROR")
            return {'overall_success': False, 'error': str(e)}
        
        finally:
            # Cleanup
            self.cleanup()


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test LLMBuilder package distribution')
    parser.add_argument('--package', help='Specific package file to test')
    
    args = parser.parse_args()
    
    tester = DistributionTester(args.package)
    
    try:
        report = tester.run_comprehensive_tests()
        sys.exit(0 if report.get('overall_success', False) else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Test error: {e}")
        sys.exit(1)
    finally:
        tester.cleanup()


if __name__ == '__main__':
    main()