#!/usr/bin/env python3
"""
Package verification and integrity checking for LLMBuilder.
"""

import sys
import subprocess
import importlib
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
try:
    from importlib.metadata import version, distributions
except ImportError:
    # Fallback for Python < 3.8
    from importlib_metadata import version, distributions


class PackageVerifier:
    """Comprehensive package verification for LLMBuilder."""
    
    def __init__(self):
        self.verification_results = {}
        self.errors = []
        self.warnings = []
    
    def _log_result(self, test_name: str, passed: bool, message: str = "", details: Any = None):
        """Log verification result."""
        self.verification_results[test_name] = {
            'passed': passed,
            'message': message,
            'details': details
        }
        
        status = "✓" if passed else "✗"
        print(f"{status} {test_name}: {message}")
        
        if not passed:
            self.errors.append(f"{test_name}: {message}")
        elif "warning" in message.lower():
            self.warnings.append(f"{test_name}: {message}")
    
    def check_package_installation(self) -> bool:
        """Check if LLMBuilder is properly installed."""
        try:
            import llmbuilder
            version = getattr(llmbuilder, '__version__', 'unknown')
            self._log_result(
                "Package Installation", 
                True, 
                f"LLMBuilder {version} is installed",
                {'version': version}
            )
            return True
        except ImportError as e:
            self._log_result(
                "Package Installation", 
                False, 
                f"LLMBuilder not installed: {e}"
            )
            return False
    
    def check_dependencies(self) -> bool:
        """Check all required and optional dependencies."""
        try:
            import llmbuilder
            
            # Core dependencies
            core_deps = [
                'torch', 'transformers', 'tokenizers', 'sentencepiece',
                'click', 'rich', 'tqdm', 'fastapi', 'uvicorn', 'pydantic',
                'pandas', 'numpy', 'requests', 'loguru', 'PyYAML'
            ]
            
            missing_core = []
            for dep in core_deps:
                try:
                    importlib.import_module(dep)
                except ImportError:
                    missing_core.append(dep)
            
            if missing_core:
                self._log_result(
                    "Core Dependencies", 
                    False, 
                    f"Missing core dependencies: {', '.join(missing_core)}",
                    {'missing': missing_core}
                )
                return False
            else:
                self._log_result(
                    "Core Dependencies", 
                    True, 
                    "All core dependencies available"
                )
            
            # Optional dependencies
            optional_deps = {
                'gpu': ['peft', 'bitsandbytes', 'accelerate'],
                'docs': ['sphinx'],
                'monitoring': ['wandb', 'tensorboard', 'mlflow']
            }
            
            optional_status = {}
            for category, deps in optional_deps.items():
                missing = []
                for dep in deps:
                    try:
                        importlib.import_module(dep)
                    except ImportError:
                        missing.append(dep)
                
                optional_status[category] = {
                    'available': len(missing) == 0,
                    'missing': missing
                }
            
            self._log_result(
                "Optional Dependencies", 
                True, 
                f"Optional dependency status checked",
                optional_status
            )
            
            return True
            
        except Exception as e:
            self._log_result(
                "Dependencies Check", 
                False, 
                f"Error checking dependencies: {e}"
            )
            return False
    
    def check_cli_functionality(self) -> bool:
        """Check CLI command functionality."""
        try:
            # Test CLI import
            from llmbuilder.cli.main import cli
            
            # Test CLI help command
            result = subprocess.run([
                sys.executable, '-c', 
                'from llmbuilder.cli.main import cli; cli(["--help"])'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self._log_result(
                    "CLI Functionality", 
                    True, 
                    "CLI commands working"
                )
                return True
            else:
                self._log_result(
                    "CLI Functionality", 
                    False, 
                    f"CLI help failed: {result.stderr}"
                )
                return False
                
        except Exception as e:
            self._log_result(
                "CLI Functionality", 
                False, 
                f"CLI test failed: {e}"
            )
            return False
    
    def check_package_integrity(self) -> bool:
        """Check package file integrity."""
        try:
            import llmbuilder
            package_path = Path(llmbuilder.__file__).parent
            
            # Check essential files
            essential_files = [
                '__init__.py',
                'cli/__init__.py',
                'cli/main.py',
                'core/__init__.py',
                'utils/__init__.py'
            ]
            
            missing_files = []
            for file_path in essential_files:
                full_path = package_path / file_path
                if not full_path.exists():
                    missing_files.append(file_path)
            
            if missing_files:
                self._log_result(
                    "Package Integrity", 
                    False, 
                    f"Missing essential files: {', '.join(missing_files)}",
                    {'missing_files': missing_files}
                )
                return False
            
            # Check templates and configs
            template_dirs = ['templates', 'configs']
            template_status = {}
            
            for template_dir in template_dirs:
                dir_path = package_path / template_dir
                if dir_path.exists():
                    file_count = len(list(dir_path.rglob('*')))
                    template_status[template_dir] = {
                        'exists': True,
                        'file_count': file_count
                    }
                else:
                    template_status[template_dir] = {
                        'exists': False,
                        'file_count': 0
                    }
            
            self._log_result(
                "Package Integrity", 
                True, 
                "Essential files present",
                {'template_status': template_status}
            )
            return True
            
        except Exception as e:
            self._log_result(
                "Package Integrity", 
                False, 
                f"Integrity check failed: {e}"
            )
            return False
    
    def check_entry_points(self) -> bool:
        """Check package entry points."""
        try:
            # Check console script entry point
            result = subprocess.run([
                sys.executable, '-m', 'llmbuilder', '--version'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self._log_result(
                    "Entry Points", 
                    True, 
                    "Console script entry point working"
                )
            else:
                # Try alternative method
                try:
                    result = subprocess.run([
                        'llmbuilder', '--version'
                    ], capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0:
                        self._log_result(
                            "Entry Points", 
                            True, 
                            "Direct command entry point working"
                        )
                    else:
                        self._log_result(
                            "Entry Points", 
                            False, 
                            "Entry points not working properly"
                        )
                        return False
                except FileNotFoundError:
                    self._log_result(
                        "Entry Points", 
                        False, 
                        "llmbuilder command not found in PATH"
                    )
                    return False
            
            return True
            
        except Exception as e:
            self._log_result(
                "Entry Points", 
                False, 
                f"Entry point check failed: {e}"
            )
            return False
    
    def check_permissions(self) -> bool:
        """Check file permissions and access."""
        try:
            import llmbuilder
            package_path = Path(llmbuilder.__file__).parent
            
            # Check read permissions
            readable = package_path.is_dir() and package_path.exists()
            
            # Check if we can create temp files (for CLI operations)
            import tempfile
            try:
                with tempfile.NamedTemporaryFile(delete=True) as tmp:
                    tmp.write(b"test")
                temp_access = True
            except Exception:
                temp_access = False
            
            if readable and temp_access:
                self._log_result(
                    "Permissions", 
                    True, 
                    "File permissions OK"
                )
                return True
            else:
                issues = []
                if not readable:
                    issues.append("package not readable")
                if not temp_access:
                    issues.append("temp file creation failed")
                
                self._log_result(
                    "Permissions", 
                    False, 
                    f"Permission issues: {', '.join(issues)}"
                )
                return False
                
        except Exception as e:
            self._log_result(
                "Permissions", 
                False, 
                f"Permission check failed: {e}"
            )
            return False
    
    def check_version_consistency(self) -> bool:
        """Check version consistency across package."""
        try:
            import llmbuilder
            
            # Get version from different sources
            versions = {}
            
            # From __init__.py
            versions['__init__'] = getattr(llmbuilder, '__version__', None)
            
            # From CLI
            try:
                result = subprocess.run([
                    sys.executable, '-c',
                    'from llmbuilder.cli.main import cli; cli(["--version"])'
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    # Extract version from output
                    output_lines = result.stdout.strip().split('\n')
                    for line in output_lines:
                        if 'version' in line.lower():
                            # Try to extract version number
                            import re
                            version_match = re.search(r'(\d+\.\d+\.\d+)', line)
                            if version_match:
                                versions['cli'] = version_match.group(1)
                            break
            except Exception:
                versions['cli'] = None
            
            # Check consistency
            unique_versions = set(v for v in versions.values() if v is not None)
            
            if len(unique_versions) <= 1:
                version = list(unique_versions)[0] if unique_versions else "unknown"
                self._log_result(
                    "Version Consistency", 
                    True, 
                    f"Version consistent: {version}",
                    versions
                )
                return True
            else:
                self._log_result(
                    "Version Consistency", 
                    False, 
                    f"Version mismatch: {versions}",
                    versions
                )
                return False
                
        except Exception as e:
            self._log_result(
                "Version Consistency", 
                False, 
                f"Version check failed: {e}"
            )
            return False
    
    def run_comprehensive_verification(self) -> Dict[str, Any]:
        """Run all verification checks."""
        print("🔍 Running LLMBuilder Package Verification")
        print("=" * 50)
        
        # Run all checks
        checks = [
            self.check_package_installation,
            self.check_dependencies,
            self.check_cli_functionality,
            self.check_package_integrity,
            self.check_entry_points,
            self.check_permissions,
            self.check_version_consistency
        ]
        
        passed_checks = 0
        total_checks = len(checks)
        
        for check in checks:
            try:
                if check():
                    passed_checks += 1
            except Exception as e:
                print(f"Error in {check.__name__}: {e}")
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 VERIFICATION SUMMARY")
        print("=" * 50)
        
        success_rate = (passed_checks / total_checks) * 100
        print(f"Passed: {passed_checks}/{total_checks} ({success_rate:.1f}%)")
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        overall_status = len(self.errors) == 0
        status_text = "✅ PASS" if overall_status else "❌ FAIL"
        print(f"\n🎯 Overall Status: {status_text}")
        
        # Create verification report
        report = {
            'timestamp': subprocess.run(['date'], capture_output=True, text=True).stdout.strip(),
            'overall_status': overall_status,
            'passed_checks': passed_checks,
            'total_checks': total_checks,
            'success_rate': success_rate,
            'results': self.verification_results,
            'errors': self.errors,
            'warnings': self.warnings
        }
        
        # Save report
        report_file = Path('llmbuilder_verification_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        return report


def main():
    """Main verification function."""
    verifier = PackageVerifier()
    report = verifier.run_comprehensive_verification()
    
    # Exit with appropriate code
    sys.exit(0 if report['overall_status'] else 1)


if __name__ == '__main__':
    main()