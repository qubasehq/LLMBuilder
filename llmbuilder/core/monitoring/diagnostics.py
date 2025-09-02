"""
System diagnostics for LLMBuilder monitoring.

This module provides automated system diagnostics including dependency checks,
data validation, configuration verification, and system resource analysis.
"""

import os
import sys
import json
import subprocess
import importlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
try:
    from importlib.metadata import version, distributions
except ImportError:
    # Fallback for Python < 3.8
    from importlib_metadata import version, distributions
import psutil
from datetime import datetime

from llmbuilder.utils.logging import get_logger
from llmbuilder.utils.config import ConfigManager

logger = get_logger(__name__)


class SystemDiagnostics:
    """
    Automated system diagnostics and issue detection.
    
    Performs comprehensive checks on dependencies, data integrity,
    configuration validity, and system resources with automated
    fix suggestions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize system diagnostics.
        
        Args:
            config: Diagnostics configuration dictionary
        """
        self.config = config
        self.check_dependencies = config.get('check_dependencies', True)
        self.check_data = config.get('check_data', True)
        self.check_models = config.get('check_models', True)
        self.check_config = config.get('check_config', True)
        self.check_system = config.get('check_system', True)
        self.fix_issues = config.get('fix_issues', False)
        self.verbose = config.get('verbose', False)
        
        self.progress_callback: Optional[Callable] = None
        self.results = {}
        
        logger.info("System diagnostics initialized")
    
    def set_progress_callback(self, callback: Callable[[str, float], None]):
        """Set progress callback for UI updates."""
        self.progress_callback = callback
    
    def run_all_checks(self) -> Dict[str, Any]:
        """
        Run all diagnostic checks.
        
        Returns:
            Dictionary containing diagnostic results
        """
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'checks_passed': 0,
            'warnings': 0,
            'critical_issues': 0,
            'fixes_applied': 0,
            'issues': []
        }
        
        checks = []
        if self.check_dependencies:
            checks.append(('Dependencies', self._check_dependencies))
        if self.check_data:
            checks.append(('Data Integrity', self._check_data_integrity))
        if self.check_models:
            checks.append(('Model Files', self._check_model_files))
        if self.check_config:
            checks.append(('Configuration', self._check_configuration))
        if self.check_system:
            checks.append(('System Resources', self._check_system_resources))
        
        total_checks = len(checks)
        
        for i, (check_name, check_func) in enumerate(checks):
            try:
                self._update_progress(f"Running {check_name} check...", (i / total_checks) * 100)
                
                check_result = check_func()
                self.results['checks'][check_name] = check_result
                
                if check_result.get('passed', False):
                    self.results['checks_passed'] += 1
                elif check_result.get('warning', False):
                    self.results['warnings'] += 1
                else:
                    self.results['critical_issues'] += 1
                    self.results['issues'].append({
                        'check': check_name,
                        'issue': check_result.get('details', ''),
                        'suggestion': check_result.get('suggestion', '')
                    })
                
                if check_result.get('fixed', False):
                    self.results['fixes_applied'] += 1
                
            except Exception as e:
                logger.error(f"Error in {check_name} check: {e}")
                self.results['checks'][check_name] = {
                    'passed': False,
                    'details': f"Check failed with error: {e}",
                    'suggestion': 'Review logs for detailed error information'
                }
                self.results['critical_issues'] += 1
        
        self._update_progress("Diagnostics complete", 100)
        
        logger.info(f"Diagnostics complete: {self.results['checks_passed']} passed, "
                   f"{self.results['warnings']} warnings, {self.results['critical_issues']} critical issues")
        
        return self.results
    
    def save_report(self, results: Dict[str, Any], report_path: Path):
        """
        Save diagnostic report to file.
        
        Args:
            results: Diagnostic results
            report_path: Path to save the report
        """
        try:
            report_data = {
                'diagnostic_report': results,
                'system_info': self._get_system_info(),
                'environment_info': self._get_environment_info()
            }
            
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Diagnostic report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error saving diagnostic report: {e}")
            raise
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check for missing or outdated dependencies."""
        try:
            required_packages = [
                'torch',
                'transformers',
                'datasets',
                'accelerate',
                'peft',
                'click',
                'rich',
                'psutil',
                'numpy',
                'pandas'
            ]
            
            optional_packages = [
                'GPUtil',
                'tensorboard',
                'wandb',
                'matplotlib',
                'seaborn'
            ]
            
            missing_required = []
            missing_optional = []
            outdated_packages = []
            
            # Check required packages
            for package in required_packages:
                try:
                    importlib.import_module(package)
                except ImportError:
                    missing_required.append(package)
            
            # Check optional packages
            for package in optional_packages:
                try:
                    importlib.import_module(package)
                except ImportError:
                    missing_optional.append(package)
            
            # Check for outdated packages (simplified check)
            try:
                result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--outdated'], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    outdated_lines = result.stdout.strip().split('\n')[2:]  # Skip header
                    for line in outdated_lines:
                        if line.strip():
                            package_name = line.split()[0]
                            if package_name in required_packages + optional_packages:
                                outdated_packages.append(package_name)
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                pass  # Skip outdated check if it fails
            
            # Determine result
            if missing_required:
                return {
                    'passed': False,
                    'details': f"Missing required packages: {', '.join(missing_required)}",
                    'suggestion': f"Install missing packages: pip install {' '.join(missing_required)}",
                    'missing_required': missing_required,
                    'missing_optional': missing_optional,
                    'outdated': outdated_packages
                }
            elif missing_optional or outdated_packages:
                return {
                    'passed': True,
                    'warning': True,
                    'details': f"Optional packages missing: {missing_optional}, Outdated: {outdated_packages}",
                    'suggestion': "Consider installing optional packages for enhanced functionality",
                    'missing_optional': missing_optional,
                    'outdated': outdated_packages
                }
            else:
                return {
                    'passed': True,
                    'details': "All required dependencies are available",
                    'suggestion': None
                }
        
        except Exception as e:
            return {
                'passed': False,
                'details': f"Dependency check failed: {e}",
                'suggestion': "Manually verify package installations"
            }
    
    def _check_data_integrity(self) -> Dict[str, Any]:
        """Check data files for corruption and accessibility."""
        try:
            data_dirs = ['data', 'datasets', 'processed_data']
            issues = []
            total_files = 0
            corrupted_files = []
            
            for data_dir in data_dirs:
                data_path = Path(data_dir)
                if not data_path.exists():
                    continue
                
                # Check common data file types
                for pattern in ['*.json', '*.jsonl', '*.txt', '*.csv', '*.parquet']:
                    for file_path in data_path.rglob(pattern):
                        total_files += 1
                        
                        try:
                            # Basic file accessibility check
                            if file_path.stat().st_size == 0:
                                corrupted_files.append(str(file_path))
                                continue
                            
                            # Format-specific checks
                            if file_path.suffix == '.json':
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    json.load(f)
                            elif file_path.suffix == '.jsonl':
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    for line_num, line in enumerate(f, 1):
                                        if line.strip():
                                            json.loads(line)
                                        if line_num > 10:  # Sample check
                                            break
                            
                        except (json.JSONDecodeError, UnicodeDecodeError, PermissionError) as e:
                            corrupted_files.append(f"{file_path}: {str(e)}")
            
            if corrupted_files:
                return {
                    'passed': False,
                    'details': f"Found {len(corrupted_files)} corrupted files out of {total_files} checked",
                    'suggestion': "Review and fix corrupted data files",
                    'corrupted_files': corrupted_files[:10]  # Limit output
                }
            elif total_files == 0:
                return {
                    'passed': True,
                    'warning': True,
                    'details': "No data files found to check",
                    'suggestion': "Ensure data files are in expected directories"
                }
            else:
                return {
                    'passed': True,
                    'details': f"All {total_files} data files passed integrity checks",
                    'suggestion': None
                }
        
        except Exception as e:
            return {
                'passed': False,
                'details': f"Data integrity check failed: {e}",
                'suggestion': "Manually verify data file accessibility"
            }
    
    def _check_model_files(self) -> Dict[str, Any]:
        """Check model files for completeness and accessibility."""
        try:
            model_dirs = ['models', 'checkpoints', 'saved_models']
            issues = []
            model_files_found = 0
            
            for model_dir in model_dirs:
                model_path = Path(model_dir)
                if not model_path.exists():
                    continue
                
                # Check for common model file patterns
                model_patterns = ['*.bin', '*.safetensors', '*.pt', '*.pth', '*.ckpt']
                config_patterns = ['config.json', 'model_config.json', 'training_args.json']
                
                for model_file in model_path.rglob('*'):
                    if any(model_file.match(pattern) for pattern in model_patterns):
                        model_files_found += 1
                        
                        # Check file accessibility and size
                        try:
                            stat = model_file.stat()
                            if stat.st_size == 0:
                                issues.append(f"Empty model file: {model_file}")
                            elif stat.st_size < 1024:  # Suspiciously small
                                issues.append(f"Suspiciously small model file: {model_file} ({stat.st_size} bytes)")
                        except (OSError, PermissionError) as e:
                            issues.append(f"Cannot access model file {model_file}: {e}")
                
                # Check for required config files in model directories
                for model_subdir in model_path.iterdir():
                    if model_subdir.is_dir():
                        has_model = any((model_subdir / pattern).exists() for pattern in model_patterns)
                        has_config = any((model_subdir / pattern).exists() for pattern in config_patterns)
                        
                        if has_model and not has_config:
                            issues.append(f"Model directory missing config: {model_subdir}")
            
            if issues:
                return {
                    'passed': False,
                    'details': f"Found {len(issues)} model file issues",
                    'suggestion': "Review and fix model file problems",
                    'issues': issues[:10]  # Limit output
                }
            elif model_files_found == 0:
                return {
                    'passed': True,
                    'warning': True,
                    'details': "No model files found",
                    'suggestion': "Model files will be created during training"
                }
            else:
                return {
                    'passed': True,
                    'details': f"All {model_files_found} model files passed checks",
                    'suggestion': None
                }
        
        except Exception as e:
            return {
                'passed': False,
                'details': f"Model file check failed: {e}",
                'suggestion': "Manually verify model file accessibility"
            }
    
    def _check_configuration(self) -> Dict[str, Any]:
        """Check configuration files for validity."""
        try:
            config_manager = ConfigManager()
            issues = []
            
            # Check main configuration
            try:
                project_config = config_manager.get_project_config()
                if not project_config:
                    issues.append("No project configuration found")
            except Exception as e:
                issues.append(f"Invalid project configuration: {e}")
            
            # Check specific config files
            config_files = [
                'config.json',
                'pyproject.toml',
                '.llmbuilder/config.json'
            ]
            
            for config_file in config_files:
                config_path = Path(config_file)
                if config_path.exists():
                    try:
                        if config_path.suffix == '.json':
                            with open(config_path, 'r') as f:
                                json.load(f)
                        elif config_path.suffix == '.toml':
                            # Basic TOML validation (could use tomli if available)
                            with open(config_path, 'r') as f:
                                content = f.read()
                                if '[' not in content:
                                    issues.append(f"Invalid TOML format: {config_file}")
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        issues.append(f"Invalid config file {config_file}: {e}")
            
            # Check for required configuration sections
            try:
                project_config = config_manager.get_project_config()
                required_sections = ['model', 'training', 'data']
                
                for section in required_sections:
                    if section not in project_config:
                        issues.append(f"Missing configuration section: {section}")
            except:
                pass  # Already handled above
            
            if issues:
                return {
                    'passed': False,
                    'details': f"Configuration issues found: {'; '.join(issues)}",
                    'suggestion': "Fix configuration files or run 'llmbuilder init' to create default config",
                    'issues': issues
                }
            else:
                return {
                    'passed': True,
                    'details': "All configuration files are valid",
                    'suggestion': None
                }
        
        except Exception as e:
            return {
                'passed': False,
                'details': f"Configuration check failed: {e}",
                'suggestion': "Manually verify configuration files"
            }
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resources and requirements."""
        try:
            issues = []
            warnings = []
            
            # Check available memory
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            if memory_gb < 4:
                issues.append(f"Low system memory: {memory_gb:.1f}GB (minimum 4GB recommended)")
            elif memory_gb < 8:
                warnings.append(f"Limited system memory: {memory_gb:.1f}GB (8GB+ recommended for training)")
            
            # Check available disk space
            disk = psutil.disk_usage('/')
            disk_free_gb = disk.free / (1024**3)
            
            if disk_free_gb < 5:
                issues.append(f"Low disk space: {disk_free_gb:.1f}GB free (minimum 5GB recommended)")
            elif disk_free_gb < 20:
                warnings.append(f"Limited disk space: {disk_free_gb:.1f}GB free (20GB+ recommended)")
            
            # Check CPU
            cpu_count = psutil.cpu_count()
            if cpu_count < 2:
                warnings.append(f"Limited CPU cores: {cpu_count} (2+ recommended)")
            
            # Check GPU availability
            gpu_available = False
            gpu_info = ""
            
            try:
from llmbuilder.utils.lazy_imports import torch
                if torch.cuda.is_available():
                    gpu_available = True
                    gpu_count = torch.cuda.device_count()
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    gpu_info = f"{gpu_count} GPU(s) available: {gpu_name} ({gpu_memory:.1f}GB)"
                    
                    if gpu_memory < 4:
                        warnings.append(f"Limited GPU memory: {gpu_memory:.1f}GB (4GB+ recommended)")
                else:
                    warnings.append("No GPU detected - training will be slower on CPU")
            except ImportError:
                warnings.append("PyTorch not available - cannot check GPU")
            
            # Check Python version
            python_version = sys.version_info
            if python_version < (3, 8):
                issues.append(f"Unsupported Python version: {python_version.major}.{python_version.minor} (3.8+ required)")
            elif python_version < (3, 9):
                warnings.append(f"Old Python version: {python_version.major}.{python_version.minor} (3.9+ recommended)")
            
            # Determine overall result
            if issues:
                return {
                    'passed': False,
                    'details': f"System resource issues: {'; '.join(issues)}",
                    'suggestion': "Upgrade system resources or adjust configuration for lower requirements",
                    'issues': issues,
                    'warnings': warnings,
                    'system_info': {
                        'memory_gb': memory_gb,
                        'disk_free_gb': disk_free_gb,
                        'cpu_count': cpu_count,
                        'gpu_available': gpu_available,
                        'gpu_info': gpu_info,
                        'python_version': f"{python_version.major}.{python_version.minor}.{python_version.micro}"
                    }
                }
            elif warnings:
                return {
                    'passed': True,
                    'warning': True,
                    'details': f"System warnings: {'; '.join(warnings)}",
                    'suggestion': "Consider upgrading system resources for better performance",
                    'warnings': warnings,
                    'system_info': {
                        'memory_gb': memory_gb,
                        'disk_free_gb': disk_free_gb,
                        'cpu_count': cpu_count,
                        'gpu_available': gpu_available,
                        'gpu_info': gpu_info,
                        'python_version': f"{python_version.major}.{python_version.minor}.{python_version.micro}"
                    }
                }
            else:
                return {
                    'passed': True,
                    'details': "System resources are adequate",
                    'suggestion': None,
                    'system_info': {
                        'memory_gb': memory_gb,
                        'disk_free_gb': disk_free_gb,
                        'cpu_count': cpu_count,
                        'gpu_available': gpu_available,
                        'gpu_info': gpu_info,
                        'python_version': f"{python_version.major}.{python_version.minor}.{python_version.micro}"
                    }
                }
        
        except Exception as e:
            return {
                'passed': False,
                'details': f"System resource check failed: {e}",
                'suggestion': "Manually verify system specifications"
            }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        try:
            return {
                'platform': sys.platform,
                'python_version': sys.version,
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'disk_total_gb': psutil.disk_usage('/').total / (1024**3),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get environment and package information."""
        try:
            env_info = {
                'python_executable': sys.executable,
                'python_path': sys.path[:5],  # Limit output
                'environment_variables': {
                    k: v for k, v in os.environ.items() 
                    if k.startswith(('CUDA', 'PATH', 'PYTHONPATH', 'TORCH'))
                }
            }
            
            # Get installed packages
            try:
                installed_packages = {}
                for dist in distributions():
                    installed_packages[dist.metadata['Name']] = dist.version
                env_info['installed_packages'] = dict(list(installed_packages.items())[:20])  # Limit output
            except Exception:
                pass
            
            return env_info
        except Exception as e:
            return {'error': str(e)}
    
    def _update_progress(self, step: str, percentage: float):
        """Update progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(step, percentage)