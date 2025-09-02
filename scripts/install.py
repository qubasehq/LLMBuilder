#!/usr/bin/env python3
"""
LLMBuilder installation script with dependency management and verification.
"""

import sys
import subprocess
import platform
import importlib.util
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json


class LLMBuilderInstaller:
    """Intelligent installer for LLMBuilder with dependency management."""
    
    def __init__(self):
        self.python_version = sys.version_info
        self.platform = platform.system().lower()
        self.architecture = platform.machine().lower()
        self.has_gpu = self._detect_gpu()
        self.installation_log = []
    
    def _detect_gpu(self) -> bool:
        """Detect if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            # Try to detect NVIDIA GPU without torch
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                return result.returncode == 0
            except (subprocess.SubprocessError, FileNotFoundError):
                return False
    
    def _log(self, message: str, level: str = "INFO"):
        """Log installation message."""
        log_entry = f"[{level}] {message}"
        print(log_entry)
        self.installation_log.append(log_entry)
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        min_version = (3, 8)
        if self.python_version < min_version:
            self._log(f"Python {'.'.join(map(str, min_version))} or higher required. "
                     f"Current: {'.'.join(map(str, self.python_version[:2]))}", "ERROR")
            return False
        
        self._log(f"Python version check passed: {'.'.join(map(str, self.python_version[:2]))}")
        return True
    
    def check_pip_version(self) -> bool:
        """Check pip version and upgrade if needed."""
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                self._log("pip not found", "ERROR")
                return False
            
            # Upgrade pip
            self._log("Upgrading pip...")
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self._log("pip upgraded successfully")
            else:
                self._log(f"pip upgrade failed: {result.stderr}", "WARNING")
            
            return True
        except Exception as e:
            self._log(f"Error checking pip: {e}", "ERROR")
            return False
    
    def get_installation_command(self, include_optional: List[str] = None) -> List[str]:
        """Generate installation command based on system configuration."""
        cmd = [sys.executable, '-m', 'pip', 'install']
        
        # Base package
        package_spec = "llmbuilder"
        
        # Add optional dependencies
        if include_optional:
            optional_deps = ','.join(include_optional)
            package_spec = f"llmbuilder[{optional_deps}]"
        
        cmd.append(package_spec)
        
        # Add platform-specific options
        if self.platform == "windows":
            cmd.extend(['--prefer-binary'])
        
        return cmd
    
    def recommend_installation_options(self) -> Dict[str, bool]:
        """Recommend installation options based on system."""
        recommendations = {
            'gpu': self.has_gpu,
            'dev': False,  # Only for development
            'docs': False,  # Only if building docs
            'monitoring': True,  # Generally useful
            'test': False,  # Only for testing
        }
        
        self._log("Installation recommendations:")
        for option, recommended in recommendations.items():
            status = "✓" if recommended else "○"
            self._log(f"  {status} {option}: {'Recommended' if recommended else 'Optional'}")
        
        return recommendations
    
    def install_system_dependencies(self) -> bool:
        """Install system dependencies if possible."""
        self._log("Checking system dependencies...")
        
        # Check for Tesseract OCR
        try:
            result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                self._log("Tesseract OCR found")
            else:
                self._log("Tesseract OCR not found - OCR features will be limited", "WARNING")
        except FileNotFoundError:
            self._log("Tesseract OCR not found - install manually for OCR support", "WARNING")
            self._provide_tesseract_instructions()
        
        return True
    
    def _provide_tesseract_instructions(self):
        """Provide platform-specific Tesseract installation instructions."""
        instructions = {
            'linux': "sudo apt-get install tesseract-ocr tesseract-ocr-eng",
            'darwin': "brew install tesseract",
            'windows': "Download from: https://github.com/UB-Mannheim/tesseract/wiki"
        }
        
        if self.platform in instructions:
            self._log(f"To install Tesseract: {instructions[self.platform]}")
    
    def install_llmbuilder(self, options: List[str] = None) -> bool:
        """Install LLMBuilder with specified options."""
        if options is None:
            recommendations = self.recommend_installation_options()
            options = [opt for opt, recommended in recommendations.items() if recommended]
        
        cmd = self.get_installation_command(options)
        self._log(f"Installing LLMBuilder with options: {options}")
        self._log(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self._log("LLMBuilder installed successfully!")
                return True
            else:
                self._log(f"Installation failed: {result.stderr}", "ERROR")
                return False
                
        except Exception as e:
            self._log(f"Installation error: {e}", "ERROR")
            return False
    
    def verify_installation(self) -> bool:
        """Verify LLMBuilder installation."""
        self._log("Verifying installation...")
        
        try:
            # Test import
            import llmbuilder
            self._log(f"LLMBuilder version {llmbuilder.__version__} imported successfully")
            
            # Test CLI
            result = subprocess.run([sys.executable, '-m', 'llmbuilder', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self._log("CLI command working")
            else:
                self._log("CLI command failed", "WARNING")
            
            # Test core functionality
            from llmbuilder.cli.main import cli
            self._log("Core modules imported successfully")
            
            return True
            
        except ImportError as e:
            self._log(f"Import failed: {e}", "ERROR")
            return False
        except Exception as e:
            self._log(f"Verification error: {e}", "ERROR")
            return False
    
    def create_installation_report(self) -> str:
        """Create installation report."""
        report = {
            'timestamp': subprocess.run(['date'], capture_output=True, text=True).stdout.strip(),
            'python_version': '.'.join(map(str, self.python_version[:3])),
            'platform': self.platform,
            'architecture': self.architecture,
            'gpu_available': self.has_gpu,
            'installation_log': self.installation_log
        }
        
        report_file = Path('llmbuilder_installation_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self._log(f"Installation report saved to: {report_file}")
        return str(report_file)
    
    def run_full_installation(self, custom_options: List[str] = None) -> bool:
        """Run complete installation process."""
        self._log("Starting LLMBuilder installation...")
        self._log("=" * 50)
        
        # System checks
        if not self.check_python_version():
            return False
        
        if not self.check_pip_version():
            return False
        
        # System dependencies
        self.install_system_dependencies()
        
        # Install LLMBuilder
        if not self.install_llmbuilder(custom_options):
            return False
        
        # Verify installation
        if not self.verify_installation():
            self._log("Installation completed but verification failed", "WARNING")
        
        # Create report
        self.create_installation_report()
        
        self._log("=" * 50)
        self._log("Installation completed!")
        self._log("Run 'llmbuilder --help' to get started")
        
        return True


def main():
    """Main installation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Install LLMBuilder with optimal configuration')
    parser.add_argument('--gpu', action='store_true', help='Include GPU dependencies')
    parser.add_argument('--dev', action='store_true', help='Include development dependencies')
    parser.add_argument('--docs', action='store_true', help='Include documentation dependencies')
    parser.add_argument('--monitoring', action='store_true', help='Include monitoring dependencies')
    parser.add_argument('--all', action='store_true', help='Include all optional dependencies')
    parser.add_argument('--auto', action='store_true', help='Auto-detect and install recommended options')
    
    args = parser.parse_args()
    
    installer = LLMBuilderInstaller()
    
    # Determine installation options
    options = []
    if args.all:
        options = ['gpu', 'dev', 'docs', 'monitoring']
    elif args.auto:
        recommendations = installer.recommend_installation_options()
        options = [opt for opt, recommended in recommendations.items() if recommended]
    else:
        if args.gpu:
            options.append('gpu')
        if args.dev:
            options.append('dev')
        if args.docs:
            options.append('docs')
        if args.monitoring:
            options.append('monitoring')
    
    # Run installation
    success = installer.run_full_installation(options)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()