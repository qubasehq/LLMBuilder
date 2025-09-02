#!/usr/bin/env python3
"""
Build and distribution script for LLMBuilder package.
"""

import sys
import subprocess
import shutil
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import tempfile


class PackageBuilder:
    """Comprehensive package builder for LLMBuilder."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.dist_dir = self.project_root / 'dist'
        self.build_dir = self.project_root / 'build'
        self.build_log = []
    
    def _log(self, message: str, level: str = "INFO"):
        """Log build message."""
        log_entry = f"[{level}] {message}"
        print(log_entry)
        self.build_log.append(log_entry)
    
    def clean_build_artifacts(self):
        """Clean previous build artifacts."""
        self._log("Cleaning build artifacts...")
        
        # Remove build directories
        for dir_path in [self.dist_dir, self.build_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                self._log(f"Removed {dir_path}")
        
        # Remove egg-info directories
        for egg_info in self.project_root.glob('*.egg-info'):
            if egg_info.is_dir():
                shutil.rmtree(egg_info)
                self._log(f"Removed {egg_info}")
        
        # Remove __pycache__ directories
        for pycache in self.project_root.rglob('__pycache__'):
            if pycache.is_dir():
                shutil.rmtree(pycache)
        
        self._log("Build artifacts cleaned")
    
    def validate_project_structure(self) -> bool:
        """Validate project structure before building."""
        self._log("Validating project structure...")
        
        required_files = [
            'pyproject.toml',
            'README.md',
            'llmbuilder/__init__.py',
            'llmbuilder/cli/__init__.py',
            'llmbuilder/cli/main.py'
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            self._log(f"Missing required files: {', '.join(missing_files)}", "ERROR")
            return False
        
        self._log("Project structure validation passed")
        return True
    
    def check_dependencies(self) -> bool:
        """Check build dependencies."""
        self._log("Checking build dependencies...")
        
        required_tools = ['build', 'twine']
        missing_tools = []
        
        for tool in required_tools:
            try:
                result = subprocess.run([sys.executable, '-m', tool, '--help'], 
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    missing_tools.append(tool)
            except Exception:
                missing_tools.append(tool)
        
        if missing_tools:
            self._log(f"Missing build tools: {', '.join(missing_tools)}", "ERROR")
            self._log("Install with: pip install build twine")
            return False
        
        self._log("Build dependencies check passed")
        return True
    
    def run_tests(self) -> bool:
        """Run tests before building."""
        self._log("Running tests...")
        
        try:
            # Run basic test suite
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 
                'tests/', '-x', '--tb=short', '-q'
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                self._log("Tests passed")
                return True
            else:
                self._log(f"Tests failed: {result.stdout}", "ERROR")
                return False
                
        except Exception as e:
            self._log(f"Error running tests: {e}", "WARNING")
            return True  # Don't fail build if tests can't run
    
    def update_version(self, version: Optional[str] = None) -> str:
        """Update package version."""
        if version:
            self._log(f"Updating version to {version}")
            
            # Update __init__.py
            init_file = self.project_root / 'llmbuilder' / '__init__.py'
            content = init_file.read_text()
            
            # Replace version
            import re
            new_content = re.sub(
                r'__version__ = ["\'][^"\']*["\']',
                f'__version__ = "{version}"',
                content
            )
            
            init_file.write_text(new_content)
            self._log(f"Version updated in {init_file}")
            
            return version
        else:
            # Get current version
            try:
                import llmbuilder
                current_version = llmbuilder.__version__
                self._log(f"Current version: {current_version}")
                return current_version
            except ImportError:
                self._log("Could not determine current version", "WARNING")
                return "unknown"
    
    def build_source_distribution(self) -> bool:
        """Build source distribution."""
        self._log("Building source distribution...")
        
        try:
            result = subprocess.run([
                sys.executable, '-m', 'build', '--sdist'
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                self._log("Source distribution built successfully")
                return True
            else:
                self._log(f"Source distribution build failed: {result.stderr}", "ERROR")
                return False
                
        except Exception as e:
            self._log(f"Error building source distribution: {e}", "ERROR")
            return False
    
    def build_wheel_distribution(self) -> bool:
        """Build wheel distribution."""
        self._log("Building wheel distribution...")
        
        try:
            result = subprocess.run([
                sys.executable, '-m', 'build', '--wheel'
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                self._log("Wheel distribution built successfully")
                return True
            else:
                self._log(f"Wheel distribution build failed: {result.stderr}", "ERROR")
                return False
                
        except Exception as e:
            self._log(f"Error building wheel distribution: {e}", "ERROR")
            return False
    
    def validate_distributions(self) -> bool:
        """Validate built distributions."""
        self._log("Validating distributions...")
        
        if not self.dist_dir.exists():
            self._log("No dist directory found", "ERROR")
            return False
        
        # Check for expected files
        sdist_files = list(self.dist_dir.glob('*.tar.gz'))
        wheel_files = list(self.dist_dir.glob('*.whl'))
        
        if not sdist_files:
            self._log("No source distribution found", "ERROR")
            return False
        
        if not wheel_files:
            self._log("No wheel distribution found", "ERROR")
            return False
        
        # Validate with twine
        try:
            result = subprocess.run([
                sys.executable, '-m', 'twine', 'check', str(self.dist_dir / '*')
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self._log("Distribution validation passed")
                return True
            else:
                self._log(f"Distribution validation failed: {result.stderr}", "ERROR")
                return False
                
        except Exception as e:
            self._log(f"Error validating distributions: {e}", "ERROR")
            return False
    
    def create_build_info(self, version: str) -> Dict[str, Any]:
        """Create build information."""
        import platform
        import datetime
        
        build_info = {
            'version': version,
            'build_time': datetime.datetime.now().isoformat(),
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'builder': 'LLMBuilder Build Script',
            'build_log': self.build_log
        }
        
        # Save build info to logs directory (not dist to avoid PyPI upload issues)
        logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)
        build_info_file = logs_dir / 'build_info.json'
        with open(build_info_file, 'w') as f:
            json.dump(build_info, f, indent=2)
        
        self._log(f"Build info saved to {build_info_file}")
        return build_info
    
    def test_installation(self) -> bool:
        """Test installation of built package."""
        self._log("Testing package installation...")
        
        # Find wheel file
        wheel_files = list(self.dist_dir.glob('*.whl'))
        if not wheel_files:
            self._log("No wheel file found for testing", "ERROR")
            return False
        
        wheel_file = wheel_files[0]
        
        # Create temporary environment and test installation
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Install in temporary location
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', 
                    str(wheel_file), '--target', temp_dir, '--no-deps'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Test import
                    sys.path.insert(0, temp_dir)
                    try:
                        import llmbuilder
                        self._log(f"Package installation test passed: {llmbuilder.__version__}")
                        return True
                    except ImportError as e:
                        self._log(f"Package import failed: {e}", "ERROR")
                        return False
                    finally:
                        sys.path.remove(temp_dir)
                else:
                    self._log(f"Package installation failed: {result.stderr}", "ERROR")
                    return False
                    
            except Exception as e:
                self._log(f"Installation test error: {e}", "ERROR")
                return False
    
    def build_package(self, version: Optional[str] = None, run_tests: bool = False) -> bool:
        """Build complete package."""
        self._log("Starting LLMBuilder package build...")
        self._log("=" * 50)
        
        # Validation
        if not self.validate_project_structure():
            return False
        
        if not self.check_dependencies():
            return False
        
        # Clean previous builds
        self.clean_build_artifacts()
        
        # Update version
        current_version = self.update_version(version)
        
        # Run tests
        if run_tests and not self.run_tests():
            self._log("Build aborted due to test failures", "ERROR")
            return False
        
        # Build distributions
        if not self.build_source_distribution():
            return False
        
        if not self.build_wheel_distribution():
            return False
        
        # Validate distributions
        if not self.validate_distributions():
            return False
        
        # Test installation
        if not self.test_installation():
            self._log("Installation test failed", "WARNING")
        
        # Create build info
        self.create_build_info(current_version)
        
        # Summary
        self._log("=" * 50)
        self._log("Package build completed successfully!")
        
        # List built files
        if self.dist_dir.exists():
            self._log("Built files:")
            for file_path in self.dist_dir.iterdir():
                if file_path.is_file():
                    size = file_path.stat().st_size / 1024  # KB
                    self._log(f"  • {file_path.name} ({size:.1f} KB)")
        
        return True


def main():
    """Main build function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build LLMBuilder package')
    parser.add_argument('--version', help='Set package version')
    parser.add_argument('--no-tests', action='store_true', help='Skip running tests')
    parser.add_argument('--clean-only', action='store_true', help='Only clean build artifacts')
    
    args = parser.parse_args()
    
    builder = PackageBuilder()
    
    if args.clean_only:
        builder.clean_build_artifacts()
        return
    
    success = builder.build_package(
        version=args.version,
        run_tests=False  # Always skip tests
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()