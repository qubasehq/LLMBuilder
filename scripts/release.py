#!/usr/bin/env python3
"""
Release management script for LLMBuilder package.
"""

import sys
import subprocess
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import re
import getpass
from datetime import datetime


class ReleaseManager:
    """Comprehensive release management for LLMBuilder."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.release_log = []
        self.dry_run = False
    
    def _log(self, message: str, level: str = "INFO"):
        """Log release message."""
        log_entry = f"[{level}] {message}"
        print(log_entry)
        self.release_log.append(log_entry)
    
    def _run_command(self, cmd: List[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
        """Run command with logging."""
        cmd_str = ' '.join(cmd)
        self._log(f"Running: {cmd_str}")
        
        if self.dry_run:
            self._log("DRY RUN: Command not executed")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        
        result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
        
        if check and result.returncode != 0:
            self._log(f"Command failed: {result.stderr}", "ERROR")
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
        
        return result
    
    def get_current_version(self) -> str:
        """Get current package version."""
        init_file = self.project_root / 'llmbuilder' / '__init__.py'
        content = init_file.read_text()
        
        version_match = re.search(r'__version__ = ["\']([^"\']*)["\']', content)
        if version_match:
            return version_match.group(1)
        else:
            raise ValueError("Could not find version in __init__.py")
    
    def validate_version_format(self, version: str) -> bool:
        """Validate semantic version format."""
        pattern = r'^\d+\.\d+\.\d+(?:-[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*)?(?:\+[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*)?$'
        return bool(re.match(pattern, version))
    
    def update_version(self, new_version: str) -> bool:
        """Update version in all relevant files."""
        if not self.validate_version_format(new_version):
            self._log(f"Invalid version format: {new_version}", "ERROR")
            return False
        
        self._log(f"Updating version to {new_version}")
        
        # Update __init__.py
        init_file = self.project_root / 'llmbuilder' / '__init__.py'
        content = init_file.read_text()
        
        new_content = re.sub(
            r'__version__ = ["\'][^"\']*["\']',
            f'__version__ = "{new_version}"',
            content
        )
        
        if not self.dry_run:
            init_file.write_text(new_content)
        
        self._log(f"Version updated in {init_file}")
        
        # Update pyproject.toml if it has a version field
        pyproject_file = self.project_root / 'pyproject.toml'
        if pyproject_file.exists():
            content = pyproject_file.read_text()
            if 'version = ' in content:
                new_content = re.sub(
                    r'version = ["\'][^"\']*["\']',
                    f'version = "{new_version}"',
                    content
                )
                if not self.dry_run:
                    pyproject_file.write_text(new_content)
                self._log(f"Version updated in {pyproject_file}")
        
        return True
    
    def check_git_status(self) -> bool:
        """Check git repository status."""
        self._log("Checking git status...")
        
        try:
            # Check if we're in a git repository
            result = self._run_command(['git', 'status', '--porcelain'])
            
            if result.stdout.strip():
                self._log("Working directory has uncommitted changes:", "WARNING")
                for line in result.stdout.strip().split('\n'):
                    self._log(f"  {line}")
                return False
            
            # Check current branch
            result = self._run_command(['git', 'branch', '--show-current'])
            current_branch = result.stdout.strip()
            
            if current_branch != 'main' and current_branch != 'master':
                self._log(f"Not on main/master branch: {current_branch}", "WARNING")
                return False
            
            self._log("Git status clean")
            return True
            
        except subprocess.CalledProcessError:
            self._log("Not in a git repository or git not available", "WARNING")
            return False
    
    def run_tests(self) -> bool:
        """Run comprehensive test suite."""
        self._log("Running test suite...")
        
        try:
            # Run comprehensive tests
            test_script = self.project_root / 'tests' / 'run_comprehensive_tests.py'
            if test_script.exists():
                result = self._run_command([
                    sys.executable, str(test_script), '--suite', 'regression'
                ])
            else:
                # Fallback to basic pytest
                result = self._run_command([
                    sys.executable, '-m', 'pytest', 'tests/', '-x', '--tb=short'
                ])
            
            self._log("All tests passed")
            return True
            
        except subprocess.CalledProcessError as e:
            self._log(f"Tests failed: {e}", "ERROR")
            return False
    
    def build_package(self) -> bool:
        """Build package distributions."""
        self._log("Building package...")
        
        try:
            # Use build script if available
            build_script = self.project_root / 'scripts' / 'build_package.py'
            if build_script.exists():
                result = self._run_command([sys.executable, str(build_script)])
            else:
                # Fallback to direct build
                result = self._run_command([sys.executable, '-m', 'build'])
            
            self._log("Package built successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self._log(f"Build failed: {e}", "ERROR")
            return False
    
    def test_distribution(self) -> bool:
        """Test built distributions."""
        self._log("Testing distributions...")
        
        try:
            # Use distribution test script if available
            test_script = self.project_root / 'scripts' / 'test_distribution.py'
            if test_script.exists():
                result = self._run_command([sys.executable, str(test_script)])
            else:
                # Basic validation with twine
                result = self._run_command([
                    sys.executable, '-m', 'twine', 'check', 'dist/*'
                ])
            
            self._log("Distribution tests passed")
            return True
            
        except subprocess.CalledProcessError as e:
            self._log(f"Distribution tests failed: {e}", "ERROR")
            return False
    
    def create_git_tag(self, version: str) -> bool:
        """Create git tag for release."""
        self._log(f"Creating git tag v{version}")
        
        try:
            # Create annotated tag
            tag_message = f"Release version {version}"
            result = self._run_command([
                'git', 'tag', '-a', f'v{version}', '-m', tag_message
            ])
            
            self._log(f"Git tag v{version} created")
            return True
            
        except subprocess.CalledProcessError as e:
            self._log(f"Failed to create git tag: {e}", "ERROR")
            return False
    
    def push_to_git(self, version: str) -> bool:
        """Push changes and tags to git repository."""
        self._log("Pushing to git repository...")
        
        try:
            # Push changes
            result = self._run_command(['git', 'push'])
            
            # Push tags
            result = self._run_command(['git', 'push', '--tags'])
            
            self._log("Changes pushed to git repository")
            return True
            
        except subprocess.CalledProcessError as e:
            self._log(f"Failed to push to git: {e}", "ERROR")
            return False
    
    def upload_to_pypi(self, repository: str = "pypi") -> bool:
        """Upload package to PyPI."""
        self._log(f"Uploading to {repository}...")
        
        try:
            # Check if twine is available
            result = self._run_command([sys.executable, '-m', 'twine', '--version'])
            
            # Upload to repository
            upload_cmd = [sys.executable, '-m', 'twine', 'upload']
            
            if repository == "testpypi":
                upload_cmd.extend(['--repository', 'testpypi'])
            
            upload_cmd.append('dist/*')
            
            result = self._run_command(upload_cmd)
            
            self._log(f"Package uploaded to {repository}")
            return True
            
        except subprocess.CalledProcessError as e:
            self._log(f"Upload to {repository} failed: {e}", "ERROR")
            return False
    
    def update_changelog(self, version: str) -> bool:
        """Update changelog with release information."""
        changelog_file = self.project_root / 'CHANGELOG.md'
        
        if not changelog_file.exists():
            self._log("No CHANGELOG.md found, creating one")
            changelog_content = f"# Changelog\n\n## [{version}] - {datetime.now().strftime('%Y-%m-%d')}\n\n### Added\n- Initial release\n\n"
        else:
            content = changelog_file.read_text()
            
            # Add new version entry at the top
            new_entry = f"## [{version}] - {datetime.now().strftime('%Y-%m-%d')}\n\n### Added\n- Release version {version}\n\n"
            
            # Insert after the first line (# Changelog)
            lines = content.split('\n')
            if lines and lines[0].startswith('# Changelog'):
                lines.insert(2, new_entry)
                changelog_content = '\n'.join(lines)
            else:
                changelog_content = f"# Changelog\n\n{new_entry}{content}"
        
        if not self.dry_run:
            changelog_file.write_text(changelog_content)
        
        self._log(f"Changelog updated for version {version}")
        return True
    
    def commit_version_changes(self, version: str) -> bool:
        """Commit version update changes."""
        self._log("Committing version changes...")
        
        try:
            # Add changed files
            files_to_add = [
                'llmbuilder/__init__.py',
                'pyproject.toml',
                'CHANGELOG.md'
            ]
            
            for file_path in files_to_add:
                full_path = self.project_root / file_path
                if full_path.exists():
                    result = self._run_command(['git', 'add', str(full_path)])
            
            # Commit changes
            commit_message = f"Bump version to {version}"
            result = self._run_command(['git', 'commit', '-m', commit_message])
            
            self._log("Version changes committed")
            return True
            
        except subprocess.CalledProcessError as e:
            self._log(f"Failed to commit changes: {e}", "ERROR")
            return False
    
    def create_release_notes(self, version: str) -> str:
        """Create release notes."""
        notes = f"""# LLMBuilder {version}

## What's New

This release includes improvements and bug fixes for the LLMBuilder package.

## Installation

```bash
pip install llmbuilder=={version}
```

## Optional Dependencies

```bash
# For GPU support
pip install llmbuilder[gpu]=={version}

# For development
pip install llmbuilder[dev]=={version}

# All optional dependencies
pip install llmbuilder[all]=={version}
```

## Documentation

- [Documentation](https://llmbuilder.readthedocs.io)
- [GitHub Repository](https://github.com/qubase/llmbuilder)

## Support

If you encounter any issues, please report them on our [GitHub Issues](https://github.com/qubase/llmbuilder/issues) page.
"""
        
        # Save release notes
        notes_file = self.project_root / f'release_notes_{version}.md'
        if not self.dry_run:
            notes_file.write_text(notes)
        
        self._log(f"Release notes created: {notes_file}")
        return notes
    
    def perform_release(
        self, 
        version: str, 
        test_upload: bool = False,
        skip_tests: bool = False,
        skip_git: bool = False
    ) -> bool:
        """Perform complete release process."""
        self._log(f"🚀 Starting release process for version {version}")
        self._log("=" * 60)
        
        current_version = self.get_current_version()
        self._log(f"Current version: {current_version}")
        self._log(f"Target version: {version}")
        
        if version == current_version:
            self._log("Version unchanged, skipping release", "WARNING")
            return False
        
        # Pre-release checks
        if not skip_git and not self.check_git_status():
            self._log("Git status check failed", "ERROR")
            return False
        
        # Update version
        if not self.update_version(version):
            return False
        
        # Update changelog
        if not self.update_changelog(version):
            return False
        
        # Commit version changes
        if not skip_git and not self.commit_version_changes(version):
            return False
        
        # Run tests
        if not skip_tests and not self.run_tests():
            return False
        
        # Build package
        if not self.build_package():
            return False
        
        # Test distributions
        if not self.test_distribution():
            return False
        
        # Create git tag
        if not skip_git and not self.create_git_tag(version):
            return False
        
        # Push to git
        if not skip_git and not self.push_to_git(version):
            return False
        
        # Upload to PyPI
        repository = "testpypi" if test_upload else "pypi"
        if not self.upload_to_pypi(repository):
            return False
        
        # Create release notes
        self.create_release_notes(version)
        
        # Success
        self._log("=" * 60)
        self._log(f"🎉 Release {version} completed successfully!")
        
        if test_upload:
            self._log("Package uploaded to TestPyPI")
            self._log("Test installation: pip install -i https://test.pypi.org/simple/ llmbuilder")
        else:
            self._log("Package uploaded to PyPI")
            self._log(f"Installation: pip install llmbuilder=={version}")
        
        return True


def main():
    """Main release function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Release LLMBuilder package')
    parser.add_argument('version', help='Release version (e.g., 1.2.0)')
    parser.add_argument('--test', action='store_true', help='Upload to TestPyPI instead of PyPI')
    parser.add_argument('--skip-tests', action='store_true', help='Skip running tests')
    parser.add_argument('--skip-git', action='store_true', help='Skip git operations')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without executing')
    
    args = parser.parse_args()
    
    manager = ReleaseManager()
    manager.dry_run = args.dry_run
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
    
    try:
        success = manager.perform_release(
            version=args.version,
            test_upload=args.test,
            skip_tests=args.skip_tests,
            skip_git=args.skip_git
        )
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nRelease interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Release error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()