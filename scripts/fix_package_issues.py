#!/usr/bin/env python3
"""
Fix package build issues for LLMBuilder.
"""

import sys
import re
from pathlib import Path
from typing import List, Tuple


def fix_import_issues(project_root: Path) -> List[str]:
    """Fix import issues in test files."""
    fixes = []
    
    # Find all test files
    test_files = list(project_root.glob('tests/**/*.py'))
    
    for test_file in test_files:
        if test_file.name == '__init__.py':
            continue
            
        try:
            content = test_file.read_text(encoding='utf-8')
            original_content = content
            
            # Fix imports from 'tools' to 'llmbuilder.core.tools'
            content = re.sub(
                r'from tools\.([a-zA-Z_][a-zA-Z0-9_]*) import',
                r'from llmbuilder.core.tools.\1 import',
                content
            )
            
            # Fix imports from 'model' to 'llmbuilder.core.model'
            content = re.sub(
                r'from model\.([a-zA-Z_][a-zA-Z0-9_]*) import',
                r'from llmbuilder.core.model.\1 import',
                content
            )
            
            # Fix imports from 'eval' to 'llmbuilder.core.eval'
            content = re.sub(
                r'from eval\.([a-zA-Z_][a-zA-Z0-9_]*) import',
                r'from llmbuilder.core.eval.\1 import',
                content
            )
            
            # Fix imports from 'training' to 'llmbuilder.core.training'
            content = re.sub(
                r'from training\.([a-zA-Z_][a-zA-Z0-9_]*) import',
                r'from llmbuilder.core.training.\1 import',
                content
            )
            
            # Fix imports from 'finetune' to 'llmbuilder.core.finetune'
            content = re.sub(
                r'from finetune\.([a-zA-Z_][a-zA-Z0-9_]*) import',
                r'from llmbuilder.core.finetune.\1 import',
                content
            )
            
            # Fix imports from 'data' to 'llmbuilder.core.data'
            content = re.sub(
                r'from data\.([a-zA-Z_][a-zA-Z0-9_]*) import',
                r'from llmbuilder.core.data.\1 import',
                content
            )
            
            if content != original_content:
                test_file.write_text(content, encoding='utf-8')
                fixes.append(f"Fixed imports in {test_file}")
                
        except Exception as e:
            print(f"Error processing {test_file}: {e}")
    
    return fixes


def add_deprecation_warnings(project_root: Path) -> List[str]:
    """Add deprecation warnings to old module imports."""
    fixes = []
    
    # Add deprecation warning to tools/__init__.py
    tools_init = project_root / 'tools' / '__init__.py'
    if tools_init.exists():
        content = tools_init.read_text(encoding='utf-8')
        
        if 'warnings.warn' not in content:
            deprecation_warning = '''
import warnings

warnings.warn(
    "Importing from 'tools' module is deprecated. "
    "Please use 'from llmbuilder.core.tools import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location for backward compatibility
try:
    from llmbuilder.core.tools import *
except ImportError:
    pass  # Graceful fallback during development
'''
            
            # Add at the beginning of the file
            new_content = deprecation_warning + '\n' + content
            tools_init.write_text(new_content, encoding='utf-8')
            fixes.append(f"Added deprecation warning to {tools_init}")
    
    return fixes


def create_installation_guide(project_root: Path) -> str:
    """Create installation guide."""
    guide_content = """# LLMBuilder Installation Guide

## Quick Installation

### From PyPI (Recommended)

```bash
pip install llmbuilder
```

### With Optional Dependencies

```bash
# For GPU support
pip install llmbuilder[gpu]

# For development
pip install llmbuilder[dev]

# For documentation building
pip install llmbuilder[docs]

# For monitoring and tracking
pip install llmbuilder[monitoring]

# All optional dependencies
pip install llmbuilder[all]
```

## System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM (8GB+ recommended)
- **Storage**: At least 2GB free space

## Optional System Dependencies

### For OCR Support (Optional)

#### Ubuntu/Debian
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

#### macOS
```bash
brew install tesseract
```

#### Windows
Download from: https://github.com/UB-Mannheim/tesseract/wiki

### For GPU Support (Optional)

Ensure you have CUDA-compatible GPU drivers installed:
- NVIDIA CUDA 11.8 or higher
- cuDNN 8.0 or higher

## Installation Methods

### 1. Standard Installation

```bash
pip install llmbuilder
```

This installs the core package with essential dependencies.

### 2. Development Installation

For contributors and developers:

```bash
git clone https://github.com/qubase/llmbuilder.git
cd llmbuilder
pip install -e .[dev]
```

### 3. Custom Installation Script

Use the provided installation script for automatic dependency detection:

```bash
python scripts/install.py --auto
```

Options:
- `--auto`: Auto-detect and install recommended dependencies
- `--gpu`: Include GPU dependencies
- `--dev`: Include development dependencies
- `--all`: Install all optional dependencies

## Verification

After installation, verify that LLMBuilder is working:

```bash
# Check version
llmbuilder --version

# Run basic help
llmbuilder --help

# Initialize a test project
llmbuilder init test-project
```

## Troubleshooting

### Common Issues

#### Import Errors
If you encounter import errors, ensure you're using the correct Python environment:

```bash
python -c "import llmbuilder; print(llmbuilder.__version__)"
```

#### Permission Errors
On some systems, you might need to use `--user` flag:

```bash
pip install --user llmbuilder
```

#### GPU Dependencies
If GPU dependencies fail to install:

```bash
# Install CPU version first
pip install llmbuilder

# Then add GPU support separately
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install llmbuilder[gpu] --no-deps
```

### Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/qubase/llmbuilder/issues)
2. Run the verification script: `python scripts/verify_package.py`
3. Check system compatibility: `python scripts/install.py --auto`

## Uninstallation

To remove LLMBuilder:

```bash
pip uninstall llmbuilder
```

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade llmbuilder
```

Or use the built-in upgrade command:

```bash
llmbuilder upgrade
```
"""
    
    guide_file = project_root / 'INSTALLATION.md'
    guide_file.write_text(guide_content, encoding='utf-8')
    
    return str(guide_file)


def create_package_manifest(project_root: Path) -> str:
    """Create or update MANIFEST.in for package distribution."""
    manifest_content = """# LLMBuilder Package Manifest

# Include documentation
include README.md
include INSTALLATION.md
include CHANGELOG.md
include LICENSE
include CONTRIBUTING.md
include USAGE.md

# Include configuration files
include pyproject.toml
include pytest.ini
include MANIFEST.in

# Include templates and configs
recursive-include llmbuilder/templates *
recursive-include llmbuilder/configs *

# Include type information
include llmbuilder/py.typed

# Exclude development and build files
exclude .gitignore
exclude .pre-commit-config.yaml
recursive-exclude * __pycache__
recursive-exclude * *.py[co]
recursive-exclude * *.so
recursive-exclude * .DS_Store

# Exclude test files from distribution
recursive-exclude tests *
exclude run_tests.py

# Exclude development scripts
recursive-exclude scripts *
recursive-exclude debug_scripts *

# Exclude build artifacts
recursive-exclude build *
recursive-exclude dist *
recursive-exclude *.egg-info *

# Exclude IDE files
recursive-exclude .vscode *
recursive-exclude .idea *
recursive-exclude *.swp
recursive-exclude *.swo

# Exclude logs and temporary files
recursive-exclude logs *
recursive-exclude *.log
recursive-exclude *.tmp
recursive-exclude *.temp

# Exclude model files and data
recursive-exclude test_models *
recursive-exclude data *
exclude *.bin
exclude *.safetensors
exclude *.gguf
"""
    
    manifest_file = project_root / 'MANIFEST.in'
    manifest_file.write_text(manifest_content, encoding='utf-8')
    
    return str(manifest_file)


def main():
    """Main fix function."""
    project_root = Path(__file__).parent.parent
    
    print("🔧 Fixing LLMBuilder package issues...")
    print("=" * 50)
    
    # Fix import issues
    print("Fixing import issues...")
    import_fixes = fix_import_issues(project_root)
    for fix in import_fixes:
        print(f"  ✓ {fix}")
    
    # Add deprecation warnings
    print("\nAdding deprecation warnings...")
    deprecation_fixes = add_deprecation_warnings(project_root)
    for fix in deprecation_fixes:
        print(f"  ✓ {fix}")
    
    # Create installation guide
    print("\nCreating installation guide...")
    guide_file = create_installation_guide(project_root)
    print(f"  ✓ Created {guide_file}")
    
    # Create package manifest
    print("\nCreating package manifest...")
    manifest_file = create_package_manifest(project_root)
    print(f"  ✓ Created {manifest_file}")
    
    print("\n" + "=" * 50)
    print("✅ Package issues fixed!")
    print("\nNext steps:")
    print("1. Run tests: python -m pytest tests/ -x")
    print("2. Build package: python scripts/build_package.py")
    print("3. Verify package: python scripts/verify_package.py")


if __name__ == '__main__':
    main()