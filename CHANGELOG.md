# Changelog

All notable changes to LLMBuilder will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-09

### Added
- Initial Python package structure with pyproject.toml
- CLI framework using Click with main command group
- Configuration management system with hierarchical config loading
- Project templates for initialization
- Package installation support (`pip install llmbuilder`)
- CLI entry point (`llmbuilder` command)
- Comprehensive logging system with loguru
- Type checking support with py.typed marker
- Development tools configuration (black, isort, mypy, pytest)

### Changed
- Restructured project as proper Python package
- Moved existing functionality to `llmbuilder.core` module
- Enhanced configuration system with YAML/JSON support

### Infrastructure
- Added pyproject.toml for modern Python packaging
- Created package directory structure
- Set up CLI framework foundation
- Implemented configuration management
- Added development and testing infrastructure