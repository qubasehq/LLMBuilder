# LLMBuilder Comprehensive Test Suite

This directory contains a comprehensive testing framework for LLMBuilder, designed to ensure code quality, reliability, and performance across all components.

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_cli_main.py    # Main CLI interface tests
│   ├── test_cli_init.py    # Project initialization tests
│   ├── test_cli_data.py    # Data processing CLI tests
│   └── test_cli_config.py  # Configuration management tests
├── integration/            # Integration tests for workflows
│   ├── test_full_pipeline.py      # End-to-end pipeline tests
│   └── test_cli_workflows.py      # CLI workflow tests
├── performance/            # Performance and benchmark tests
│   └── test_benchmarks.py         # Performance benchmarks
├── fixtures/               # Test data and utilities
│   ├── test_data.py       # Test data generators
│   └── __init__.py
├── conftest.py            # Pytest configuration and fixtures
├── run_comprehensive_tests.py     # Test runner script
├── validate_test_suite.py         # Test suite validation
└── README.md              # This file
```

## Test Categories

### Unit Tests (`tests/unit/`)
- Test individual components in isolation
- Fast execution (< 1 second per test)
- Mock external dependencies
- High code coverage target (95%+)

### Integration Tests (`tests/integration/`)
- Test component interactions
- End-to-end workflow validation
- Real file system operations
- Moderate execution time (< 30 seconds per test)

### Performance Tests (`tests/performance/`)
- Benchmark critical operations
- Memory usage validation
- Throughput measurements
- Concurrency testing

## Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.cli` - CLI-specific tests
- `@pytest.mark.core` - Core functionality tests
- `@pytest.mark.gpu` - Tests requiring GPU
- `@pytest.mark.ocr` - Tests requiring OCR/Tesseract

## Running Tests

### Quick Test Run
```bash
# Run all fast tests
pytest tests/ -m "not slow"

# Run specific test category
pytest tests/unit/ -m unit
pytest tests/integration/ -m integration
```

### Comprehensive Test Suite
```bash
# Run all tests with detailed reporting
python tests/run_comprehensive_tests.py

# Run specific test suite
python tests/run_comprehensive_tests.py --suite unit
python tests/run_comprehensive_tests.py --suite integration
python tests/run_comprehensive_tests.py --suite performance

# Include slow tests
python tests/run_comprehensive_tests.py --include-slow
```

### Coverage Analysis
```bash
# Run tests with coverage
python tests/run_comprehensive_tests.py --suite coverage

# Generate HTML coverage report
pytest tests/ --cov=llmbuilder --cov-report=html
```

## Test Configuration

### pytest.ini
Main pytest configuration with:
- Test discovery patterns
- Marker definitions
- Coverage settings
- Output formatting

### conftest.py
Shared fixtures and configuration:
- Project root setup
- Temporary workspace creation
- Mock data generators
- Test environment setup

## Continuous Integration

### GitHub Actions
Automated testing on:
- Multiple Python versions (3.8, 3.9, 3.10, 3.11)
- Multiple operating systems (Ubuntu, Windows, macOS)
- Pull requests and main branch pushes
- Scheduled daily runs

### Test Jobs
- **Unit Tests**: Fast feedback on basic functionality
- **Integration Tests**: Workflow validation
- **CLI Tests**: Cross-platform CLI testing
- **Performance Tests**: Benchmark tracking
- **Coverage Analysis**: Code coverage reporting

## Test Data and Fixtures

### Test Data Generators
Located in `tests/fixtures/test_data.py`:
- Sample datasets (JSONL, conversation, multilingual)
- Mock configurations
- Test project structures
- Noisy data for cleaning tests

### Fixtures
Common test fixtures in `conftest.py`:
- `temp_workspace`: Temporary project workspace
- `sample_config`: Test configuration data
- `sample_documents`: Test document collection
- `mock_model_checkpoint`: Mock model data

## Writing Tests

### Unit Test Example
```python
import pytest
from click.testing import CliRunner
from llmbuilder.cli.main import cli

class TestMyFeature:
    def setup_method(self):
        self.runner = CliRunner()
    
    def test_basic_functionality(self):
        result = self.runner.invoke(cli, ['my-command'])
        assert result.exit_code == 0
        assert 'expected output' in result.output
    
    @pytest.mark.unit
    def test_with_mock(self, mocker):
        mock_func = mocker.patch('module.function')
        mock_func.return_value = 'mocked'
        # Test implementation
```

### Integration Test Example
```python
@pytest.mark.integration
def test_full_workflow(temp_workspace):
    # Setup test data
    data_dir = temp_workspace / 'data'
    data_dir.mkdir()
    
    # Run workflow
    runner = CliRunner()
    result = runner.invoke(cli, ['init', str(temp_workspace)])
    assert result.exit_code == 0
    
    # Verify results
    assert (temp_workspace / 'config.json').exists()
```

## Performance Testing

### Benchmarks
Performance tests measure:
- Data processing throughput
- Memory usage patterns
- CLI startup time
- Concurrent operation handling

### Thresholds
Performance assertions ensure:
- Processing speed > 0.1 MB/s
- Memory usage < 5x file size
- CLI startup < 2 seconds
- No memory leaks in repeated operations

## Test Validation

### Validation Script
Run `python tests/validate_test_suite.py` to check:
- Test discovery works correctly
- All required dependencies are available
- Test structure is complete
- Markers are properly configured
- CLI commands are accessible

### Health Checks
Regular validation ensures:
- All tests can be discovered
- No import errors
- Configuration is valid
- Dependencies are satisfied

## Best Practices

### Test Organization
- One test file per module/component
- Clear test class and method names
- Logical grouping of related tests
- Proper use of fixtures and markers

### Test Quality
- Fast unit tests (< 1s each)
- Reliable integration tests
- Meaningful assertions
- Good error messages
- Proper cleanup

### Maintenance
- Regular test review and updates
- Performance regression monitoring
- Coverage gap analysis
- Flaky test identification and fixing

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure project is installed in development mode (`pip install -e .`)
2. **Missing Dependencies**: Install test dependencies (`pip install -e .[dev]`)
3. **Permission Errors**: Check file system permissions for temp directories
4. **Timeout Issues**: Increase timeout for slow tests or mark as slow

### Debug Mode
```bash
# Run with verbose output
pytest tests/ -v -s

# Run specific test with debugging
pytest tests/unit/test_cli_main.py::TestMainCLI::test_cli_help -v -s --pdb
```

### Test Isolation
- Each test should be independent
- Use temporary directories for file operations
- Clean up resources in teardown
- Mock external dependencies

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure good test coverage (>90%)
3. Add appropriate markers
4. Update documentation
5. Run full test suite before submitting

### Test Review Checklist
- [ ] Tests cover happy path and edge cases
- [ ] Appropriate markers are used
- [ ] Tests are fast and reliable
- [ ] Good test names and documentation
- [ ] Proper fixture usage
- [ ] No hardcoded paths or values
- [ ] Cleanup is handled properly