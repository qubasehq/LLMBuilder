# Pull Request

## Description
Brief description of the changes made in this pull request.

## Type of Change
Please select the relevant option:

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🔧 Refactoring (no functional changes)
- [ ] ⚡ Performance improvement
- [ ] 🧪 Test improvements

## Related Issues
Closes #(issue number)
Fixes #(issue number)
Related to #(issue number)

## Changes Made
Detailed list of changes:

- [ ] Added/modified component X
- [ ] Updated configuration for Y
- [ ] Fixed bug in Z
- [ ] Added tests for new functionality
- [ ] Updated documentation

## Testing
### Test Coverage
- [ ] Unit tests pass (`python run_tests.py --unit`)
- [ ] Integration tests pass (`python run_tests.py --integration`)
- [ ] Performance tests pass (if applicable)
- [ ] New tests added for new functionality
- [ ] All existing tests still pass

### Manual Testing
Describe the manual testing performed:
- [ ] Tested on Linux/macOS
- [ ] Tested on Windows
- [ ] Tested with different Python versions
- [ ] Tested with different configurations
- [ ] Tested edge cases

### Test Results
```
# Paste relevant test output here
```

## Performance Impact
If applicable, describe the performance impact:

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Processing Speed | X files/sec | Y files/sec | +Z% |
| Memory Usage | X MB | Y MB | ±Z MB |
| Test Duration | X seconds | Y seconds | ±Z seconds |

## Documentation
- [ ] Code is self-documenting with clear variable names
- [ ] Complex logic is commented
- [ ] Docstrings added/updated for public functions
- [ ] README.md updated (if needed)
- [ ] USAGE.md updated (if needed)
- [ ] Examples added (if needed)
- [ ] Configuration documentation updated (if needed)

## Code Quality
- [ ] Code follows the project's style guidelines
- [ ] Code has been formatted with `black`
- [ ] Imports have been sorted with `isort`
- [ ] Code passes `flake8` linting
- [ ] Type hints added where appropriate
- [ ] No new warnings introduced

## Backward Compatibility
- [ ] Changes are backward compatible
- [ ] If breaking changes, they are clearly documented
- [ ] Migration guide provided (if needed)
- [ ] Deprecation warnings added (if applicable)

## Security
- [ ] No sensitive information exposed
- [ ] Input validation added where needed
- [ ] No new security vulnerabilities introduced
- [ ] Dependencies are from trusted sources

## Deployment
- [ ] Changes work in development environment
- [ ] Changes work in production-like environment
- [ ] No additional deployment steps required
- [ ] Configuration changes documented

## Screenshots/Examples
If applicable, add screenshots or examples of the new functionality:

```python
# Example usage of new feature
result = new_feature.process(input_data)
print(result)
```

## Checklist
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## Additional Notes
Any additional information that reviewers should know:

- Special considerations for review
- Known limitations or trade-offs
- Future improvements planned
- Dependencies on other PRs or issues

---

**For Reviewers:**
- [ ] Code review completed
- [ ] Functionality tested
- [ ] Documentation reviewed
- [ ] Performance impact assessed (if applicable)
- [ ] Security implications considered