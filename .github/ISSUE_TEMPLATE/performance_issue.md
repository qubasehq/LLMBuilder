---
name: Performance Issue
about: Report performance problems or regressions
title: '[PERFORMANCE] '
labels: ['performance', 'needs-triage']
assignees: ''
---

## Performance Issue Description
A clear description of the performance problem you're experiencing.

## Performance Metrics
Provide specific performance measurements:

### Current Performance
- **Processing Speed**: [e.g. 2.5 files/second]
- **Memory Usage**: [e.g. 2.1GB peak memory]
- **Processing Time**: [e.g. 45 minutes for 1000 files]
- **CPU Usage**: [e.g. 85% average CPU utilization]

### Expected Performance
- **Expected Speed**: [e.g. 5+ files/second]
- **Expected Memory**: [e.g. <1GB memory usage]
- **Expected Time**: [e.g. <20 minutes for 1000 files]

## Test Environment
- **Hardware**: [e.g. Intel i7-9700K, 16GB RAM, SSD]
- **OS**: [e.g. Ubuntu 20.04]
- **Python Version**: [e.g. 3.9.7]
- **Dataset Size**: [e.g. 1000 files, 500MB total]
- **Configuration**: [e.g. config_gpu.json with batch_size=16]

## Reproduction Steps
Steps to reproduce the performance issue:
1. Prepare dataset with [specific characteristics]
2. Run command: `[specific command]`
3. Monitor performance with: `[monitoring tools]`
4. Observe: [specific performance problem]

## Performance Profiling
If you've done any profiling, include the results:

```
# Example profiling output
Function                     Calls    Time    Per Call
process_documents()          1000     45.2s   0.045s
extract_text()              1000     30.1s   0.030s
deduplicate()               1        8.7s    8.700s
```

## Comparison Data
If you have comparison data (e.g., previous versions, different configurations):

| Version/Config | Files/sec | Memory (GB) | Total Time |
|----------------|-----------|-------------|------------|
| v1.0.0         | 5.2       | 0.8         | 18m        |
| v1.1.0         | 2.5       | 2.1         | 45m        |

## System Resources
During the performance issue:
- **CPU Usage**: [percentage and pattern]
- **Memory Usage**: [peak and pattern]
- **Disk I/O**: [read/write patterns]
- **Network**: [if applicable]

## Potential Causes
If you have ideas about what might be causing the performance issue:
- Recent changes that might have introduced the regression
- Specific components that seem to be bottlenecks
- Configuration settings that might be suboptimal

## Workarounds
Any workarounds you've found:
- Configuration changes that improve performance
- Alternative approaches that work better

## Additional Context
- **Logs**: Include relevant log excerpts
- **Monitoring Data**: Screenshots of performance monitoring tools
- **Environment Details**: Any other relevant system information

## Checklist
- [ ] I have provided specific performance measurements
- [ ] I have included system specifications
- [ ] I have tested with different configurations (if applicable)
- [ ] I have checked for similar existing issues