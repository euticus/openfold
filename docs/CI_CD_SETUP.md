# OpenFold++ CI/CD Setup Guide

This document describes the automated GPU benchmarking system for OpenFold++ using GitHub Actions.

## Overview

The CI/CD pipeline automatically benchmarks OpenFold++ performance on every commit, ensuring:
- **Quality**: TM-score ≥ 0.66 on CASP targets
- **Speed**: Runtime ≤ 5.5s for 300 AA sequences  
- **Efficiency**: Memory usage ≤ 8GB

## Pipeline Architecture

### Triggers
- **Push** to `main` or `develop` branches
- **Pull Requests** to `main` branch
- **Daily Schedule** at 2 AM UTC
- **Manual Dispatch** with configurable options

### Benchmark Configurations

#### Production Configuration
- TM-score threshold: ≥ 0.66
- Runtime threshold: ≤ 5.5s (300 AA)
- Memory threshold: ≤ 8GB
- Target: Production deployment readiness

#### Research Configuration  
- TM-score threshold: ≥ 0.70
- Runtime threshold: ≤ 4.0s (300 AA)
- Memory threshold: ≤ 6GB
- Target: Research-grade performance

## Setup Requirements

### Self-Hosted Runner

The pipeline requires a self-hosted GitHub Actions runner with:

```yaml
# Runner labels required
runs-on: [self-hosted, gpu, A100]
```

#### Hardware Requirements
- **GPU**: NVIDIA A100 (or equivalent)
- **VRAM**: ≥ 40GB
- **RAM**: ≥ 64GB
- **Storage**: ≥ 500GB SSD
- **CUDA**: 11.8+

#### Software Requirements
- **OS**: Ubuntu 20.04+ or CentOS 8+
- **Docker**: Latest version
- **NVIDIA Drivers**: 525.60.13+
- **CUDA Toolkit**: 11.8+

### Runner Setup

1. **Install GitHub Actions Runner**
```bash
# Download and configure runner
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz

# Configure with repository
./config.sh --url https://github.com/your-org/openfold --token YOUR_TOKEN --labels self-hosted,gpu,A100

# Install as service
sudo ./svc.sh install
sudo ./svc.sh start
```

2. **Install CUDA and Dependencies**
```bash
# Install CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Install Python and PyTorch
conda create -n openfold python=3.9
conda activate openfold
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. **Configure Environment**
```bash
# Add to ~/.bashrc
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Benchmark Components

### CASP Benchmark
Tests structure prediction quality on CASP targets:
- **Targets**: 5 representative CASP13-14 sequences
- **Metric**: Mean TM-score across all targets
- **Threshold**: ≥ 0.66 (production) or ≥ 0.70 (research)
- **Timeout**: 30 minutes

### Performance Benchmark  
Tests runtime and memory efficiency:
- **Sequences**: 100, 200, 300 AA test cases
- **Runtime Metric**: Time for 300 AA sequence
- **Memory Metric**: Peak GPU memory usage
- **Thresholds**: ≤ 5.5s runtime, ≤ 8GB memory

### System Monitoring
Collects comprehensive system information:
- GPU model and memory
- CUDA version
- PyTorch version
- CPU and RAM specs

## Workflow Configuration

### Environment Variables
```yaml
env:
  PYTHON_VERSION: "3.9"
  CUDA_VERSION: "11.8" 
  PYTORCH_VERSION: "2.0.1"
```

### Secrets Required
- `SLACK_WEBHOOK`: For failure notifications (optional)

### Artifacts
- Benchmark results (JSON)
- Detailed logs
- Performance reports
- System information

## Usage

### Automatic Triggers
The pipeline runs automatically on:
- Every push to main/develop
- Every pull request to main
- Daily at 2 AM UTC

### Manual Execution
Trigger manually with options:
```bash
# Via GitHub UI: Actions → GPU Benchmark → Run workflow
# Options:
# - quick: Fast benchmark (2 targets)
# - full: Complete benchmark (all targets)
# - casp-only: Only CASP quality test
# - performance-only: Only speed/memory test
```

### Local Testing
Run benchmarks locally:
```bash
# Quick test
python openfoldpp/scripts/evaluation/gpu_ci_benchmark.py --quick

# Full benchmark
python openfoldpp/scripts/evaluation/gpu_ci_benchmark.py \
  --config production \
  --casp-benchmark \
  --performance-benchmark

# Custom thresholds
python openfoldpp/scripts/evaluation/gpu_ci_benchmark.py \
  --tm-threshold 0.70 \
  --runtime-threshold 4.0 \
  --memory-threshold 6.0
```

## Results and Reporting

### Pass/Fail Criteria
The pipeline **FAILS** if ANY target is not met:
- TM-score below threshold
- Runtime above threshold  
- Memory usage above threshold

### Automated Reporting
- **PR Comments**: Results posted to pull requests
- **Slack Notifications**: Alerts on main branch failures
- **Badges**: Dynamic badges showing current status
- **Artifacts**: Detailed reports and logs

### Result Interpretation

#### ✅ PASS Example
```
TM-Score: 0.685 (≥0.66) ✅
Runtime: 4.8s (≤5.5s) ✅  
Memory: 7.2GB (≤8GB) ✅
Overall: ✅ PASS
```

#### ❌ FAIL Example
```
TM-Score: 0.642 (≥0.66) ❌
Runtime: 6.2s (≤5.5s) ❌
Memory: 7.1GB (≤8GB) ✅
Overall: ❌ FAIL
```

## Troubleshooting

### Common Issues

#### GPU Not Available
```bash
# Check GPU status
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Fix: Restart CUDA services
sudo systemctl restart nvidia-persistenced
```

#### Out of Memory
```bash
# Check GPU memory
nvidia-smi

# Fix: Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

#### Runner Offline
```bash
# Check runner status
sudo ./svc.sh status

# Restart runner
sudo ./svc.sh stop
sudo ./svc.sh start
```

### Performance Issues

#### Slow Benchmarks
- Check GPU utilization with `nvidia-smi`
- Verify CUDA version compatibility
- Monitor CPU and memory usage
- Check for thermal throttling

#### Inconsistent Results
- Ensure GPU warmup is enabled
- Check for background processes
- Verify consistent CUDA/PyTorch versions
- Monitor system load

## Maintenance

### Regular Tasks
- **Weekly**: Check runner health and logs
- **Monthly**: Update dependencies and drivers
- **Quarterly**: Review benchmark thresholds

### Updates
- **CUDA**: Test compatibility before updating
- **PyTorch**: Verify benchmark consistency
- **Dependencies**: Update gradually with testing

### Monitoring
- **Runner Uptime**: Monitor via GitHub Actions
- **Performance Trends**: Track benchmark results over time
- **Resource Usage**: Monitor GPU/CPU/memory trends

## Security Considerations

### Runner Security
- Isolate runner in dedicated environment
- Regular security updates
- Monitor access logs
- Use dedicated service account

### Secrets Management
- Store sensitive data in GitHub Secrets
- Rotate tokens regularly
- Limit secret access scope
- Audit secret usage

## Integration with Development

### Branch Protection
Configure branch protection rules:
```yaml
# .github/branch-protection.yml
protection_rules:
  main:
    required_status_checks:
      - "GPU Performance Benchmark (production)"
    enforce_admins: true
    dismiss_stale_reviews: true
```

### Development Workflow
1. **Feature Development**: Work on feature branches
2. **Local Testing**: Run quick benchmarks locally
3. **Pull Request**: Automatic benchmark on PR
4. **Review**: Check benchmark results before merge
5. **Merge**: Full benchmark on main branch

---

*For questions or issues, contact the OpenFold++ development team.*
