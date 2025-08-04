# OpenFold++ Production Makefile
# 
# This Makefile provides commands for testing, benchmarking, and deploying
# the complete OpenFold++ optimization pipeline.

.PHONY: help install test benchmark benchmark-quick benchmark-full clean docs

# Default target
help:
	@echo "OpenFold++ Production Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install dependencies and setup environment"
	@echo "  make install-dev      Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run unit tests"
	@echo "  make test-quick       Run quick smoke tests"
	@echo "  make test-integration Run integration tests"
	@echo ""
	@echo "Benchmarking:"
	@echo "  make benchmark        Run complete production benchmark"
	@echo "  make benchmark-quick  Run quick benchmark (no external deps)"
	@echo "  make benchmark-full   Run full benchmark with real models"
	@echo "  make benchmark-phases Run all phase benchmarks (A, B, C, D)"
	@echo "  make benchmark-casp   Run CASP dataset evaluation with TM/RMSD"
	@echo ""
	@echo "Analysis:"
	@echo "  make profile          Profile performance bottlenecks"
	@echo "  make memory-test      Test memory usage patterns"
	@echo "  make accuracy-test    Test accuracy vs baseline"
	@echo ""
	@echo "Deployment:"
	@echo "  make package          Package for deployment"
	@echo "  make docker           Build Docker image"
	@echo "  make deploy-test      Test deployment readiness"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean            Clean build artifacts"
	@echo "  make docs             Generate documentation"
	@echo "  make format           Format code"

# Installation targets
install:
	@echo "🔧 Installing OpenFold++ dependencies..."
	cd openfoldpp && pip install -r requirements.txt
	cd openfoldpp && pip install -e .
	@echo "✅ Installation complete"

install-dev: install
	@echo "🔧 Installing development dependencies..."
	pip install pytest pytest-cov black flake8 mypy
	@echo "✅ Development setup complete"

# Testing targets
test: test-quick test-integration
	@echo "✅ All tests passed"

test-quick:
	@echo "🧪 Running quick smoke tests..."
	cd openfoldpp && python -m pytest tests/ -v --tb=short -x
	@echo "✅ Quick tests passed"

test-integration:
	@echo "🧪 Running integration tests..."
	cd openfoldpp && python scripts/evaluation/quick_phase_a_benchmark.py
	cd openfoldpp && python scripts/evaluation/quick_phase_b_benchmark.py
	cd openfoldpp && python scripts/evaluation/quick_phase_d_benchmark.py
	@echo "✅ Integration tests passed"

# Benchmarking targets
benchmark: benchmark-quick
	@echo "🏆 Production benchmark complete!"

benchmark-quick:
	@echo "🚀 Running quick production benchmark..."
	@echo "This runs without external dependencies (ESM, etc.)"
	@echo ""
	cd openfoldpp && python scripts/evaluation/production_benchmark.py --mode quick
	@echo ""
	@echo "📊 Quick benchmark results available in openfoldpp/reports/production/"

benchmark-full:
	@echo "🚀 Running full production benchmark..."
	@echo "This requires ESM models and full dependencies"
	@echo ""
	cd openfoldpp && python scripts/evaluation/production_benchmark.py --mode full
	@echo ""
	@echo "📊 Full benchmark results available in openfoldpp/reports/production/"

benchmark-phases:
	@echo "🚀 Running all phase benchmarks..."
	cd openfoldpp && python scripts/evaluation/quick_phase_a_benchmark.py
	cd openfoldpp && python scripts/evaluation/quick_phase_b_benchmark.py
	cd openfoldpp && python scripts/evaluation/distillation_report.py
	cd openfoldpp && python scripts/evaluation/quick_phase_d_benchmark.py
	@echo "📊 Phase benchmark results available in openfoldpp/reports/"

benchmark-casp:
	@echo "🧬 Running CASP dataset benchmark..."
	cd openfoldpp && python scripts/evaluation/simple_casp_benchmark.py
	@echo "📊 CASP benchmark results available in openfoldpp/reports/casp/"

# Analysis targets
profile:
	@echo "📈 Profiling OpenFold++ performance..."
	cd openfoldpp && python scripts/evaluation/profile_performance.py
	@echo "📊 Profile results available in openfoldpp/reports/profiling/"

memory-test:
	@echo "💾 Testing memory usage patterns..."
	cd openfoldpp && python scripts/evaluation/memory_benchmark.py
	@echo "📊 Memory test results available in openfoldpp/reports/memory/"

accuracy-test:
	@echo "🎯 Testing accuracy vs baseline..."
	cd openfoldpp && python scripts/evaluation/accuracy_comparison.py
	@echo "📊 Accuracy test results available in openfoldpp/reports/accuracy/"

# Deployment targets
package:
	@echo "📦 Packaging OpenFold++ for deployment..."
	cd openfoldpp && python setup.py sdist bdist_wheel
	@echo "✅ Package created in openfoldpp/dist/"

docker:
	@echo "🐳 Building Docker image..."
	docker build -t openfoldpp:latest -f openfoldpp/Dockerfile openfoldpp/
	@echo "✅ Docker image built: openfoldpp:latest"

deploy-test:
	@echo "🚀 Testing deployment readiness..."
	cd openfoldpp && python scripts/evaluation/deployment_test.py
	@echo "✅ Deployment test complete"

# Utility targets
clean:
	@echo "🧹 Cleaning build artifacts..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf openfoldpp/build/
	rm -rf openfoldpp/dist/
	rm -rf openfoldpp/.pytest_cache/
	@echo "✅ Clean complete"

docs:
	@echo "📚 Generating documentation..."
	cd docs && make html
	@echo "✅ Documentation available in docs/build/html/"

format:
	@echo "🎨 Formatting code..."
	black openfoldpp/src/ openfoldpp/scripts/ openfoldpp/tests/
	@echo "✅ Code formatted"

# Special benchmark targets for CI/CD
ci-test:
	@echo "🤖 Running CI tests..."
	make test-quick
	make benchmark-quick
	@echo "✅ CI tests passed"

performance-regression:
	@echo "📊 Checking for performance regressions..."
	cd openfoldpp && python scripts/evaluation/regression_test.py
	@echo "✅ Performance regression test complete"

# Development helpers
dev-setup: install-dev
	@echo "🛠️  Setting up development environment..."
	pre-commit install || echo "pre-commit not available"
	@echo "✅ Development environment ready"

quick-check: test-quick benchmark-quick
	@echo "⚡ Quick development check complete"

# Production validation
validate-production:
	@echo "🔍 Validating production readiness..."
	make test
	make benchmark
	make accuracy-test
	make memory-test
	@echo "✅ Production validation complete"

# Show system info
info:
	@echo "🖥️  System Information:"
	@echo "Python: $$(python --version)"
	@echo "PyTorch: $$(python -c 'import torch; print(torch.__version__)')"
	@echo "CUDA available: $$(python -c 'import torch; print(torch.cuda.is_available())')"
	@echo "GPU count: $$(python -c 'import torch; print(torch.cuda.device_count())')"
	@echo "Working directory: $$(pwd)"
