#!/usr/bin/env python3
"""
Test Suite for Mutation Scanning System

Tests the high-performance async mutation scanning web backend and benchmarking system.
"""

import pytest
import asyncio
import time
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.mutation_benchmark import MutationBenchmark, MutationScanConfig
from benchmarks.web_backend_test import WebBackendTester, LoadTestConfig

# Try to import web backend components (optional for testing)
try:
    from web_backend.mutation_server import MutationPredictor
    WEB_BACKEND_AVAILABLE = True
except ImportError:
    WEB_BACKEND_AVAILABLE = False
    print("Warning: Web backend not available for testing")


class TestMutationBenchmark:
    """Test mutation benchmark functionality."""
    
    def test_mutation_benchmark_init(self):
        """Test MutationBenchmark initialization."""
        
        benchmark = MutationBenchmark()
        
        assert len(benchmark.amino_acids) == 20
        assert len(benchmark.test_sequences) > 0
        assert all(len(seq) >= 50 for seq in benchmark.test_sequences)
        
        print(f"✅ Generated {len(benchmark.test_sequences)} test sequences")
        print(f"✅ Sequence lengths: {[len(seq) for seq in benchmark.test_sequences[:5]]}")
    
    def test_generate_random_mutations(self):
        """Test random mutation generation."""
        
        benchmark = MutationBenchmark()
        sequence = "MKWVTFISLLFLFSSAYS"
        num_mutations = 10
        
        mutations = benchmark._generate_random_mutations(sequence, num_mutations)
        
        assert len(mutations) == num_mutations
        
        for pos, from_aa, to_aa in mutations:
            assert 0 <= pos < len(sequence)
            assert sequence[pos] == from_aa
            assert to_aa in benchmark.amino_acids
            assert from_aa != to_aa
        
        print(f"✅ Generated {len(mutations)} valid mutations")
        print(f"✅ Sample mutations: {mutations[:3]}")
    
    def test_mock_mutation_scanning(self):
        """Test mock mutation scanning."""
        
        benchmark = MutationBenchmark()
        sequence = "MKWVTFISLLFLFSSAYS"
        mutations = benchmark._generate_random_mutations(sequence, 5)
        
        start_time = time.time()
        ddg_predictions = benchmark._mock_mutation_scanning(sequence, mutations, "cpu")
        scan_time = time.time() - start_time
        
        assert len(ddg_predictions) == len(mutations)
        assert isinstance(ddg_predictions, np.ndarray)
        assert scan_time < 2.0  # Should be fast for mock
        
        print(f"✅ Mock scanning completed in {scan_time:.3f}s")
        print(f"✅ ΔΔG predictions: {ddg_predictions}")
    
    def test_benchmark_mutation_scanning(self):
        """Test mutation scanning benchmark."""
        
        benchmark = MutationBenchmark()
        
        results = benchmark.benchmark_mutation_scanning(
            num_proteins=2,
            mutations_per_protein=10,
            model_path="mock_model.pt",
            device="cpu"
        )
        
        assert 'config' in results
        assert 'protein_results' in results
        assert 'summary' in results
        
        assert len(results['protein_results']) == 2
        
        for protein_result in results['protein_results']:
            assert 'sequence_length' in protein_result
            assert 'num_mutations' in protein_result
            assert 'scan_time_seconds' in protein_result
            assert 'mutations_per_second' in protein_result
            assert protein_result['mutations_per_second'] > 0
        
        summary = results['summary']
        assert summary['total_proteins'] == 2
        assert summary['total_mutations'] == 20
        assert summary['overall_mutations_per_second'] > 0
        
        print(f"✅ Benchmark completed: {summary['overall_mutations_per_second']:.1f} mutations/s")
    
    def test_batch_mutation_benchmark(self):
        """Test batch mutation benchmark."""
        
        benchmark = MutationBenchmark()
        
        results = benchmark.benchmark_batch_mutations(batch_sizes=[1, 4, 8])
        
        assert 'batch_results' in results
        assert 'summary' in results
        
        for batch_size in [1, 4, 8]:
            assert batch_size in results['batch_results']
            batch_result = results['batch_results'][batch_size]
            assert batch_result['mutations_per_second'] > 0
        
        assert 'optimal_batch_size' in results['summary']
        assert results['summary']['optimal_batch_size'] in [1, 4, 8]
        
        print(f"✅ Optimal batch size: {results['summary']['optimal_batch_size']}")


@pytest.mark.skipif(not WEB_BACKEND_AVAILABLE, reason="Web backend not available")
class TestMutationPredictor:
    """Test mutation predictor functionality."""

    def test_mutation_predictor_init(self):
        """Test MutationPredictor initialization."""

        predictor = MutationPredictor()

        assert predictor.model_path == "models/odinfold.pt"
        assert len(predictor.amino_acids) == 20
        assert len(predictor.aa_to_idx) == 20

        print(f"✅ Predictor initialized with device: {predictor.device}")

    @pytest.mark.asyncio
    async def test_load_model(self):
        """Test model loading."""

        predictor = MutationPredictor()

        assert predictor.model is None

        await predictor.load_model()

        assert predictor.model is not None

        print("✅ Model loaded successfully")

    def test_mock_ddg_prediction(self):
        """Test mock ΔΔG prediction."""

        predictor = MutationPredictor()

        # Create mock mutation object
        class MockMutation:
            def __init__(self, position, from_aa, to_aa):
                self.position = position
                self.from_aa = from_aa
                self.to_aa = to_aa

        mutation = MockMutation(position=0, from_aa='A', to_aa='V')
        sequence = "AKWVTFISLLFLFSSAYS"

        ddg = predictor._mock_ddg_prediction(sequence, mutation)

        assert isinstance(ddg, float)
        assert -10.0 <= ddg <= 10.0  # Reasonable range

        print(f"✅ Mock ΔΔG prediction: {ddg:.3f} kcal/mol")

    def test_mock_confidence_prediction(self):
        """Test mock confidence prediction."""

        predictor = MutationPredictor()

        confidence = predictor._mock_confidence_prediction()

        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

        print(f"✅ Mock confidence: {confidence:.3f}")


@pytest.mark.skipif(not WEB_BACKEND_AVAILABLE, reason="Web backend not available")
class TestWebBackend:
    """Test web backend functionality (requires FastAPI dependencies)."""

    def test_web_backend_components(self):
        """Test web backend components are importable."""

        # Test that we can import the main components
        from web_backend.mutation_server import MutationPredictor

        predictor = MutationPredictor()
        assert predictor is not None

        print("✅ Web backend components available")


class TestWebBackendTester:
    """Test web backend testing functionality."""
    
    def test_web_backend_tester_init(self):
        """Test WebBackendTester initialization."""
        
        tester = WebBackendTester("http://localhost:8000")
        
        assert tester.backend_url == "http://localhost:8000"
        assert len(tester.amino_acids) == 20
        
        print("✅ WebBackendTester initialized")
    
    def test_generate_test_request(self):
        """Test test request generation."""
        
        tester = WebBackendTester()
        
        request = tester._generate_test_request(sequence_length=100, num_mutations=10)
        
        assert "sequence" in request
        assert "mutations" in request
        assert len(request["sequence"]) == 100
        assert len(request["mutations"]) == 10
        
        for mutation in request["mutations"]:
            assert "position" in mutation
            assert "from_aa" in mutation
            assert "to_aa" in mutation
            assert 0 <= mutation["position"] < 100
            assert mutation["from_aa"] in tester.amino_acids
            assert mutation["to_aa"] in tester.amino_acids
            assert mutation["from_aa"] != mutation["to_aa"]
        
        print(f"✅ Generated test request with {len(request['mutations'])} mutations")


def test_mutation_scanning_integration():
    """Test integration of mutation scanning components."""
    
    print("🧬 Testing Mutation Scanning Integration...")
    
    # Test benchmark and web backend integration
    benchmark = MutationBenchmark()
    tester = WebBackendTester()
    
    # Generate test data
    sequence = "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL"
    
    mutations = benchmark._generate_random_mutations(sequence, 50)
    
    print(f"✅ Generated test sequence: {len(sequence)} residues")
    print(f"✅ Generated mutations: {len(mutations)}")
    
    # Test mock prediction
    ddg_predictions = benchmark._mock_mutation_scanning(sequence, mutations, "cpu")
    
    print(f"✅ Mock predictions: mean ΔΔG = {np.mean(ddg_predictions):.3f} kcal/mol")
    
    # Test web request generation
    web_request = tester._generate_test_request(200, 25)
    
    print(f"✅ Web request generated: {len(web_request['sequence'])} residues, {len(web_request['mutations'])} mutations")
    
    print("🎉 Mutation scanning integration test passed!")


def test_performance_expectations():
    """Test that performance meets expectations."""
    
    print("⚡ Testing Performance Expectations...")
    
    benchmark = MutationBenchmark()
    
    # Test small-scale performance
    sequence = "A" * 100
    mutations = benchmark._generate_random_mutations(sequence, 100)
    
    start_time = time.time()
    ddg_predictions = benchmark._mock_mutation_scanning(sequence, mutations, "cpu")
    scan_time = time.time() - start_time
    
    mutations_per_second = len(mutations) / scan_time
    
    print(f"✅ Performance test: {mutations_per_second:.1f} mutations/s")
    
    # Performance expectations
    assert mutations_per_second > 50, f"Performance too slow: {mutations_per_second:.1f} mutations/s"
    assert scan_time < 5.0, f"Scan took too long: {scan_time:.2f}s"
    
    print("🎉 Performance expectations met!")


if __name__ == "__main__":
    # Run integration tests
    test_mutation_scanning_integration()
    test_performance_expectations()
    
    # Run all tests
    pytest.main([__file__, "-v"])
