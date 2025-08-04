#!/usr/bin/env python3
"""
Test script to verify the migration worked correctly.
"""

import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all imports work correctly."""
    
    print("🧪 Testing OpenFold++ Migration")
    print("=" * 40)
    
    try:
        # Test main package import
        import openfoldpp
        print("✅ Main package import: SUCCESS")
        
        # Test pipeline imports
        from openfoldpp.pipelines import FullInfrastructurePipeline
        print("✅ FullInfrastructurePipeline import: SUCCESS")
        
        from openfoldpp.pipelines import TrainedOpenFoldPipeline
        print("✅ TrainedOpenFoldPipeline import: SUCCESS")
        
        from openfoldpp.pipelines import WorkingOpenFoldPipeline
        print("✅ WorkingOpenFoldPipeline import: SUCCESS")
        
        print(f"\n📦 Package version: {openfoldpp.__version__}")
        print(f"📁 Package location: {openfoldpp.__file__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_file_structure():
    """Test that the file structure is correct."""
    
    print(f"\n🗂️  Testing File Structure")
    print("=" * 30)
    
    base_path = Path(__file__).parent
    
    # Required directories
    required_dirs = [
        "src/openfoldpp",
        "src/openfoldpp/pipelines",
        "src/openfoldpp/models", 
        "src/openfoldpp/modules",
        "src/openfoldpp/utils",
        "tests/integration",
        "tests/benchmarks",
        "scripts/data",
        "scripts/evaluation",
        "data/casp14",
        "data/databases",
        "data/weights",
        "results/predictions",
        "results/benchmarks",
        "results/analysis"
    ]
    
    all_exist = True
    
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path}")
            all_exist = False
    
    return all_exist


def test_key_files():
    """Test that key files exist."""
    
    print(f"\n📄 Testing Key Files")
    print("=" * 25)
    
    base_path = Path(__file__).parent
    
    # Key files
    key_files = [
        "src/openfoldpp/__init__.py",
        "src/openfoldpp/pipelines/__init__.py",
        "src/openfoldpp/pipelines/complete_pipeline.py",
        "src/openfoldpp/pipelines/trained_pipeline.py",
        "src/openfoldpp/pipelines/basic_pipeline.py",
        "requirements.txt",
        "setup.py",
        "README.md"
    ]
    
    all_exist = True
    
    for file_path in key_files:
        full_path = base_path / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"✅ {file_path} ({size} bytes)")
        else:
            print(f"❌ {file_path}")
            all_exist = False
    
    return all_exist


def test_data_migration():
    """Test that data was migrated correctly."""
    
    print(f"\n💾 Testing Data Migration")
    print("=" * 30)
    
    base_path = Path(__file__).parent
    
    # Check data directories
    data_checks = [
        ("data/casp14", "CASP14 data"),
        ("data/databases", "Sequence databases"),
        ("data/weights", "Model weights"),
        ("results/predictions", "Prediction results"),
        ("results/benchmarks", "Benchmark results"),
        ("tests/integration", "Integration tests")
    ]
    
    all_good = True
    
    for dir_path, description in data_checks:
        full_path = base_path / dir_path
        if full_path.exists():
            file_count = len(list(full_path.rglob("*")))
            print(f"✅ {description}: {file_count} files")
        else:
            print(f"❌ {description}: Missing")
            all_good = False
    
    return all_good


def main():
    """Main test function."""
    
    print("🚀 OPENFOLD++ MIGRATION VERIFICATION")
    print("=" * 50)
    
    # Run all tests
    tests = [
        ("File Structure", test_file_structure),
        ("Key Files", test_key_files), 
        ("Data Migration", test_data_migration),
        ("Package Imports", test_imports)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n📊 MIGRATION SUMMARY")
    print("=" * 25)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 MIGRATION SUCCESSFUL!")
        print("✅ All components migrated correctly")
        print("🚀 Ready for Phase 2 refactoring")
    else:
        print("⚠️  MIGRATION ISSUES DETECTED")
        print("🔧 Some components need attention")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
