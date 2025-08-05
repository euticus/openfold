#!/usr/bin/env python3
"""
Test OdinFold API without Docker

Quick test to validate the API server works locally.
"""

import sys
import time
import requests
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_api_locally():
    """Test the OdinFold API server locally."""
    
    print("🧪 Testing OdinFold API locally")
    print("=" * 40)
    
    # Start the server in background
    print("🚀 Starting API server...")
    
    try:
        # Import and test the API directly
        from odinfold.api.server import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test health endpoint
        print("📊 Testing health endpoint...")
        response = client.get("/health")
        assert response.status_code == 200
        health_data = response.json()
        print(f"   Status: {health_data['status']}")
        print(f"   Device: {health_data['device']}")
        print(f"   Model loaded: {health_data['model_loaded']}")
        
        # Test root endpoint
        print("🏠 Testing root endpoint...")
        response = client.get("/")
        assert response.status_code == 200
        root_data = response.json()
        print(f"   Message: {root_data['message']}")
        
        # Test folding endpoint
        print("🧬 Testing protein folding...")
        fold_request = {
            "sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "confidence": True,
            "relax": False
        }
        
        response = client.post("/fold", json=fold_request)
        assert response.status_code == 200
        fold_data = response.json()
        
        print(f"   Job ID: {fold_data['job_id']}")
        print(f"   Sequence length: {fold_data['sequence_length']}")
        print(f"   Runtime: {fold_data['runtime_seconds']:.3f}s")
        print(f"   TM estimate: {fold_data['tm_score_estimate']:.3f}")
        print(f"   Coordinates: {len(fold_data['coordinates'])} atoms")
        
        if fold_data['confidence_scores']:
            avg_confidence = sum(fold_data['confidence_scores']) / len(fold_data['confidence_scores'])
            print(f"   Avg confidence: {avg_confidence:.1f}")
        
        print("✅ All API tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Missing dependencies. Install with: pip install fastapi uvicorn")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_api_locally()
    sys.exit(0 if success else 1)
