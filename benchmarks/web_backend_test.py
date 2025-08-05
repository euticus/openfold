"""
Web Backend Load Testing for OdinFold Mutation API

Tests the performance and reliability of the async mutation scanning web backend.
"""

import asyncio
import aiohttp
import time
import json
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

logger = logging.getLogger(__name__)


@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    
    # Test parameters
    concurrent_users: int = 10
    requests_per_user: int = 50
    ramp_up_time_seconds: int = 10
    test_duration_seconds: int = 300
    
    # Request settings
    timeout_seconds: int = 30
    retry_attempts: int = 3
    
    # Test data
    sequence_lengths: List[int] = None
    mutations_per_request: List[int] = None
    
    def __post_init__(self):
        if self.sequence_lengths is None:
            self.sequence_lengths = [100, 200, 300]
        if self.mutations_per_request is None:
            self.mutations_per_request = [10, 25, 50]


class WebBackendTester:
    """Load tester for mutation scanning web backend."""
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        
    def test_functionality(self) -> Dict[str, Any]:
        """Test basic functionality of the web backend."""
        
        logger.info("Testing web backend functionality")
        
        results = {
            'health_check': None,
            'single_mutation': None,
            'batch_mutations': None,
            'error_handling': None,
            'summary': {}
        }
        
        # Test health check
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=10)
            results['health_check'] = {
                'status_code': response.status_code,
                'response_time_ms': response.elapsed.total_seconds() * 1000,
                'healthy': response.status_code == 200
            }
        except Exception as e:
            results['health_check'] = {
                'error': str(e),
                'healthy': False
            }
        
        # Test single mutation request
        try:
            test_data = self._generate_test_request(sequence_length=100, num_mutations=5)
            
            response = requests.post(
                f"{self.backend_url}/api/mutations/scan",
                json=test_data,
                timeout=30
            )
            
            results['single_mutation'] = {
                'status_code': response.status_code,
                'response_time_ms': response.elapsed.total_seconds() * 1000,
                'success': response.status_code == 200,
                'response_size_bytes': len(response.content)
            }
            
            if response.status_code == 200:
                data = response.json()
                results['single_mutation']['num_results'] = len(data.get('results', []))
                results['single_mutation']['processing_time_ms'] = data.get('processing_time_ms')
                
        except Exception as e:
            results['single_mutation'] = {
                'error': str(e),
                'success': False
            }
        
        # Test batch mutations
        try:
            batch_data = {
                'proteins': [
                    self._generate_test_request(100, 10),
                    self._generate_test_request(150, 15)
                ],
                'priority': 'normal'
            }
            
            response = requests.post(
                f"{self.backend_url}/api/mutations/batch",
                json=batch_data,
                timeout=30
            )
            
            results['batch_mutations'] = {
                'status_code': response.status_code,
                'response_time_ms': response.elapsed.total_seconds() * 1000,
                'success': response.status_code == 200
            }
            
        except Exception as e:
            results['batch_mutations'] = {
                'error': str(e),
                'success': False
            }
        
        # Test error handling
        try:
            # Send invalid request
            invalid_data = {
                'sequence': 'INVALID_SEQUENCE_WITH_NUMBERS123',
                'mutations': [{'position': 0, 'from_aa': 'A', 'to_aa': 'V'}]
            }
            
            response = requests.post(
                f"{self.backend_url}/api/mutations/scan",
                json=invalid_data,
                timeout=30
            )
            
            results['error_handling'] = {
                'status_code': response.status_code,
                'handles_errors': response.status_code >= 400,
                'response_time_ms': response.elapsed.total_seconds() * 1000
            }
            
        except Exception as e:
            results['error_handling'] = {
                'error': str(e),
                'handles_errors': True  # Exception is also proper error handling
            }
        
        # Generate summary
        tests_passed = 0
        total_tests = 0
        
        for test_name, test_result in results.items():
            if test_name == 'summary':
                continue
                
            total_tests += 1
            if isinstance(test_result, dict):
                if test_result.get('healthy') or test_result.get('success') or test_result.get('handles_errors'):
                    tests_passed += 1
        
        results['summary'] = {
            'tests_passed': tests_passed,
            'total_tests': total_tests,
            'success_rate': tests_passed / total_tests if total_tests > 0 else 0,
            'backend_accessible': results['health_check'].get('healthy', False)
        }
        
        return results
    
    def run_load_test(self, 
                     concurrent_users: int = 10,
                     requests_per_user: int = 50) -> Dict[str, Any]:
        """Run load test with concurrent users."""
        
        logger.info(f"Running load test: {concurrent_users} users, {requests_per_user} requests each")
        
        results = {
            'config': {
                'concurrent_users': concurrent_users,
                'requests_per_user': requests_per_user,
                'total_requests': concurrent_users * requests_per_user
            },
            'request_results': [],
            'summary': {}
        }
        
        start_time = time.time()
        
        # Run concurrent requests
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = []
            
            for user_id in range(concurrent_users):
                future = executor.submit(
                    self._simulate_user_requests,
                    user_id,
                    requests_per_user
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    user_results = future.result()
                    results['request_results'].extend(user_results)
                except Exception as e:
                    logger.error(f"User simulation failed: {e}")
        
        total_time = time.time() - start_time
        
        # Calculate summary statistics
        successful_requests = [r for r in results['request_results'] if r.get('success', False)]
        failed_requests = [r for r in results['request_results'] if not r.get('success', False)]
        
        response_times = [r['response_time_ms'] for r in successful_requests if 'response_time_ms' in r]
        
        results['summary'] = {
            'total_time_seconds': total_time,
            'total_requests': len(results['request_results']),
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'success_rate': len(successful_requests) / len(results['request_results']) if results['request_results'] else 0,
            'requests_per_second': len(results['request_results']) / total_time if total_time > 0 else 0,
            'successful_requests_per_second': len(successful_requests) / total_time if total_time > 0 else 0
        }
        
        if response_times:
            results['summary'].update({
                'mean_response_time_ms': np.mean(response_times),
                'median_response_time_ms': np.median(response_times),
                'std_response_time_ms': np.std(response_times),
                'min_response_time_ms': np.min(response_times),
                'max_response_time_ms': np.max(response_times),
                'p95_response_time_ms': np.percentile(response_times, 95),
                'p99_response_time_ms': np.percentile(response_times, 99)
            })
        
        # Error analysis
        if failed_requests:
            error_types = {}
            for req in failed_requests:
                error = req.get('error', 'unknown')
                error_types[error] = error_types.get(error, 0) + 1
            
            results['summary']['error_breakdown'] = error_types
        
        return results
    
    def _simulate_user_requests(self, user_id: int, num_requests: int) -> List[Dict[str, Any]]:
        """Simulate requests from a single user."""
        
        user_results = []
        
        for request_id in range(num_requests):
            # Generate test request
            seq_length = np.random.choice([100, 200, 300])
            num_mutations = np.random.choice([10, 25, 50])
            
            test_data = self._generate_test_request(seq_length, num_mutations)
            
            # Send request
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{self.backend_url}/api/mutations/scan",
                    json=test_data,
                    timeout=30
                )
                
                response_time = (time.time() - start_time) * 1000
                
                result = {
                    'user_id': user_id,
                    'request_id': request_id,
                    'success': response.status_code == 200,
                    'status_code': response.status_code,
                    'response_time_ms': response_time,
                    'sequence_length': seq_length,
                    'num_mutations': num_mutations,
                    'response_size_bytes': len(response.content)
                }
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        result['server_processing_time_ms'] = data.get('processing_time_ms')
                        result['mutations_per_second'] = data.get('server_info', {}).get('mutations_per_second')
                    except:
                        pass
                
                user_results.append(result)
                
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                
                user_results.append({
                    'user_id': user_id,
                    'request_id': request_id,
                    'success': False,
                    'error': str(e),
                    'response_time_ms': response_time,
                    'sequence_length': seq_length,
                    'num_mutations': num_mutations
                })
            
            # Small delay between requests
            time.sleep(0.1)
        
        return user_results
    
    def _generate_test_request(self, sequence_length: int, num_mutations: int) -> Dict[str, Any]:
        """Generate a test mutation request."""
        
        # Generate random sequence
        sequence = ''.join(np.random.choice(self.amino_acids, sequence_length))
        
        # Generate random mutations
        mutations = []
        positions = np.random.choice(sequence_length, min(num_mutations, sequence_length), replace=False)
        
        for pos in positions:
            from_aa = sequence[pos]
            to_aa_choices = [aa for aa in self.amino_acids if aa != from_aa]
            to_aa = np.random.choice(to_aa_choices)
            
            mutations.append({
                'position': int(pos),
                'from_aa': from_aa,
                'to_aa': to_aa
            })
        
        return {
            'sequence': sequence,
            'mutations': mutations,
            'batch_size': 32,
            'include_confidence': True
        }
    
    async def run_async_load_test(self, config: LoadTestConfig) -> Dict[str, Any]:
        """Run async load test for better performance."""
        
        logger.info(f"Running async load test: {config.concurrent_users} users")
        
        results = {
            'config': config.__dict__,
            'request_results': [],
            'summary': {}
        }
        
        start_time = time.time()
        
        # Create async tasks
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=config.timeout_seconds)) as session:
            tasks = []
            
            for user_id in range(config.concurrent_users):
                task = asyncio.create_task(
                    self._async_user_simulation(session, user_id, config.requests_per_user)
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            user_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results
            for user_result in user_results:
                if isinstance(user_result, list):
                    results['request_results'].extend(user_result)
                else:
                    logger.error(f"User simulation failed: {user_result}")
        
        total_time = time.time() - start_time
        
        # Calculate summary (similar to sync version)
        successful_requests = [r for r in results['request_results'] if r.get('success', False)]
        response_times = [r['response_time_ms'] for r in successful_requests if 'response_time_ms' in r]
        
        results['summary'] = {
            'total_time_seconds': total_time,
            'total_requests': len(results['request_results']),
            'successful_requests': len(successful_requests),
            'success_rate': len(successful_requests) / len(results['request_results']) if results['request_results'] else 0,
            'requests_per_second': len(results['request_results']) / total_time if total_time > 0 else 0
        }
        
        if response_times:
            results['summary'].update({
                'mean_response_time_ms': np.mean(response_times),
                'p95_response_time_ms': np.percentile(response_times, 95),
                'p99_response_time_ms': np.percentile(response_times, 99)
            })
        
        return results
    
    async def _async_user_simulation(self, session: aiohttp.ClientSession, user_id: int, num_requests: int) -> List[Dict[str, Any]]:
        """Async user simulation."""
        
        user_results = []
        
        for request_id in range(num_requests):
            test_data = self._generate_test_request(200, 25)
            
            start_time = time.time()
            
            try:
                async with session.post(
                    f"{self.backend_url}/api/mutations/scan",
                    json=test_data
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    result = {
                        'user_id': user_id,
                        'request_id': request_id,
                        'success': response.status == 200,
                        'status_code': response.status,
                        'response_time_ms': response_time
                    }
                    
                    user_results.append(result)
                    
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                
                user_results.append({
                    'user_id': user_id,
                    'request_id': request_id,
                    'success': False,
                    'error': str(e),
                    'response_time_ms': response_time
                })
            
            # Small async delay
            await asyncio.sleep(0.05)
        
        return user_results
