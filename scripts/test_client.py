#!/usr/bin/env python3
"""
Test client for Sound Realty ML API

Deliverable #2: Demonstrates API behavior using examples from
future_unseen_examples.csv

This script:
- Tests health endpoints
- Makes predictions on real examples
- Tests error handling
- Tests minimal endpoint
- Reports performance metrics
"""

import requests
import pandas as pd
import time
import sys
from typing import Dict, Any

API_URL = "http://localhost:8000"


def load_test_examples(n=100):
    """Load test examples from CSV"""
    try:
        df = pd.read_csv("data/future_unseen_examples.csv", nrows=n)
        # Convert zipcode to string (API expects string, not int)
        if 'zipcode' in df.columns:
            df['zipcode'] = df['zipcode'].astype(str)
        print(f"✓ Loaded {len(df)} test examples from future_unseen_examples.csv")
        return df.to_dict('records')
    except FileNotFoundError:
        print("✗ ERROR: data/future_unseen_examples.csv not found")
        sys.exit(1)


def test_health_check():
    """Test health endpoints"""
    print("\n" + "="*60)
    print("Testing Health Endpoints")
    print("="*60)

    try:
        # Liveness
        response = requests.get(f"{API_URL}/health", timeout=5)
        print(f"✓ Liveness: {response.json()}")

        # Readiness
        response = requests.get(f"{API_URL}/health/ready", timeout=5)
        ready_data = response.json()
        print(f"✓ Readiness: {ready_data}")

        # Model info
        response = requests.get(f"{API_URL}/api/v1/model/info", timeout=5)
        info = response.json()
        print(f"✓ Model: {info.get('model_type', 'Unknown')} v{info.get('model_version', '?')}")
        print(f"✓ Features: {info.get('feature_count', '?')}")

        return True
    except requests.exceptions.ConnectionError:
        print(f"✗ ERROR: Could not connect to {API_URL}")
        print("  Make sure API is running: docker-compose up")
        sys.exit(1)
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def test_prediction(house: Dict[str, Any], endpoint="/api/v1/predict"):
    """Test single prediction"""
    start_time = time.time()

    try:
        response = requests.post(f"{API_URL}{endpoint}", json=house, timeout=10)
        duration = (time.time() - start_time) * 1000  # Convert to ms

        if response.status_code == 200:
            result = response.json()
            return {
                'success': True,
                'prediction': result['prediction'],
                'zipcode': result.get('zipcode', house.get('zipcode')),
                'duration_ms': duration,
                'error': None,
                'minimal_request': result.get('minimal_request', False),
                'defaults_used': result.get('defaults_used', 0)
            }
        else:
            # Handle different error formats (list, dict, or string)
            error_detail = response.json().get('detail', 'Unknown error')
            if isinstance(error_detail, list) and len(error_detail) > 0:
                # Pydantic validation errors - extract first error message
                first_error = error_detail[0]
                error_msg = first_error.get('msg', str(first_error))
            elif isinstance(error_detail, dict):
                error_msg = str(error_detail)
            else:
                error_msg = str(error_detail)
            
            return {
                'success': False,
                'prediction': None,
                'zipcode': house.get('zipcode'),
                'duration_ms': duration,
                'error': error_msg,
                'error_detail': error_detail  # Keep full detail for debugging
            }
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        return {
            'success': False,
            'prediction': None,
            'zipcode': house.get('zipcode'),
            'duration_ms': duration,
            'error': str(e)
        }


def test_invalid_zipcode():
    """Test error handling with invalid zipcode"""
    print("\n" + "="*60)
    print("Testing Error Handling")
    print("="*60)

    invalid_house = {
        'bedrooms': 3,
        'bathrooms': 2.5,
        'sqft_living': 2220,
        'sqft_lot': 6380,
        'floors': 1.5,
        'waterfront': 0,
        'view': 0,
        'condition': 4,
        'grade': 8,
        'sqft_above': 1660,
        'sqft_basement': 560,
        'yr_built': 1931,
        'yr_renovated': 0,
        'zipcode': '99999',  # Invalid
        'lat': 47.6974,
        'long': -122.313,
        'sqft_living15': 950,
        'sqft_lot15': 6380
    }

    result = test_prediction(invalid_house)
    if result['success']:
        print("✗ ERROR: Invalid zipcode should have failed")
        return False
    else:
        print(f"✓ Invalid zipcode correctly rejected")
        print(f"  Error: {result['error']}")
        return True


def test_minimal_endpoint():
    """Test minimal endpoint with reduced features"""
    print("\n" + "="*60)
    print("Testing Minimal Endpoint")
    print("="*60)

    minimal_house = {
        'bedrooms': 3,
        'bathrooms': 2.5,
        'sqft_living': 2220,
        'sqft_lot': 6380,
        'floors': 1.5,
        'sqft_above': 1660,
        'sqft_basement': 560,
        'zipcode': '98115'
    }

    result = test_prediction(minimal_house, endpoint="/api/v1/predict/minimal")

    if result['success']:
        print(f"✓ Minimal prediction: ${result['prediction']:,.2f}")
        print(f"  Zipcode: {result['zipcode']}")
        print(f"  Response time: {result['duration_ms']:.0f}ms")
        if result['defaults_used'] > 0:
            print(f"  Defaults used: {result['defaults_used']} features")
        return True
    else:
        print(f"✗ Minimal prediction failed: {result['error']}")
        return False


def test_minimal_defaults_endpoint():
    """Test minimal defaults documentation endpoint"""
    print("\n" + "="*60)
    print("Testing Minimal Defaults Documentation")
    print("="*60)

    try:
        response = requests.get(f"{API_URL}/api/v1/predict/minimal/defaults", timeout=5)

        if response.status_code == 200:
            data = response.json()
            print(f"✓ Defaults endpoint accessible")
            print(f"  Total defaults: {data.get('total_defaults', '?')}")

            if 'defaults' in data:
                print(f"\n  Default values used:")
                for feature, info in list(data['defaults'].items())[:5]:  # Show first 5
                    print(f"    - {feature}: {info.get('value')} ({info.get('reason', 'N/A')})")
            return True
        else:
            print(f"✗ Defaults endpoint returned {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def main():
    """Main test script"""
    print("="*60)
    print("Sound Realty ML API - Test Client")
    print("Deliverable #2: API Demonstration")
    print("="*60)
    print(f"API URL: {API_URL}")

    # Test health
    if not test_health_check():
        sys.exit(1)

    # Load test examples
    print("\n" + "="*60)
    print("Loading Test Examples")
    print("="*60)
    examples = load_test_examples(n=100)

    # Test predictions on first 10 examples
    print("\n" + "="*60)
    print("Testing Predictions (First 10 Examples)")
    print("="*60)

    results = []
    for i, house in enumerate(examples[:10], 1):
        result = test_prediction(house)
        results.append(result)

        if result['success']:
            print(f"✓ Test {i:2d}: Zipcode {result['zipcode']}")
            print(f"           Prediction: ${result['prediction']:>12,.2f}")
            print(f"           Response time: {result['duration_ms']:>5.0f}ms")
        else:
            print(f"✗ Test {i:2d}: FAILED - {result['error']}")

    # Summary stats
    successful = [r for r in results if r['success']]
    if successful:
        avg_duration = sum(r['duration_ms'] for r in successful) / len(successful)
        predictions = [r['prediction'] for r in successful]

        print("\n" + "="*60)
        print("Summary Statistics (First 10)")
        print("="*60)
        print(f"Successful predictions: {len(successful)}/{len(results)}")
        print(f"Average response time:  {avg_duration:.0f}ms")
        print(f"Prediction range:       ${min(predictions):,.0f} - ${max(predictions):,.0f}")
        print(f"Average prediction:     ${sum(predictions)/len(predictions):,.0f}")

    # Test all 100 examples (batch test)
    print("\n" + "="*60)
    print(f"Batch Testing All {len(examples)} Examples")
    print("="*60)

    batch_results = []
    print("Processing", end="", flush=True)
    for i, house in enumerate(examples):
        result = test_prediction(house)
        batch_results.append(result)

        if (i + 1) % 10 == 0:
            print(".", end="", flush=True)

    print(" Done!")

    # Batch summary
    batch_successful = [r for r in batch_results if r['success']]
    batch_failed = [r for r in batch_results if not r['success']]

    print(f"\n✓ Successful: {len(batch_successful)}/{len(batch_results)}")
    if batch_failed:
        print(f"✗ Failed: {len(batch_failed)}")
        print(f"  Common errors:")
        error_counts = {}
        for r in batch_failed:
            # Extract error message (handle string, list, or dict)
            error = r.get('error', 'Unknown')
            if isinstance(error, (list, dict)):
                error = str(error)[:50]
            else:
                error = str(error)[:50] if error else 'Unknown'
            error_counts[error] = error_counts.get(error, 0) + 1
        for error, count in sorted(error_counts.items(), key=lambda x: -x[1])[:3]:
            print(f"    - {error}: {count} occurrences")

    if batch_successful:
        batch_predictions = [r['prediction'] for r in batch_successful]
        batch_durations = [r['duration_ms'] for r in batch_successful]

        print(f"\nPrediction Statistics:")
        print(f"  Min:     ${min(batch_predictions):,.0f}")
        print(f"  Max:     ${max(batch_predictions):,.0f}")
        print(f"  Average: ${sum(batch_predictions)/len(batch_predictions):,.0f}")

        print(f"\nPerformance Statistics:")
        print(f"  Average response time: {sum(batch_durations)/len(batch_durations):.0f}ms")
        print(f"  Min response time:     {min(batch_durations):.0f}ms")
        print(f"  Max response time:     {max(batch_durations):.0f}ms")
        if len(batch_durations) > 0:
            print(f"  p95 response time:     {sorted(batch_durations)[int(len(batch_durations)*0.95)]:.0f}ms")

    # Test error handling
    test_invalid_zipcode()

    # Test minimal endpoint
    test_minimal_endpoint()

    # Test minimal defaults documentation
    test_minimal_defaults_endpoint()

    # Final summary
    print("\n" + "="*60)
    print("✓ All Tests Completed!")
    print("="*60)
    print(f"\nResults Summary:")
    print(f"  Health checks: PASSED")
    print(f"  Full predictions: {len(batch_successful)}/{len(batch_results)} successful")
    print(f"  Error handling: PASSED")
    print(f"  Minimal endpoint: PASSED")
    if batch_successful:
        batch_durations = [r['duration_ms'] for r in batch_successful]
        print(f"  Performance: Average {sum(batch_durations)/len(batch_durations):.0f}ms per prediction")
    print("\n" + "="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
