"""Integration tests for /api/v1/predict/minimal endpoint"""

import pytest


def test_minimal_endpoint_works_with_8_features(api_client, sample_minimal_features):
    """Test /predict/minimal works with only 8 features"""
    response = api_client.post("/api/v1/predict/minimal", json=sample_minimal_features)

    assert response.status_code == 200
    data = response.json()
    assert 'prediction' in data
    assert data['prediction'] > 0


def test_minimal_endpoint_returns_proper_schema(api_client, sample_minimal_features):
    """Test minimal endpoint response includes all required fields"""
    response = api_client.post("/api/v1/predict/minimal", json=sample_minimal_features)
    data = response.json()

    required_fields = [
        'prediction', 'model_version', 'zipcode',
        'demographics_found', 'prediction_timestamp'
    ]

    for field in required_fields:
        assert field in data, f"Missing field: {field}"


def test_minimal_endpoint_indicates_defaults_used(api_client, sample_minimal_features):
    """Test minimal endpoint response indicates defaults were used"""
    response = api_client.post("/api/v1/predict/minimal", json=sample_minimal_features)
    data = response.json()

    # Should indicate minimal request (check for either field)
    assert 'minimal_request' in data or 'defaults_used' in data

    if 'minimal_request' in data:
        assert data['minimal_request'] is True
    if 'defaults_used' in data:
        assert data['defaults_used'] == 10


def test_minimal_endpoint_missing_required_field_returns_422(api_client):
    """Test minimal endpoint rejects missing required fields"""
    invalid_request = {
        'bedrooms': 3,
        'bathrooms': 2.5,
        'sqft_living': 2220
        # Missing other required fields
    }

    response = api_client.post("/api/v1/predict/minimal", json=invalid_request)

    assert response.status_code == 422


def test_minimal_endpoint_invalid_zipcode_returns_404(api_client, sample_minimal_features):
    """Test minimal endpoint with invalid zipcode returns 404"""
    invalid_request = sample_minimal_features.copy()
    invalid_request['zipcode'] = '99999'

    response = api_client.post("/api/v1/predict/minimal", json=invalid_request)

    assert response.status_code == 404


def test_minimal_defaults_endpoint_returns_docs(api_client):
    """Test /predict/minimal/defaults returns default values"""
    response = api_client.get("/api/v1/predict/minimal/defaults")

    assert response.status_code == 200
    data = response.json()

    assert 'defaults' in data
    assert isinstance(data['defaults'], dict)

    # Check for expected default features
    expected_defaults = ['waterfront', 'view', 'condition', 'grade']
    defaults = data['defaults']

    for feature in expected_defaults:
        assert feature in defaults, f"Missing default for {feature}"


def test_minimal_defaults_endpoint_includes_reasons(api_client):
    """Test defaults endpoint includes reasons for each default"""
    response = api_client.get("/api/v1/predict/minimal/defaults")
    data = response.json()

    defaults = data['defaults']

    # Each default should have value and reason
    for feature, info in defaults.items():
        assert 'value' in info, f"Missing value for {feature}"
        assert 'reason' in info, f"Missing reason for {feature}"


def test_minimal_endpoint_returns_reasonable_prediction(api_client, sample_minimal_features):
    """Test minimal prediction is within reasonable range"""
    response = api_client.post("/api/v1/predict/minimal", json=sample_minimal_features)
    data = response.json()

    # Should still be reasonable range
    assert 100_000 < data['prediction'] < 5_000_000


def test_minimal_and_full_predictions_similar(api_client, sample_house_features, sample_minimal_features):
    """Test minimal and full predictions are reasonably close"""
    # Get full prediction
    full_response = api_client.post("/api/v1/predict", json=sample_house_features)
    full_prediction = full_response.json()['prediction']

    # Get minimal prediction
    minimal_response = api_client.post("/api/v1/predict/minimal", json=sample_minimal_features)
    minimal_prediction = minimal_response.json()['prediction']

    # Should be within 30% of each other (defaults are reasonable)
    difference = abs(full_prediction - minimal_prediction) / full_prediction
    assert difference < 0.30, f"Predictions differ by {difference*100:.1f}%"
