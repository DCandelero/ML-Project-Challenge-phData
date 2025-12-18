"""Integration tests for /api/v1/predict endpoint"""

import pytest


def test_predict_endpoint_valid_input_returns_200(api_client, sample_house_features):
    """Test /predict with valid input returns 200"""
    response = api_client.post("/api/v1/predict", json=sample_house_features)

    assert response.status_code == 200

    data = response.json()
    assert 'prediction' in data
    assert 'zipcode' in data
    assert data['zipcode'] == '98115'
    assert data['prediction'] > 0


def test_predict_endpoint_returns_proper_schema(api_client, sample_house_features):
    """Test response matches PredictionResponse schema"""
    response = api_client.post("/api/v1/predict", json=sample_house_features)
    data = response.json()

    required_fields = [
        'prediction', 'model_version', 'zipcode',
        'demographics_found', 'prediction_timestamp'
    ]

    for field in required_fields:
        assert field in data, f"Missing field: {field}"


def test_predict_endpoint_missing_field_returns_422(api_client):
    """Test missing required field returns 422"""
    invalid_request = {'bedrooms': 3, 'bathrooms': 2.5}  # Missing most fields

    response = api_client.post("/api/v1/predict", json=invalid_request)

    assert response.status_code == 422
    assert 'detail' in response.json()


def test_predict_endpoint_invalid_zipcode_returns_404(api_client, sample_house_features):
    """Test invalid zipcode returns 404"""
    invalid_request = sample_house_features.copy()
    invalid_request['zipcode'] = '99999'

    response = api_client.post("/api/v1/predict", json=invalid_request)

    assert response.status_code == 404
    assert '99999' in response.json()['detail']


def test_predict_endpoint_invalid_types_returns_422(api_client, sample_house_features):
    """Test invalid data types return 422"""
    invalid_request = sample_house_features.copy()
    invalid_request['bedrooms'] = "three"  # String instead of int

    response = api_client.post("/api/v1/predict", json=invalid_request)

    assert response.status_code == 422


def test_predict_endpoint_negative_values_rejected(api_client, sample_house_features):
    """Test negative values are rejected"""
    invalid_request = sample_house_features.copy()
    invalid_request['bedrooms'] = -5  # Negative

    response = api_client.post("/api/v1/predict", json=invalid_request)

    assert response.status_code == 422


def test_predict_endpoint_invalid_zipcode_format(api_client, sample_house_features):
    """Test invalid zipcode format is rejected"""
    invalid_request = sample_house_features.copy()
    invalid_request['zipcode'] = '1234'  # Only 4 digits

    response = api_client.post("/api/v1/predict", json=invalid_request)

    assert response.status_code == 422


def test_predict_endpoint_returns_reasonable_prediction(api_client, sample_house_features):
    """Test prediction is within reasonable range"""
    response = api_client.post("/api/v1/predict", json=sample_house_features)
    data = response.json()

    # Seattle area houses should be between $100k and $5M
    assert 100_000 < data['prediction'] < 5_000_000


def test_predict_endpoint_demographics_found_flag(api_client, sample_house_features):
    """Test demographics_found flag is True for valid zipcode"""
    response = api_client.post("/api/v1/predict", json=sample_house_features)
    data = response.json()

    assert data['demographics_found'] is True


def test_predict_endpoint_has_timestamp(api_client, sample_house_features):
    """Test response includes timestamp"""
    response = api_client.post("/api/v1/predict", json=sample_house_features)
    data = response.json()

    assert 'prediction_timestamp' in data
    # Should be ISO 8601 format
    assert 'T' in data['prediction_timestamp']
    assert 'Z' in data['prediction_timestamp']
