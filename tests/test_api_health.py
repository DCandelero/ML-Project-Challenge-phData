"""Integration tests for health and info endpoints"""

import pytest


def test_health_endpoint_returns_healthy(api_client):
    """Test /health returns healthy status"""
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'


def test_readiness_endpoint_returns_ready(api_client):
    """Test /health/ready indicates model loaded"""
    response = api_client.get("/health/ready")

    assert response.status_code == 200
    data = response.json()

    assert data['status'] == 'ready'
    assert 'model_loaded' in data
    assert data['model_loaded'] is True


def test_readiness_endpoint_includes_demographics_count(api_client):
    """Test /health/ready includes demographics information"""
    response = api_client.get("/health/ready")
    data = response.json()

    assert 'demographics_loaded' in data
    assert data['demographics_loaded'] is True

    if 'demographics_count' in data:
        assert data['demographics_count'] == 70


def test_model_info_endpoint_returns_metadata(api_client):
    """Test /api/v1/model/info returns model metadata"""
    response = api_client.get("/api/v1/model/info")

    assert response.status_code == 200
    data = response.json()

    assert 'model_version' in data
    assert 'features' in data
    assert 'feature_count' in data
    assert isinstance(data['features'], list)
    assert data['feature_count'] > 0


def test_model_info_includes_feature_list(api_client):
    """Test model info includes complete feature list"""
    response = api_client.get("/api/v1/model/info")
    data = response.json()

    features = data['features']
    assert isinstance(features, list)
    assert len(features) == data['feature_count']

    # Check for expected features
    expected_features = ['bedrooms', 'bathrooms', 'sqft_living']
    for feature in expected_features:
        assert feature in features


def test_root_endpoint_redirects_to_docs(api_client):
    """Test root endpoint provides helpful information"""
    response = api_client.get("/")

    # Should either redirect to /docs or return helpful message
    assert response.status_code in [200, 307, 308]


def test_docs_endpoint_accessible(api_client):
    """Test /docs endpoint is accessible"""
    response = api_client.get("/docs")

    # Swagger UI should be available
    assert response.status_code == 200


def test_openapi_json_accessible(api_client):
    """Test /openapi.json is accessible"""
    response = api_client.get("/openapi.json")

    assert response.status_code == 200
    data = response.json()

    assert 'openapi' in data
    assert 'info' in data
    assert 'paths' in data


def test_health_endpoint_fast_response(api_client):
    """Test health endpoint responds quickly"""
    import time
    start = time.time()

    response = api_client.get("/health")

    duration = time.time() - start

    assert response.status_code == 200
    assert duration < 0.1  # Should respond in less than 100ms
