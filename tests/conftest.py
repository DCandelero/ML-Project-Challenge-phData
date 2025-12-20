"""Pytest fixtures for testing"""

import pytest
import pandas as pd
from fastapi.testclient import TestClient


@pytest.fixture
def sample_demographics():
    """Sample demographics data for testing"""
    return {
        '98115': {
            'ppltn_qty': 43263.0,
            'medn_hshld_incm_amt': 58475.0,
            'hous_val_amt': 412345.0,
            'per_urbn': 98.5,
            'per_bchlr': 45.2
        },
        '98042': {
            'ppltn_qty': 38249.0,
            'medn_hshld_incm_amt': 66051.0,
            'hous_val_amt': 458900.0,
            'per_urbn': 85.3,
            'per_bchlr': 32.1
        }
    }


@pytest.fixture
def sample_house_features():
    """Sample house features for testing"""
    return {
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
        'zipcode': '98115',
        'lat': 47.6974,
        'long': -122.313,
        'sqft_living15': 950,
        'sqft_lot15': 6380
    }


@pytest.fixture
def sample_minimal_features():
    """Sample minimal house features for testing"""
    return {
        'bedrooms': 3,
        'bathrooms': 2.5,
        'sqft_living': 2220,
        'sqft_lot': 6380,
        'floors': 1.5,
        'sqft_above': 1660,
        'sqft_basement': 560,
        'zipcode': '98115'
    }


@pytest.fixture
def api_client():
    """FastAPI test client"""
    from app.main import app
    return TestClient(app)


@pytest.fixture
def demographics_service():
    """Demographics service for testing"""
    from ml.demographics_service import DemographicsService
    return DemographicsService("data/zipcode_demographics.csv")


@pytest.fixture
def model_service():
    """Model service for testing (uses v1 by default)"""
    from ml.model_loader import ModelService
    from app.config import settings
    return ModelService(settings.model_path, settings.features_path)


@pytest.fixture
def feature_defaults_service(demographics_service):
    """Feature defaults service for testing"""
    from ml.feature_defaults import FeatureDefaultsService
    return FeatureDefaultsService(demographics_service)


@pytest.fixture
def preprocessor(model_service):
    """Feature preprocessor for testing"""
    from ml.preprocessor import FeaturePreprocessor
    return FeaturePreprocessor(model_service.get_features())


@pytest.fixture
def prediction_service(model_service, demographics_service, preprocessor, feature_defaults_service):
    """Full prediction service for testing"""
    from ml.predictor import PredictionService
    return PredictionService(
        model_service,
        demographics_service,
        preprocessor,
        feature_defaults_service
    )
