"""Pytest fixtures for testing"""

import pytest
import pandas as pd
from fastapi.testclient import TestClient


@pytest.fixture
def sample_demographics():
    """Sample demographics data for testing (all 26 columns)"""
    return {
        '98042': {
            'ppltn_qty': 38249.0,
            'urbn_ppltn_qty': 37394.0,
            'sbrbn_ppltn_qty': 0.0,
            'farm_ppltn_qty': 0.0,
            'non_farm_qty': 855.0,
            'medn_hshld_incm_amt': 66051.0,
            'medn_incm_per_prsn_amt': 25219.0,
            'hous_val_amt': 192000.0,
            'edctn_less_than_9_qty': 437.0,
            'edctn_9_12_qty': 2301.0,
            'edctn_high_schl_qty': 7135.0,
            'edctn_some_clg_qty': 7787.0,
            'edctn_assoc_dgre_qty': 2202.0,
            'edctn_bchlr_dgre_qty': 4964.0,
            'edctn_prfsnl_qty': 1783.0,
            'per_urbn': 97.0,
            'per_sbrbn': 0.0,
            'per_farm': 0.0,
            'per_non_farm': 2.0,
            'per_less_than_9': 1.0,
            'per_9_to_12': 6.0,
            'per_hsd': 18.0,
            'per_some_clg': 20.0,
            'per_assoc': 5.0,
            'per_bchlr': 12.0,
            'per_prfsnl': 4.0
        },
        '98115': {
            'ppltn_qty': 43263.0,
            'urbn_ppltn_qty': 42500.0,
            'sbrbn_ppltn_qty': 763.0,
            'farm_ppltn_qty': 0.0,
            'non_farm_qty': 0.0,
            'medn_hshld_incm_amt': 58475.0,
            'medn_incm_per_prsn_amt': 35000.0,
            'hous_val_amt': 412345.0,
            'edctn_less_than_9_qty': 500.0,
            'edctn_9_12_qty': 2500.0,
            'edctn_high_schl_qty': 8000.0,
            'edctn_some_clg_qty': 8500.0,
            'edctn_assoc_dgre_qty': 2500.0,
            'edctn_bchlr_dgre_qty': 19550.0,
            'edctn_prfsnl_qty': 1713.0,
            'per_urbn': 98.5,
            'per_sbrbn': 1.5,
            'per_farm': 0.0,
            'per_non_farm': 0.0,
            'per_less_than_9': 1.0,
            'per_9_to_12': 6.0,
            'per_hsd': 18.0,
            'per_some_clg': 20.0,
            'per_assoc': 6.0,
            'per_bchlr': 45.2,
            'per_prfsnl': 4.0
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
    """Model service for testing (uses v1 model)"""
    from ml.model_loader import ModelService
    return ModelService(
        "model/v1/model.pkl",
        "model/v1/model_features.json"
    )


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
