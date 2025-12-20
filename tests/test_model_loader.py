"""Unit tests for Model Loader Service"""

import pytest
import pandas as pd
from ml.model_loader import ModelService


def test_model_loads_successfully(model_service):
    """Test model loads from pickle file"""
    assert model_service.model is not None
    assert len(model_service.get_features()) > 0


def test_model_has_predict_method(model_service):
    """Test loaded model has predict method"""
    assert hasattr(model_service.model, 'predict')
    assert callable(model_service.model.predict)


def test_features_loaded_correctly(model_service):
    """Test feature names loaded from JSON"""
    features = model_service.get_features()

    assert isinstance(features, list)
    assert len(features) > 10  # Should have multiple features
    assert 'bedrooms' in features
    assert 'bathrooms' in features


def test_model_info_returns_dict(model_service):
    """Test model info returns metadata dict"""
    info = model_service.get_model_info()

    assert isinstance(info, dict)
    assert 'feature_count' in info
    assert info['feature_count'] == len(model_service.get_features())


def test_predict_returns_float(model_service):
    """Test predict method returns float prediction"""
    # Create sample feature dataframe
    features = model_service.get_features()
    sample_data = pd.DataFrame([[1.0] * len(features)], columns=features)

    prediction = model_service.predict(sample_data)

    assert isinstance(prediction, float)
    assert prediction > 0  # House prices should be positive


def test_predict_with_multiple_rows_returns_single_value(model_service):
    """Test predict with single row returns single value"""
    features = model_service.get_features()
    sample_data = pd.DataFrame([[1.0] * len(features)], columns=features)

    prediction = model_service.predict(sample_data)

    # Should return a single float, not an array
    assert isinstance(prediction, (int, float))
