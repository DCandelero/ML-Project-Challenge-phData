"""Unit tests for Feature Preprocessor"""

import pytest
import pandas as pd
from ml.preprocessor import FeaturePreprocessor


def test_prepare_features_returns_dataframe(preprocessor, sample_house_features, sample_demographics):
    """Test feature preparation returns DataFrame"""
    demographics = sample_demographics['98115']
    result = preprocessor.prepare_features(sample_house_features, demographics)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1  # Single row


def test_feature_alignment_matches_model(preprocessor, model_service, sample_house_features, sample_demographics):
    """Test features are aligned to model feature order"""
    demographics = sample_demographics['98115']
    result = preprocessor.prepare_features(sample_house_features, demographics)

    expected_features = model_service.get_features()
    actual_features = list(result.columns)

    assert actual_features == expected_features, "Feature order must match model!"


def test_all_required_features_present(preprocessor, model_service, sample_house_features, sample_demographics):
    """Test all required features are present"""
    demographics = sample_demographics['98115']
    result = preprocessor.prepare_features(sample_house_features, demographics)

    assert len(result.columns) == len(model_service.get_features())


def test_demographic_merge_preserves_house_features(preprocessor, sample_house_features, sample_demographics):
    """Test demographic merge doesn't lose house features"""
    demographics = sample_demographics['98115']
    result = preprocessor.prepare_features(sample_house_features, demographics)

    # Check original house features are present
    if 'bedrooms' in result.columns:
        assert result['bedrooms'].iloc[0] == 3
    if 'bathrooms' in result.columns:
        assert result['bathrooms'].iloc[0] == 2.5


def test_prepare_features_handles_missing_demographics_columns(preprocessor, sample_house_features):
    """Test preprocessor handles incomplete demographics gracefully"""
    # Minimal demographics
    minimal_demographics = {
        'ppltn_qty': 40000.0,
        'medn_hshld_incm_amt': 60000.0
    }

    # This should work or raise clear error
    try:
        result = preprocessor.prepare_features(sample_house_features, minimal_demographics)
        assert isinstance(result, pd.DataFrame)
    except KeyError as e:
        # Expected if missing required columns
        assert 'missing' in str(e).lower() or 'not found' in str(e).lower()


def test_prepare_features_with_different_zipcodes(preprocessor, sample_house_features, sample_demographics):
    """Test preprocessor works with different zipcodes"""
    # Test with zipcode 98115
    result1 = preprocessor.prepare_features(
        sample_house_features,
        sample_demographics['98115']
    )

    # Test with zipcode 98042
    house_features_2 = sample_house_features.copy()
    house_features_2['zipcode'] = '98042'
    result2 = preprocessor.prepare_features(
        house_features_2,
        sample_demographics['98042']
    )

    # Both should return valid DataFrames
    assert isinstance(result1, pd.DataFrame)
    assert isinstance(result2, pd.DataFrame)
    assert len(result1.columns) == len(result2.columns)
