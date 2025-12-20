"""Unit tests for Feature Engineering"""

import pytest
import pandas as pd
import numpy as np
from ml.feature_engineering import engineer_features, get_engineered_feature_names


def test_house_age_calculated():
    """Test house_age feature is calculated correctly"""
    df = pd.DataFrame({
        'yr_built': [1990, 2000, 2020],
        'yr_renovated': [0, 0, 0],
        'bedrooms': [3, 3, 3],
        'bathrooms': [2.0, 2.0, 2.0],
        'sqft_living': [2000, 2000, 2000],
        'sqft_lot': [5000, 5000, 5000],
        'sqft_basement': [0, 0, 0]
    })
    result = engineer_features(df)

    assert 'house_age' in result.columns
    assert result['house_age'].iloc[0] == 2025 - 1990
    assert result['house_age'].iloc[1] == 2025 - 2000
    assert result['house_age'].iloc[2] == 2025 - 2020


def test_renovation_flag_created():
    """Test is_renovated binary flag"""
    df = pd.DataFrame({
        'yr_built': [1990, 1990],
        'yr_renovated': [0, 2010],
        'bedrooms': [3, 3],
        'bathrooms': [2.0, 2.0],
        'sqft_living': [2000, 2000],
        'sqft_lot': [5000, 5000],
        'sqft_basement': [0, 0]
    })
    result = engineer_features(df)

    assert 'is_renovated' in result.columns
    assert result['is_renovated'].iloc[0] == 0
    assert result['is_renovated'].iloc[1] == 1


def test_years_since_renovation_calculated():
    """Test years_since_renovation feature"""
    df = pd.DataFrame({
        'yr_built': [1990, 1990, 1990],
        'yr_renovated': [0, 2010, 2020],
        'bedrooms': [3, 3, 3],
        'bathrooms': [2.0, 2.0, 2.0],
        'sqft_living': [2000, 2000, 2000],
        'sqft_lot': [5000, 5000, 5000],
        'sqft_basement': [0, 0, 0]
    })
    result = engineer_features(df)

    assert 'years_since_renovation' in result.columns
    assert result['years_since_renovation'].iloc[0] == 0  # Never renovated
    assert result['years_since_renovation'].iloc[1] == 2025 - 2010
    assert result['years_since_renovation'].iloc[2] == 2025 - 2020


def test_bathroom_per_bedroom_ratio():
    """Test bath_per_bed ratio calculation"""
    df = pd.DataFrame({
        'bedrooms': [3, 4, 0],  # Include 0 to test division by zero protection
        'bathrooms': [2.0, 3.0, 1.0],
        'yr_built': [1990, 2000, 2010],
        'yr_renovated': [0, 0, 0],
        'sqft_living': [2000, 2000, 2000],
        'sqft_lot': [5000, 5000, 5000],
        'sqft_basement': [0, 0, 0]
    })
    result = engineer_features(df)

    assert 'bath_per_bed' in result.columns
    # +1 to bedrooms to avoid division by zero
    assert result['bath_per_bed'].iloc[0] == pytest.approx(2.0 / (3 + 1))
    assert result['bath_per_bed'].iloc[1] == pytest.approx(3.0 / (4 + 1))
    assert result['bath_per_bed'].iloc[2] == pytest.approx(1.0 / (0 + 1))  # Studio apartment


def test_living_to_lot_ratio():
    """Test living_to_lot_ratio calculation"""
    df = pd.DataFrame({
        'sqft_living': [2000, 3000],
        'sqft_lot': [5000, 6000],
        'bedrooms': [3, 3],
        'bathrooms': [2.0, 2.0],
        'yr_built': [1990, 2000],
        'yr_renovated': [0, 0],
        'sqft_basement': [0, 0]
    })
    result = engineer_features(df)

    assert 'living_to_lot_ratio' in result.columns
    assert result['living_to_lot_ratio'].iloc[0] == pytest.approx(2000 / (5000 + 1))
    assert result['living_to_lot_ratio'].iloc[1] == pytest.approx(3000 / (6000 + 1))


def test_basement_flag_created():
    """Test has_basement flag"""
    df = pd.DataFrame({
        'sqft_basement': [0, 500, 1000],
        'sqft_living': [2000, 2500, 3000],
        'bedrooms': [3, 3, 3],
        'bathrooms': [2.0, 2.0, 2.0],
        'sqft_lot': [5000, 5000, 5000],
        'yr_built': [1990, 2000, 2010],
        'yr_renovated': [0, 0, 0]
    })
    result = engineer_features(df)

    assert 'has_basement' in result.columns
    assert result['has_basement'].iloc[0] == 0
    assert result['has_basement'].iloc[1] == 1
    assert result['has_basement'].iloc[2] == 1


def test_basement_percentage_calculated():
    """Test basement_pct calculation"""
    df = pd.DataFrame({
        'sqft_basement': [0, 500, 1000],
        'sqft_living': [2000, 2000, 2000],
        'bedrooms': [3, 3, 3],
        'bathrooms': [2.0, 2.0, 2.0],
        'sqft_lot': [5000, 5000, 5000],
        'yr_built': [1990, 2000, 2010],
        'yr_renovated': [0, 0, 0]
    })
    result = engineer_features(df)

    assert 'basement_pct' in result.columns
    assert result['basement_pct'].iloc[0] == pytest.approx(0 / (2000 + 1))
    assert result['basement_pct'].iloc[1] == pytest.approx(500 / (2000 + 1))
    assert result['basement_pct'].iloc[2] == pytest.approx(1000 / (2000 + 1))


def test_total_rooms_calculated():
    """Test total_rooms feature"""
    df = pd.DataFrame({
        'bedrooms': [3, 4, 2],
        'bathrooms': [2.0, 2.5, 1.5],
        'yr_built': [1990, 2000, 2010],
        'yr_renovated': [0, 0, 0],
        'sqft_living': [2000, 2000, 2000],
        'sqft_lot': [5000, 5000, 5000],
        'sqft_basement': [0, 0, 0]
    })
    result = engineer_features(df)

    assert 'total_rooms' in result.columns
    assert result['total_rooms'].iloc[0] == 3 + 2.0
    assert result['total_rooms'].iloc[1] == 4 + 2.5
    assert result['total_rooms'].iloc[2] == 2 + 1.5


def test_quality_score_when_grade_and_condition_present():
    """Test quality_score is created when grade and condition available"""
    df = pd.DataFrame({
        'bedrooms': [3, 4],
        'bathrooms': [2.0, 2.5],
        'sqft_living': [2000, 2500],
        'sqft_lot': [5000, 6000],
        'sqft_basement': [0, 500],
        'yr_built': [1990, 2000],
        'yr_renovated': [0, 0],
        'grade': [7, 8],
        'condition': [3, 4]
    })
    result = engineer_features(df)

    assert 'quality_score' in result.columns
    assert result['quality_score'].iloc[0] == 7 * 3
    assert result['quality_score'].iloc[1] == 8 * 4


def test_quality_score_not_created_when_columns_missing():
    """Test quality_score is not created when grade or condition missing"""
    df = pd.DataFrame({
        'bedrooms': [3],
        'bathrooms': [2.0],
        'sqft_living': [2000],
        'sqft_lot': [5000],
        'sqft_basement': [0],
        'yr_built': [1990],
        'yr_renovated': [0]
    })
    result = engineer_features(df)

    # quality_score should not be created if grade/condition missing
    assert 'quality_score' not in result.columns


def test_get_engineered_feature_names():
    """Test get_engineered_feature_names returns correct list"""
    feature_names = get_engineered_feature_names()

    assert isinstance(feature_names, list)
    assert 'house_age' in feature_names
    assert 'is_renovated' in feature_names
    assert 'bath_per_bed' in feature_names
    assert 'total_rooms' in feature_names


def test_original_features_preserved():
    """Test original features are not removed"""
    df = pd.DataFrame({
        'bedrooms': [3],
        'bathrooms': [2.0],
        'sqft_living': [2000],
        'sqft_lot': [5000],
        'sqft_basement': [0],
        'yr_built': [1990],
        'yr_renovated': [0]
    })
    result = engineer_features(df)

    # Original features should still be present
    assert 'bedrooms' in result.columns
    assert 'bathrooms' in result.columns
    assert 'sqft_living' in result.columns
    assert 'yr_built' in result.columns
