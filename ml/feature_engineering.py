"""Feature engineering for improved house price predictions."""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features for better predictions.

    Strategy:
    - Age-related features (depreciation)
    - Renovation indicators (value adds)
    - Ratios (quality/density indicators)
    - Binary flags (basement, etc.)
    - Composite scores

    Args:
        df: DataFrame with original features

    Returns:
        DataFrame with original + engineered features
    """
    df = df.copy()

    # 1. House age (strong predictor - older = less valuable generally)
    df['house_age'] = 2025 - df['yr_built']

    # 2. Renovation flag (binary indicator)
    df['is_renovated'] = (df['yr_renovated'] > 0).astype(int)

    # 3. Years since renovation (0 if never renovated)
    df['years_since_renovation'] = np.where(
        df['yr_renovated'] > 0,
        2025 - df['yr_renovated'],
        0
    )

    # 4. Bathroom to bedroom ratio (quality indicator)
    # More bathrooms per bedroom = luxury
    # Add 1 to bedrooms to avoid division by zero
    df['bath_per_bed'] = df['bathrooms'] / (df['bedrooms'] + 1)

    # 5. Living space to lot ratio (density indicator)
    # High ratio = urban/dense, low = suburban with big yard
    # Add 1 to lot to avoid division by zero
    df['living_to_lot_ratio'] = df['sqft_living'] / (df['sqft_lot'] + 1)

    # 6. Basement flag (has basement or not)
    df['has_basement'] = (df['sqft_basement'] > 0).astype(int)

    # 7. Basement percentage (how much of house is basement)
    # Add 1 to living to avoid division by zero
    df['basement_pct'] = df['sqft_basement'] / (df['sqft_living'] + 1)

    # 8. Total rooms estimate (bedrooms + bathrooms)
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    # Optional: Quality score if grade and condition available
    # (These might not be in minimal features but useful for full model)
    if 'grade' in df.columns and 'condition' in df.columns:
        df['quality_score'] = df['grade'] * df['condition']

    return df


def get_engineered_feature_names() -> list:
    """
    Return list of engineered feature names.

    Useful for tracking which features are derived vs. original.
    """
    return [
        'house_age',
        'is_renovated',
        'years_since_renovation',
        'bath_per_bed',
        'living_to_lot_ratio',
        'has_basement',
        'basement_pct',
        'total_rooms',
        'quality_score'  # Optional (only if grade/condition available)
    ]


def get_feature_descriptions() -> dict:
    """
    Return descriptions of engineered features.

    Useful for documentation and interpretability.
    """
    return {
        'house_age': {
            'formula': '2025 - yr_built',
            'rationale': 'Older houses typically depreciate in value',
            'type': 'derived_numeric'
        },
        'is_renovated': {
            'formula': '1 if yr_renovated > 0 else 0',
            'rationale': 'Renovations typically increase home value',
            'type': 'derived_binary'
        },
        'years_since_renovation': {
            'formula': '2025 - yr_renovated (0 if never)',
            'rationale': 'Recent renovations have more impact',
            'type': 'derived_numeric'
        },
        'bath_per_bed': {
            'formula': 'bathrooms / (bedrooms + 1)',
            'rationale': 'Higher ratio indicates luxury/quality',
            'type': 'derived_ratio'
        },
        'living_to_lot_ratio': {
            'formula': 'sqft_living / (sqft_lot + 1)',
            'rationale': 'Indicates urban density vs. suburban space',
            'type': 'derived_ratio'
        },
        'has_basement': {
            'formula': '1 if sqft_basement > 0 else 0',
            'rationale': 'Basement adds value and space',
            'type': 'derived_binary'
        },
        'basement_pct': {
            'formula': 'sqft_basement / (sqft_living + 1)',
            'rationale': 'Percentage of house that is basement',
            'type': 'derived_ratio'
        },
        'total_rooms': {
            'formula': 'bedrooms + bathrooms',
            'rationale': 'Overall size indicator',
            'type': 'derived_numeric'
        },
        'quality_score': {
            'formula': 'grade * condition',
            'rationale': 'Composite quality indicator',
            'type': 'derived_composite'
        }
    }
