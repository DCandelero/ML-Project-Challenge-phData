"""Feature preprocessing for model input preparation."""

import pandas as pd
from typing import Dict, List, Any


class FeaturePreprocessor:
    """Preprocessor for merging and aligning features for model input."""

    def __init__(self, model_features: List[str]):
        """Initialize preprocessor with expected model features.

        Args:
            model_features: List of feature names in the order expected by model
        """
        self.model_features = model_features

    def prepare_features(
        self,
        house_data: Dict[str, Any],
        demographics: Dict[str, Any]
    ) -> pd.DataFrame:
        """Prepare features for model prediction.

        Combines house features with demographics, creates a DataFrame,
        and ensures column order matches model expectations.

        Args:
            house_data: Dictionary of house features (without demographics)
            demographics: Dictionary of demographic features for the zipcode

        Returns:
            DataFrame with single row, columns aligned to model_features

        Raises:
            ValueError: If required features are missing
        """
        # Combine house data and demographics
        combined_features = {**house_data, **demographics}

        # Remove zipcode if present (not used as a feature)
        combined_features.pop('zipcode', None)

        # Create DataFrame with single row
        df = pd.DataFrame([combined_features])

        # Select and reorder columns to match model features
        try:
            # This will raise KeyError if any required feature is missing
            df_aligned = df[self.model_features]
        except KeyError as e:
            missing_features = set(self.model_features) - set(df.columns)
            raise ValueError(
                f"Missing required features: {missing_features}. "
                f"Available features: {set(df.columns)}"
            )

        return df_aligned
