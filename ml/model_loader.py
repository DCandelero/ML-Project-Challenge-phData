"""Model loading service for pickle-based ML models."""

import json
import pickle
from typing import List, Dict, Any
import pandas as pd


class ModelService:
    """Service for loading and using pickled ML models."""

    def __init__(self, model_path: str, features_path: str):
        """Load model and feature names from disk.

        Args:
            model_path: Path to model.pkl file
            features_path: Path to model_features.json file
        """
        # Load pickled model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Load feature names
        with open(features_path, 'r') as f:
            self.features = json.load(f)

        # Validate model has predict method
        if not hasattr(self.model, 'predict'):
            raise ValueError("Loaded model does not have a predict method")

    def predict(self, features_df: pd.DataFrame) -> float:
        """Make prediction using loaded model.

        Args:
            features_df: DataFrame with features in correct order

        Returns:
            Predicted value (house price)
        """
        try:
            prediction = self.model.predict(features_df)
            return float(prediction[0])
        except Exception as e:
            raise ValueError(f"Model prediction failed: {str(e)}")

    def get_features(self) -> List[str]:
        """Get list of feature names required by the model.

        Returns:
            List of feature name strings
        """
        return self.features

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata.

        Returns:
            Dictionary with model information
        """
        return {
            'feature_count': len(self.features),
            'model_type': type(self.model).__name__
        }
