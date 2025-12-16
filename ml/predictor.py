"""Prediction orchestration service."""

from typing import Dict, Any
from ml.model_loader import ModelService
from ml.demographics_service import DemographicsService, ZipcodeNotFoundError
from ml.preprocessor import FeaturePreprocessor


class PredictionService:
    """Orchestrates the full prediction workflow."""

    def __init__(
        self,
        model_service: ModelService,
        demographics_service: DemographicsService,
        preprocessor: FeaturePreprocessor
    ):
        """Initialize prediction service with dependencies.

        Args:
            model_service: Service for model loading and prediction
            demographics_service: Service for demographics lookup
            preprocessor: Service for feature preprocessing
        """
        self.model_service = model_service
        self.demographics_service = demographics_service
        self.preprocessor = preprocessor

    def predict_price(self, house_features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict house price from house features.

        This method:
        1. Extracts zipcode from house features
        2. Looks up demographics for that zipcode
        3. Merges and preprocesses features
        4. Makes prediction using model
        5. Returns result with metadata

        Args:
            house_features: Dictionary of house features including zipcode

        Returns:
            Dictionary with prediction and metadata:
            {
                'prediction': float,
                'zipcode': str,
                'demographics_found': bool,
                'model_version': str
            }

        Raises:
            ZipcodeNotFoundError: If zipcode not in demographics data
            ValueError: If required features are missing
        """
        # Extract zipcode
        zipcode = house_features.get('zipcode')
        if not zipcode:
            raise ValueError("Zipcode is required in house features")

        # Lookup demographics for this zipcode
        demographics = self.demographics_service.get_demographics(zipcode)

        # Prepare features (merge house + demographics, align columns)
        features_df = self.preprocessor.prepare_features(
            house_features,
            demographics
        )

        # Get prediction
        prediction = self.model_service.predict(features_df)

        # Return result with metadata
        return {
            'prediction': prediction,
            'zipcode': zipcode,
            'demographics_found': True,
            'model_version': 'v1'
        }
