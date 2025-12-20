"""Prediction orchestration service."""

from typing import Dict, Any, Optional
from ml.model_loader import ModelService
from ml.demographics_service import DemographicsService, ZipcodeNotFoundError
from ml.preprocessor import FeaturePreprocessor
from ml.feature_defaults import FeatureDefaultsService


class PredictionService:
    """Orchestrates the full prediction workflow."""

    def __init__(
        self,
        model_service: ModelService,
        demographics_service: DemographicsService,
        preprocessor: FeaturePreprocessor,
        feature_defaults: Optional[FeatureDefaultsService] = None
    ):
        """Initialize prediction service with dependencies.

        Args:
            model_service: Service for model loading and prediction
            demographics_service: Service for demographics lookup
            preprocessor: Service for feature preprocessing
            feature_defaults: Optional service for feature defaults (minimal endpoint)
        """
        self.model_service = model_service
        self.demographics_service = demographics_service
        self.preprocessor = preprocessor
        self.feature_defaults = feature_defaults

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

    def predict_price_minimal(self, minimal_features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict price from minimal features (8 fields).

        This method:
        1. Expands minimal features to full features using defaults
        2. Calls predict_price with expanded features
        3. Adds metadata about defaults used

        Args:
            minimal_features: Dictionary with 8 essential features:
                - bedrooms, bathrooms, sqft_living, sqft_lot, floors
                - sqft_above, sqft_basement, zipcode

        Returns:
            Dictionary with prediction and metadata, including:
            - All fields from predict_price()
            - minimal_request: True
            - defaults_used: 10 (number of features set to defaults)

        Raises:
            ValueError: If feature defaults service not configured
            ZipcodeNotFoundError: If zipcode not in demographics data
        """
        if not self.feature_defaults:
            raise ValueError("Feature defaults service not configured for minimal predictions")

        # Expand minimal to full features using defaults
        full_features = self.feature_defaults.expand_minimal_features(minimal_features)

        # Use existing prediction logic
        result = self.predict_price(full_features)

        # Add metadata about minimal request
        result['minimal_request'] = True
        result['defaults_used'] = 10  # Number of features that were defaulted

        return result
