"""Sensible defaults for missing features in minimal prediction endpoint."""

from typing import Dict, Any


class FeatureDefaultsService:
    """Provides default values for features not in minimal request."""

    def __init__(self, demographics_service=None):
        """
        Initialize feature defaults service.

        Args:
            demographics_service: Optional demographics service for zipcode lookups
        """
        self.demographics_service = demographics_service

    def expand_minimal_features(self, minimal_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expand minimal features (8) to full features (18).

        Strategy:
        - Binary flags (waterfront, view) → 0 (most common)
        - Ratings (condition, grade) → median values
        - Year built → 1980 (median year in dataset)
        - Year renovated → 0 (most houses never renovated)
        - Lat/long → Seattle city center
        - Living15/lot15 → same as living/lot (reasonable approximation)

        Args:
            minimal_features: Dict with 8 required features

        Returns:
            Dict with all 18 features
        """
        # Start with minimal features
        full_features = minimal_features.copy()

        # Add binary flags (most houses don't have these)
        full_features['waterfront'] = 0
        full_features['view'] = 0

        # Add ratings (use typical/median values)
        full_features['condition'] = 3  # Average condition (1-5 scale)
        full_features['grade'] = 7      # Typical grade (1-13 scale)

        # Year built (median from dataset is around 1980)
        full_features['yr_built'] = 1980

        # Year renovated (0 = never renovated, most common)
        full_features['yr_renovated'] = 0

        # Lat/long - use Seattle city center as approximation
        # In a production system, we'd look up by zipcode
        full_features['lat'] = 47.6062
        full_features['long'] = -122.3321

        # Living space / lot size of neighbors
        # Approximate as same as the property (reasonable default)
        full_features['sqft_living15'] = minimal_features['sqft_living']
        full_features['sqft_lot15'] = minimal_features['sqft_lot']

        return full_features

    def get_defaults_documentation(self) -> Dict[str, Dict[str, Any]]:
        """
        Return documentation of what defaults are used.

        Returns:
            Dict mapping feature name to {value, reason}
        """
        return {
            'waterfront': {
                'value': 0,
                'reason': 'Most houses are not waterfront properties'
            },
            'view': {
                'value': 0,
                'reason': 'Most houses have no special view rating'
            },
            'condition': {
                'value': 3,
                'reason': 'Average condition on 1-5 scale (3 = average)'
            },
            'grade': {
                'value': 7,
                'reason': 'Typical building grade on 1-13 scale (7 = average)'
            },
            'yr_built': {
                'value': 1980,
                'reason': 'Median year built in dataset'
            },
            'yr_renovated': {
                'value': 0,
                'reason': 'Most houses have never been renovated (0 = never)'
            },
            'lat': {
                'value': 47.6062,
                'reason': 'Seattle city center latitude (approximation)'
            },
            'long': {
                'value': -122.3321,
                'reason': 'Seattle city center longitude (approximation)'
            },
            'sqft_living15': {
                'value': 'same as sqft_living',
                'reason': 'Approximate neighborhood average as property value'
            },
            'sqft_lot15': {
                'value': 'same as sqft_lot',
                'reason': 'Approximate neighborhood average as property value'
            }
        }
