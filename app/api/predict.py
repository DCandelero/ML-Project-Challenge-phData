"""Prediction API endpoints."""

from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
import logging

from app.models.request import HouseFeaturesRequest, MinimalFeaturesRequest
from app.models.response import PredictionResponse
from ml.predictor import PredictionService
from ml.demographics_service import ZipcodeNotFoundError


router = APIRouter(prefix="/api/v1", tags=["predictions"])
logger = logging.getLogger("housing.api")

# Global prediction service (will be set by main.py)
_prediction_service: PredictionService = None


def set_prediction_service(service: PredictionService):
    """Set the global prediction service."""
    global _prediction_service
    _prediction_service = service


def get_prediction_service() -> PredictionService:
    """Dependency to get prediction service."""
    return _prediction_service


@router.post("/predict", response_model=PredictionResponse)
async def predict_price(
    request: HouseFeaturesRequest,
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Predict house price based on property features.

    Demographics are automatically added on the backend using zipcode.
    The API validates all inputs and returns a prediction with metadata.
    """
    try:
        # Convert Pydantic model to dict
        house_features = request.model_dump()

        # Get prediction
        result = prediction_service.predict_price(house_features)

        # Add timestamp
        result["prediction_timestamp"] = datetime.utcnow().isoformat() + "Z"

        logger.info(
            f"Prediction successful: zipcode={result['zipcode']} "
            f"prediction=${result['prediction']:.2f}"
        )

        return PredictionResponse(**result)

    except ZipcodeNotFoundError as e:
        logger.error(f"Zipcode not found: {request.zipcode}")
        raise HTTPException(
            status_code=404,
            detail=f"Demographics not found for zipcode {request.zipcode}"
        )
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction"
        )


@router.post("/predict/minimal", response_model=PredictionResponse)
async def predict_price_minimal(
    request: MinimalFeaturesRequest,
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Predict house price from minimal features (BONUS endpoint).

    Only requires 8 essential features:
    - bedrooms, bathrooms, sqft_living, sqft_lot, floors
    - sqft_above, sqft_basement, zipcode

    Other features are set to sensible defaults:
    - waterfront=0, view=0, condition=3, grade=7
    - yr_built=1980, yr_renovated=0
    - lat/long from Seattle center, sqft_living15/lot15 approximated

    Demographics are automatically added using zipcode (same as /predict).
    """
    try:
        # Convert Pydantic model to dict
        minimal_features = request.model_dump()

        # Use minimal prediction method
        result = prediction_service.predict_price_minimal(minimal_features)

        # Add timestamp
        result["prediction_timestamp"] = datetime.utcnow().isoformat() + "Z"

        logger.info(
            f"Minimal prediction successful: zipcode={result['zipcode']} "
            f"prediction=${result['prediction']:.2f} (minimal request, {result['defaults_used']} defaults)"
        )

        return PredictionResponse(**result)

    except ZipcodeNotFoundError as e:
        logger.error(f"Zipcode not found: {request.zipcode}")
        raise HTTPException(
            status_code=404,
            detail=f"Demographics not found for zipcode {request.zipcode}"
        )
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Minimal prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction"
        )


@router.get("/predict/minimal/defaults")
async def get_minimal_defaults():
    """
    Get documentation of default values used by minimal endpoint.

    Returns information about what default values are used for the 10 features
    not required in the minimal prediction endpoint.
    """
    from ml.feature_defaults import FeatureDefaultsService

    defaults_service = FeatureDefaultsService(None)

    return {
        "description": "Default values used when features not provided in minimal request",
        "total_defaults": 10,
        "defaults": defaults_service.get_defaults_documentation()
    }
