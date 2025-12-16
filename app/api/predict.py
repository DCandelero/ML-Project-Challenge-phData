"""Prediction API endpoints."""

from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
import logging

from app.models.request import HouseFeaturesRequest
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
