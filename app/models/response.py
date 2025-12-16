"""Pydantic models for API response validation."""

from pydantic import BaseModel, Field
from typing import Optional


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""

    prediction: float = Field(..., description="Predicted house price in USD")
    model_version: str = Field(..., description="Model version used")
    zipcode: str = Field(..., description="Property zipcode")
    demographics_found: bool = Field(..., description="Whether demographics were found")
    prediction_timestamp: str = Field(..., description="ISO 8601 timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 675234.50,
                "model_version": "v1",
                "zipcode": "98115",
                "demographics_found": True,
                "prediction_timestamp": "2025-12-15T10:23:45.123Z"
            }
        }


class ErrorResponse(BaseModel):
    """Response model for errors."""

    detail: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    zipcode: Optional[str] = Field(None, description="Zipcode if applicable")
