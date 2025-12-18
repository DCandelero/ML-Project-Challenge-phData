"""Pydantic models for API request validation."""

from pydantic import BaseModel, Field


class HouseFeaturesRequest(BaseModel):
    """Request model for house features (18 fields from future_unseen_examples.csv)."""

    bedrooms: int = Field(..., ge=0, le=33, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, le=10, description="Number of bathrooms")
    sqft_living: int = Field(..., gt=0, description="Square feet of living space")
    sqft_lot: int = Field(..., gt=0, description="Square feet of lot")
    floors: float = Field(..., ge=1, le=3.5, description="Number of floors")
    waterfront: int = Field(..., ge=0, le=1, description="Waterfront property (0/1)")
    view: int = Field(..., ge=0, le=4, description="View rating (0-4)")
    condition: int = Field(..., ge=1, le=5, description="Condition rating (1-5)")
    grade: int = Field(..., ge=1, le=13, description="Building grade (1-13)")
    sqft_above: int = Field(..., ge=0, description="Square feet above ground")
    sqft_basement: int = Field(..., ge=0, description="Square feet of basement")
    yr_built: int = Field(..., ge=1800, le=2025, description="Year built")
    yr_renovated: int = Field(..., ge=0, le=2025, description="Year renovated (0 if never)")
    zipcode: str = Field(..., pattern=r'^\d{5}$', description="5-digit zipcode")
    lat: float = Field(..., ge=47.0, le=48.0, description="Latitude")
    long: float = Field(..., ge=-123.0, le=-121.0, description="Longitude")
    sqft_living15: int = Field(..., gt=0, description="Living space of nearest 15 neighbors")
    sqft_lot15: int = Field(..., gt=0, description="Lot size of nearest 15 neighbors")

    class Config:
        json_schema_extra = {
            "example": {
                "bedrooms": 3,
                "bathrooms": 2.5,
                "sqft_living": 2220,
                "sqft_lot": 6380,
                "floors": 1.5,
                "waterfront": 0,
                "view": 0,
                "condition": 4,
                "grade": 8,
                "sqft_above": 1660,
                "sqft_basement": 560,
                "yr_built": 1931,
                "yr_renovated": 0,
                "zipcode": "98115",
                "lat": 47.6974,
                "long": -122.313,
                "sqft_living15": 950,
                "sqft_lot15": 6380
            }
        }


class MinimalFeaturesRequest(BaseModel):
    """Minimal request model - only 8 essential features (BONUS endpoint)."""

    bedrooms: int = Field(..., ge=0, le=33, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, le=10, description="Number of bathrooms")
    sqft_living: int = Field(..., gt=0, description="Square feet of living space")
    sqft_lot: int = Field(..., gt=0, description="Square feet of lot")
    floors: float = Field(..., ge=1, le=3.5, description="Number of floors")
    sqft_above: int = Field(..., ge=0, description="Square feet above ground")
    sqft_basement: int = Field(..., ge=0, description="Square feet of basement")
    zipcode: str = Field(..., pattern=r'^\d{5}$', description="5-digit zipcode")

    class Config:
        json_schema_extra = {
            "example": {
                "bedrooms": 3,
                "bathrooms": 2.5,
                "sqft_living": 2220,
                "sqft_lot": 6380,
                "floors": 1.5,
                "sqft_above": 1660,
                "sqft_basement": 560,
                "zipcode": "98115"
            }
        }
