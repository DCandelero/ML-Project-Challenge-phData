from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import time

from app.config import settings
from app.api import predict
from ml.model_loader import ModelService
from ml.demographics_service import DemographicsService
from ml.preprocessor import FeaturePreprocessor
from ml.predictor import PredictionService
from ml.feature_defaults import FeatureDefaultsService


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("housing.api")

# FastAPI app
app = FastAPI(
    title="Housing ML API",
    description="REST API for predicting house prices in Seattle area. Supports full predictions (18 features) and minimal predictions (8 features).",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load model and demographics on startup."""
    logger.info("Starting Housing ML API...")
    logger.info(f"Model version: {settings.model_version}")

    try:
        # Load model
        logger.info(f"Loading model from {settings.model_path}")
        model_service = ModelService(settings.model_path, settings.features_path)
        logger.info(f"Model loaded successfully with {len(model_service.get_features())} features")

        # Load demographics
        logger.info(f"Loading demographics from {settings.demographics_path}")
        demographics_service = DemographicsService(settings.demographics_path)
        logger.info(f"Demographics loaded: {len(demographics_service.get_available_zipcodes())} zipcodes")

        # Create preprocessor
        preprocessor = FeaturePreprocessor(model_service.get_features())

        # Create feature defaults service (for minimal endpoint)
        feature_defaults_service = FeatureDefaultsService(demographics_service)
        logger.info("Feature defaults service initialized")

        # Create prediction service
        prediction_service = PredictionService(
            model_service,
            demographics_service,
            preprocessor,
            feature_defaults_service
        )

        # Set prediction service in the predict module
        predict.set_prediction_service(prediction_service)

        logger.info("API startup complete ✓")

    except Exception as e:
        logger.error(f"Startup failed: {str(e)}", exc_info=True)
        raise


@app.middleware("http")
async def log_requests(request, call_next):
    """Log every HTTP request with timing."""
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time

    logger.info(
        f"method={request.method} "
        f"path={request.url.path} "
        f"status={response.status_code} "
        f"duration={duration:.3f}s"
    )

    return response


# Root endpoint
@app.get("/")
async def root():
    """Redirect to API documentation."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


# Health check endpoints
@app.get("/health")
async def health_check():
    """Liveness probe."""
    return {"status": "healthy"}


@app.get("/health/ready")
async def readiness_check():
    """Readiness probe."""
    prediction_service = predict.get_prediction_service()

    if prediction_service is None:
        return {
            "status": "not ready",
            "model_loaded": False,
            "demographics_loaded": False
        }

    return {
        "status": "ready",
        "model_loaded": True,
        "demographics_loaded": True,
        "demographics_count": len(prediction_service.demographics_service.get_available_zipcodes())
    }


@app.get("/api/v1/model/info")
async def model_info():
    """Get model metadata including evaluation metrics."""
    import json
    from pathlib import Path

    prediction_service = predict.get_prediction_service()

    if prediction_service is None:
        return {"error": "Model not loaded"}

    model_info = prediction_service.model_service.get_model_info()

    # Load evaluation metrics if they exist
    metrics_path = Path("model/evaluation_metrics.json")
    evaluation_metrics = {}
    cross_validation = {}

    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                metrics_data = json.load(f)
                evaluation_metrics = metrics_data.get('test_set_metrics', {})
                cross_validation = metrics_data.get('cross_validation', {})
        except Exception as e:
            logger.warning(f"Failed to load evaluation metrics: {e}")

    response = {
        "model_version": settings.model_version,
        "model_type": model_info.get('model_type'),
        "features": prediction_service.model_service.get_features(),
        "feature_count": model_info.get('feature_count')
    }

    # Add metrics if available
    if evaluation_metrics:
        response["evaluation_metrics"] = evaluation_metrics

    if cross_validation:
        response["cross_validation"] = cross_validation

    return response


# Register routers
app.include_router(predict.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True  # Hot reload for development
    )
