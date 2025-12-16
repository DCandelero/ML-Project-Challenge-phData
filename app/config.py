"""Application configuration settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    log_level: str = "info"
    model_version: str = "v1"
    model_path: str = "model/model.pkl"
    features_path: str = "model/model_features.json"
    demographics_path: str = "data/zipcode_demographics.csv"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
