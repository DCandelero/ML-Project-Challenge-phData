"""Application configuration settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    log_level: str = "info"
    model_version: str = "v1"
    demographics_path: str = "data/zipcode_demographics.csv"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def model_path(self) -> str:
        """Get model path based on version."""
        if self.model_version == "v2":
            return "model/v2/model.pkl"
        elif self.model_version == "v1":
            return "model/v1/model.pkl"
        else:
            # Legacy/fallback
            return "model/model.pkl"

    @property
    def features_path(self) -> str:
        """Get features path based on version."""
        if self.model_version == "v2":
            return "model/v2/model_features.json"
        elif self.model_version == "v1":
            return "model/v1/model_features.json"
        else:
            # Legacy/fallback
            return "model/model_features.json"


settings = Settings()
