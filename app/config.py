"""Application configuration settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    log_level: str = "info"
    model_version: str = "v1"
    demographics_path: str = "data/zipcode_demographics.csv"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Allow direct override of model paths (useful for testing)
    model_path: str = ""
    features_path: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set default paths based on version if not explicitly provided
        if not self.model_path:
            if self.model_version == "v2":
                self.model_path = "model/v2/model.pkl"
            elif self.model_version == "v1":
                self.model_path = "model/v1/model.pkl"
            else:
                self.model_path = "model/model.pkl"

        if not self.features_path:
            if self.model_version == "v2":
                self.features_path = "model/v2/model_features.json"
            elif self.model_version == "v1":
                self.features_path = "model/v1/model_features.json"
            else:
                self.features_path = "model/model_features.json"


settings = Settings()
