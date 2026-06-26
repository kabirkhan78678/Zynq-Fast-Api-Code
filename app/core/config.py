from app.core.env import ROOT_ENV_FILE
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "FastAPI Structured App"
    
    # We can add more configuration properties here (like database URIs, API keys, etc.)
    # and they can be overridden by environment variables or a .env file.
    
    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=ROOT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
