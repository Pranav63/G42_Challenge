from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings with defaults.
    Can be overridden by environment variables or .env file.
    """

    app_name: str = "Coin Detection API"
    app_version: str = "1.0.0"
    debug: bool = Field(False, env="DEBUG")

    api_prefix: str = "/api/v1"

    database_url: str = Field("sqlite:///./coin_detection.db", env="DATABASE_URL")

    storage_path: str = Field("./api_uploads", env="STORAGE_PATH")
    max_upload_size: int = Field(10 * 1024 * 1024, env="MAX_UPLOAD_SIZE")

    min_radius: int = Field(20, env="MIN_RADIUS")
    max_radius: int = Field(200, env="MAX_RADIUS")
    param1: int = Field(70, env="PARAM1")
    param2: int = Field(35, env="PARAM2")
    min_dist: int = Field(60, env="MIN_DIST")
    confidence_threshold: float = Field(0.4, env="CONFIDENCE_THRESHOLD")

    class Config:
        env_file = ".env"
        case_sensitive = False

    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)


settings = Settings()
settings.ensure_directories()
