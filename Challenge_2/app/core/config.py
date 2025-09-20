from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Image Processing API"
    app_version: str = "1.0.0"

    database_url: str = Field("sqlite:///./image_frames.db", env="DATABASE_URL")

    original_width: int = 200
    target_width: int = 150

    max_upload_size: int = Field(50 * 1024 * 1024, env="MAX_UPLOAD_SIZE")  # 50MB

    class Config:
        env_file = ".env"


settings = Settings()
