from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
from datetime import datetime


class CoinResponse(BaseModel):
    """Basic coin information response."""

    id: str
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    centroid: Tuple[int, int]  # (x, y)
    radius: int
    confidence: float = Field(ge=0, le=1)


class ImageUploadResponse(BaseModel):
    """Response after uploading and processing an image."""

    image_id: str
    filename: Optional[str]
    coin_count: int
    coins: List[CoinResponse]
    processing_time_ms: float
    message: str = "Image processed successfully"


class CoinDetailResponse(BaseModel):
    """Detailed information about a specific coin."""

    id: str
    image_id: str
    bounding_box: Tuple[int, int, int, int]
    centroid: Tuple[int, int]
    radius: int
    confidence: float
    created_at: datetime


class EvaluationMetrics(BaseModel):
    """Model evaluation metrics."""

    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1_score: float = Field(ge=0, le=1)
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)


class ImageListResponse(BaseModel):
    """Brief image information for listing."""

    id: str
    filename: Optional[str]
    coin_count: int
    upload_time: datetime


class ImageResponse(BaseModel):
    """Complete image information with all detected coins."""

    id: str
    filename: Optional[str]
    width: int
    height: int
    coin_count: int
    coins: List[CoinResponse]
    upload_time: datetime
    file_path: Optional[str] = None
