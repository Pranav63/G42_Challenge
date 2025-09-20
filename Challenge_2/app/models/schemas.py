from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class FrameResponse(BaseModel):
    depth: float
    width: int
    height: int
    data: List[List[float]]


class FrameVisualization(BaseModel):
    depth: float
    width: int
    height: int
    data: List[List[List[int]]]  # RGB values for colored image


class UploadResponse(BaseModel):
    success: bool
    message: str
    rows_processed: int
    processing_time_ms: float
    errors: List[str] = []


class FrameQuery(BaseModel):
    depth_min: float = Field(..., description="Minimum depth value")
    depth_max: float = Field(..., description="Maximum depth value")
    colormap: Optional[str] = Field(None, description="Optional colormap name")


class ProcessingStatus(BaseModel):
    total_frames: int
    depth_range: tuple
    storage_size_mb: float
