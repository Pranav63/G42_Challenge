from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from typing import List, Optional
from sqlalchemy.orm import Session
import tempfile
import os
import logging

from app.models import schemas, database
from app.services import ImageProcessor

router = APIRouter()
logger = logging.getLogger(__name__)

processor = ImageProcessor()


@router.post("/upload-csv", response_model=schemas.UploadResponse)
async def upload_csv_file(
    file: UploadFile = File(...), db: Session = Depends(database.get_db)
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "File must be CSV format")

    if file.size > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(400, "File too large (max 50MB)")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = processor.process_csv_file(tmp_path, db)

        return schemas.UploadResponse(
            success=True,
            message=f"Successfully processed {result['rows_processed']} rows",
            rows_processed=result["rows_processed"],
            processing_time_ms=result["processing_time_ms"],
            errors=result["errors"][:10],  # Limit errors in response
        )

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

    finally:
        os.unlink(tmp_path)


@router.get("/frames")
async def get_frames(
    depth_min: float = Query(..., description="Minimum depth"),
    depth_max: float = Query(..., description="Maximum depth"),
    colormap: Optional[str] = Query(None, description="Colormap name"),
    limit: int = Query(100, le=1000, description="Max frames to return"),
    db: Session = Depends(database.get_db),
):
    if depth_min > depth_max:
        raise HTTPException(400, "depth_min must be <= depth_max")

    frames = processor.get_frames_in_range(db, depth_min, depth_max, colormap, limit)

    if not frames:
        raise HTTPException(404, f"No frames found in range [{depth_min}, {depth_max}]")

    return frames


@router.get("/frames/statistics", response_model=schemas.ProcessingStatus)
async def get_frame_statistics(db: Session = Depends(database.get_db)):
    stats = processor.get_statistics(db)
    return schemas.ProcessingStatus(**stats)


@router.delete("/frames/all")
async def delete_all_frames(db: Session = Depends(database.get_db)):
    count = db.query(database.ImageFrame).count()
    db.query(database.ImageFrame).delete()
    db.commit()
    return {"message": f"Deleted {count} frames"}


@router.get("/colormaps")
async def get_available_colormaps():
    return {
        "colormaps": [
            "viridis",
            "jet",
            "hot",
            "cool",
            "hsv",
            "rainbow",
            "plasma",
            "inferno",
            "magma",
            "twilight",
        ]
    }
