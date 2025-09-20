"""
FastAPI endpoints for coin detection.
Why FastAPI: Modern, fast, automatic API documentation, type safety.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
import time
import cv2
import numpy as np
from fastapi.responses import Response
import logging
from pathlib import Path
from typing import List

from app.models import database, schemas
from app.services.storage import StorageService
from app.services.coin_detector import CoinDetector
from app.services.evaluation import Evaluator

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
storage = StorageService()
fallback_detector = CoinDetector()

# Load YOLO if available
try:
    from app.services.yolo_detector import YOLOFinalDetector

    yolo_detector = YOLOFinalDetector()
    has_yolo = True
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.warning(f"YOLO not available, using fallback: {e}")
    has_yolo = False


@router.post("/images/upload", response_model=schemas.ImageUploadResponse)
async def upload_image(
    file: UploadFile = File(...), db: Session = Depends(database.get_db)
):
    """
    Upload image and detect coins using trained YOLO model.
    Warns if testing on training data.
    """
    start_time = time.time()

    # Validation
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    if file.size > 10 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 10MB)")

    try:
        # Read and save image
        image_bytes = await file.read()
        image_id, file_path = storage.save_image(image_bytes, file.filename)

        # Convert to numpy array
        image_array = storage.bytes_to_array(image_bytes)
        height, width = image_array.shape[:2]

        # Detect coins
        warning_message = ""

        if has_yolo:
            # Use trained YOLO
            detected_coins, is_training_image = yolo_detector.detect(
                image_array, file.filename
            )

            if is_training_image:
                warning_message = " ⚠️ WARNING: This image was in the training set!"
                logger.warning(
                    f"Data leakage detected: {file.filename} was used in training"
                )
        else:
            # Fallback detector
            detected_coins = fallback_detector.detect(image_array)
            warning_message = " (Using fallback - YOLO not available)"

        # Save to database
        image_record = database.ImageModel(
            id=image_id,
            filename=file.filename,
            file_path=file_path,
            width=width,
            height=height,
            coin_count=len(detected_coins),
        )
        db.add(image_record)

        # Save coins
        coin_responses = []
        for coin in detected_coins:
            coin_record = database.CoinModel(
                id=coin.id,
                image_id=image_id,
                bbox_x=coin.bounding_box[0],
                bbox_y=coin.bounding_box[1],
                bbox_width=coin.bounding_box[2],
                bbox_height=coin.bounding_box[3],
                centroid_x=coin.centroid[0],
                centroid_y=coin.centroid[1],
                radius=coin.radius,
                confidence=coin.confidence,
            )
            db.add(coin_record)

            coin_responses.append(
                schemas.CoinResponse(
                    id=coin.id,
                    bounding_box=coin.bounding_box,
                    centroid=coin.centroid,
                    radius=coin.radius,
                    confidence=coin.confidence,
                )
            )

        db.commit()

        processing_time = (time.time() - start_time) * 1000

        return schemas.ImageUploadResponse(
            image_id=image_id,
            filename=file.filename,
            coin_count=len(detected_coins),
            coins=coin_responses,
            processing_time_ms=processing_time,
            message=f"Detected {len(detected_coins)} coins{warning_message}",
        )

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        db.rollback()
        raise HTTPException(500, f"Error: {str(e)}")


@router.get("/images", response_model=List[schemas.ImageListResponse])
async def list_images(db: Session = Depends(database.get_db)):
    """List all uploaded images."""
    images = db.query(database.ImageModel).all()

    return [
        schemas.ImageListResponse(
            id=image.id,
            filename=image.filename,
            coin_count=image.coin_count,
            upload_time=image.upload_time,  # Use upload_time from your model
        )
        for image in images
    ]


@router.get("/images/{image_id}", response_model=schemas.ImageResponse)
async def get_image(image_id: str, db: Session = Depends(database.get_db)):
    """Get complete image details with all detected coins."""
    image = db.query(database.ImageModel).filter_by(id=image_id).first()

    if not image:
        raise HTTPException(404, f"Image {image_id} not found")

    coins = db.query(database.CoinModel).filter_by(image_id=image_id).all()

    coin_responses = [
        schemas.CoinResponse(
            id=coin.id,
            bounding_box=(coin.bbox_x, coin.bbox_y, coin.bbox_width, coin.bbox_height),
            centroid=(
                coin.centroid_x,
                coin.centroid_y,
            ),  # Keep as int since your model uses Integer
            radius=coin.radius,
            confidence=coin.confidence,
        )
        for coin in coins
    ]

    return schemas.ImageResponse(
        id=image.id,
        filename=image.filename,
        width=image.width,
        height=image.height,
        coin_count=image.coin_count,
        coins=coin_responses,
        upload_time=image.upload_time,  # Use upload_time from your model
        file_path=image.file_path,
    )


@router.get("/images/{image_id}/coins", response_model=List[schemas.CoinResponse])
async def get_image_coins(image_id: str, db: Session = Depends(database.get_db)):
    """Get all coins detected in an image."""
    coins = db.query(database.CoinModel).filter_by(image_id=image_id).all()

    if not coins:
        raise HTTPException(404, f"No coins found for image {image_id}")

    return [
        schemas.CoinResponse(
            id=coin.id,
            bounding_box=(coin.bbox_x, coin.bbox_y, coin.bbox_width, coin.bbox_height),
            centroid=(coin.centroid_x, coin.centroid_y),
            radius=coin.radius,
            confidence=coin.confidence,
        )
        for coin in coins
    ]


@router.get("/coins/{coin_id}", response_model=schemas.CoinDetailResponse)
async def get_coin_details(coin_id: str, db: Session = Depends(database.get_db)):
    """Get detailed information about a specific coin."""
    coin = db.query(database.CoinModel).filter_by(id=coin_id).first()

    if not coin:
        raise HTTPException(404, f"Coin {coin_id} not found")

    return schemas.CoinDetailResponse(
        id=coin.id,
        image_id=coin.image_id,
        bounding_box=(coin.bbox_x, coin.bbox_y, coin.bbox_width, coin.bbox_height),
        centroid=(coin.centroid_x, coin.centroid_y),
        radius=coin.radius,
        confidence=coin.confidence,
        created_at=coin.created_at,  # This field exists in CoinModel
    )


@router.get("/images/{image_id}/mask")
async def get_image_mask(image_id: str, db: Session = Depends(database.get_db)):
    """Generate and return mask visualization showing detected circular objects."""
    image = db.query(database.ImageModel).filter_by(id=image_id).first()
    if not image:
        raise HTTPException(404, f"Image {image_id} not found")

    coins = db.query(database.CoinModel).filter_by(image_id=image_id).all()

    try:
        # Load original image
        image_array = storage.load_image_by_path(image.file_path)

        # Create mask
        mask = np.zeros((image.height, image.width), dtype=np.uint8)

        # Draw circles on mask for each detected coin
        for coin in coins:
            center = (int(coin.centroid_x), int(coin.centroid_y))
            radius = int(coin.radius)
            cv2.circle(mask, center, radius, 255, -1)  # Filled circle
            cv2.circle(mask, center, radius, 128, 2)  # Border

        # Create side-by-side visualization
        combined = create_mask_visualization(image_array, mask)

        # Encode as PNG
        _, buffer = cv2.imencode(".png", combined)

        return Response(
            content=buffer.tobytes(),
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename=mask_{image_id}.png"},
        )

    except Exception as e:
        logger.error(f"Error generating mask: {str(e)}")
        raise HTTPException(500, f"Error generating mask: {str(e)}")


@router.post("/evaluate", response_model=schemas.EvaluationMetrics)
async def evaluate_detection(
    predictions: List[dict], ground_truth: List[dict], iou_threshold: float = 0.5
):
    """
    Evaluate detection performance using IoU metrics.

    Example request body:
    {
        "predictions": [
            {"bbox": [10, 20, 50, 50]}
        ],
        "ground_truth": [
            {"bbox": [12, 22, 48, 48]}
        ]
    }
    """
    evaluator = Evaluator(iou_threshold=iou_threshold)
    metrics = evaluator.evaluate(predictions, ground_truth)
    return schemas.EvaluationMetrics(**metrics)


# Helper function for mask visualization
def create_mask_visualization(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create side-by-side visualization of image and mask."""
    # Ensure image is in color
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Convert mask to color for visualization
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Create overlay version
    overlay = image.copy()
    overlay[mask > 0] = [0, 255, 0]  # Green for detected areas
    blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    # Create side-by-side comparison
    combined = np.hstack([image, mask_colored, blended])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(
        combined, "Mask", (image.shape[1] + 10, 30), font, 1, (255, 255, 255), 2
    )
    cv2.putText(
        combined, "Overlay", (image.shape[1] * 2 + 10, 30), font, 1, (255, 255, 255), 2
    )

    return combined
