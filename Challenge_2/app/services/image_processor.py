import numpy as np
import pandas as pd
import cv2
from typing import List, Dict, Optional, Tuple
import logging
from sqlalchemy.orm import Session
import time

from app.models.database import ImageFrame
from app.core.config import settings

logger = logging.getLogger(__name__)


class ImageProcessor:
    def __init__(self):
        self.original_width = settings.original_width
        self.target_width = settings.target_width
        self.colormaps = {
            "viridis": cv2.COLORMAP_VIRIDIS,
            "jet": cv2.COLORMAP_JET,
            "hot": cv2.COLORMAP_HOT,
            "cool": cv2.COLORMAP_COOL,
            "hsv": cv2.COLORMAP_HSV,
            "rainbow": cv2.COLORMAP_RAINBOW,
            "plasma": cv2.COLORMAP_PLASMA,
            "inferno": cv2.COLORMAP_INFERNO,
            "magma": cv2.COLORMAP_MAGMA,
            "twilight": cv2.COLORMAP_TWILIGHT,
        }

    def resize_image_row(self, row_data: np.ndarray) -> np.ndarray:
        if len(row_data) == self.target_width:
            return row_data

        row_2d = row_data.reshape(1, -1)
        resized = cv2.resize(
            row_2d.astype(np.float32),
            (self.target_width, 1),
            interpolation=cv2.INTER_LINEAR,
        )

        return resized.flatten()

    def process_csv_file(self, csv_path: str, db: Session) -> Dict:
        start_time = time.time()
        errors = []
        rows_processed = 0

        try:
            logger.info(f"Starting CSV processing: {csv_path}")

            df = pd.read_csv(csv_path)

            if "depth" not in df.columns:
                raise ValueError("CSV must contain 'depth' column")

            image_columns = [col for col in df.columns if col.startswith("col")]

            if len(image_columns) == 0:
                raise ValueError("No image columns (col1, col2, ...) found")

            logger.info(f"Found {len(image_columns)} image columns")

            batch_size = 100
            batch = []

            for idx, row in df.iterrows():
                try:
                    depth_str = row["depth"]
                    logger.debug(f"Row {idx} content: {row.to_dict()}")

                    depth = float(depth_str)

                    if np.isnan(depth):
                        error_msg = f"Row {idx}: depth is NaN, skipping. Raw depth value: {depth_str}"
                        logger.warning(error_msg)
                        errors.append(error_msg)
                        continue

                    pixel_values = row[image_columns].values.astype(np.float32)

                    if np.isnan(pixel_values).any():
                        pixel_values = np.nan_to_num(pixel_values, 0)

                    if pixel_values.max() <= 1.0:
                        pixel_values = pixel_values * 255

                    pixel_values = np.clip(pixel_values, 0, 255)

                    resized = self.resize_image_row(pixel_values)

                    existing = db.query(ImageFrame).filter_by(depth=depth).first()

                    if existing:
                        existing.image_data = resized.astype(np.uint8).tobytes()
                        existing.width = self.target_width
                    else:
                        frame = ImageFrame.from_array(
                            depth, resized, len(image_columns)
                        )
                        batch.append(frame)

                    if len(batch) >= batch_size:
                        db.add_all(batch)
                        db.commit()
                        batch = []

                    rows_processed += 1

                    if rows_processed % 500 == 0:
                        logger.info(f"Processed {rows_processed} rows")

                except Exception as e:
                    error_msg = f"Row {idx}: {str(e)}"
                    errors.append(error_msg)
                    logger.warning(error_msg)
                    continue

            if batch:
                db.add_all(batch)
                db.commit()

            processing_time = (time.time() - start_time) * 1000

            logger.info(
                f"Processing complete: {rows_processed} rows in {processing_time:.2f}ms"
            )

            return {
                "rows_processed": rows_processed,
                "processing_time_ms": processing_time,
                "errors": errors,
            }

        except Exception as e:
            logger.error(f"CSV processing failed: {str(e)}")
            raise

    def apply_colormap(
        self, grayscale_data: np.ndarray, colormap_name: str
    ) -> np.ndarray:
        if colormap_name not in self.colormaps:
            colormap_name = "viridis"

        if grayscale_data.max() <= 1.0:
            grayscale_data = grayscale_data * 255

        grayscale_uint8 = grayscale_data.astype(np.uint8)

        colored = cv2.applyColorMap(grayscale_uint8, self.colormaps[colormap_name])

        return colored

    def get_frames_in_range(
        self,
        db: Session,
        depth_min: float,
        depth_max: float,
        colormap: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict]:

        frames = (
            db.query(ImageFrame)
            .filter(ImageFrame.depth >= depth_min, ImageFrame.depth <= depth_max)
            .order_by(ImageFrame.depth)
            .limit(limit)
            .all()
        )

        results = []

        for frame in frames:
            image_array = frame.to_array()

            if colormap:
                image_array = self.apply_colormap(image_array, colormap)
                data = image_array.tolist()
            else:
                data = [[float(val) for val in image_array[0]]]

            results.append(
                {
                    "depth": frame.depth,
                    "width": frame.width,
                    "height": frame.height,
                    "data": data,
                }
            )

        return results

    def get_statistics(self, db: Session) -> Dict:
        count = db.query(ImageFrame).count()

        if count == 0:
            return {"total_frames": 0, "depth_range": (0, 0), "storage_size_mb": 0}

        min_depth = db.query(ImageFrame.depth).order_by(ImageFrame.depth).first()[0]
        max_depth = (
            db.query(ImageFrame.depth).order_by(ImageFrame.depth.desc()).first()[0]
        )

        storage_size = count * self.target_width / (1024 * 1024)

        return {
            "total_frames": count,
            "depth_range": (min_depth, max_depth),
            "storage_size_mb": round(storage_size, 2),
        }
