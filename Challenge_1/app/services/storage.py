"""
Local file storage service.
Why local storage: No external dependencies, simple deployment, cost-free.
"""

import uuid
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class StorageService:
    """Handle image file storage operations."""

    def __init__(self):
        self.storage_path = Path(settings.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def save_image(
        self, image_bytes: bytes, filename: Optional[str] = None
    ) -> tuple[str, str]:
        """
        Save image to local storage.

        Returns:
            Tuple of (image_id, file_path)
        """
        # Generate unique ID for image
        image_id = str(uuid.uuid4())

        # Determine file extension
        ext = ".jpg"
        if filename:
            ext = Path(filename).suffix or ".jpg"

        # Create file path
        file_name = f"{image_id}{ext}"
        file_path = self.storage_path / file_name

        # Save image
        with open(file_path, "wb") as f:
            f.write(image_bytes)

        logger.info(f"Saved image: {file_path}")
        return image_id, str(file_path)

    def load_image(self, image_id: str) -> Optional[np.ndarray]:
        """Load image from storage as numpy array."""
        # Try common extensions
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            file_path = self.storage_path / f"{image_id}{ext}"
            if file_path.exists():
                image = cv2.imread(str(file_path))
                return image

        logger.warning(f"Image not found: {image_id}")
        return None

    def load_image_by_path(self, file_path: str) -> np.ndarray:
        """Load image from file path and return as numpy array."""
        full_path = Path(file_path)

        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {file_path}")

        # Load image
        image = cv2.imread(str(full_path))
        if image is None:
            raise ValueError(f"Could not read image: {file_path}")

        return image

    def delete_image(self, image_id: str) -> bool:
        """Delete image from storage."""
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            file_path = self.storage_path / f"{image_id}{ext}"
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted image: {file_path}")
                return True
        return False

    @staticmethod
    def bytes_to_array(image_bytes: bytes) -> np.ndarray:
        """Convert image bytes to numpy array."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    @staticmethod
    def array_to_bytes(image_array: np.ndarray) -> bytes:
        """Convert numpy array to image bytes."""
        _, buffer = cv2.imencode(".jpg", image_array)
        return buffer.tobytes()
