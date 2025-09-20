"""
Core coin detection logic using OpenCV.
Why Hough Circle Transform: Purpose-built for circle detection,
efficient, and works well with clear circular objects.
"""

import cv2
import numpy as np
from typing import List
import logging
import uuid

logger = logging.getLogger(__name__)


class DetectedCoin:
    def __init__(self, x: int, y: int, radius: int, confidence: float = 1.0):
        self.id = str(uuid.uuid4())
        self.centroid = (x, y)
        self.radius = radius
        self.bounding_box = (x - radius, y - radius, 2 * radius, 2 * radius)
        self.confidence = confidence


class CoinDetector:
    """
    Simplified detector - focus on what works, remove what doesn't.
    """

    def detect(self, image: np.ndarray) -> List[DetectedCoin]:
        """
        Simple, reliable coin detection.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Simple preprocessing - just blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        height, width = gray.shape[:2]

        # Run detection with reasonable parameters
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(min(height, width) * 0.08),  # 8% of image size
            param1=50,  # Canny edge threshold
            param2=30,  # Accumulator threshold
            minRadius=int(min(height, width) * 0.02),  # 2% of image
            maxRadius=int(min(height, width) * 0.25),  # 25% of image
        )

        coins = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            # Simple deduplication based on position
            seen_positions = []
            for x, y, r in circles:
                # Check if we already have a coin near this position
                is_duplicate = False
                for sx, sy, sr in seen_positions:
                    distance = np.sqrt((x - sx) ** 2 + (y - sy) ** 2)
                    if distance < max(r, sr) * 0.5:  # Within 50% of radius
                        is_duplicate = True
                        break

                if not is_duplicate:
                    seen_positions.append((x, y, r))
                    # Don't filter by confidence - just accept what Hough finds
                    coin = DetectedCoin(x, y, r, confidence=0.8)
                    coins.append(coin)

        logger.info(f"Detected {len(coins)} coins")
        return coins
