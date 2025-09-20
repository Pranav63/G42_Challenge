from ultralytics import YOLO
import numpy as np
from typing import List, Tuple
import uuid
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class DetectedCoin:
    def __init__(self, x: int, y: int, radius: int, confidence: float = 1.0):
        self.id = str(uuid.uuid4())
        self.centroid = (x, y)
        self.radius = radius
        self.bounding_box = (x - radius, y - radius, 2 * radius, 2 * radius)
        self.confidence = confidence


class YOLOFinalDetector:
    def __init__(self):
        try:
            self.model = YOLO("coin_model_final.pt")

            # Load training images list
            with open("training_images_list.json", "r") as f:
                self.training_images = set(json.load(f))

            logger.info(
                f"YOLO loaded. Training set has {len(self.training_images)} images"
            )
        except:
            raise Exception(
                "No trained model found! Run train_with_annotations.py first"
            )

    def detect(
        self, image: np.ndarray, filename: str = None
    ) -> Tuple[List[DetectedCoin], bool]:
        """
        Detect coins and check for data leakage.
        """
        is_training_image = False
        if filename:
            basename = Path(filename).name
            if basename in self.training_images:
                is_training_image = True

        # Run detection
        results = self.model(image, conf=0.25, iou=0.45)

        coins = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])

                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    radius = int(max(x2 - x1, y2 - y1) / 2)

                    coin = DetectedCoin(center_x, center_y, radius, conf)
                    coins.append(coin)

        logger.info(f"Detected {len(coins)} coins in {filename}")
        return coins, is_training_image
