"""
Model evaluation using Intersection over Union (IoU).
Why IoU: Standard metric for object detection,
measures overlap between predicted and actual bounding boxes.
"""

from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate coin detection performance."""

    def __init__(self, iou_threshold: float = 0.5):
        """
        Args:
            iou_threshold: Minimum IoU to consider detection as correct.
                         0.5 is standard for object detection.
        """
        self.iou_threshold = iou_threshold

    def calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.

        IoU = Area of Overlap / Area of Union

        Why IoU: Industry standard metric that handles both position
        and size accuracy in a single measure.
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def evaluate(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """
        Evaluate predictions against ground truth.

        Returns:
            Dictionary with precision, recall, and F1 score.
        """
        true_positives = 0
        false_positives = 0
        matched_gt = set()

        # Match each prediction to ground truth
        for pred in predictions:
            best_iou = 0
            best_gt_idx = -1

            for idx, gt in enumerate(ground_truth):
                if idx in matched_gt:
                    continue

                iou = self.calculate_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou >= self.iou_threshold:
                true_positives += 1
                matched_gt.add(best_gt_idx)
            else:
                false_positives += 1

        false_negatives = len(ground_truth) - len(matched_gt)

        # Calculate metrics
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1_score, 3),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }
