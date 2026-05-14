"""ADE20K evaluator: mIoU and pixel accuracy on the validation set.

The predict.py saves prediction masks as PNG files in a directory,
and writes a JSON file pointing to that directory:
  {"predictions_dir": "/path/to/masks/"}

Each PNG mask is a single-channel uint8 image with pixel values 0-149 (class indices).
Ground truth masks use ADE20K convention: pixel value 0=unlabeled, 1-150=classes.
We subtract 1 from GT so both use 0-149, and treat GT -1 as ignore.
"""

from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig


class ADE20KEvaluator(MetricEvaluatorBase):
    """Evaluate semantic segmentation predictions on ADE20K val set (150 classes)."""

    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        # Load metadata
        with open(os.path.join(test_data_dir, "meta.json")) as f:
            meta = json.load(f)
        num_classes = meta["num_classes"]

        # Load predictions JSON to get the mask directory
        with open(predictions_path) as f:
            pred_data = json.load(f)
        if "predictions_dir" not in pred_data:
            raise ValueError("Predictions JSON must contain 'predictions_dir'")
        pred_dir = pred_data["predictions_dir"]

        if not os.path.isdir(pred_dir):
            raise ValueError(f"Predictions directory not found: {pred_dir}")

        # Load ground truth masks from the label side of test_data_dir.
        gt_dir = os.path.join(test_data_dir, "labels", "annotations", "validation")
        if not os.path.isdir(gt_dir):
            # Backward-compatible fallback for older local prepared data.
            gt_dir = os.path.join(test_data_dir, "annotations", "validation")
        gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".png")])

        # Check prediction count
        pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".png")])
        if len(pred_files) != len(gt_files):
            raise ValueError(
                f"Prediction count mismatch: got {len(pred_files)} masks, "
                f"expected {len(gt_files)}"
            )

        # Accumulate confusion matrix
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        total_pixels = 0
        correct_pixels = 0

        for gt_file in gt_files:
            name = os.path.splitext(gt_file)[0]
            pred_file = os.path.join(pred_dir, f"{name}.png")

            if not os.path.exists(pred_file):
                raise ValueError(f"Missing prediction mask: {name}.png")

            # Load ground truth: ADE20K pixel values 0=unlabeled, 1-150=classes
            # Subtract 1 → 0-149 valid, -1 = ignore
            gt_mask = np.array(Image.open(os.path.join(gt_dir, gt_file)),
                               dtype=np.int64) - 1

            # Load prediction: pixel values must be 0-149 class indices.
            pred_mask = np.array(Image.open(pred_file), dtype=np.int64)

            if pred_mask.shape != gt_mask.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: prediction {pred_mask.shape} "
                    f"vs ground truth {gt_mask.shape}"
                )

            pred_min = int(pred_mask.min())
            pred_max = int(pred_mask.max())
            if pred_min < 0 or pred_max >= num_classes:
                raise ValueError(
                    f"Invalid class index in {name}.png: values must be in "
                    f"[0, {num_classes - 1}], got min={pred_min}, max={pred_max}"
                )

            # Mask out ignored pixels (gt == -1)
            valid = gt_mask >= 0
            gt_valid = gt_mask[valid]
            pred_valid = pred_mask[valid]

            # Pixel accuracy
            total_pixels += len(gt_valid)
            correct_pixels += int(np.sum(gt_valid == pred_valid))

            # Update confusion matrix (vectorized)
            np.add.at(confusion, (gt_valid, pred_valid), 1)

        # Compute per-class IoU
        intersection = np.diag(confusion)
        union = confusion.sum(axis=1) + confusion.sum(axis=0) - intersection

        valid_classes = union > 0
        per_class_iou = np.zeros(num_classes)
        per_class_iou[valid_classes] = (
            intersection[valid_classes] / union[valid_classes]
        )

        mIoU = float(per_class_iou[valid_classes].mean())
        pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0

        metrics = {
            "mIoU": round(mIoU, 4),
            "pixel_accuracy": round(float(pixel_accuracy), 4),
        }

        return EvalResult(
            metrics=metrics,
            primary_metric_name="mIoU",
            primary_metric_value=mIoU,
        )
