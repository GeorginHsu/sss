"""DIV2K 4x SR evaluator: PSNR and SSIM on Y channel with border crop.

Standard SR evaluation protocol:
  1. Convert both SR and HR images to YCbCr color space
  2. Extract Y channel only
  3. Crop `scale` pixels from each border (4 pixels for x4)
  4. Compute PSNR = 10 * log10(255^2 / MSE)
  5. Compute SSIM on Y channel (11x11 Gaussian window, sigma=1.5, matching MATLAB ssim())
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig


def rgb_to_ycbcr(img: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 image to YCbCr. Returns float64 array."""
    img = img.astype(np.float64)
    y = 16.0 + (65.481 * img[..., 0] + 128.553 * img[..., 1] + 24.966 * img[..., 2]) / 255.0
    return y


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images (float64, same shape)."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(255.0 ** 2 / mse)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two single-channel images (float64).

    Uses 11x11 Gaussian window with sigma=1.5, matching MATLAB ssim()
    and the standard used in SR papers (BasicSR, EDSR, SwinIR, HAT, etc.).
    """
    return float(structural_similarity(
        img1, img2, data_range=255.0, gaussian_weights=True, sigma=1.5,
    ))


class DIV2KSRx4Evaluator(MetricEvaluatorBase):
    """Evaluate super-resolution predictions: PSNR and SSIM on Y channel."""

    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        # Load metadata
        with open(os.path.join(test_data_dir, "meta.json")) as f:
            meta = json.load(f)
        border = meta.get("eval_border_crop", 4)

        # Load predictions JSON to get SR image directory
        with open(predictions_path) as f:
            pred_data = json.load(f)
        pred_dir = pred_data["predictions_dir"]

        if not os.path.isdir(pred_dir):
            raise ValueError(f"Predictions directory not found: {pred_dir}")

        # Load ground truth HR images
        gt_dir = os.path.join(test_data_dir, "hr")
        gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(".png")])

        if not gt_files:
            raise ValueError(f"No HR ground truth images found in {gt_dir}")

        # Check prediction count
        pred_files = sorted([f for f in os.listdir(pred_dir) if f.lower().endswith(".png")])
        if len(pred_files) != len(gt_files):
            raise ValueError(
                f"Prediction count mismatch: got {len(pred_files)} SR images, "
                f"expected {len(gt_files)}"
            )

        psnr_values = []
        ssim_values = []

        for gt_file in gt_files:
            pred_file = os.path.join(pred_dir, gt_file)
            if not os.path.exists(pred_file):
                raise ValueError(f"Missing SR image: {gt_file}")

            # Load images as RGB uint8
            gt_img = np.array(Image.open(os.path.join(gt_dir, gt_file)).convert("RGB"))
            sr_img = np.array(Image.open(pred_file).convert("RGB"))

            if sr_img.shape != gt_img.shape:
                raise ValueError(
                    f"Shape mismatch for {gt_file}: SR {sr_img.shape} vs GT {gt_img.shape}"
                )

            # Convert to Y channel
            gt_y = rgb_to_ycbcr(gt_img)
            sr_y = rgb_to_ycbcr(sr_img)

            # Crop border
            if border > 0:
                gt_y = gt_y[border:-border, border:-border]
                sr_y = sr_y[border:-border, border:-border]

            # Compute metrics
            psnr_values.append(compute_psnr(sr_y, gt_y))
            ssim_values.append(compute_ssim(sr_y, gt_y))

        avg_psnr = sum(psnr_values) / len(psnr_values)
        avg_ssim = sum(ssim_values) / len(ssim_values)

        metrics = {
            "psnr_y": round(avg_psnr, 4),
            "ssim_y": round(avg_ssim, 4),
        }

        return EvalResult(
            metrics=metrics,
            primary_metric_name="psnr_y",
            primary_metric_value=avg_psnr,
        )
