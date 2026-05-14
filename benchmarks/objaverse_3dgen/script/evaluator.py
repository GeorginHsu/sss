"""Objaverse 3D Generation evaluator: LPIPS, PSNR, and SSIM on novel views.

Evaluation protocol:
  1. Load ground truth target views (11 per object, 256x256 RGBA)
  2. Load predicted views from the agent's output directory
  3. Convert RGBA to RGB (composite over white background)
  4. Compute per-view metrics:
     - LPIPS (VGG): perceptual similarity (lower is better)
     - PSNR: pixel-level fidelity (higher is better)
     - SSIM: structural similarity (higher is better)
  5. Report averages across all views and objects
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig

N_TARGET_VIEWS = 11


def rgba_to_rgb(img: np.ndarray) -> np.ndarray:
    """Composite RGBA image over white background, return RGB uint8."""
    if img.shape[-1] == 4:
        alpha = img[..., 3:4].astype(np.float32) / 255.0
        rgb = img[..., :3].astype(np.float32)
        white = np.full_like(rgb, 255.0)
        composited = rgb * alpha + white * (1.0 - alpha)
        return np.clip(composited, 0, 255).astype(np.uint8)
    return img[..., :3]


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two RGB uint8 images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(255.0 ** 2 / mse)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two RGB uint8 images (channel_axis=-1)."""
    return float(structural_similarity(
        img1, img2,
        data_range=255.0,
        channel_axis=-1,
        gaussian_weights=True,
        sigma=1.5,
    ))


class Objaverse3DGenEvaluator(MetricEvaluatorBase):
    """Evaluate 3D generation via novel view synthesis: LPIPS, PSNR, SSIM."""

    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        # Load predictions JSON
        with open(predictions_path) as f:
            pred_data = json.load(f)
        pred_dir = pred_data["predictions_dir"]

        if not os.path.isdir(pred_dir):
            raise ValueError(f"Predictions directory not found: {pred_dir}")

        # Load ground truth directory
        gt_dir = os.path.join(test_data_dir, "test_gt")
        if not os.path.isdir(gt_dir):
            raise ValueError(f"Ground truth directory not found: {gt_dir}")

        # Get test object UIDs
        gt_objects = sorted(os.listdir(gt_dir))
        if not gt_objects:
            raise ValueError(f"No ground truth objects found in {gt_dir}")

        # Initialize LPIPS
        import lpips
        lpips_fn = lpips.LPIPS(net="vgg")
        lpips_fn.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lpips_fn = lpips_fn.to(device)

        all_psnr = []
        all_ssim = []
        all_lpips = []
        missing_objects = []

        for obj_uid in gt_objects:
            pred_obj_dir = os.path.join(pred_dir, obj_uid)
            gt_obj_dir = os.path.join(gt_dir, obj_uid)

            if not os.path.isdir(pred_obj_dir):
                missing_objects.append(obj_uid)
                continue

            for vi in range(1, N_TARGET_VIEWS + 1):
                view_name = f"view_{vi:02d}.png"
                gt_path = os.path.join(gt_obj_dir, view_name)
                pred_path = os.path.join(pred_obj_dir, view_name)

                if not os.path.exists(gt_path):
                    continue
                if not os.path.exists(pred_path):
                    # Missing prediction: assign worst-case metrics
                    all_psnr.append(0.0)
                    all_ssim.append(0.0)
                    all_lpips.append(1.0)
                    continue

                # Load images
                gt_img = np.array(Image.open(gt_path).convert("RGBA"))
                pred_img = np.array(Image.open(pred_path))

                # Handle both RGBA and RGB predictions
                if pred_img.shape[-1] == 4:
                    pred_rgb = rgba_to_rgb(pred_img)
                elif pred_img.shape[-1] == 3:
                    pred_rgb = pred_img
                else:
                    raise ValueError(
                        f"Unexpected channels for {obj_uid}/{view_name}: {pred_img.shape}"
                    )
                gt_rgb = rgba_to_rgb(gt_img)

                # Resize prediction if needed
                if pred_rgb.shape[:2] != gt_rgb.shape[:2]:
                    pred_pil = Image.fromarray(pred_rgb).resize(
                        (gt_rgb.shape[1], gt_rgb.shape[0]), Image.LANCZOS,
                    )
                    pred_rgb = np.array(pred_pil)

                # PSNR
                all_psnr.append(compute_psnr(pred_rgb, gt_rgb))

                # SSIM
                all_ssim.append(compute_ssim(pred_rgb, gt_rgb))

                # LPIPS
                gt_tensor = torch.from_numpy(gt_rgb).permute(2, 0, 1).float() / 127.5 - 1.0
                pred_tensor = torch.from_numpy(pred_rgb).permute(2, 0, 1).float() / 127.5 - 1.0
                gt_tensor = gt_tensor.unsqueeze(0).to(device)
                pred_tensor = pred_tensor.unsqueeze(0).to(device)

                with torch.no_grad():
                    lpips_val = lpips_fn(pred_tensor, gt_tensor).item()
                all_lpips.append(lpips_val)

        if missing_objects:
            n_missing = len(missing_objects)
            print(
                f"Warning: {n_missing}/{len(gt_objects)} objects missing from predictions. "
                f"First 5: {missing_objects[:5]}"
            )

        if not all_lpips:
            raise ValueError("No valid predictions found to evaluate.")

        avg_lpips = sum(all_lpips) / len(all_lpips)
        avg_psnr = sum(all_psnr) / len(all_psnr)
        avg_ssim = sum(all_ssim) / len(all_ssim)

        n_evaluated = len(gt_objects) - len(missing_objects)

        metrics = {
            "lpips": round(avg_lpips, 4),
            "psnr": round(avg_psnr, 4),
            "ssim": round(avg_ssim, 4),
            "n_objects_evaluated": n_evaluated,
            "n_objects_total": len(gt_objects),
            "n_views_evaluated": len(all_lpips),
        }

        return EvalResult(
            metrics=metrics,
            primary_metric_name="lpips",
            primary_metric_value=avg_lpips,
        )
