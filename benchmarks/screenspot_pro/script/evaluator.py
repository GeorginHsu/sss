from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _coerce_pred_map(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    payload = raw.get("predictions")
    if not isinstance(payload, list):
        return {}
    mapping: dict[str, dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        sid = item.get("id")
        if sid is None:
            continue
        entry: dict[str, Any] = {}
        if isinstance(item.get("bbox"), list) and len(item["bbox"]) >= 4:
            entry["bbox"] = [float(x) for x in item["bbox"][:4]]
        if isinstance(item.get("point"), list) and len(item["point"]) >= 2:
            entry["point"] = [float(x) for x in item["point"][:2]]
        if entry:
            mapping[str(sid)] = entry
    return mapping


class ScreenSpotProEvaluator(MetricEvaluatorBase):
    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        del task_config
        pred_map = _coerce_pred_map(_read_json(Path(predictions_path)))
        labels = _read_jsonl(Path(test_data_dir) / "test_labels.jsonl")

        total = len(labels)
        if total == 0:
            return EvalResult(
                metrics={"grounding_score": 0.0, "total": 0, "error": "empty test labels"},
                primary_metric_name="grounding_score",
                primary_metric_value=0.0,
            )

        hit = 0
        bbox_eval_count = 0
        point_eval_count = 0
        missing = 0
        iou_sum = 0.0
        point_dist_sum = 0.0

        for row in labels:
            sid = str(row.get("id", ""))
            pred = pred_map.get(sid)
            if pred is None:
                missing += 1
                continue

            gt_bbox = row.get("target_bbox")
            gt_point = row.get("target_point")

            if isinstance(gt_bbox, list) and len(gt_bbox) >= 4 and "bbox" in pred:
                iou = _bbox_iou([float(x) for x in gt_bbox[:4]], pred["bbox"])
                iou_sum += iou
                bbox_eval_count += 1
                if iou >= 0.5:
                    hit += 1
                continue

            if isinstance(gt_point, list) and len(gt_point) >= 2 and "point" in pred:
                gx, gy = float(gt_point[0]), float(gt_point[1])
                px, py = pred["point"]
                dist = math.sqrt((gx - px) ** 2 + (gy - py) ** 2)
                point_dist_sum += dist
                point_eval_count += 1
                if dist <= 14.0:
                    hit += 1
                continue

            missing += 1

        grounding_score = hit / total
        metrics: dict[str, Any] = {
            "grounding_score": round(grounding_score, 4),
            "num_correct": hit,
            "total": total,
            "num_missing_or_invalid": missing,
            "bbox_eval_count": bbox_eval_count,
            "point_eval_count": point_eval_count,
        }
        if bbox_eval_count > 0:
            metrics["mean_iou"] = round(iou_sum / bbox_eval_count, 4)
        if point_eval_count > 0:
            metrics["mean_point_distance_px"] = round(point_dist_sum / point_eval_count, 4)

        return EvalResult(
            metrics=metrics,
            primary_metric_name="grounding_score",
            primary_metric_value=grounding_score,
        )
