"""Clotho audio captioning evaluator.

Computes SPIDEr (primary), CIDEr, SPICE, METEOR, ROUGE-L, and
recent DCASE secondary metrics (SPIDEr-FL and FENSE).

SPIDEr = (CIDEr-D + SPICE) / 2 is the standard audio captioning metric.
Uses the aac-metrics package which handles SPICE/METEOR via Stanford CoreNLP.
"""

from __future__ import annotations

import csv
import json
import os

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig

# Keep aac-metrics / transformers on the PyTorch path only. Some host or base
# environments include TensorFlow with incompatible protobuf pins; evaluator
# metrics do not need TF/Flax.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")


def _load_reference_captions(csv_path: str) -> dict[str, list[str]]:
    """Load reference captions from Clotho CSV.

    Returns: {filename: [caption_1, ..., caption_5]}
    """
    refs = {}
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["file_name"].strip()
            captions = []
            for i in range(1, 6):
                key = f"caption_{i}"
                if key in row and row[key].strip():
                    captions.append(row[key].strip())
            if captions:
                refs[fname] = captions
    return refs


def _score_to_float(value) -> float:
    """Convert torch/numpy scalar metric outputs to plain float."""
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


class ClothoCaptionEvaluator(MetricEvaluatorBase):
    """Evaluate audio captioning using SPIDEr and related NLG metrics."""

    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        # 1. Load predictions
        with open(predictions_path) as f:
            preds_data = json.load(f)

        predictions = preds_data["predictions"]

        # 2. Load test file order
        test_files_path = os.path.join(test_data_dir, "test_files.txt")
        with open(test_files_path) as f:
            filenames = [line.strip() for line in f if line.strip()]

        # 3. Validate prediction count
        if len(predictions) != len(filenames):
            raise ValueError(
                f"Prediction count mismatch: got {len(predictions)}, "
                f"expected {len(filenames)}"
            )

        # 4. Load reference captions
        ref_csv = os.path.join(test_data_dir, "reference_captions.csv")
        all_refs = _load_reference_captions(ref_csv)

        # Build parallel lists: candidates and multi-references
        candidates = []
        mult_references = []
        missing_refs = []

        for i, fname in enumerate(filenames):
            if fname not in all_refs:
                missing_refs.append(fname)
                continue
            candidates.append(str(predictions[i]).strip())
            mult_references.append(all_refs[fname])

        if missing_refs:
            raise ValueError(
                f"Missing reference captions for {len(missing_refs)} files. "
                f"First 5: {missing_refs[:5]}"
            )

        # 5. Compute metrics using aac-metrics
        print(f"Computing SPIDEr on {len(candidates)} captions ...")
        from aac_metrics import evaluate as aac_evaluate

        # "spider_fl" internally computes SPIDEr, CIDEr-D, SPICE, and FER, so
        # don't request those separately (avoids redundant SPICE Java runs).
        corpus_scores, _ = aac_evaluate(
            candidates,
            mult_references,
            metrics=["spider_fl", "meteor", "rouge_l", "fense"],
        )

        spider = _score_to_float(corpus_scores["spider"])
        cider = _score_to_float(corpus_scores["cider_d"])
        spice = _score_to_float(corpus_scores["spice"])
        meteor = _score_to_float(corpus_scores["meteor"])
        rouge_l = _score_to_float(corpus_scores["rouge_l"])

        metrics = {
            "spider": round(spider, 4),
            "cider_d": round(cider, 4),
            "spice": round(spice, 4),
            "meteor": round(meteor, 4),
            "rouge_l": round(rouge_l, 4),
            "spider_fl": round(_score_to_float(corpus_scores["spider_fl"]), 4),
            "fense": round(_score_to_float(corpus_scores["fense"]), 4),
            "sbert_sim": round(_score_to_float(corpus_scores["sbert_sim"]), 4),
            "fer": round(_score_to_float(corpus_scores["fer"]), 4),
            "num_evaluated": float(len(candidates)),
        }

        return EvalResult(
            metrics=metrics,
            primary_metric_name="spider",
            primary_metric_value=spider,
        )
