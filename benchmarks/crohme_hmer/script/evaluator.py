"""CROHME HMER evaluator: ExpRate, ExpRate<=1, ExpRate<=2 on token sequences.

Primary metric **ExpRate** (Expression Recognition Rate) is the fraction of
predictions whose canonical LaTeX token sequence exactly matches the ground
truth. ExpRate<=k allows up to k token-level edit-distance errors
(insertion/deletion/substitution), following the reporting convention of
CoMER (ECCV 2022), CAN (ECCV 2022), BTTR, ICAL (CVPR 2024), TAMER (AAAI
2025), NAMER (ECCV 2024), PosFormer (ECCV 2024).

LaTeX normalization:
  Both ground truth and agent predictions are canonicalized with the same
  normalizer before comparison (see `_normalize_tokens` below):
    1. \\mathrm{op} → \\op for standard operator names
       (sin, cos, tan, log, lim, max, det, ...).
    2. Redundant single-leaf-token braces stripped iteratively:
       `{C}` → `C`, `{\\sum}` → `\\sum`, but `{2n}` (multi-token) kept.
  This matches the CoMER/CAN community's preprocessed-label convention;
  numbers differ from the official CROHME symLG / lgeval protocol by
  roughly 1–2 percentage points but preserve relative method ranking.
"""

from __future__ import annotations

import json
import os
import re

import Levenshtein

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig


# ── LaTeX normalization (mirrors prepare.py) ───────────────────────────────

_TOKEN_RE = re.compile(
    r"\\[a-zA-Z]+|"
    r"\\[^a-zA-Z\s]|"
    r"[^\s]"
)
_MATHRM_OP_RE = re.compile(
    r"\\mathrm\s*\{\s*"
    r"(sin|cos|tan|cot|sec|csc|"
    r"arcsin|arccos|arctan|"
    r"sinh|cosh|tanh|coth|"
    r"log|ln|lg|lim|exp|"
    r"min|max|sup|inf|det|arg|gcd|mod|hom|ker|deg|dim|"
    r"Pr)"
    r"\s*\}"
)


def _normalize_tokens(latex: str | None) -> list[str]:
    """Canonicalize a LaTeX string into a space-separated token list."""
    if not latex:
        return []
    s = latex.strip()
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2]
    elif s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    s = _MATHRM_OP_RE.sub(r"\\\1 ", s)
    tokens = _TOKEN_RE.findall(s)
    while True:
        out: list[str] = []
        i = 0
        changed = False
        n = len(tokens)
        while i < n:
            if (i + 2 < n and tokens[i] == "{"
                    and tokens[i + 1] not in ("{", "}")
                    and tokens[i + 2] == "}"):
                out.append(tokens[i + 1])
                i += 3
                changed = True
            else:
                out.append(tokens[i])
                i += 1
        tokens = out
        if not changed:
            break
    return tokens


def _load_labels(path: str) -> dict[str, list[str]]:
    """Load tab-separated labels file into {id: token_list}."""
    gt: dict[str, list[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            sample_id, token_str = parts
            gt[sample_id] = token_str.split() if token_str else []
    return gt


def _token_edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance over token sequences.

    python-Levenshtein operates on strings; to measure edit distance over
    token sequences we map each unique token to a unique unicode character.
    """
    vocab: dict[str, str] = {}
    def encode(seq: list[str]) -> str:
        chars: list[str] = []
        for tok in seq:
            if tok not in vocab:
                vocab[tok] = chr(0x100 + len(vocab))
            chars.append(vocab[tok])
        return "".join(chars)
    return Levenshtein.distance(encode(a), encode(b))


class CROHMEHMEREvaluator(MetricEvaluatorBase):
    """Compare agent LaTeX predictions against CROHME 2019 test labels."""

    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        # 1. Load predictions
        with open(predictions_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pred_list = data.get("predictions", [])
        if not isinstance(pred_list, list):
            raise ValueError("predictions must be a list of {id, latex} objects")

        pred_map: dict[str, list[str]] = {}
        for entry in pred_list:
            if not isinstance(entry, dict):
                raise ValueError(f"prediction entry must be a dict, got {type(entry)}")
            sid = entry.get("id")
            latex = entry.get("latex", "")
            if sid is None:
                raise ValueError(f"prediction entry missing 'id': {entry}")
            if not isinstance(latex, str):
                raise ValueError(f"prediction 'latex' must be a string, got {type(latex)}")
            # Apply the same canonicalization as prepare.py uses on GT labels,
            # so agents can output any equivalent LaTeX style
            # (`\frac{1}{2}`, `\frac { 1 } { 2 }`, `\frac 1 2` all match).
            pred_map[sid] = _normalize_tokens(latex)

        # 2. Load ground truth
        gt = _load_labels(os.path.join(test_data_dir, "labels.txt"))
        if not gt:
            raise ValueError(f"No ground truth labels found in {test_data_dir}/labels.txt")

        # 3. Validate count
        if len(pred_map) != len(gt):
            raise ValueError(
                f"Prediction count mismatch: got {len(pred_map)} unique ids, "
                f"expected {len(gt)} (test set size)"
            )
        missing = set(gt.keys()) - set(pred_map.keys())
        if missing:
            raise ValueError(
                f"Predictions missing {len(missing)} ids; first missing: "
                f"{sorted(missing)[:3]}"
            )

        # 4. Compute ExpRate and ExpRate<=k
        n = len(gt)
        exact = 0
        le1 = 0
        le2 = 0
        for sid, gt_toks in gt.items():
            pred_toks = pred_map[sid]
            if pred_toks == gt_toks:
                exact += 1
                le1 += 1
                le2 += 1
                continue
            d = _token_edit_distance(pred_toks, gt_toks)
            if d <= 1:
                le1 += 1
            if d <= 2:
                le2 += 1

        exprate = exact / n
        metrics = {
            "exprate": round(exprate, 4),
            "exprate_le1": round(le1 / n, 4),
            "exprate_le2": round(le2 / n, 4),
            "n_test": n,
        }
        return EvalResult(
            metrics=metrics,
            primary_metric_name="exprate",
            primary_metric_value=exprate,
        )
