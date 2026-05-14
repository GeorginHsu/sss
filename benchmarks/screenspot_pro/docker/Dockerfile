ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}

# ---------------------------------------------------------------------------
# ScreenSpot-Pro task image
#
# Same philosophy as humanoidbench: prepare.py is a thin file-copy layer; the
# image bakes everything it needs under /opt/farbench/screenspot_pro/ at build time.
#
# Layout written by this Dockerfile:
#   /opt/farbench/screenspot_pro/
#     train_data/
#       unified_train.jsonl        # ~7100 rows (95% of ShowUI-desktop)
#       unified_val.jsonl          # ~400  rows (5%, image-disjoint split)
#       images/<image_id>.png      # ~1500 unique source screenshots
#       dataset_info.json
#     test_data/
#       test_prompts.jsonl         # visible to agent (no labels)
#       test_labels.jsonl          # hidden labels for evaluator
#       images/<image_id>.png
#       dataset_info.json
#     base_model/
#       Qwen2.5-VL-3B-Instruct/    # snapshotted, load offline
#
# Training data: showlab/ShowUI-desktop (a.k.a. ShowUI-desktop-8K, sourced from
# OmniAct, GPT-4o re-captioned). Chosen because:
#   * Test set (likaixin/ScreenSpot-Pro) is professional desktop software
#     screenshots — same modality, so train/test are well-aligned.
#   * It is small enough (~3 GB raw images + 7496 instruction rows) to bake
#     into the pre-built Docker image archive without pain.
#   * Annotations are clean (bbox + point per instruction). We unify them
#     into the same JSONL schema as the test set so the agent only has to
#     learn one format.
#
# Test set: likaixin/ScreenSpot-Pro (HF). Materialized into prompts/labels
# JSONL + per-sample PNG.
#
# Base model: Qwen/Qwen2.5-VL-3B-Instruct (~7 GB safetensors).
#
# All downloads happen at `docker build` time. The container itself runs
# with `network_access: false`, so the agent only ever sees the already-
# laid-out fixtures.
# ---------------------------------------------------------------------------

# Build args (override with `--build-arg` if you re-host data/model elsewhere).
ARG HF_TOKEN=""
ARG TRAIN_DATA_REPO="showlab/ShowUI-desktop"
ARG TEST_DATA_REPO="likaixin/ScreenSpot-Pro"
ARG BASE_MODEL_REPO="Qwen/Qwen2.5-VL-3B-Instruct"
ARG VAL_FRACTION="0.05"
ENV HF_TOKEN=${HF_TOKEN}

# ── System deps for Pillow/cv2 image handling ──────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps for both prep + agent training/eval ────────────────────────
RUN pip install --no-cache-dir \
    "datasets>=2.19" \
    "huggingface_hub>=0.24" \
    "hf_transfer>=0.1.6" \
    "pillow" \
    "pyarrow" \
    "transformers>=4.45" \
    "accelerate>=0.33" \
    "peft>=0.12" \
    "trl>=0.10" \
    "qwen-vl-utils" \
    "tqdm"

ENV HF_HUB_ENABLE_HF_TRANSFER=1

# ── Bake assets under /opt/farbench/screenspot_pro/ ─────────────────────────────
RUN mkdir -p /opt/farbench/screenspot_pro

# ----------------------------------------------------------------------------
# (1) Training mix: showlab/ShowUI-desktop -> unified_train/val.jsonl + images/
#
# Source schema (per row):
#   image_url:   str (e.g. "screenshots/message/screen_1.png", reused across
#                rows to bind multiple instructions to the same screenshot)
#   image:       PIL Image (high-res RGB, often 3360x2100)
#   instruction: str (text query)
#   bbox:        [x1, y1, x2, y2] in NORMALIZED [0,1] coords  <-- IMPORTANT
#   point:       [x, y]            in NORMALIZED [0,1] coords  <-- IMPORTANT
#   type:        "original" | "appearance" | "spatial" | "intent" (augmentations)
#
# Unified target schema (per row), aligned with the ScreenSpot-Pro test set:
#   id:           str ("showui-XXXXXX")
#   image_path:   str ("images/<sha1>.png", relative to data_dir)
#   instruction:  str
#   target_bbox:  [x1, y1, x2, y2] in ABSOLUTE pixels, XYXY (clipped)
#   target_point: [x, y]            in ABSOLUTE pixels, XY    (clipped)
#   image_width:  int
#   image_height: int
#   source:       "ShowUI-desktop"
#   type:         <pass-through>
#
# Train/val split is image-disjoint (we hash image_url, not row index) so
# augmented variants of the same screenshot never straddle the split — that
# would inflate val accuracy due to memorized layouts.
# ----------------------------------------------------------------------------
RUN python - <<'PY'
import hashlib
import io
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset
from PIL import Image

repo = os.environ.get("TRAIN_DATA_REPO", "showlab/ShowUI-desktop")
val_frac = float(os.environ.get("VAL_FRACTION", "0.05"))
out = Path("/opt/farbench/screenspot_pro/train_data")
img_dir = out / "images"
img_dir.mkdir(parents=True, exist_ok=True)

print(f"[image-build] training mix <- {repo} (val_frac={val_frac:.3f})", flush=True)
ds = load_dataset(repo, split="train")
print(f"[image-build]   raw rows: {len(ds)}", flush=True)


def split_bucket(image_url: str) -> str:
    h = hashlib.md5(image_url.encode("utf-8")).hexdigest()
    return "val" if (int(h[:8], 16) / 0xFFFFFFFF) < val_frac else "train"


def stable_image_id(image_url: str) -> str:
    return "showui-" + hashlib.sha1(image_url.encode("utf-8")).hexdigest()[:12]


def to_abs_xyxy(bbox, w, h):
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if max(x1, y1, x2, y2) <= 1.5:
        x1 *= w; y1 *= h; x2 *= w; y2 *= h
    x1 = max(0.0, min(float(w), x1))
    y1 = max(0.0, min(float(h), y1))
    x2 = max(0.0, min(float(w), x2))
    y2 = max(0.0, min(float(h), y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def to_abs_xy(point, w, h):
    if not (isinstance(point, (list, tuple)) and len(point) == 2):
        return None
    x, y = float(point[0]), float(point[1])
    if max(x, y) <= 1.5:
        x *= w; y *= h
    x = max(0.0, min(float(w), x))
    y = max(0.0, min(float(h), y))
    return [round(x, 2), round(y, 2)]


saved_images: dict[str, tuple[int, int]] = {}
counts = {"train": 0, "val": 0, "skipped": 0}

train_f = (out / "unified_train.jsonl").open("w", encoding="utf-8")
val_f = (out / "unified_val.jsonl").open("w", encoding="utf-8")

for i, row in enumerate(ds):
    img = row.get("image")
    image_url = row.get("image_url") or f"row-{i}"
    if img is None:
        counts["skipped"] += 1
        continue
    if not isinstance(img, Image.Image):
        try:
            img = Image.open(io.BytesIO(img["bytes"]))
        except Exception:
            counts["skipped"] += 1
            continue
    img = img.convert("RGB")
    w, h = img.size

    sid = stable_image_id(image_url)
    img_path = img_dir / f"{sid}.png"
    if sid not in saved_images:
        img.save(img_path, format="PNG", optimize=False)
        saved_images[sid] = (w, h)
    else:
        w, h = saved_images[sid]

    bbox_abs = to_abs_xyxy(row.get("bbox"), w, h)
    point_abs = to_abs_xy(row.get("point"), w, h)
    if bbox_abs is None and point_abs is None:
        counts["skipped"] += 1
        continue

    rec = {
        "id": f"{sid}-r{i:06d}",
        "image_path": f"images/{sid}.png",
        "instruction": row.get("instruction") or "",
        "image_width": w,
        "image_height": h,
        "source": "ShowUI-desktop",
        "type": row.get("type") or "original",
    }
    if bbox_abs is not None:
        rec["target_bbox"] = bbox_abs
    if point_abs is not None:
        rec["target_point"] = point_abs

    bucket = split_bucket(image_url)
    (train_f if bucket == "train" else val_f).write(
        json.dumps(rec, ensure_ascii=False) + "\n"
    )
    counts[bucket] += 1

train_f.close(); val_f.close()

(out / "dataset_info.json").write_text(
    json.dumps(
        {
            "source": repo,
            "license": "see source repo",
            "split": {
                "method": "image-disjoint MD5(image_url) hash",
                "val_fraction": val_frac,
                "n_train": counts["train"],
                "n_val": counts["val"],
                "n_skipped": counts["skipped"],
                "n_unique_images": len(saved_images),
            },
            "row_schema": {
                "id": "str",
                "image_path": "relative path to images/<id>.png",
                "instruction": "str",
                "target_bbox": "[x1,y1,x2,y2] absolute pixels, XYXY (optional)",
                "target_point": "[x,y] absolute pixels, XY (optional)",
                "image_width": "int",
                "image_height": "int",
                "source": "ShowUI-desktop",
                "type": "original|appearance|spatial|intent",
            },
        },
        indent=2,
    )
)

print(
    f"[image-build]   wrote train={counts['train']} val={counts['val']} "
    f"skipped={counts['skipped']} unique_images={len(saved_images)}",
    flush=True,
)
if counts["train"] == 0 or counts["val"] == 0:
    print("[image-build] FATAL: empty split; aborting build", file=sys.stderr)
    sys.exit(1)
PY

# ----------------------------------------------------------------------------
# (2) ScreenSpot-Pro test set (likaixin/ScreenSpot-Pro):
#
# This repo is NOT a HF `datasets` standard layout (no parquet, no loading
# script) — it ships as 26 per-application annotation JSONs plus a flat
# `images/<application>_<platform>/<screenshot>.png` tree, so we have to
# pull the raw files via snapshot_download and assemble the unified jsonl
# ourselves.
#
# Each annotation row already provides:
#   id           - globally unique (e.g. "blender_windows_0")
#   img_filename - path inside images/, e.g. "blender_windows/foo.png"
#   bbox         - [x1, y1, x2, y2], ABSOLUTE pixels XYXY (no rescale needed)
#   img_size     - [w, h]
#   instruction  - English text query (we drop instruction_cn)
#   application/platform/ui_type/group - metadata (passed through to prompts)
#
# We flatten images/<id>.png in the output dir so test_data/images/ has no
# nested subfolders and predict.py can resolve every prompt with a simple
# os.path.join(data_path, prompt["image_path"]).
# ----------------------------------------------------------------------------
RUN python - <<'PY'
import json
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo = os.environ.get("TEST_DATA_REPO", "likaixin/ScreenSpot-Pro")
print(f"[image-build] test set <- {repo}", flush=True)
src = Path(
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN") or None,
        max_workers=8,
        allow_patterns=["annotations/*.json", "images/**", "eval.yaml", "README.md"],
    )
)

ann_dir = src / "annotations"
ann_files = sorted(ann_dir.glob("*.json"))
if not ann_files:
    print(f"[image-build] FATAL: no annotation JSONs under {ann_dir}", file=sys.stderr)
    sys.exit(1)
print(f"[image-build]   annotation files: {len(ann_files)}", flush=True)

out = Path("/opt/farbench/screenspot_pro/test_data")
imgs_out = out / "images"
imgs_out.mkdir(parents=True, exist_ok=True)
prompts_f = (out / "test_prompts.jsonl").open("w", encoding="utf-8")
labels_f = (out / "test_labels.jsonl").open("w", encoding="utf-8")

n = 0
n_orphan = 0
seen_ids: set[str] = set()
for ann_path in ann_files:
    try:
        rows = json.loads(ann_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[image-build]   WARN: bad json {ann_path.name}: {exc}", flush=True)
        continue
    if not isinstance(rows, list):
        print(f"[image-build]   WARN: {ann_path.name} is not a list", flush=True)
        continue
    for row in rows:
        sid = row.get("id")
        img_rel = row.get("img_filename")
        bbox = row.get("bbox")
        img_size = row.get("img_size") or [0, 0]
        if not (sid and img_rel and isinstance(bbox, list) and len(bbox) == 4):
            n_orphan += 1
            continue
        if sid in seen_ids:
            n_orphan += 1
            continue
        seen_ids.add(sid)

        src_img = src / "images" / img_rel
        if not src_img.is_file():
            n_orphan += 1
            continue
        dst_img = imgs_out / f"{sid}.png"
        if not dst_img.exists():
            shutil.copyfile(src_img, dst_img)

        w, h = int(img_size[0]), int(img_size[1])
        prompt = {
            "id": sid,
            "image_path": f"images/{sid}.png",
            "instruction": row.get("instruction") or "",
            "image_width": w,
            "image_height": h,
            "application": row.get("application", ""),
            "platform": row.get("platform", ""),
            "ui_type": row.get("ui_type", ""),
            "group": row.get("group", ""),
        }
        label = {
            "id": sid,
            "image_width": w,
            "image_height": h,
            "target_bbox": [float(x) for x in bbox],
        }
        prompts_f.write(json.dumps(prompt, ensure_ascii=False) + "\n")
        labels_f.write(json.dumps(label, ensure_ascii=False) + "\n")
        n += 1

prompts_f.close(); labels_f.close()

(out / "dataset_info.json").write_text(
    json.dumps(
        {
            "source": repo,
            "split": "test",
            "num_examples": n,
            "num_orphan_or_malformed": n_orphan,
            "num_annotation_files": len(ann_files),
        },
        indent=2,
    )
)
print(
    f"[image-build]   test set materialized: {n} examples (skipped {n_orphan})",
    flush=True,
)
if n == 0:
    print("[image-build] FATAL: empty test set; aborting build", file=sys.stderr)
    sys.exit(1)
PY

# ----------------------------------------------------------------------------
# (3) Base model: Qwen2.5-VL-3B-Instruct (~7 GB safetensors).
# ----------------------------------------------------------------------------
RUN python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo = os.environ.get("BASE_MODEL_REPO", "Qwen/Qwen2.5-VL-3B-Instruct")
dst = "/opt/farbench/screenspot_pro/base_model/Qwen2.5-VL-3B-Instruct"
os.makedirs(dst, exist_ok=True)
print(f"[image-build] base model <- {repo}", flush=True)
snapshot_download(
    repo_id=repo,
    repo_type="model",
    local_dir=dst,
    token=os.environ.get("HF_TOKEN") or None,
    max_workers=8,
    ignore_patterns=["*.bin", "*.msgpack", "flax_model.*", "tf_model.*", "*.gguf"],
)
print("[image-build]   base model files (top 8):", sorted(os.listdir(dst))[:8])
PY

# ----------------------------------------------------------------------------
# (4) Build-time sanity check — fail the build if any expected file is missing
# or any JSONL row violates the documented schema. Catches regressions early.
# ----------------------------------------------------------------------------
RUN python - <<'PY'
import json
import os
import sys
from pathlib import Path

ROOT = Path("/opt/farbench/screenspot_pro")
problems: list[str] = []

REQUIRED_TRAIN = [
    "train_data/unified_train.jsonl",
    "train_data/unified_val.jsonl",
    "train_data/dataset_info.json",
]
REQUIRED_TEST = [
    "test_data/test_prompts.jsonl",
    "test_data/test_labels.jsonl",
    "test_data/dataset_info.json",
]
for rel in REQUIRED_TRAIN + REQUIRED_TEST:
    if not (ROOT / rel).is_file():
        problems.append(f"missing file: {rel}")

for d in ("train_data/images", "test_data/images"):
    p = ROOT / d
    if not p.is_dir() or not any(p.iterdir()):
        problems.append(f"empty or missing dir: {d}")

bm = ROOT / "base_model" / "Qwen2.5-VL-3B-Instruct"
if not (bm / "config.json").is_file():
    problems.append("base_model/Qwen2.5-VL-3B-Instruct/config.json missing")
weights = list(bm.glob("*.safetensors")) if bm.is_dir() else []
if not weights:
    problems.append("base_model has no .safetensors weights")


def _check_jsonl(rel: str, required: list[str], optional_one_of: list[list[str]]) -> None:
    path = ROOT / rel
    if not path.is_file():
        return
    n_rows = 0
    n_with_bbox = 0
    n_with_point = 0
    with path.open() as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                problems.append(f"{rel}:{ln}: invalid JSON ({exc})")
                continue
            for k in required:
                if k not in row:
                    problems.append(f"{rel}:{ln}: missing required field '{k}'")
            for group in optional_one_of:
                if not any(k in row for k in group):
                    problems.append(
                        f"{rel}:{ln}: at least one of {group} required"
                    )
            n_rows += 1
            if "target_bbox" in row:
                n_with_bbox += 1
            if "target_point" in row:
                n_with_point += 1
    if n_rows == 0:
        problems.append(f"{rel}: zero rows")
    print(
        f"[sanity] {rel}: rows={n_rows} with_bbox={n_with_bbox} with_point={n_with_point}",
        flush=True,
    )


_check_jsonl(
    "train_data/unified_train.jsonl",
    required=["id", "image_path", "instruction", "image_width", "image_height"],
    optional_one_of=[["target_bbox", "target_point"]],
)
_check_jsonl(
    "train_data/unified_val.jsonl",
    required=["id", "image_path", "instruction", "image_width", "image_height"],
    optional_one_of=[["target_bbox", "target_point"]],
)
_check_jsonl(
    "test_data/test_prompts.jsonl",
    required=["id", "image_path", "instruction", "image_width", "image_height"],
    optional_one_of=[],
)
_check_jsonl(
    "test_data/test_labels.jsonl",
    required=["id", "image_width", "image_height"],
    optional_one_of=[["target_bbox", "target_point"]],
)

if problems:
    print("[sanity] BUILD FAILED:", file=sys.stderr)
    for p in problems:
        print("  -", p, file=sys.stderr)
    sys.exit(1)

# Print final layout for the build log.
print("\n[sanity] /opt/farbench/screenspot_pro/ layout:", flush=True)
for sub in ("train_data", "test_data", "base_model"):
    p = ROOT / sub
    if p.is_dir():
        items = sorted(p.iterdir())[:8]
        print(f"  {sub}/  ({len(list(p.iterdir()))} entries) -> {[x.name for x in items]}")
print("[sanity] OK")
PY

WORKDIR /workspace
