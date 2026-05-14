"""CROHME 2019 HMER data preparation: download InkML, render strokes, split.

Split strategy (literature-standard):
  Training: CROHME 2019 MainTask_formula/Train.zip (~9993 expressions —
  aggregated from CROHME 2012/2013/2016 train+test), labels from InkML truth.
  Validation: CROHME 2014 test (986 expressions) — the de-facto literature
  validation set used by CoMER (ECCV 2022), CAN (ECCV 2022), BTTR, PosFormer.
  This is shipped inside the training package as MainTask_formula/valid.zip.
  Test: CROHME 2019 official test set (1199 expressions). The RIT-hosted
  test InkMLs have labels stripped (they were withheld for the 2019
  competition), so we fetch the released ground-truth LaTeX labels from
  Zenodo record 17122781 ("CROHME2019st.json", published 2025) and join by
  InkML filename.

Archive layout notes:
  The CROHME 2019 archives are nested zips. The outer Task1_and_Task2.zip
  bundles per-task zips; we only extract the online-handwriting InkML for the
  MainTask_formula sub-task (the HMER benchmark, not the symbol/structure
  sub-tasks). See CROHME 2019 data description for the full layout.

References:
  - https://www.cs.rit.edu/~crohme2019/dataANDtools.html (InkML archives)
  - https://zenodo.org/records/17122781 (test GT release)
License: CC BY-NC-SA 3.0 (research use only).

Output layout:
    FARBENCH_DATA_DIR/
        meta.json
        vocab.txt                            — one LaTeX token per line
        train/images/<id>.png                — grayscale PNGs
        train/labels.txt                     — "<id>\t<tokens>" per line
        val/images/<id>.png
        val/labels.txt
    FARBENCH_TEST_DATA_DIR/
        meta.json
        vocab.txt
        images/<id>.png                      — 1199 test images (no labels)
        labels.txt                           — test ground truth (evaluator only)
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image, ImageDraw


DOWNLOAD_URLS = {
    "train": "https://www.cs.rit.edu/~crohme2019/downloads/Task1_and_Task2.zip",
    "test": "https://www.cs.rit.edu/~crohme2019/downloads/zipped_CROHME2019_testData.zip",
    # Test-set ground-truth labels (withheld from the RIT archive; Zenodo 2025 release)
    "test_gt": "https://zenodo.org/records/17122781/files/CROHME2019st.json?download=1",
}

# Inner zip paths / filters within the downloaded archives.
# Each entry yields a stream of InkML members for the given split.
TRAIN_INNER_ZIP = "Task1_and_Task2/Task1_and_Task2/Task1_onlineRec/MainTask_formula/Train.zip"
VAL_INNER_ZIP = "Task1_and_Task2/Task1_and_Task2/Task1_onlineRec/MainTask_formula/valid.zip"
TEST_OUTER_INNER_ZIP = "zipped_CROHME2019_testData/Task1_onlineRec.zip"
TEST_INNER_PREFIX = "Task1_onlineRec/MainTask_formula/TestSet2019/"

MIN_FILE_BYTES = 1024

# Rendering parameters — produces images compatible with CoMER/CAN style.
RENDER_PADDING = 20           # px padding around strokes
RENDER_STROKE_WIDTH = 3
RENDER_MAX_WIDTH = 1600
RENDER_MAX_HEIGHT = 320

# Reserved vocabulary entries (must match evaluator expectations).
SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]


# ── Download ────────────────────────────────────────────────────────────────

def download_file(url: str, dest: str) -> None:
    print(f"  Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "FARBench/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0 and downloaded % (50 * 65536) == 0:
                pct = downloaded * 100 // total
                print(f"\r  {pct}% ({downloaded // (1024*1024)}MB / {total // (1024*1024)}MB)",
                      end="", flush=True)
        print()


# ── InkML parsing ───────────────────────────────────────────────────────────

# Strip {http://...}tag → tag
NS_RE = re.compile(r"\{[^}]+\}")


def _local(tag: str) -> str:
    return NS_RE.sub("", tag)


def parse_inkml_bytes(data: bytes) -> tuple[list[list[tuple[float, float]]], str | None]:
    """Parse InkML from raw bytes (so we can read straight from zip members).

    Returns:
        strokes: list of strokes; each stroke is a list of (x, y) float points.
        latex:   the ground-truth LaTeX string (None if missing).
    """
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return [], None

    # Strokes live under <trace id="...">x,y x,y ...</trace>
    strokes: list[list[tuple[float, float]]] = []
    for tr in root.iter():
        if _local(tr.tag) != "trace":
            continue
        text = (tr.text or "").strip()
        if not text:
            continue
        pts: list[tuple[float, float]] = []
        for pt_str in text.split(","):
            pt_str = pt_str.strip()
            if not pt_str:
                continue
            parts = pt_str.split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            pts.append((x, y))
        if len(pts) >= 1:
            strokes.append(pts)

    # LaTeX ground truth is in
    # <annotation type="truth">$ \frac{1}{2} $</annotation>
    latex = None
    for ann in root.iter():
        if _local(ann.tag) != "annotation":
            continue
        t = ann.get("type") or ""
        if t.lower() == "truth":
            latex = (ann.text or "").strip()
            break
    return strokes, latex


# ── Rendering ───────────────────────────────────────────────────────────────

def render_strokes(
    strokes: list[list[tuple[float, float]]],
    max_w: int = RENDER_MAX_WIDTH,
    max_h: int = RENDER_MAX_HEIGHT,
    pad: int = RENDER_PADDING,
    stroke_w: int = RENDER_STROKE_WIDTH,
) -> Image.Image | None:
    """Render stroke coordinates to a grayscale PNG (black on white).

    Scaling preserves aspect ratio so the expression fits within (max_w, max_h).
    Returns None if strokes are empty or degenerate.
    """
    all_pts = [pt for s in strokes for pt in s]
    if not all_pts:
        return None
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w = x1 - x0
    h = y1 - y0
    if w <= 0 and h <= 0:
        return None

    # Target aspect-preserving size
    scale_w = (max_w - 2 * pad) / max(w, 1e-6)
    scale_h = (max_h - 2 * pad) / max(h, 1e-6)
    scale = min(scale_w, scale_h)
    scale = min(scale, 2.0)  # don't blow up tiny expressions
    if scale <= 0:
        return None

    out_w = max(int(math.ceil(w * scale)) + 2 * pad, 32)
    out_h = max(int(math.ceil(h * scale)) + 2 * pad, 32)
    out_w = min(out_w, max_w)
    out_h = min(out_h, max_h)

    img = Image.new("L", (out_w, out_h), color=255)
    draw = ImageDraw.Draw(img)

    for stroke in strokes:
        if len(stroke) == 0:
            continue
        if len(stroke) == 1:
            x, y = stroke[0]
            px = int((x - x0) * scale) + pad
            py = int((y - y0) * scale) + pad
            draw.ellipse(
                [px - stroke_w, py - stroke_w, px + stroke_w, py + stroke_w],
                fill=0,
            )
            continue
        pts_xy = [
            (int((x - x0) * scale) + pad, int((y - y0) * scale) + pad)
            for (x, y) in stroke
        ]
        draw.line(pts_xy, fill=0, width=stroke_w, joint="curve")
    return img


# ── LaTeX tokenization & normalization ─────────────────────────────────────

# Tokens we recognize: \commands (with letters), \\ single symbols (\{, \|, etc.),
# single LaTeX control chars, or any single non-whitespace character.
TOKEN_RE = re.compile(
    r"\\[a-zA-Z]+|"           # multi-letter commands like \frac, \sum, \alpha
    r"\\[^a-zA-Z\s]|"          # single-char commands like \{, \}, \\, \|, \,
    r"[^\s]"                    # any other non-whitespace char (digits, +, =, {, })
)

# \mathrm{op} → \op for standard operator names (CoMER/CAN preprocessing convention).
# Raw CROHME annotations sometimes encode operators as \mathrm{sin} / \mathrm{cos};
# the HMER community's canonical form uses \sin / \cos directly.
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


def tokenize_latex(latex: str | None) -> list[str]:
    """Tokenize and canonicalize a LaTeX expression for HMER evaluation.

    Applies three normalization steps that match the CoMER/CAN/BTTR community
    preprocessing convention, so predictions can be compared by exact-match
    ExpRate against published-paper-style labels:

      1. Strip outer $ / $$ delimiters.
      2. Rewrite \\mathrm{op} → \\op for standard operator names (sin, cos, log,
         lim, max, det, ...).
      3. Iteratively remove redundant single-leaf-token braces: `{X}` → `X`
         where X is exactly one non-brace token. This collapses CROHME's raw
         `{C}={\\sum }_{n=1}{c}_{n}` into the canonical
         `C = \\sum _ { n = 1 } c _ n n ^ 2` form widely used in HMER papers.
         Multi-token braces like `x^{2n}` stay as `x ^ { 2 n }` because
         `{2n}` is not a single leaf token.

    Note: token-level ExpRate on this canonical form is the metric CoMER/CAN
    and other recent papers report during training/validation. The official
    CROHME 2019 competition ExpRate (via lgeval on symLG format) may differ
    by ~1-2 percentage points but the relative ranking of methods is stable.
    """
    if not latex:
        return []
    s = latex.strip()
    # 1. Strip $…$ / $$…$$ delimiters
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2]
    elif s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    # 2. \mathrm{op} → \op
    # Trailing space avoids `\mathrm{sin}x` → `\sinx` (which would be a single
    # undefined command after tokenization); preserve operator-argument boundary.
    s = _MATHRM_OP_RE.sub(r"\\\1 ", s)
    # 3. Tokenize
    tokens = TOKEN_RE.findall(s)
    # 4. Iteratively strip { X } where X is a single non-brace token.
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


# ── Processing pipeline ─────────────────────────────────────────────────────

def _find_inkml_files(root_dir: str) -> list[str]:
    out: list[str] = []
    for dirpath, _dirs, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(".inkml"):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def _iter_inkml_from_zip(
    outer_zip_path: str,
    inner_zip_member: str,
    inner_name_filter=None,
) -> list[tuple[str, bytes]]:
    """Extract InkML bytes from a nested-zip archive.

    Opens outer_zip_path, reads inner_zip_member (another zip), then returns
    (member_name, content_bytes) for each .inkml entry matching inner_name_filter
    (if given). Uses in-memory streams to avoid spilling intermediate zips to disk.
    """
    out: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(outer_zip_path, "r") as outer:
        inner_bytes = outer.read(inner_zip_member)
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        for info in inner.infolist():
            name = info.filename
            if not name.lower().endswith(".inkml"):
                continue
            if inner_name_filter and not inner_name_filter(name):
                continue
            out.append((name, inner.read(info)))
    return out


def _load_test_gt_map(gt_json_path: str) -> dict[str, str]:
    """Load Zenodo CROHME 2019 test ground truth JSON and return {basename: latex}.

    The JSON is a list of entries with fields "input_file" (InkML basename)
    and "input_latex" (ground truth LaTeX string).
    """
    import json
    with open(gt_json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise ValueError(f"Unexpected GT JSON shape in {gt_json_path}")
    return {e["input_file"]: e["input_latex"] for e in entries if "input_file" in e}


def process_inkml_members(
    members: list[tuple[str, bytes]],
    out_img_dir: str,
    id_prefix: str,
    external_gt: dict[str, str] | None = None,
) -> list[tuple[str, list[str]]]:
    """Render each InkML member to PNG and return (id, tokens) for valid samples.

    If external_gt is provided, LaTeX labels are looked up by InkML basename
    instead of parsing `<annotation type="truth">` from the InkML itself. This
    is needed for the CROHME 2019 test set (RIT-hosted InkMLs have stripped GT).
    """
    os.makedirs(out_img_dir, exist_ok=True)
    results: list[tuple[str, list[str]]] = []
    skipped_no_tokens = 0
    skipped_render = 0
    for i, (name, data) in enumerate(members):
        strokes, inkml_latex = parse_inkml_bytes(data)
        if external_gt is not None:
            basename = name.rsplit("/", 1)[-1]
            latex = external_gt.get(basename)
        else:
            latex = inkml_latex
        tokens = tokenize_latex(latex)
        if not strokes or not tokens:
            skipped_no_tokens += 1
            continue
        img = render_strokes(strokes)
        if img is None:
            skipped_render += 1
            continue
        sample_id = f"{id_prefix}_{i:05d}"
        img.save(os.path.join(out_img_dir, f"{sample_id}.png"), optimize=True)
        results.append((sample_id, tokens))
        if (i + 1) % 1000 == 0:
            print(f"    processed {i+1}/{len(members)} "
                  f"(skipped_no_tokens={skipped_no_tokens}, skipped_render={skipped_render})")
    print(f"    done: {len(results)} valid, "
          f"{skipped_no_tokens} skipped (no tokens), {skipped_render} skipped (render)")
    return results


def build_vocab(train_samples: list[tuple[str, list[str]]]) -> list[str]:
    """Build vocabulary from training tokens, prepended with special tokens."""
    token_set: set[str] = set()
    for _id, toks in train_samples:
        token_set.update(toks)
    # Deterministic order: specials first, then lexicographic
    return SPECIAL_TOKENS + sorted(token_set)


def save_labels(samples: list[tuple[str, list[str]]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for sample_id, toks in samples:
            f.write(f"{sample_id}\t{' '.join(toks)}\n")


def save_vocab(vocab: list[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for tok in vocab:
            f.write(f"{tok}\n")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    # Idempotency
    marker_files = [
        os.path.join(data_dir, "meta.json"),
        os.path.join(data_dir, "vocab.txt"),
        os.path.join(data_dir, "train", "labels.txt"),
        os.path.join(data_dir, "val", "labels.txt"),
        os.path.join(test_data_dir, "meta.json"),
        os.path.join(test_data_dir, "labels.txt"),
    ]
    if all(os.path.exists(p) and os.path.getsize(p) > MIN_FILE_BYTES for p in marker_files):
        train_imgs_dir = os.path.join(data_dir, "train", "images")
        test_imgs_dir = os.path.join(test_data_dir, "images")
        if (os.path.isdir(train_imgs_dir) and len(os.listdir(train_imgs_dir)) > 1000
                and os.path.isdir(test_imgs_dir) and len(os.listdir(test_imgs_dir)) > 500):
            print("CROHME 2019 HMER data already prepared, skipping.")
            return

    raw_dir = os.path.join(data_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Download outer archives + test GT JSON
    outer_paths: dict[str, str] = {}
    for key, url in DOWNLOAD_URLS.items():
        ext = "json" if key == "test_gt" else "zip"
        local_path = os.path.join(raw_dir, f"crohme_{key}.{ext}")
        if not os.path.exists(local_path) or os.path.getsize(local_path) < MIN_FILE_BYTES:
            download_file(url, local_path)
        outer_paths[key] = local_path

    # Extract InkML members directly from nested zips (no intermediate tmp dir).
    print("Reading training InkML from nested zip...")
    train_members = _iter_inkml_from_zip(outer_paths["train"], TRAIN_INNER_ZIP)
    print(f"  {len(train_members)} training InkML")

    print("Reading validation InkML from nested zip...")
    val_members = _iter_inkml_from_zip(outer_paths["train"], VAL_INNER_ZIP)
    print(f"  {len(val_members)} validation InkML (CROHME 2014 test, literature-standard val)")

    print("Reading test InkML from nested zip...")
    test_members = _iter_inkml_from_zip(
        outer_paths["test"],
        TEST_OUTER_INNER_ZIP,
        inner_name_filter=lambda n: n.startswith(TEST_INNER_PREFIX),
    )
    print(f"  {len(test_members)} test InkML (CROHME 2019 official test set)")

    print("Loading test ground-truth LaTeX from Zenodo release...")
    test_gt_map = _load_test_gt_map(outer_paths["test_gt"])
    print(f"  {len(test_gt_map)} test GT entries")

    if len(train_members) < 5000 or len(val_members) < 500 or len(test_members) < 800:
        raise RuntimeError(
            "Unexpected InkML counts — CROHME archive layout may have changed. "
            f"Got train={len(train_members)}, val={len(val_members)}, test={len(test_members)}."
        )
    if len(test_gt_map) < 1000:
        raise RuntimeError(
            f"Test ground-truth JSON has only {len(test_gt_map)} entries — "
            "Zenodo release may have changed."
        )

    # Render and save each split
    print("Rendering training images...")
    train_samples = process_inkml_members(
        train_members, os.path.join(data_dir, "train", "images"), id_prefix="train"
    )
    save_labels(train_samples, os.path.join(data_dir, "train", "labels.txt"))

    print("Rendering validation images...")
    val_samples = process_inkml_members(
        val_members, os.path.join(data_dir, "val", "images"), id_prefix="val"
    )
    save_labels(val_samples, os.path.join(data_dir, "val", "labels.txt"))

    # Build vocab from training portion only (val/test unseen tokens become <unk>).
    vocab = build_vocab(train_samples)

    print("Rendering test images (labels joined from Zenodo GT by InkML basename)...")
    test_img_dir = os.path.join(test_data_dir, "images")
    test_samples = process_inkml_members(
        test_members, test_img_dir, id_prefix="test", external_gt=test_gt_map
    )
    save_labels(test_samples, os.path.join(test_data_dir, "labels.txt"))

    # Write vocab to both dirs
    save_vocab(vocab, os.path.join(data_dir, "vocab.txt"))
    save_vocab(vocab, os.path.join(test_data_dir, "vocab.txt"))

    # Compute image size stats on train for meta
    sample_for_stats = train_samples[:500]
    max_h = 0
    max_w = 0
    for sample_id, _t in sample_for_stats:
        p = os.path.join(data_dir, "train", "images", f"{sample_id}.png")
        if os.path.exists(p):
            with Image.open(p) as im:
                w, h = im.size
                max_h = max(max_h, h)
                max_w = max(max_w, w)

    meta = {
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "n_test": len(test_samples),
        "vocab_size": len(vocab),
        "max_img_h": max_h,
        "max_img_w": max_w,
        "special_tokens": {tok: i for i, tok in enumerate(SPECIAL_TOKENS)},
        "image_mode": "L",
        "render_stroke_width": RENDER_STROKE_WIDTH,
        "render_padding": RENDER_PADDING,
    }
    for d in [data_dir, test_data_dir]:
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # Cleanup
    shutil.rmtree(raw_dir, ignore_errors=True)

    print("\nCROHME 2019 HMER data ready:")
    print(f"  Train:     {data_dir}/train/ ({len(train_samples)} samples)")
    print(f"  Val:       {data_dir}/val/ ({len(val_samples)} samples)")
    print(f"  Test:      {test_data_dir}/ ({len(test_samples)} samples)")
    print(f"  Vocab:     {len(vocab)} tokens (incl. {len(SPECIAL_TOKENS)} specials)")


if __name__ == "__main__":
    main()
