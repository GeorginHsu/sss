"""LJSpeech TTS evaluator: assess synthesized speech quality.

Primary metric: UTMOS (automated MOS prediction, 1-5 scale, higher is better).
  Uses SpeechMOS v1.2.0 (wav2vec2-based predictor trained on MOS labels).
  Correlation with human MOS: SRCC ~0.90 at utterance level.

Secondary metric: MCD on a deterministic subset (Mel Cepstral Distortion, dB,
lower is better).
  Measures spectral distance between synthesized and reference audio using
  DTW-aligned 13-order MFCCs. It is capped because pyworld/DTW MCD is much
  slower than the primary UTMOS metric.
"""

from __future__ import annotations

import json
import os
import warnings
from collections import defaultdict

import numpy as np
import torch
import torchaudio

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig

SAMPLE_RATE = 22050
UTMOS_BATCH_SIZE = 16
MCD_MAX_FILES = 25


# ---------------------------------------------------------------------------
# UTMOS: Automated MOS Prediction
# ---------------------------------------------------------------------------

def _load_utmos(device: str = "cpu"):
    """Load UTMOS predictor from pre-cached SpeechMOS repo."""
    hub_dir = os.path.join(
        os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch")),
        "hub",
    )
    # Find the cached SpeechMOS directory
    repo_dir = None
    try:
        for name in os.listdir(hub_dir):
            if "SpeechMOS" in name or "speechmos" in name.lower():
                repo_dir = os.path.join(hub_dir, name)
                break
    except FileNotFoundError:
        pass

    if repo_dir and os.path.isdir(repo_dir):
        predictor = torch.hub.load(
            repo_dir, "utmos22_strong", source="local", trust_repo=True
        )
    else:
        # Fallback: try normal hub load (works if network or cache available)
        predictor = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
        )

    predictor = predictor.to(device)
    predictor.eval()
    return predictor


def compute_utmos_scores(
    wav_paths: list[str], device: str = "cpu", batch_size: int = 16
) -> list[float]:
    """Compute UTMOS score for each WAV file.

    SpeechMOS/UTMOS accepts batched waveforms only when they share the same
    sample count. We batch exact-length groups to speed up common cases without
    padding, because padding changes the utterance-level average score.
    """
    predictor = _load_utmos(device)
    scores = [float("nan")] * len(wav_paths)
    groups: dict[
        tuple[int, int], list[tuple[int, torch.Tensor, str]]
    ] = defaultdict(list)

    for idx, path in enumerate(wav_paths):
        try:
            wave, sr = torchaudio.load(path)
            # UTMOS expects (batch, samples); handles any sample rate internally.
            if wave.shape[0] > 1:
                wave = wave.mean(dim=0, keepdim=True)
            groups[(int(sr), int(wave.shape[-1]))].append((idx, wave, path))
        except Exception as e:
            warnings.warn(f"UTMOS failed for {os.path.basename(path)}: {e}")

    for (sr, _num_samples), items in groups.items():
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            try:
                batch = torch.cat([wave for _, wave, _ in chunk], dim=0).to(device)
                with torch.no_grad():
                    chunk_scores = predictor(batch, sr=sr).detach().cpu().view(-1)
                for (idx, _wave, _path), score in zip(chunk, chunk_scores):
                    scores[idx] = float(score.item())
            except Exception as e:
                # Fall back to one-by-one scoring for this chunk so one bad or
                # too-large file does not discard the rest of the submission.
                warnings.warn(f"UTMOS batch failed for {len(chunk)} files: {e}")
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                for idx, wave, path in chunk:
                    try:
                        with torch.no_grad():
                            score = predictor(wave.to(device), sr=sr)
                        scores[idx] = float(score.item())
                    except Exception as item_e:
                        warnings.warn(
                            f"UTMOS failed for {os.path.basename(path)}: {item_e}"
                        )

    return scores


# ---------------------------------------------------------------------------
# MCD: Mel Cepstral Distortion (with DTW alignment)
# ---------------------------------------------------------------------------

def compute_mcd_scores(
    synth_paths: list[str],
    ref_paths: list[str],
) -> list[float]:
    """Compute MCD between synthesized and reference audio pairs using DTW."""
    try:
        from pymcd.mcd import Calculate_MCD
        mcd_calculator = Calculate_MCD(MCD_mode="dtw")
    except ImportError:
        warnings.warn("pymcd not installed, skipping MCD computation")
        return []

    scores = []
    for synth_path, ref_path in zip(synth_paths, ref_paths):
        try:
            mcd_val = mcd_calculator.calculate_mcd(ref_path, synth_path)
            scores.append(float(mcd_val))
        except Exception as e:
            warnings.warn(
                f"MCD failed for {os.path.basename(synth_path)}: {e}"
            )
            scores.append(float("nan"))

    return scores


# ---------------------------------------------------------------------------
# Main Evaluator
# ---------------------------------------------------------------------------

class LJSpeechTTSEvaluator(MetricEvaluatorBase):
    """Evaluate TTS quality using UTMOS (primary) and MCD (secondary)."""

    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        # 1. Load predictions manifest
        with open(predictions_path) as f:
            manifest = json.load(f)

        predictions_dir = manifest["predictions_dir"]
        if not os.path.isdir(predictions_dir):
            raise ValueError(
                f"Synthesized WAV directory not found: {predictions_dir}"
            )

        # 2. Load test file list
        test_texts_path = os.path.join(test_data_dir, "test_texts.txt")
        test_ids = []
        with open(test_texts_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    uid = line.split("|")[0].strip()
                    test_ids.append(uid)

        # 3. Verify all synthesized files exist
        synth_paths = []
        ref_paths = []
        ref_wav_dir = os.path.join(test_data_dir, "reference_wavs")
        missing = []

        for uid in test_ids:
            synth_path = os.path.join(predictions_dir, f"{uid}.wav")
            ref_path = os.path.join(ref_wav_dir, f"{uid}.wav")

            if not os.path.isfile(synth_path):
                missing.append(f"{uid}.wav")
                continue

            synth_paths.append(synth_path)
            ref_paths.append(ref_path)

        if missing:
            raise ValueError(
                f"Missing {len(missing)} synthesized files out of "
                f"{len(test_ids)}. First 5: {missing[:5]}"
            )

        # 4. Compute UTMOS (primary metric)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Computing UTMOS on {len(synth_paths)} files (device={device}) ...")
        utmos_scores = compute_utmos_scores(
            synth_paths, device=device, batch_size=UTMOS_BATCH_SIZE
        )

        valid_utmos = [s for s in utmos_scores if not np.isnan(s)]
        if not valid_utmos:
            raise ValueError("No valid UTMOS scores computed — check WAV files.")
        avg_utmos = float(np.mean(valid_utmos))

        # 5. Compute MCD (secondary metric)
        n_mcd = min(MCD_MAX_FILES, len(synth_paths))
        print(f"Computing MCD on {n_mcd} deterministic pairs ...")
        mcd_scores = compute_mcd_scores(synth_paths[:n_mcd], ref_paths[:n_mcd])

        valid_mcd = [s for s in mcd_scores if not np.isnan(s)]
        avg_mcd = float(np.mean(valid_mcd)) if valid_mcd else 0.0

        metrics = {
            "utmos": round(avg_utmos, 4),
            "mcd": round(avg_mcd, 4),
            "num_evaluated": len(valid_utmos),
            "num_mcd_evaluated": len(valid_mcd),
        }

        return EvalResult(
            metrics=metrics,
            primary_metric_name="utmos",
            primary_metric_value=avg_utmos,
        )
