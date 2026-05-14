"""VoiceBank+DEMAND speech enhancement evaluator.

Computes PESQ (wideband), STOI, and composite measures (CSIG, CBAK, COVL)
following the standard evaluation protocol for VoiceBank+DEMAND.

Composite measures follow Hu & Loizou (2008):
  CSIG = 3.093 - 1.029*LLR + 0.603*PESQ - 0.009*WSS
  CBAK = 1.634 + 0.478*PESQ - 0.007*WSS + 0.063*segSNR
  COVL = 1.594 + 0.805*PESQ - 0.512*LLR - 0.007*WSS

Helper implementations adapted from github.com/schmiph2/pysepm (Loizou's book).
"""

from __future__ import annotations

import json
import os

import numpy as np
import soundfile as sf
from pesq import pesq as compute_pesq
from pystoi import stoi as compute_stoi
from scipy.linalg import toeplitz
from scipy.signal import stft as scipy_stft

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig

SAMPLE_RATE = 16000

_EPS = np.finfo(np.float64).eps

# 25 critical-band center frequencies and bandwidths (Hz) — Zwicker bands
_CENT_FREQ = np.array([
    50.0, 120.0, 190.0, 260.0, 330.0, 400.0, 470.0, 540.0,
    617.372, 703.378, 798.717, 904.128, 1020.38, 1148.30, 1288.72,
    1442.54, 1610.70, 1794.16, 1993.93, 2211.08, 2446.71, 2701.97,
    2978.04, 3276.17, 3597.63,
])
_BANDWIDTH = np.array([
    70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 77.3724,
    86.0056, 95.3398, 105.411, 116.256, 127.914, 140.423, 153.823,
    168.154, 183.457, 199.776, 217.153, 235.631, 255.255, 276.072,
    298.126, 321.465, 346.136,
])


# ---------------------------------------------------------------------------
# Helper: overlapped windowed frames
# ---------------------------------------------------------------------------

def _extract_frames(x, win_len, hop, win):
    n_frames = 1 + (len(x) - win_len) // hop
    frames = np.zeros((n_frames, win_len))
    for i in range(n_frames):
        frames[i] = x[i * hop: i * hop + win_len] * win
    return frames


def _hann(win_len):
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, win_len + 1) / (win_len + 1)))


# ---------------------------------------------------------------------------
# Segmental SNR
# ---------------------------------------------------------------------------

def _snrseg(clean, processed, fs, frame_len=0.03, overlap=0.75):
    win_len = round(frame_len * fs)
    hop = int(np.floor((1 - overlap) * frame_len * fs))
    win = _hann(win_len)

    cf = _extract_frames(clean, win_len, hop, win)
    pf = _extract_frames(processed, win_len, hop, win)

    sig_e = np.sum(cf ** 2, axis=1)
    noise_e = np.sum((cf - pf) ** 2, axis=1)

    seg = 10 * np.log10(sig_e / (noise_e + _EPS) + _EPS)
    seg = np.clip(seg, -10, 35)
    return float(np.mean(seg[:-1]))


# ---------------------------------------------------------------------------
# Log-Likelihood Ratio (LLR)
# ---------------------------------------------------------------------------

def _lpcoeff(frame, order):
    """Levinson-Durbin LPC coefficients."""
    n = len(frame)
    R = np.array([np.sum(frame[:n - k] * frame[k:]) for k in range(order + 1)])

    a = np.ones(order)
    a_past = np.ones(order)
    E = np.zeros(order + 1)
    E[0] = R[0]

    for i in range(order):
        a_past[:i] = a[:i]
        s = np.sum(a_past[:i] * R[i:0:-1])
        rc = (R[i + 1] - s) / (E[i] + _EPS)
        a[i] = rc
        if i > 0:
            a[:i] = a_past[:i] - rc * a_past[i - 1::-1]
        E[i + 1] = (1 - rc * rc) * E[i]

    lp = np.ones(order + 1)
    lp[1:] = -a
    return lp, R


def _llr(clean, processed, fs, frame_len=0.03, overlap=0.75):
    alpha = 0.95
    win_len = round(frame_len * fs)
    hop = int(np.floor((1 - overlap) * frame_len * fs))
    P = 16 if fs >= 10000 else 10

    win = _hann(win_len)
    cf = _extract_frames(clean + _EPS, win_len, hop, win)
    pf = _extract_frames(processed + _EPS, win_len, hop, win)

    n_frames = cf.shape[0]
    dist = np.zeros(n_frames - 1)

    for i in range(n_frames - 1):
        a_c, r_c = _lpcoeff(cf[i], P)
        a_p, _ = _lpcoeff(pf[i], P)

        T = toeplitz(r_c)
        num = float(a_p @ T @ a_p)
        den = float(a_c @ T @ a_c)

        ratio = num / (den + _EPS)
        if ratio <= 0 or np.isnan(ratio):
            ratio = 1000.0
        dist[i] = np.log(ratio)

    # composite mode: no clipping at 2
    dist = np.sort(dist)
    dist = dist[: int(round(len(dist) * alpha))]
    return float(np.mean(dist))


# ---------------------------------------------------------------------------
# Weighted Spectral Slope (WSS)
# ---------------------------------------------------------------------------

def _find_loc_peaks(slope, energy):
    n = len(energy)
    peaks = np.zeros_like(slope)
    for i in range(len(slope)):
        if slope[i] > 0:
            j = i
            while j < n - 1 and slope[j] > 0:
                j += 1
            peaks[i] = energy[j - 1]
        else:
            j = i
            while j >= 0 and slope[j] <= 0:
                j -= 1
            peaks[i] = energy[j + 1]
    return peaks


def _wss(clean, processed, fs, frame_len=0.03, overlap=0.75):
    Kmax = 20
    Klocmax = 1
    alpha = 0.95
    num_crit = 25

    clean = clean.astype(np.float64) + _EPS
    processed = processed.astype(np.float64) + _EPS

    win_len = round(frame_len * fs)
    hop = int(np.floor((1 - overlap) * frame_len * fs))
    max_freq = fs / 2
    n_fft = int(2 ** np.ceil(np.log2(2 * win_len)))
    n_fftby2 = n_fft // 2

    # Build critical-band filter bank
    bw_min = _BANDWIDTH[0]
    min_factor = np.exp(-30.0 / (2.0 * 2.303))
    j = np.arange(n_fftby2)
    crit_filter = np.zeros((num_crit, n_fftby2))
    for i in range(num_crit):
        f0 = (_CENT_FREQ[i] / max_freq) * n_fftby2
        bw = (_BANDWIDTH[i] / max_freq) * n_fftby2
        norm = np.log(bw_min) - np.log(_BANDWIDTH[i])
        crit_filter[i] = np.exp(-11 * ((j - np.floor(f0)) / bw) ** 2 + norm)
        crit_filter[i] *= (crit_filter[i] > min_factor)

    win = _hann(win_len)
    scale = np.sqrt(1.0 / win.sum() ** 2)
    num_frames = len(clean) // hop - (win_len // hop)
    n_samples = int(num_frames) * hop + win_len - hop

    _, _, Zc = scipy_stft(
        clean[:n_samples], fs=fs, window=win, nperseg=win_len,
        noverlap=win_len - hop, nfft=n_fft, detrend=False,
        return_onesided=True, boundary=None, padded=False,
    )
    clean_spec = np.abs(Zc[:-1, :]) ** 2 / scale ** 2

    _, _, Zp = scipy_stft(
        processed[:n_samples], fs=fs, window=win, nperseg=win_len,
        noverlap=win_len - hop, nfft=n_fft, detrend=False,
        return_onesided=True, boundary=None, padded=False,
    )
    proc_spec = np.abs(Zp[:-1, :]) ** 2 / scale ** 2

    clean_energy = crit_filter @ clean_spec
    log_clean = 10 * np.log10(clean_energy + _EPS)
    log_clean = np.clip(log_clean, -100, None)

    proc_energy = crit_filter @ proc_spec
    log_proc = 10 * np.log10(proc_energy + _EPS)
    log_proc = np.clip(log_proc, -100, None)

    clean_slope = np.diff(log_clean, axis=0)
    proc_slope = np.diff(log_proc, axis=0)

    dBMax_clean = np.max(log_clean, axis=0)
    dBMax_proc = np.max(log_proc, axis=0)

    n_fr = clean_slope.shape[1]
    c_peaks = np.zeros_like(clean_slope)
    p_peaks = np.zeros_like(proc_slope)
    for ii in range(n_fr):
        c_peaks[:, ii] = _find_loc_peaks(clean_slope[:, ii], log_clean[:, ii])
        p_peaks[:, ii] = _find_loc_peaks(proc_slope[:, ii], log_proc[:, ii])

    Wmax_c = Kmax / (Kmax + dBMax_clean - log_clean[:-1, :])
    Wloc_c = Klocmax / (Klocmax + c_peaks - log_clean[:-1, :])
    W_c = Wmax_c * Wloc_c

    Wmax_p = Kmax / (Kmax + dBMax_proc - log_proc[:-1, :])
    Wloc_p = Klocmax / (Klocmax + p_peaks - log_proc[:-1, :])
    W_p = Wmax_p * Wloc_p

    W = (W_c + W_p) / 2.0

    d = np.sum(W * (clean_slope - proc_slope) ** 2, axis=0) / np.sum(W, axis=0)
    d = np.sort(d)
    d = d[: int(round(len(d) * alpha))]
    return float(np.mean(d))


# ---------------------------------------------------------------------------
# Composite measures
# ---------------------------------------------------------------------------

def _composite(clean, processed, fs, pesq_val):
    """Hu & Loizou (2008) composite measures from PESQ + WSS + LLR + segSNR."""
    wss_val = _wss(clean, processed, fs)
    llr_val = _llr(clean, processed, fs)
    snr_val = _snrseg(clean, processed, fs)

    csig = 3.093 - 1.029 * llr_val + 0.603 * pesq_val - 0.009 * wss_val
    cbak = 1.634 + 0.478 * pesq_val - 0.007 * wss_val + 0.063 * snr_val
    covl = 1.594 + 0.805 * pesq_val - 0.512 * llr_val - 0.007 * wss_val

    return np.clip(csig, 1, 5), np.clip(cbak, 1, 5), np.clip(covl, 1, 5)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class VoiceBankDemandEvaluator(MetricEvaluatorBase):
    """Evaluate speech enhancement using PESQ, STOI, CSIG, CBAK, COVL."""

    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        with open(predictions_path) as f:
            manifest = json.load(f)

        predictions_dir = manifest["predictions_dir"]
        if not os.path.isdir(predictions_dir):
            raise ValueError(f"Enhanced WAV directory not found: {predictions_dir}")

        # Load test file list
        test_files_path = os.path.join(test_data_dir, "test_files.txt")
        with open(test_files_path) as f:
            filenames = [line.strip() for line in f if line.strip()]

        clean_dir = os.path.join(test_data_dir, "clean")
        if not os.path.isdir(clean_dir):
            raise ValueError(f"Clean reference directory not found: {clean_dir}")

        pesq_scores, stoi_scores = [], []
        csig_scores, cbak_scores, covl_scores = [], [], []
        missing = []

        for fname in filenames:
            enhanced_path = os.path.join(predictions_dir, fname)
            clean_path = os.path.join(clean_dir, fname)

            if not os.path.isfile(enhanced_path):
                missing.append(fname)
                continue

            enhanced, _ = sf.read(enhanced_path)
            clean, _ = sf.read(clean_path)

            if enhanced.ndim > 1:
                enhanced = enhanced[:, 0]
            if clean.ndim > 1:
                clean = clean[:, 0]

            min_len = min(len(enhanced), len(clean))
            enhanced = enhanced[:min_len]
            clean = clean[:min_len]

            # PESQ (wideband, 16 kHz)
            try:
                p = compute_pesq(SAMPLE_RATE, clean, enhanced, "wb")
                pesq_scores.append(p)
            except Exception:
                continue  # skip utterances where PESQ fails

            # STOI
            try:
                s = compute_stoi(clean, enhanced, SAMPLE_RATE, extended=False)
                stoi_scores.append(s)
            except Exception:
                pass

            # Composite measures (CSIG, CBAK, COVL)
            try:
                cs, cb, co = _composite(clean, enhanced, SAMPLE_RATE, p)
                csig_scores.append(float(cs))
                cbak_scores.append(float(cb))
                covl_scores.append(float(co))
            except Exception:
                pass

        if missing:
            raise ValueError(
                f"Missing {len(missing)} enhanced files out of {len(filenames)}. "
                f"First 5: {missing[:5]}"
            )

        if not pesq_scores:
            raise ValueError("No valid PESQ scores computed — check enhanced WAV files.")

        avg_pesq = float(np.mean(pesq_scores))
        avg_stoi = float(np.mean(stoi_scores)) if stoi_scores else 0.0
        avg_csig = float(np.mean(csig_scores)) if csig_scores else 0.0
        avg_cbak = float(np.mean(cbak_scores)) if cbak_scores else 0.0
        avg_covl = float(np.mean(covl_scores)) if covl_scores else 0.0

        metrics = {
            "pesq": round(avg_pesq, 4),
            "csig": round(avg_csig, 4),
            "cbak": round(avg_cbak, 4),
            "covl": round(avg_covl, 4),
            "stoi": round(avg_stoi, 4),
            "num_evaluated": len(pesq_scores),
        }

        return EvalResult(
            metrics=metrics,
            primary_metric_name="pesq",
            primary_metric_value=avg_pesq,
        )
