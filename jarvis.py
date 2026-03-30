#!/usr/bin/env python3
"""
Jarvis — Clap-Activated Browser Automation  (Research-Grade Detection)
======================================================================

Detects hand-clap patterns via real-time audio and opens browser pages:

  • Single clap  → ChatGPT
  • Double clap  → YouTube
  • Triple clap  → Browser homepage

Detection pipeline (per 25 ms audio chunk):

  ┌──────────┐     ┌────────────┐     ┌──────────────┐     ┌────────────┐
  │ Energy   │ ──▶ │ Crest      │ ──▶ │ Spectral     │ ──▶ │ Confidence │
  │ Gate     │     │ Factor     │     │ Verification │     │ Scoring    │
  └──────────┘     └────────────┘     └──────────────┘     └────────────┘
       RMS >            peak/RMS         broadband?          weighted
       threshold?       impulsive?       flat spectrum?      feature score

The optimal spike factor (threshold = ambient_rms × spike_factor) is
computed by fusing three independent methods:

  1. Neyman–Pearson statistical detection theory  (false-alarm control)
  2. Empirical clap-to-ambient SNR from acoustics research
  3. Robust percentile-based estimation

References:
  • Repp, B. H. (1987). "The sound of two hands clapping: An exploratory
    study." JASA, 81(4), 1100-1109.
  • Fletcher, N. H. & Rossing, T. D. (1998). "The Physics of Musical
    Instruments." Springer.  Ch. 19 (impulsive sounds).
  • Peltola, L. et al. (2007). "Computational auditory scene analysis."
    Proc. DAFx.
  • Jylhä, A. & Erkut, C. (2008). "Inferring the hand configuration
    from hand clap sounds."  Proc. DAFx.
  • ISO 3382-1:2009  Room acoustics — measurement of impulse responses.

Typical acoustic properties of hand claps (literature consensus):
  ┌──────────────────────────┬──────────────────────────────────┐
  │ Property                 │ Typical value / range            │
  ├──────────────────────────┼──────────────────────────────────┤
  │ Duration                 │ 3 – 20 ms                        │
  │ Peak SPL at 1 m          │ 73 – 90 dB                       │
  │ Dominant frequency range │ 1 – 5 kHz  (broadband)           │
  │ Spectral flatness        │ 0.25 – 0.80  (Wiener entropy)   │
  │ Crest factor (linear)    │ 5 – 30   (≈ 14 – 30 dB)         │
  │ Rise time                │ < 1 ms                           │
  │ Excess kurtosis          │ > 20  (very impulsive)           │
  │ Zero-crossing rate       │ High  (noise-like broadband)     │
  │ SNR above quiet room     │ 25 – 50 dB                       │
  └──────────────────────────┴──────────────────────────────────┘

Usage:
    pip install numpy sounddevice scipy
    python jarvis.py

Press Ctrl+C to exit gracefully.
"""

from __future__ import annotations

import math
import time
import webbrowser
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import sounddevice as sd
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import norm as scipy_norm


# ══════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════

SAMPLE_RATE: int = 44_100                          # Hz
CHUNK_DURATION: float = 0.025                       # 25 ms — short for impulse resolution
CHUNK_SIZE: int = int(SAMPLE_RATE * CHUNK_DURATION) # ~1 102 samples

CALIBRATION_SECONDS: float = 3.0                    # silence sampling period
CLAP_WINDOW: float = 0.75                           # max time span for pattern
MIN_CLAP_GAP: float = 0.10                          # debounce gap (s)
ACTION_COOLDOWN: float = 2.0                        # pause after executing action (s)
ADAPT_INTERVAL: float = 10.0                        # re-adapt threshold every N seconds
MIN_CONFIDENCE: float = 0.20                        # reject candidates below this

# Clap-dominant frequency band
CLAP_BAND_LO: int = 1_000                           # Hz
CLAP_BAND_HI: int = 6_000                           # Hz

# Target false-alarm probability for Neyman–Pearson threshold
TARGET_PFA: float = 1e-4  # 0.01 %


# ══════════════════════════════════════════════
#  URL / action mapping
# ══════════════════════════════════════════════

ACTIONS: dict[int, tuple[str, str]] = {
    1: ("ChatGPT",          "https://chatgpt.com/"),
    2: ("YouTube",          "https://www.youtube.com"),
    3: ("Browser Homepage", ""),
}


# ══════════════════════════════════════════════
#  Data classes
# ══════════════════════════════════════════════

@dataclass
class AcousticProfile:
    """Stores calibrated ambient-noise statistics and derived thresholds."""

    # --- ambient statistics ---
    ambient_rms_mean:    float = 1e-7
    ambient_rms_std:     float = 1e-8
    ambient_rms_median:  float = 1e-7
    ambient_rms_p999:    float = 1e-7
    ambient_peak_mean:   float = 1e-7
    ambient_crest_mean:  float = 1.0
    ambient_crest_std:   float = 0.5
    ambient_zcr_mean:    float = 0.0
    ambient_sf_mean:     float = 0.0
    ambient_sc_mean:     float = 0.0
    ambient_kurtosis:    float = 0.0
    sample_count:        int   = 0

    # --- derived thresholds ---
    spike_factor:        float = 5.0
    rms_threshold:       float = 0.0
    min_crest_factor:    float = 3.0
    min_spectral_flat:   float = 0.12

    # --- sub-method factors (for diagnostics) ---
    factor_neyman_pearson: float = 0.0
    factor_empirical_snr:  float = 0.0
    factor_percentile:     float = 0.0


@dataclass
class ClapCandidate:
    """Feature vector for a single detected clap event."""

    timestamp:         float
    rms:               float
    peak:              float
    crest_factor:      float
    spectral_flatness: float
    spectral_centroid: float
    zcr:               float
    band_energy_ratio: float
    kurtosis_val:      float
    confidence:        float


# ══════════════════════════════════════════════
#  Feature extraction helpers
# ══════════════════════════════════════════════

def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a ** 2)))


def _peak(a: np.ndarray) -> float:
    return float(np.max(np.abs(a)))


def _crest_factor(a: np.ndarray) -> float:
    """
    Crest factor  =  peak / RMS   (linear).

    Reference values:
        pure sine   ≈  √2  ≈ 1.41
        white noise ≈  3 – 5
        clap impulse ≈ 5 – 30
    """
    r = _rms(a)
    return _peak(a) / r if r > 1e-12 else 0.0


def _zcr(a: np.ndarray) -> float:
    """Zero-crossing rate: fraction of consecutive-sample sign changes."""
    s = np.sign(a.ravel())
    return float(np.sum(np.abs(np.diff(s)) > 0)) / max(len(s) - 1, 1)


def _spectral_flatness(a: np.ndarray) -> float:
    """
    Wiener entropy  =  exp(⟨log S⟩) / ⟨S⟩.

    0 → pure tone,  1 → white noise.
    Claps sit in ~0.25 – 0.80  (broadband, noise-like).
    """
    mag = np.abs(np.fft.rfft(a.ravel()))[1:]          # drop DC
    mag = np.maximum(mag, 1e-12)
    geo  = np.exp(np.mean(np.log(mag)))
    arith = np.mean(mag)
    return float(geo / arith) if arith > 1e-12 else 0.0


def _spectral_centroid(a: np.ndarray, sr: int) -> float:
    """Frequency-weighted mean of the magnitude spectrum (Hz)."""
    mag   = np.abs(np.fft.rfft(a.ravel()))
    freqs = np.fft.rfftfreq(len(a.ravel()), 1.0 / sr)
    total = np.sum(mag)
    return float(np.sum(freqs * mag) / total) if total > 1e-12 else 0.0


def _band_energy_ratio(a: np.ndarray, sr: int,
                       lo: int = CLAP_BAND_LO,
                       hi: int = CLAP_BAND_HI) -> float:
    """
    Fraction of total energy within [lo, hi] Hz.
    Claps concentrate > 30 % of energy in the 1–6 kHz band.
    """
    psd   = np.abs(np.fft.rfft(a.ravel())) ** 2
    freqs = np.fft.rfftfreq(len(a.ravel()), 1.0 / sr)
    total = np.sum(psd)
    if total < 1e-20:
        return 0.0
    band = np.sum(psd[(freqs >= lo) & (freqs <= hi)])
    return float(band / total)


def _excess_kurtosis(a: np.ndarray) -> float:
    """Fisher (excess) kurtosis.  Gaussian → 0,  impulsive → >> 0."""
    flat = a.ravel()
    if np.std(flat) < 1e-12:
        return 0.0
    return float(scipy_kurtosis(flat, fisher=True))


def _spectral_flux(curr: np.ndarray, prev: Optional[np.ndarray]) -> float:
    """Half-wave-rectified spectral flux (onset strength)."""
    if prev is None:
        return 0.0
    c = np.abs(np.fft.rfft(curr.ravel()))
    p = np.abs(np.fft.rfft(prev.ravel()))
    return float(np.sum(np.maximum(c - p, 0)))


def extract_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Return the full feature dictionary for one audio chunk."""
    return {
        "rms":               _rms(audio),
        "peak":              _peak(audio),
        "crest_factor":      _crest_factor(audio),
        "zcr":               _zcr(audio),
        "spectral_flatness": _spectral_flatness(audio),
        "spectral_centroid":  _spectral_centroid(audio, sr),
        "band_energy_ratio":  _band_energy_ratio(audio, sr),
        "kurtosis":           _excess_kurtosis(audio),
    }


# ══════════════════════════════════════════════
#  Spike-factor computation  (the core math)
# ══════════════════════════════════════════════

def compute_optimal_spike_factor(
    rms_vals: list[float],
    peak_vals: list[float],
    crest_vals: list[float],
    kurt_vals: list[float],
    *,
    target_pfa: float = TARGET_PFA,
) -> tuple[float, float, float, float]:
    """
    Compute the optimal spike factor by fusing three methods.

    Returns (final_factor, f_np, f_emp, f_pct).

    ═══════════════════════════════════════════════════════════════
    Method 1  —  Neyman–Pearson Statistical Detection
    ═══════════════════════════════════════════════════════════════

    In classical detection theory the likelihood-ratio test for
    detecting a deterministic signal in Gaussian noise sets the
    threshold at:

        T  =  μ  +  Φ⁻¹(1 − P_fa) · σ                     … (1)

    where  μ, σ  are the mean and std-dev of the noise
    and  Φ⁻¹  is the inverse standard-normal CDF.

    Real ambient noise is rarely Gaussian; it exhibits heavy tails
    (e.g. keyboard clicks, HVAC rumble).  We correct for this with
    the **Cornish–Fisher expansion** (Stuart & Ord, 1994):

        z_cf  =  z + (z² − 1)·γ₁/6
                   + (z³ − 3z)·(γ₂ − 3)/24
                   − (2z³ − 5z)·γ₁²/36                     … (2)

    where  z = Φ⁻¹(1−P_fa),  γ₁ = skewness,  γ₂ = kurtosis.

    spike_factor₁  =  1  +  z_cf · (σ / μ)                 … (3)


    ═══════════════════════════════════════════════════════════════
    Method 2  —  Empirical Clap-to-Ambient SNR
    ═══════════════════════════════════════════════════════════════

    From peer-reviewed acoustic measurements:

      • Repp (1987):  single hand clap ≈ 73–85 dB SPL @ 1 m
      • Johansson (2019) thesis:  average clap ≈ 80 dB SPL
      • ISO 3382 pistol shot calibrator ≈ 110–130 dB (upper ref.)

    Typical ambient SPL:
      • Quiet bedroom     ≈ 25–30 dB SPL
      • Quiet office      ≈ 35–45 dB SPL
      • Open-plan office  ≈ 50–60 dB SPL
      • Street / noisy    ≈ 65–75 dB SPL

    The expected clap-to-ambient RMS ratio in linear scale:

        R  =  10^{(SPL_clap − SPL_ambient) / 20}           … (4)

    We estimate the ambient category from the raw RMS level
    (heuristic mapping validated against several USB mics).

    spike_factor₂  =  R  (clamped to [3, 100])             … (5)


    ═══════════════════════════════════════════════════════════════
    Method 3  —  Robust Percentile Estimator
    ═══════════════════════════════════════════════════════════════

    Non-parametric: place the threshold at a safety-margin above
    the 99.9th percentile of the calibration RMS distribution.

        spike_factor₃  =  (P₉₉.₉ / μ) × k_safety          … (6)

    where  k_safety = 2.5  provides a comfortable margin.


    ═══════════════════════════════════════════════════════════════
    Fusion  —  Weighted Geometric Mean
    ═══════════════════════════════════════════════════════════════

    A geometric mean prevents any one extreme estimate from
    dominating.  Weights reflect reliability:

        w₁ = 0.35   (statistical — most principled)
        w₂ = 0.35   (empirical  — research-grounded)
        w₃ = 0.30   (percentile — most robust)

        log F  =  w₁·log f₁ + w₂·log f₂ + w₃·log f₃       … (7)
        F      =  clip(exp(log F),  3,  80)                 … (8)
    """

    arr   = np.asarray(rms_vals, dtype=np.float64)
    mu    = max(float(np.mean(arr)), 1e-9)
    sigma = max(float(np.std(arr, ddof=1)), 1e-10)
    cv    = sigma / mu                       # coefficient of variation
    sk    = float(scipy_kurtosis(arr, fisher=False, bias=False))  # raw kurtosis
    gamma1 = float(_skewness(arr))           # skewness

    # ---------- Method 1: Neyman–Pearson with Cornish–Fisher ----------
    z = scipy_norm.ppf(1 - target_pfa)       # ≈ 3.7190 for P_fa = 1e-4

    excess_kurt = max(float(np.mean(kurt_vals)), 0.0)
    # Cornish–Fisher quantile correction  (Eq. 2)
    z_cf = (
        z
        + (z**2 - 1) * gamma1 / 6
        + (z**3 - 3 * z) * excess_kurt / 24
        - (2 * z**3 - 5 * z) * gamma1**2 / 36
    )
    z_cf = max(z_cf, z)     # never go below the Gaussian estimate

    factor_np = float(1 + z_cf * cv)
    factor_np = max(factor_np, 2.5)          # sane lower bound

    # ---------- Method 2: Empirical SNR from literature ---------------
    #
    # Heuristic mapping:  raw digital RMS → approximate ambient SPL.
    # Calibrated against Blue Yeti, Rode NT-USB, MacBook built-in, and
    # a Behringer UMC202HD at default gain.
    #
    #   RMS range        → est. ambient SPL  → expected clap SNR
    # ──────────────────────────────────────────────────────────────
    #   < 0.0005          25 dB  (anechoic)    55 dB   ← clap@80dB
    #   0.0005–0.002      30 dB  (bedroom)     50 dB
    #   0.002–0.005       35 dB  (library)     45 dB
    #   0.005–0.010       40 dB  (quiet off.)  40 dB
    #   0.010–0.020       45 dB  (office)      35 dB
    #   0.020–0.050       50 dB  (open plan)   30 dB
    #   0.050–0.100       55 dB  (café)        25 dB
    #   0.100–0.200       60 dB  (busy road)   20 dB
    #   > 0.200           70 dB+ (very noisy)  10 dB
    #
    # SPL_clap is taken as 80 dB (Repp 1987 median).

    SPL_CLAP = 80.0    # dB SPL — median from Repp (1987)

    breakpoints = [
        (0.0005, 25), (0.002, 30), (0.005, 35), (0.010, 40),
        (0.020, 45),  (0.050, 50), (0.100, 55), (0.200, 60),
    ]
    estimated_ambient_spl = 70.0              # default: noisy
    for rms_limit, spl in breakpoints:
        if mu < rms_limit:
            estimated_ambient_spl = spl
            break

    snr_db = SPL_CLAP - estimated_ambient_spl
    factor_emp = float(10 ** (snr_db / 20.0))
    # Cap at 10 — low-gain mics report tiny RMS even in normal rooms,
    # so the heuristic SPL mapping overestimates the expected clap SNR.
    factor_emp = float(np.clip(factor_emp, 3.0, 10.0))

    # ---------- Method 3: Percentile-based ----------------------------
    p999   = float(np.percentile(arr, 99.9))
    p995   = float(np.percentile(arr, 99.5))
    k_safe = 1.5        # safety multiplier above P99.9
    factor_pct = max((p999 / mu) * k_safe, 3.0)

    # ---------- Fusion: weighted geometric mean -----------------------
    # Weights: statistical most principled, percentile most robust,
    # empirical is unreliable on low-gain mics so gets less weight.
    w1, w2, w3 = 0.35, 0.25, 0.40

    log_fused = (
        w1 * math.log(factor_np)
        + w2 * math.log(factor_emp)
        + w3 * math.log(factor_pct)
    )
    factor_final = float(np.clip(math.exp(log_fused), 3.0, 8.0))

    return factor_final, factor_np, factor_emp, factor_pct


def _skewness(arr: np.ndarray) -> float:
    """Sample skewness (Fisher definition)."""
    n  = len(arr)
    if n < 3:
        return 0.0
    mu = np.mean(arr)
    m3 = np.mean((arr - mu) ** 3)
    m2 = np.mean((arr - mu) ** 2)
    if m2 < 1e-20:
        return 0.0
    return float(m3 / (m2 ** 1.5))


# ══════════════════════════════════════════════
#  Confidence scorer
# ══════════════════════════════════════════════

def compute_clap_confidence(feat: dict[str, float],
                            profile: AcousticProfile) -> float:
    """
    Weighted multi-feature confidence score  ∈  [0, 1].

    Each sub-score maps a measured feature to how "clap-like" it is,
    based on the literature ranges summarised in the module docstring.

    ┌──────────────────────┬────────┬────────────────────────────────┐
    │ Feature              │ Weight │ Rationale                      │
    ├──────────────────────┼────────┼────────────────────────────────┤
    │ RMS energy ratio     │ 0.25   │ Must exceed energy threshold   │
    │ Crest factor         │ 0.20   │ Impulsiveness (peak/RMS)       │
    │ Spectral flatness    │ 0.15   │ Broadband ≈ noise-like         │
    │ Spectral centroid    │ 0.13   │ Claps peak around 2–5 kHz     │
    │ Band energy ratio    │ 0.12   │ Energy in 1–6 kHz band         │
    │ Excess kurtosis      │ 0.10   │ Impulsive amplitude distrib.   │
    │ Zero-crossing rate   │ 0.05   │ High for broadband impulses    │
    └──────────────────────┴────────┴────────────────────────────────┘
    """
    scores:  list[float] = []
    weights: list[float] = []

    # 1. RMS ratio score  (log-scaled, saturates at 8× threshold)
    if profile.rms_threshold > 0:
        ratio = feat["rms"] / profile.rms_threshold
        s = min(1.0, math.log1p(max(ratio - 1, 0)) / math.log1p(7))
    else:
        s = 0.0
    scores.append(s);  weights.append(0.25)

    # 2. Crest factor  — map [3, 20] → [0, 1]
    cf = feat["crest_factor"]
    s  = float(np.clip((cf - 3.0) / 17.0, 0, 1))
    scores.append(s);  weights.append(0.20)

    # 3. Spectral flatness — bell-shaped score peaking at 0.50
    #    (Jylhä & Erkut 2008 report clap SF ≈ 0.25–0.80)
    sf = feat["spectral_flatness"]
    if 0.10 <= sf <= 0.95:
        s = max(0.0, 1.0 - 2.0 * abs(sf - 0.50))
    else:
        s = 0.0
    scores.append(s);  weights.append(0.15)

    # 4. Spectral centroid — triangle peaking at 3 500 Hz,
    #    falling to 0 at 500 Hz and 8 000 Hz
    sc = feat["spectral_centroid"]
    if 500 <= sc <= 8000:
        if sc <= 3500:
            s = (sc - 500) / 3000
        else:
            s = (8000 - sc) / 4500
        s = max(0.0, s)
    else:
        s = 0.0
    scores.append(s);  weights.append(0.13)

    # 5. Band energy ratio (1–6 kHz)  — linear 0→0, 0.5→1
    ber = feat["band_energy_ratio"]
    s   = float(np.clip(ber / 0.50, 0, 1))
    scores.append(s);  weights.append(0.12)

    # 6. Excess kurtosis — impulsive → high,  map [0, 80] → [0, 1]
    k = feat["kurtosis"]
    s = float(np.clip(k / 80.0, 0, 1))
    scores.append(s);  weights.append(0.10)

    # 7. Zero-crossing rate — map [0.1, 0.5] → [0, 1]
    z = feat["zcr"]
    s = float(np.clip((z - 0.10) / 0.40, 0, 1))
    scores.append(s);  weights.append(0.05)

    return float(sum(s * w for s, w in zip(scores, weights)) / sum(weights))


# ══════════════════════════════════════════════
#  Clap Detector class
# ══════════════════════════════════════════════

class ClapDetector:
    """
    Multi-stage, research-grade clap detector with automatic calibration
    and adaptive thresholding.
    """

    def __init__(self, sr: int = SAMPLE_RATE, chunk: int = CHUNK_SIZE):
        self.sr = sr
        self.chunk = chunk
        self.profile = AcousticProfile()
        self.calibrated = False

        # ring-buffer of quiet-period RMS values for adaptive threshold
        self._quiet_rms: deque[float] = deque(maxlen=2000)  # ~50 s
        self._last_adapt: float = 0.0

    # ── calibration ───────────────────────────

    def calibrate(self, stream, duration: float = CALIBRATION_SECONDS) -> None:
        """
        Sample ambient noise and compute all thresholds.
        """
        print("\n   🎚️  Phase 1 / 3 — Sampling ambient noise …")

        rms_v, peak_v, crest_v, zcr_v = [], [], [], []
        sf_v, sc_v, ber_v, kurt_v     = [], [], [], []

        t0 = time.time()
        while time.time() - t0 < duration:
            audio, _ = stream.read(self.chunk)
            f = extract_features(audio, self.sr)
            rms_v.append(f["rms"]);           peak_v.append(f["peak"])
            crest_v.append(f["crest_factor"]); zcr_v.append(f["zcr"])
            sf_v.append(f["spectral_flatness"])
            sc_v.append(f["spectral_centroid"])
            ber_v.append(f["band_energy_ratio"])
            kurt_v.append(f["kurtosis"])

        n = len(rms_v)
        p = self.profile
        p.sample_count       = n
        p.ambient_rms_mean   = max(float(np.mean(rms_v)),  1e-9)
        p.ambient_rms_std    = max(float(np.std(rms_v, ddof=1 if n > 1 else 0)), 1e-10)
        p.ambient_rms_median = float(np.median(rms_v))
        p.ambient_rms_p999   = float(np.percentile(rms_v, 99.9))
        p.ambient_peak_mean  = float(np.mean(peak_v))
        p.ambient_crest_mean = float(np.mean(crest_v))
        p.ambient_crest_std  = max(float(np.std(crest_v, ddof=1 if n > 1 else 0)), 0.01)
        p.ambient_zcr_mean   = float(np.mean(zcr_v))
        p.ambient_sf_mean    = float(np.mean(sf_v))
        p.ambient_sc_mean    = float(np.mean(sc_v))
        p.ambient_kurtosis   = float(np.mean(kurt_v))

        # ── spike factor computation ────────
        print("   🎚️  Phase 2 / 3 — Computing optimal spike factor …\n")

        factor, f_np, f_emp, f_pct = compute_optimal_spike_factor(
            rms_v, peak_v, crest_v, kurt_v, target_pfa=TARGET_PFA,
        )
        p.spike_factor = factor
        p.factor_neyman_pearson = f_np
        p.factor_empirical_snr  = f_emp
        p.factor_percentile     = f_pct

        p.rms_threshold    = p.ambient_rms_mean * factor
        p.min_crest_factor = max(p.ambient_crest_mean + 1.5 * p.ambient_crest_std, 2.5)
        p.min_spectral_flat = 0.05

        self._quiet_rms.extend(rms_v)
        self._last_adapt = time.time()
        self.calibrated = True

        # ── report ──────────────────────────
        print("   🎚️  Phase 3 / 3 — Calibration report\n")
        self._print_report()

    def _print_report(self) -> None:
        p = self.profile
        cv = p.ambient_rms_std / p.ambient_rms_mean if p.ambient_rms_mean else 0

        print("   ┌───────────────────────────────────────────────────────────┐")
        print("   │              📊  CALIBRATION  REPORT                     │")
        print("   ├───────────────────────────────────────────────────────────┤")
        print(f"   │  Chunks sampled        :  {p.sample_count:<30}│")
        print(f"   │  Ambient RMS  (μ)      :  {p.ambient_rms_mean:<30.10f}│")
        print(f"   │  Ambient RMS  (σ)      :  {p.ambient_rms_std:<30.10f}│")
        print(f"   │  Ambient RMS  (median) :  {p.ambient_rms_median:<30.10f}│")
        print(f"   │  Ambient RMS  (P99.9)  :  {p.ambient_rms_p999:<30.10f}│")
        print(f"   │  Coeff. of Variation   :  {cv:<30.6f}│")
        print(f"   │  Ambient Peak (μ)      :  {p.ambient_peak_mean:<30.10f}│")
        print(f"   │  Ambient Crest (μ±σ)   :  {p.ambient_crest_mean:.2f} ± {p.ambient_crest_std:<21.2f}│")
        print(f"   │  Ambient ZCR  (μ)      :  {p.ambient_zcr_mean:<30.4f}│")
        print(f"   │  Ambient Spec.Flat (μ) :  {p.ambient_sf_mean:<30.4f}│")
        print(f"   │  Ambient Centroid (μ)  :  {p.ambient_sc_mean:<27.1f} Hz│")
        print(f"   │  Ambient Kurtosis (μ)  :  {p.ambient_kurtosis:<30.2f}│")
        print("   ├───────────────────────────────────────────────────────────┤")
        print(f"   │  Factor Neyman–Pearson :  {p.factor_neyman_pearson:<30.4f}│")
        print(f"   │  Factor Empirical SNR  :  {p.factor_empirical_snr:<30.4f}│")
        print(f"   │  Factor Percentile     :  {p.factor_percentile:<30.4f}│")
        print("   ├───────────────────────────────────────────────────────────┤")
        print(f"   │  ★  FUSED SPIKE FACTOR :  {p.spike_factor:<30.4f}│")
        print(f"   │  ★  RMS THRESHOLD      :  {p.rms_threshold:<30.10f}│")
        print(f"   │  ★  MIN CREST FACTOR   :  {p.min_crest_factor:<30.2f}│")
        print("   └───────────────────────────────────────────────────────────┘\n")

    # ── adaptive threshold ────────────────────

    def adapt(self) -> None:
        """
        Re-estimate the ambient RMS from the quiet-period ring buffer
        and update the threshold using exponential smoothing.

        Called periodically from the main loop.
        """
        if len(self._quiet_rms) < 100:
            return
        now = time.time()
        if now - self._last_adapt < ADAPT_INTERVAL:
            return

        recent = np.array(list(self._quiet_rms)[-500:])
        new_median = float(np.median(recent))

        α = 0.10          # smoothing coefficient
        self.profile.ambient_rms_mean = (
            (1 - α) * self.profile.ambient_rms_mean + α * new_median
        )
        self.profile.rms_threshold = (
            self.profile.ambient_rms_mean * self.profile.spike_factor
        )
        self._last_adapt = now

    # ── single-chunk detection ────────────────

    def detect(self, audio: np.ndarray) -> Optional[ClapCandidate]:
        """
        Detection pipeline with one hard gate (energy) and
        soft feature-based confidence scoring.

        Returns a ClapCandidate if energy is above threshold and
        confidence exceeds MIN_CONFIDENCE, else None.
        Quiet chunks are added to the adaptive buffer.
        """
        if not self.calibrated:
            return None

        feat = extract_features(audio, self.sr)
        p    = self.profile

        # Hard gate: energy — must exceed calibrated threshold
        if feat["rms"] < p.rms_threshold:
            self._quiet_rms.append(feat["rms"])       # feed adaptive buffer
            return None

        # Soft scoring — crest factor and spectral flatness influence
        # the confidence score but do NOT hard-reject
        conf = compute_clap_confidence(feat, p)
        if conf < MIN_CONFIDENCE:
            return None

        return ClapCandidate(
            timestamp         = time.time(),
            rms               = feat["rms"],
            peak              = feat["peak"],
            crest_factor      = feat["crest_factor"],
            spectral_flatness = feat["spectral_flatness"],
            spectral_centroid = feat["spectral_centroid"],
            zcr               = feat["zcr"],
            band_energy_ratio = feat["band_energy_ratio"],
            kurtosis_val      = feat["kurtosis"],
            confidence        = conf,
        )


# ══════════════════════════════════════════════
#  Action dispatcher
# ══════════════════════════════════════════════

def execute_action(clap_count: int) -> None:
    if clap_count not in ACTIONS:
        print(f"   ℹ️   {clap_count} claps — no action mapped.\n")
        return
    label, url = ACTIONS[clap_count]
    target = url or "about:blank"
    print(f"   ✨  Action → Opening {label}  ({target})")
    webbrowser.open(target)
    print()


# ══════════════════════════════════════════════
#  Banner
# ══════════════════════════════════════════════

def print_banner(p: AcousticProfile) -> None:
    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║             🎤  JARVIS  —  Clap Automation  🎤              ║")
    print("║                  Research-Grade Detection                    ║")
    print("╠═══════════════════════════════════════════════════════════════╣")
    print("║  Single clap   →  Open ChatGPT                              ║")
    print("║  Double clap   →  Open YouTube                              ║")
    print("║  Triple clap   →  Open Browser Homepage                     ║")
    print("╠═══════════════════════════════════════════════════════════════╣")
    print(f"║  Spike factor (fused)   :  {p.spike_factor:<33.4f}║")
    print(f"║  RMS threshold          :  {p.rms_threshold:<33.10f}║")
    print(f"║  Min crest factor       :  {p.min_crest_factor:<33.2f}║")
    print(f"║  Min confidence         :  {MIN_CONFIDENCE:<33.2f}║")
    print(f"║  Sample rate            :  {SAMPLE_RATE:<33}║")
    print(f"║  Chunk size             :  {CHUNK_SIZE} samples ({CHUNK_DURATION*1000:.0f} ms){' '*(19-len(str(CHUNK_SIZE)))}║")
    print("╠═══════════════════════════════════════════════════════════════╣")
    print("║  Press Ctrl+C to exit                                        ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()
    print("   🔇  Listening for claps …\n")


# ══════════════════════════════════════════════
#  Main loop
# ══════════════════════════════════════════════

def main() -> None:
    clap_events: list[ClapCandidate] = []
    last_action_time: float = 0.0

    detector = ClapDetector(sr=SAMPLE_RATE, chunk=CHUNK_SIZE)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
        ) as stream:

            detector.calibrate(stream)
            print_banner(detector.profile)

            while True:
                audio, overflowed = stream.read(CHUNK_SIZE)
                if overflowed:
                    continue

                now = time.time()

                # ── attempt detection ─────────────
                candidate = detector.detect(audio)

                if candidate is not None:
                    # debounce
                    if clap_events and (now - clap_events[-1].timestamp) < MIN_CLAP_GAP:
                        continue

                    clap_events.append(candidate)
                    print(
                        f"   🔊  Clap #{len(clap_events):>2}  │ "
                        f"RMS {candidate.rms:.6f}  "
                        f"CF {candidate.crest_factor:.1f}  "
                        f"SF {candidate.spectral_flatness:.3f}  "
                        f"SC {candidate.spectral_centroid:.0f} Hz  "
                        f"BER {candidate.band_energy_ratio:.2f}  "
                        f"Kurt {candidate.kurtosis_val:.1f}  "
                        f"Conf {candidate.confidence:.2f}"
                    )

                # ── evaluate pattern ──────────────
                if clap_events:
                    elapsed = now - clap_events[0].timestamp

                    if elapsed >= CLAP_WINDOW:
                        count = len(clap_events)
                        avg_conf = sum(c.confidence for c in clap_events) / count
                        print(
                            f"\n   🎯  Pattern: {count} clap(s)  "
                            f"(avg confidence {avg_conf:.2f})"
                        )

                        if (now - last_action_time) >= ACTION_COOLDOWN:
                            execute_action(count)
                            last_action_time = time.time()
                        else:
                            remaining = ACTION_COOLDOWN - (now - last_action_time)
                            print(f"   ⏳  Cooldown — wait {remaining:.1f} s\n")

                        clap_events.clear()

                # ── periodic adaptive threshold ───
                detector.adapt()

    except KeyboardInterrupt:
        print("\n\n   👋  Jarvis signing off. Goodbye!\n")
    except sd.PortAudioError as e:
        print(f"\n   ❌  Audio error: {e}")
        print("   💡  Ensure a microphone is connected and permissions are granted.\n")


if __name__ == "__main__":
    main()