"""
Bellows Test Data Analysis Toolkit
==================================

A Python/NumPy/SciPy engineering-analysis library for bellows QA test data
(hydrostatic pressure, cyclic fatigue, spring rate) — the numeric core behind
the MB Engineering Data Analyzer web app, reimplemented for engineers who
want a scriptable, MATLAB-style workflow instead of (or alongside) the
browser UI.

Every function here mirrors the equivalent calculation in the web app
(index.html) so results match between the two: same FFT convention, same
rainflow method, same Cp/Cpk formulas, same hold-phase / fatigue-life /
hysteresis check logic.

Dependencies: numpy, scipy, pandas (see requirements.txt)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy import optimize as sp_optimize
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Signal processing: FFT / spectrogram
# ---------------------------------------------------------------------------

def compute_fft(y: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Detrended magnitude spectrum of a signal.

    Parameters
    ----------
    y : samples (evenly spaced in x)
    dt : sample spacing (e.g. seconds between rows, or cycles between rows)

    Returns
    -------
    freqs, magnitude — one-sided spectrum (DC and Nyquist excluded)
    """
    y = np.asarray(y, dtype=float)
    y = y - np.mean(y)
    n = len(y)
    if n < 4:
        return np.array([]), np.array([])
    spectrum = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, d=dt)
    magnitude = np.abs(spectrum) * 2 / n
    # drop DC (index 0) and Nyquist (last) the same way the web app does
    return freqs[1:-1], magnitude[1:-1]


def compute_spectrogram(y: np.ndarray, fs: float, nperseg: int = 32):
    """Time-varying frequency content via scipy.signal.spectrogram."""
    y = np.asarray(y, dtype=float)
    nperseg = min(nperseg, max(4, len(y) // 4))
    f, t, Sxx = sp_signal.spectrogram(y - np.mean(y), fs=fs, nperseg=nperseg,
                                       noverlap=nperseg // 2)
    return f, t, Sxx


# ---------------------------------------------------------------------------
# Rainflow cycle counting (ASTM E1049-85, simplified 4-point algorithm)
# ---------------------------------------------------------------------------

def rainflow_count(y: np.ndarray) -> pd.DataFrame:
    """Rainflow-counts full and half cycles in a load/stress signal.

    Returns a DataFrame with columns: range, mean, count (1.0 = full cycle,
    0.5 = half cycle) — the standard rainflow output used for fatigue-life
    calculations instead of naive peak-to-peak counting.
    """
    y = np.asarray(y, dtype=float)
    # reduce to turning points (local extrema) first
    turning = [y[0]]
    for i in range(1, len(y) - 1):
        if (y[i] - y[i - 1]) * (y[i + 1] - y[i]) < 0:
            turning.append(y[i])
    turning.append(y[-1])

    points = list(turning)
    cycles = []
    stack: list[float] = []
    i = 0
    while i < len(points):
        stack.append(points[i])
        while len(stack) >= 3:
            x1, x2, x3 = stack[-3], stack[-2], stack[-1]
            r1, r2 = abs(x2 - x1), abs(x3 - x2)
            if r1 <= r2:
                if len(stack) == 3:
                    # half cycle at the start, pop the oldest and continue
                    cycles.append((r1, (x1 + x2) / 2, 0.5))
                    stack.pop(0)
                else:
                    cycles.append((r1, (x1 + x2) / 2, 1.0))
                    del stack[-3:-1]
            else:
                break
        i += 1
    # flush remaining as half cycles
    for j in range(len(stack) - 1):
        r = abs(stack[j + 1] - stack[j])
        cycles.append((r, (stack[j] + stack[j + 1]) / 2, 0.5))

    return pd.DataFrame(cycles, columns=["range", "mean", "count"])


# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------

def fit_linear(x: np.ndarray, y: np.ndarray) -> dict:
    coeffs = np.polyfit(x, y, 1)
    fn = np.poly1d(coeffs)
    return {"label": "linear", "coeffs": coeffs, "fn": fn, "r2": _r2(y, fn(x))}


def fit_poly2(x: np.ndarray, y: np.ndarray) -> dict:
    coeffs = np.polyfit(x, y, 2)
    fn = np.poly1d(coeffs)
    return {"label": "quadratic", "coeffs": coeffs, "fn": fn, "r2": _r2(y, fn(x))}


def fit_exponential(x: np.ndarray, y: np.ndarray) -> Optional[dict]:
    mask = y > 0
    if mask.sum() < 3:
        return None
    def model(x, a, b):
        return a * np.exp(b * x)
    try:
        popt, _ = sp_optimize.curve_fit(model, x[mask], y[mask], p0=(y[mask][0], -0.0001), maxfev=5000)
    except Exception:
        return None
    fn = lambda xv: model(xv, *popt)
    return {"label": "exponential", "a": popt[0], "b": popt[1], "fn": fn, "r2": _r2(y, fn(x))}


def _r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 1.0


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def cross_correlate(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Returns (best_lag, pearson_r) maximizing |correlation| within ±max_lag."""
    best_lag, best_r = 0, 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa, bb = a[-lag:], b[: len(b) + lag]
        elif lag > 0:
            aa, bb = a[: len(a) - lag], b[lag:]
        else:
            aa, bb = a, b
        if len(aa) < 3:
            continue
        r = np.corrcoef(aa, bb)[0, 1]
        if not np.isnan(r) and abs(r) > abs(best_r):
            best_lag, best_r = lag, r
    return best_lag, best_r


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(numeric_only=True)


# ---------------------------------------------------------------------------
# SPC / process capability
# ---------------------------------------------------------------------------

@dataclass
class SpcResult:
    mean: float
    std: float
    ucl: float
    lcl: float
    out_of_control: np.ndarray  # boolean mask
    runs: int  # count of 8+ point runs on one side of the mean


def spc_control_limits(y: np.ndarray) -> SpcResult:
    y = np.asarray(y, dtype=float)
    mean, std = float(np.mean(y)), float(np.std(y, ddof=1))
    ucl, lcl = mean + 3 * std, mean - 3 * std
    out = (y > ucl) | (y < lcl)

    # count runs of 8+ consecutive points on one side of the mean
    sign = np.sign(y - mean)
    runs = 0
    run_len = 1
    for i in range(1, len(sign)):
        if sign[i] == sign[i - 1] and sign[i] != 0:
            run_len += 1
        else:
            if run_len >= 8:
                runs += 1
            run_len = 1
    if run_len >= 8:
        runs += 1

    return SpcResult(mean=mean, std=std, ucl=ucl, lcl=lcl, out_of_control=out, runs=runs)


@dataclass
class CapabilityResult:
    cp: Optional[float]
    cpk: Optional[float]
    mean: float
    std: float


def process_capability(y: np.ndarray, lower: Optional[float], upper: Optional[float]) -> CapabilityResult:
    y = np.asarray(y, dtype=float)
    mean, std = float(np.mean(y)), float(np.std(y, ddof=1))
    if std == 0:
        return CapabilityResult(None, None, mean, std)
    cp = None
    if lower is not None and upper is not None:
        cp = (upper - lower) / (6 * std)
        cpk = min((upper - mean) / (3 * std), (mean - lower) / (3 * std))
    elif upper is not None:
        cpk = (upper - mean) / (3 * std)
    elif lower is not None:
        cpk = (mean - lower) / (3 * std)
    else:
        cpk = None
    return CapabilityResult(cp=cp, cpk=cpk, mean=mean, std=std)


# ---------------------------------------------------------------------------
# Bellows-specific QA checks (mirrors the web app's checks exactly)
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    label: str
    status: str  # 'pass' | 'warn' | 'fail'
    detail: str


def hold_phase_leak_check(x: np.ndarray, pressure: np.ndarray, hold_from: float,
                           hold_to: float, max_decay_pct: float = 2.0) -> CheckResult:
    mask = (x >= hold_from) & (x <= hold_to)
    hp, hx = pressure[mask], x[mask]
    if mask.sum() < 4:
        return CheckResult("Hold-phase leak check", "warn", "Not enough data in hold window.")
    order = np.argsort(hx)
    hp = hp[order]
    start_val, end_val = hp[0], hp[-1]
    decay_pct = (start_val - end_val) / start_val * 100 if start_val != 0 else 0
    if decay_pct <= max_decay_pct:
        status = "pass"
    elif decay_pct <= max_decay_pct * 1.5:
        status = "warn"
    else:
        status = "fail"
    return CheckResult(
        "Hold-phase leak check", status,
        f"Pressure held {start_val:.2f} -> {end_val:.2f} ({decay_pct:.2f}% decay, threshold {max_decay_pct}%)"
    )


def fatigue_life_projection(x: np.ndarray, y: np.ndarray, target_cycles: float,
                             threshold: float) -> CheckResult:
    fit = fit_linear(x, y)
    fn = fit["fn"]
    declining = fn(x.max()) < fn(x.min())
    xs = np.linspace(x.min(), x.max(), 400)
    ys = fn(xs)
    crossing = None
    for xv, yv in zip(xs, ys):
        if (declining and yv < threshold) or (not declining and yv > threshold):
            crossing = xv
            break
    if crossing is None:
        # extrapolate forward
        xmax = x.max()
        rng = x.max() - x.min() or 1
        for xv in np.arange(xmax, xmax + rng * 3, rng / 300):
            yv = fn(xv)
            if (declining and yv < threshold) or (not declining and yv > threshold):
                crossing = xv
                break
    if crossing is None:
        return CheckResult("Fatigue life projection", "pass",
                            "No degradation trend toward the threshold within a reasonable projection.")
    if crossing >= target_cycles:
        margin = crossing - target_cycles
        return CheckResult("Fatigue life projection", "pass",
                            f"Projected crossing at ~{crossing:,.0f} cycles "
                            f"({margin:,.0f} beyond the {target_cycles:,.0f}-cycle target)")
    shortfall = target_cycles - crossing
    return CheckResult("Fatigue life projection", "fail",
                        f"Projected crossing at ~{crossing:,.0f} cycles "
                        f"({shortfall:,.0f} short of the {target_cycles:,.0f}-cycle target)")


def spring_hysteresis_check(x: np.ndarray, load: np.ndarray, unload: np.ndarray,
                             r2_threshold: float = 0.98,
                             hysteresis_threshold: Optional[float] = None) -> CheckResult:
    fit = fit_linear(x, load)
    r2 = fit["r2"]
    work = float(np.trapezoid(load - unload, x))
    work = abs(work)

    lin_fail = r2 < r2_threshold
    hyst_fail = hysteresis_threshold is not None and work > hysteresis_threshold
    if not lin_fail and not hyst_fail:
        status = "pass"
    elif lin_fail and r2 >= r2_threshold - 0.02 and (not hyst_fail or work <= (hysteresis_threshold or 0) * 1.15):
        status = "warn"
    else:
        status = "fail" if (lin_fail or hyst_fail) else "pass"

    detail = f"Loading R^2={r2:.4f} (min {r2_threshold}); hysteresis work={work:.1f}"
    if hysteresis_threshold is not None:
        detail += f" (max {hysteresis_threshold})"
    return CheckResult("Spring linearity & hysteresis check", status, detail)


def squirm_stability_check(wall_mm: float, bore_mm: float, convolutions: int,
                            operating_pressure_bar: float) -> CheckResult:
    """Illustrative column-squirm margin — same simplified model as the web
    app's Design Simulator (not a certified EJMA calculation)."""
    C = 4793.0
    pcr_bar = C * wall_mm ** 1.5 / (bore_mm ** 0.5 * convolutions ** 1.2)
    margin = pcr_bar / operating_pressure_bar
    status = "pass" if margin >= 1.5 else ("warn" if margin >= 1.0 else "fail")
    return CheckResult(
        "Column squirm stability check", status,
        f"Estimated critical squirm pressure ~{pcr_bar:.1f} bar at {operating_pressure_bar:.1f} bar "
        f"operating pressure (margin {margin:.2f}x)"
    )


def safety_factor(worst_observed: float, allowable: float) -> CheckResult:
    if allowable == 0 or worst_observed == 0:
        return CheckResult("Safety factor", "warn", "Non-zero allowable/observed value required.")
    sf = abs(allowable / worst_observed)
    status = "pass" if sf >= 2 else ("warn" if sf >= 1 else "fail")
    return CheckResult("Safety factor", status,
                        f"Worst observed {worst_observed:.2f} vs allowable {allowable:.2f} -> {sf:.2f}x margin")
