"""
Test suite for bellows_analysis.py

Each test validates a function against synthetic data with a known,
hand-checkable answer (a pure sine wave for FFT, an exact line for curve
fitting, a manually-shifted signal for cross-correlation, etc.) rather than
just checking "it doesn't crash" — so a regression in the math itself gets
caught, not just a crash.

Run with:  pytest test_bellows_analysis.py -v
"""

import numpy as np
import pandas as pd
import pytest

import bellows_analysis as ba


# ---------------------------------------------------------------------------
# FFT / spectrogram
# ---------------------------------------------------------------------------

def test_fft_recovers_known_frequency():
    fs = 100.0  # samples/sec
    dt = 1 / fs
    t = np.arange(0, 10, dt)
    true_freq = 5.0
    y = 3.0 * np.sin(2 * np.pi * true_freq * t)

    freqs, mag = ba.compute_fft(y, dt)
    peak_freq = freqs[np.argmax(mag)]

    assert peak_freq == pytest.approx(true_freq, abs=0.05)
    # amplitude of a sine is recovered (within FFT/windowing tolerance)
    assert mag.max() == pytest.approx(3.0, rel=0.05)


def test_fft_too_short_returns_empty():
    freqs, mag = ba.compute_fft(np.array([1.0, 2.0]), dt=1.0)
    assert len(freqs) == 0 and len(mag) == 0


def test_spectrogram_shapes_are_consistent():
    fs = 50.0
    t = np.arange(0, 4, 1 / fs)
    y = np.sin(2 * np.pi * 5 * t)
    f, tt, Sxx = ba.compute_spectrogram(y, fs=fs, nperseg=16)
    assert Sxx.shape == (len(f), len(tt))
    assert Sxx.min() >= 0  # power spectral density is non-negative


# ---------------------------------------------------------------------------
# Rainflow
# ---------------------------------------------------------------------------

def test_rainflow_on_pure_sine_finds_one_cycle_per_period():
    # 10 full periods, amplitude 5 (peak-to-peak range 10)
    t = np.linspace(0, 10 * 2 * np.pi, 2000)
    y = 5 * np.sin(t)
    rf = ba.rainflow_count(y)

    total_cycles = rf["count"].sum()
    assert total_cycles == pytest.approx(10, abs=1.5)

    # every extracted range should be close to peak-to-peak = 10
    dominant = rf[rf["count"] >= 0.5]
    assert dominant["range"].median() == pytest.approx(10, rel=0.05)


def test_rainflow_conserves_reversal_count():
    # every half-cycle consumes exactly one reversal interval; every full
    # cycle consumes two — so 2*sum(count) must equal (turning points - 1)
    y = np.array([0, 3, -2, 5, -1, 4, -3, 2, 0], dtype=float)
    turning = [y[0]]
    for i in range(1, len(y) - 1):
        if (y[i] - y[i - 1]) * (y[i + 1] - y[i]) < 0:
            turning.append(y[i])
    turning.append(y[-1])

    rf = ba.rainflow_count(y)
    assert 2 * rf["count"].sum() == pytest.approx(len(turning) - 1)


def test_rainflow_flat_signal_has_no_meaningful_cycles():
    # a flat signal has no local extrema, so the only possible output is a
    # residual zero-range half-cycle (no actual load reversal happened)
    y = np.full(20, 5.0)
    rf = ba.rainflow_count(y)
    assert (rf["range"] == 0).all()


# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------

def test_fit_linear_recovers_exact_line():
    x = np.linspace(0, 10, 50)
    y = 2.0 * x + 3.0
    fit = ba.fit_linear(x, y)
    assert fit["coeffs"][0] == pytest.approx(2.0, abs=1e-9)
    assert fit["coeffs"][1] == pytest.approx(3.0, abs=1e-9)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)


def test_fit_poly2_recovers_exact_quadratic():
    x = np.linspace(-5, 5, 50)
    y = 2 * x**2 - 3 * x + 1
    fit = ba.fit_poly2(x, y)
    np.testing.assert_allclose(fit["coeffs"], [2, -3, 1], atol=1e-8)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)


def test_fit_exponential_recovers_known_decay():
    x = np.linspace(0, 50, 100)
    a_true, b_true = 5.0, -0.1
    y = a_true * np.exp(b_true * x)
    fit = ba.fit_exponential(x, y)
    assert fit is not None
    assert fit["a"] == pytest.approx(a_true, rel=0.01)
    assert fit["b"] == pytest.approx(b_true, rel=0.01)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-6)


def test_fit_exponential_returns_none_for_nonpositive_data():
    x = np.linspace(0, 10, 20)
    y = np.full(20, -1.0)  # never positive -> can't fit exponential
    assert ba.fit_exponential(x, y) is None


def test_r2_is_perfect_for_exact_fit_and_zero_for_mean_only():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert ba._r2(y, y) == pytest.approx(1.0)
    assert ba._r2(y, np.full_like(y, y.mean())) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def test_cross_correlate_recovers_known_shift():
    rng = np.random.default_rng(42)
    a = rng.normal(size=200)
    shift = 7
    b = np.roll(a, shift)  # b lags a by `shift` samples... a[i] == b[i+shift]

    lag, r = ba.cross_correlate(a, b, max_lag=20)
    assert abs(r) > 0.9
    assert lag in (shift, -shift)  # sign convention aside, magnitude must match


def test_correlation_matrix_diagonal_and_signs():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "double": [2, 4, 6, 8, 10],
        "negated": [-1, -2, -3, -4, -5],
    })
    corr = ba.correlation_matrix(df)
    assert np.allclose(np.diag(corr), 1.0)
    assert corr.loc["x", "double"] == pytest.approx(1.0)
    assert corr.loc["x", "negated"] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# SPC / process capability
# ---------------------------------------------------------------------------

def test_spc_control_limits_formula_and_outlier_detection():
    y = np.array([10, 11, 9, 10, 12, 8, 10, 11, 9, 10, 50], dtype=float)  # 50 is an outlier
    spc = ba.spc_control_limits(y)
    assert spc.mean == pytest.approx(y.mean())
    assert spc.ucl == pytest.approx(spc.mean + 3 * spc.std)
    assert spc.lcl == pytest.approx(spc.mean - 3 * spc.std)
    assert spc.out_of_control[-1] == True  # the 50 must be flagged
    assert spc.out_of_control[:-1].sum() == 0


def test_spc_detects_a_run_of_eight():
    # 8 points above the mean in a row, then some below, forces a clear run
    y = np.array([10]*8 + [0]*8, dtype=float)
    spc = ba.spc_control_limits(y)
    assert spc.runs >= 1


def test_process_capability_matches_manual_formula():
    y = np.array([9, 10, 11, 10, 9, 11, 10, 10, 9, 11], dtype=float)
    lower, upper = 5.0, 15.0
    cap = ba.process_capability(y, lower, upper)
    std = y.std(ddof=1)
    mean = y.mean()
    assert cap.cp == pytest.approx((upper - lower) / (6 * std))
    assert cap.cpk == pytest.approx(min((upper - mean) / (3 * std), (mean - lower) / (3 * std)))


def test_process_capability_handles_zero_std():
    y = np.full(10, 5.0)
    cap = ba.process_capability(y, 0, 10)
    assert cap.cp is None and cap.cpk is None


def test_process_capability_one_sided_limits():
    y = np.array([9, 10, 11, 10, 9, 11, 10], dtype=float)
    cap_upper = ba.process_capability(y, None, 15.0)
    cap_lower = ba.process_capability(y, 5.0, None)
    assert cap_upper.cp is None and cap_upper.cpk is not None
    assert cap_lower.cp is None and cap_lower.cpk is not None


# ---------------------------------------------------------------------------
# Bellows-specific checks
# ---------------------------------------------------------------------------

def test_hold_phase_leak_check_pass_and_fail():
    x = np.linspace(0, 100, 101)
    # steady pressure during hold (30-90) -> should pass
    steady = np.where((x >= 30) & (x <= 90), 25.0, 10.0)
    ok = ba.hold_phase_leak_check(x, steady, hold_from=30, hold_to=90, max_decay_pct=2.0)
    assert ok.status == "pass"

    # decaying pressure during hold -> should fail
    hold_mask = (x >= 30) & (x <= 90)
    decaying = steady.copy()
    decay_curve = np.linspace(25.0, 20.0, hold_mask.sum())  # 20% decay
    decaying[hold_mask] = decay_curve
    leaking = ba.hold_phase_leak_check(x, decaying, hold_from=30, hold_to=90, max_decay_pct=2.0)
    assert leaking.status == "fail"
    assert "20.00% decay" in leaking.detail


def test_fatigue_life_projection_pass_and_fail():
    # y declines linearly from 142 at x=0 to 100 at x=10000
    x = np.linspace(0, 10000, 50)
    y = 142 - (42 / 10000) * x

    # generous target well before the threshold crossing -> pass
    easy = ba.fatigue_life_projection(x, y, target_cycles=1000, threshold=105)
    assert easy.status == "pass"

    # target far beyond where the line actually crosses the threshold -> fail
    hard = ba.fatigue_life_projection(x, y, target_cycles=50000, threshold=105)
    assert hard.status == "fail"


def test_fatigue_life_projection_crossing_matches_algebra():
    # y = 142 - 0.0042*x crosses y=105 at x = (142-105)/0.0042
    slope = -0.0042
    intercept = 142.0
    threshold = 105.0
    expected_crossing = (threshold - intercept) / slope

    x = np.linspace(0, 12000, 60)
    y = intercept + slope * x
    res = ba.fatigue_life_projection(x, y, target_cycles=expected_crossing - 500, threshold=threshold)
    assert res.status == "pass"
    # pull the numeric crossing back out of the detail string
    import re
    m = re.search(r"~([\d,]+) cycles", res.detail)
    assert m is not None
    got_crossing = float(m.group(1).replace(",", ""))
    assert got_crossing == pytest.approx(expected_crossing, rel=0.02)


def test_spring_hysteresis_check_linear_no_hysteresis_passes():
    x = np.linspace(0, 10, 50)
    load = 20 * x
    unload = 20 * x  # identical -> zero hysteresis, perfectly linear
    res = ba.spring_hysteresis_check(x, load, unload, r2_threshold=0.98, hysteresis_threshold=10)
    assert res.status == "pass"
    assert "hysteresis work=0.0" in res.detail


def test_spring_hysteresis_check_known_rectangular_gap():
    # constant gap `g` between load and unload over range `L` -> hysteresis
    # work (area between the curves) = g * L exactly
    x = np.linspace(0, 10, 200)
    g, L = 5.0, 10.0
    load = 20 * x
    unload = 20 * x - g
    res = ba.spring_hysteresis_check(x, load, unload, r2_threshold=0.98, hysteresis_threshold=1000)
    import re
    m = re.search(r"hysteresis work=([\d.]+)", res.detail)
    assert float(m.group(1)) == pytest.approx(g * L, rel=0.01)


def test_spring_hysteresis_check_nonlinear_fails():
    x = np.linspace(0, 10, 50)
    load = 3 * x**2  # strongly nonlinear -> low R^2 against a linear fit
    unload = load.copy()
    res = ba.spring_hysteresis_check(x, load, unload, r2_threshold=0.98)
    assert res.status == "fail"


def test_squirm_stability_check_more_convolutions_increases_risk():
    low_conv = ba.squirm_stability_check(wall_mm=1.2, bore_mm=300, convolutions=6, operating_pressure_bar=25)
    high_conv = ba.squirm_stability_check(wall_mm=1.2, bore_mm=300, convolutions=10, operating_pressure_bar=25)
    assert "~" in low_conv.detail and "~" in high_conv.detail
    # more convolutions -> lower critical pressure -> worse (or equal-or-worse) status ranking
    rank = {"pass": 2, "warn": 1, "fail": 0}
    assert rank[high_conv.status] <= rank[low_conv.status]


def test_squirm_stability_check_thicker_wall_reduces_risk():
    thin = ba.squirm_stability_check(wall_mm=1.2, bore_mm=300, convolutions=6, operating_pressure_bar=25)
    thick = ba.squirm_stability_check(wall_mm=2.0, bore_mm=300, convolutions=6, operating_pressure_bar=25)
    rank = {"pass": 2, "warn": 1, "fail": 0}
    assert rank[thick.status] >= rank[thin.status]


def test_safety_factor_boundaries():
    assert ba.safety_factor(worst_observed=10, allowable=30).status == "pass"   # 3x
    assert ba.safety_factor(worst_observed=10, allowable=15).status == "warn"   # 1.5x
    assert ba.safety_factor(worst_observed=10, allowable=5).status == "fail"    # 0.5x


def test_safety_factor_handles_zero_inputs():
    res = ba.safety_factor(worst_observed=0, allowable=10)
    assert res.status == "warn"
