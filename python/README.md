# Bellows Test Data Analysis Toolkit (Python)

A NumPy/SciPy/pandas engineering-analysis library and CLI for bellows QA test
data — hydrostatic pressure, cyclic fatigue, and spring rate — for engineers
who want a scriptable, MATLAB-style workflow instead of (or alongside) the
web app.

Every function in `bellows_analysis.py` mirrors the equivalent calculation in
the web app (`../index.html`) so results match between the two: same FFT
convention, same rainflow method, same Cp/Cpk formulas, same hold-phase /
fatigue-life / hysteresis check logic.

## Install

```bash
pip install -r requirements.txt
```

## Use as a CLI

```bash
# Hydro test — spec limits, safety factor, hold-phase leak check
python analyze.py ../sample-data/Bellows_Hydro_Test_04.csv \
  --spec-lower 0 --spec-upper 25 --allowable 30 --hold-from 30 --hold-to 90 --plot

# Fatigue test — projected life vs a target
python analyze.py ../sample-data/Bellows_Fatigue_Test_11.csv \
  --target-cycles 10000 --threshold 125

# Spring rate test — linearity + hysteresis
python analyze.py ../sample-data/Bellows_SpringRate_B2026-11.csv --plot
```

`--plot` saves a `<file>_analysis.png` next to the CSV with the primary
chart, FFT spectrum, and SPC control chart.

## Use as a library

```python
import pandas as pd
import bellows_analysis as ba

df = pd.read_csv("test.csv")
x, y = df["Time (s)"].to_numpy(), df["Pressure (bar)"].to_numpy()

freqs, mag = ba.compute_fft(y, dt=1.0)
spc = ba.spc_control_limits(y)
cap = ba.process_capability(y, lower=0, upper=25)
check = ba.hold_phase_leak_check(x, y, hold_from=30, hold_to=90)
print(check.status, check.detail)
```

## What's here

| Function | Purpose |
|---|---|
| `compute_fft` | Detrended magnitude spectrum (NumPy FFT) |
| `compute_spectrogram` | Time-varying frequency content (SciPy) |
| `rainflow_count` | ASTM E1049-style rainflow cycle counting |
| `fit_linear` / `fit_poly2` / `fit_exponential` | Curve fitting (NumPy/SciPy) |
| `cross_correlate` | Lag-aware correlation between two signals |
| `correlation_matrix` | Pearson correlation across all numeric columns (pandas) |
| `spc_control_limits` | Mean ± 3σ control limits + run detection |
| `process_capability` | Cp / Cpk |
| `hold_phase_leak_check` | Hydro hold-phase pressure decay check |
| `fatigue_life_projection` | Linear-trend cycles-to-failure vs. target |
| `spring_hysteresis_check` | Loading-curve linearity + hysteresis work |
| `squirm_stability_check` | Illustrative column-squirm margin |
| `safety_factor` | Worst observed vs. allowable limit |

Everything here is illustrative/engineering-estimate math, same as the web
app — not a certified EJMA/ASME calculation.

## Tests

```bash
pytest test_bellows_analysis.py -v
```

Every function is checked against synthetic data with a known, hand-checkable
answer rather than just "did it crash" — a sine wave for FFT frequency
recovery, an exact line/quadratic/exponential for curve fitting, a
known-shift signal for cross-correlation, a manually-solved crossing point
for fatigue life projection, a known rectangular gap for hysteresis work,
and formula cross-checks for SPC/Cp/Cpk. 28 tests, all passing.
