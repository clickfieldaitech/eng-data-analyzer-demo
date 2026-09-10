#!/usr/bin/env python3
"""
CLI for the Bellows Test Data Analysis Toolkit.

Usage
-----
    python analyze.py sample-data/Bellows_Hydro_Test_04.csv
    python analyze.py sample-data/Bellows_Fatigue_Test_11.csv --target-cycles 10000 --threshold 125
    python analyze.py sample-data/Bellows_SpringRate_B2026-11.csv --plot

Auto-detects the test type (hydro / fatigue / spring) from column names,
the same way the web app's CSV-upload heuristics do, so the same file
that works in the browser works here.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import bellows_analysis as ba


def _strip_unit(label: str) -> str:
    """Drop a trailing '(unit)' suffix, e.g. 'Force Loading (N)' -> 'Force Loading'."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", label).strip()


def detect_columns(df: pd.DataFrame):
    x_key = df.columns[0]
    pressure_col = next((c for c in df.columns if re.search(r"pressure", c, re.I)
                          and pd.api.types.is_numeric_dtype(df[c])), None)
    load_col = next((c for c in df.columns if re.search(r"load(ing)?$", _strip_unit(c), re.I)
                      and pd.api.types.is_numeric_dtype(df[c])), None)
    unload_col = next((c for c in df.columns if re.search(r"unload(ing)?$", _strip_unit(c), re.I)
                        and pd.api.types.is_numeric_dtype(df[c])), None)
    is_time_like = bool(re.search(r"time|second|min(ute)?", x_key, re.I))
    is_cycle_like = bool(re.search(r"cycle", x_key, re.I))
    return {
        "x_key": x_key,
        "pressure_col": pressure_col,
        "load_col": load_col,
        "unload_col": unload_col,
        "is_time_like": is_time_like,
        "is_cycle_like": is_cycle_like,
    }


def pick_primary_y(df: pd.DataFrame, x_key: str) -> str:
    numeric_cols = [c for c in df.columns if c != x_key and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise SystemExit("No numeric Y columns found in this file.")
    # prefer a stress/amplitude column (the meaningful fatigue variable) when present
    preferred = next((c for c in numeric_cols if re.search(r"stress|amplitude", c, re.I)), None)
    return preferred or numeric_cols[0]


def section(title: str):
    print()
    print(title)
    print("-" * len(title))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv_file", type=Path)
    p.add_argument("--x", help="X-axis column name (default: first column)")
    p.add_argument("--y", help="Primary Y column name (default: first numeric non-X column)")
    p.add_argument("--spec-lower", type=float, default=None)
    p.add_argument("--spec-upper", type=float, default=None)
    p.add_argument("--allowable", type=float, default=None, help="Safety-factor allowable limit")
    p.add_argument("--target-cycles", type=float, default=None, help="Target fatigue life")
    p.add_argument("--threshold", type=float, default=None, help="Fatigue end-of-life threshold")
    p.add_argument("--hold-from", type=float, default=None)
    p.add_argument("--hold-to", type=float, default=None)
    p.add_argument("--max-decay-pct", type=float, default=2.0)
    p.add_argument("--plot", action="store_true", help="Save a PNG chart + FFT alongside the CSV")
    args = p.parse_args()

    if not args.csv_file.exists():
        raise SystemExit(f"File not found: {args.csv_file}")
    df = pd.read_csv(args.csv_file)

    det = detect_columns(df)
    x_key = args.x or det["x_key"]
    y_key = args.y or pick_primary_y(df, x_key)
    x = df[x_key].to_numpy(dtype=float)
    y = df[y_key].to_numpy(dtype=float)
    dt = float(np.median(np.diff(np.sort(x)))) or 1.0

    print(f"=== {args.csv_file.name} ===")
    print(f"Rows: {len(df)}   Columns: {list(df.columns)}")
    print(f"X = {x_key}   Primary Y = {y_key}")

    section("Statistics")
    print(f"max={y.max():.3f}  min={y.min():.3f}  mean={y.mean():.3f}  "
          f"median={np.median(y):.3f}  std={y.std(ddof=1):.3f}")
    print(f"5th pct={np.percentile(y,5):.3f}  95th pct={np.percentile(y,95):.3f}  99th pct={np.percentile(y,99):.3f}")

    section("FFT (detrended magnitude spectrum)")
    freqs, mag = ba.compute_fft(y, dt)
    if len(freqs):
        peak_idx = int(np.argmax(mag))
        print(f"{len(freqs)} frequency bins; dominant frequency = {freqs[peak_idx]:.4g} "
              f"(magnitude {mag[peak_idx]:.4g})")
    else:
        print("Not enough samples for a meaningful FFT.")

    section("Rainflow cycle counting")
    rf = ba.rainflow_count(y)
    if len(rf):
        total_cycles = rf["count"].sum()
        top = rf.sort_values("range", ascending=False).head(3)
        print(f"{total_cycles:.1f} equivalent full cycles from {len(y)} readings")
        print("Largest ranges:")
        for _, row in top.iterrows():
            print(f"  range={row['range']:.3f}  mean={row['mean']:.3f}  count={row['count']}")
    else:
        print("Not enough turning points to count cycles.")

    section("SPC control chart (individuals, mean +/- 3 sigma)")
    spc = ba.spc_control_limits(y)
    print(f"mean={spc.mean:.3f}  UCL={spc.ucl:.3f}  LCL={spc.lcl:.3f}  "
          f"out-of-control points={int(spc.out_of_control.sum())}  8+ point runs={spc.runs}")

    if args.spec_lower is not None or args.spec_upper is not None:
        section("Spec compliance")
        violations = ((y < args.spec_lower) if args.spec_lower is not None else np.zeros_like(y, dtype=bool)) | \
                     ((y > args.spec_upper) if args.spec_upper is not None else np.zeros_like(y, dtype=bool))
        status = "PASS" if violations.sum() == 0 else "FAIL"
        print(f"{status} - {violations.sum()} of {len(y)} points out of range "
              f"[{args.spec_lower}, {args.spec_upper}]")

        section("Process capability (Cp/Cpk)")
        cap = ba.process_capability(y, args.spec_lower, args.spec_upper)
        print(f"Cp={cap.cp}  Cpk={cap.cpk}  mean={cap.mean:.3f}  std={cap.std:.3f}")

    if args.allowable is not None:
        section("Safety factor")
        worst = y[np.argmax(np.abs(y))]
        res = ba.safety_factor(worst, args.allowable)
        print(f"[{res.status.upper()}] {res.detail}")

    if det["pressure_col"] and det["is_time_like"] and args.hold_from is not None and args.hold_to is not None:
        section("Hold-phase leak check")
        pressure = df[det["pressure_col"]].to_numpy(dtype=float)
        res = ba.hold_phase_leak_check(x, pressure, args.hold_from, args.hold_to, args.max_decay_pct)
        print(f"[{res.status.upper()}] {res.detail}")

    if det["is_cycle_like"] and args.target_cycles is not None and args.threshold is not None:
        section("Fatigue life projection")
        res = ba.fatigue_life_projection(x, y, args.target_cycles, args.threshold)
        print(f"[{res.status.upper()}] {res.detail}")

    if det["load_col"] and det["unload_col"]:
        section("Spring linearity & hysteresis check")
        load = df[det["load_col"]].to_numpy(dtype=float)
        unload = df[det["unload_col"]].to_numpy(dtype=float)
        res = ba.spring_hysteresis_check(x, load, unload)
        print(f"[{res.status.upper()}] {res.detail}")

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 2:
        section("Correlation matrix")
        print(ba.correlation_matrix(numeric_df).round(2).to_string())

    if args.plot:
        save_plots(args.csv_file, x, y, x_key, y_key, freqs, mag, spc)


def save_plots(csv_path: Path, x, y, x_key, y_key, freqs, mag, spc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = csv_path.with_suffix("").with_name(csv_path.stem + "_analysis.png")
    fig, axes = plt.subplots(3, 1, figsize=(9, 10))

    axes[0].plot(x, y, color="#2a78d6")
    axes[0].set_title(f"{y_key} vs {x_key}")
    axes[0].set_xlabel(x_key)
    axes[0].set_ylabel(y_key)
    axes[0].grid(alpha=0.3)

    if len(freqs):
        axes[1].plot(freqs, mag, color="#eb6834")
    axes[1].set_title("FFT magnitude spectrum")
    axes[1].set_xlabel("frequency")
    axes[1].grid(alpha=0.3)

    axes[2].plot(x, y, color="#1baf7a", marker=".", linewidth=1)
    axes[2].axhline(spc.mean, color="gray", linestyle="--", linewidth=1, label="mean")
    axes[2].axhline(spc.ucl, color="#d03b3b", linestyle="--", linewidth=1, label="UCL/LCL")
    axes[2].axhline(spc.lcl, color="#d03b3b", linestyle="--", linewidth=1)
    axes[2].set_title("SPC control chart")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved plots to {out_path}")


if __name__ == "__main__":
    main()
