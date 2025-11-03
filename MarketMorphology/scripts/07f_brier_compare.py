# scripts/07f_brier_compare.py
# Compare probabilistic signals using the Brier score.
# - Calibrates P(Y=1 | τ_c) on a rolling look-back window via decile bins (configurable).
# - Compares against two baselines: unconditional base rate and 1-step persistence.
# - Optionally saves a reliability (calibration) curve for the calibrated model.
#
# Example:
#   python -m scripts.07f_brier_compare \
#       --tau artifacts/tau_c_per_window_dense.csv \
#       --labels artifacts/labels_softdtw_K2.csv \
#       --h 1 --lb 78 --bins 10 \
#       --out artifacts/fig_5_11_brier.png \
#       --reliability-out artifacts/fig_5_11_reliability.png

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def brier(y, p):
    """Compute Brier score = mean((y - p)^2) over finite pairs."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.mean((y[m] - p[m]) ** 2)) if m.any() else np.nan


def find_label_column(df: pd.DataFrame) -> str:
    """
    Try to locate a binary label column.
    Priority: 'label' (case-insensitive); otherwise last numeric column.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    if "label" in cols_lower:
        return cols_lower["label"]
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise SystemExit("No numeric column found for labels.")
    return numeric_cols[-1]


def rolling_calibrated_probs(
    tau: np.ndarray,
    y: np.ndarray,
    h: int = 1,
    lb: int = 78,
    bins: int = 10,
    min_per_bin: int = 5,
) -> np.ndarray:
    """
    Produce calibrated probabilities P(Y=1 | τ_c in bin) at horizon h using a rolling window:
      - For each t ≥ lb, form a look-back window [t-lb, t)
      - Bin τ_c within that window into 'bins' quantile bins
      - Estimate event rate per bin using Y in the same window (no leak)
      - Map current τ_c[t] to its bin and assign that bin's historical event rate
      - Store at index t+h (predicting h steps ahead)
    Returns an array p_tau of length n with NaN where not computable.
    """
    n = len(y)
    p_tau = np.full(n, np.nan)
    tau = np.asarray(tau, float)
    y = np.asarray(y, float)

    if bins < 2:
        raise ValueError("--bins must be ≥ 2")

    for t in range(lb, n - h):
        past_tau = tau[t - lb : t]
        past_y = y[t - lb : t]

        # Quantile edges (unique, sorted)
        edges = np.nanquantile(past_tau, np.linspace(0, 1, bins + 1))
        edges = np.unique(edges)
        if len(edges) <= 2:
            # not enough dispersion to bin; skip this time
            continue

        # Digitize into bins-1 internal cut points (right=True => left-closed bins)
        # We keep 'bins' nominal target bins, but digitize uses len(edges)-1 actually.
        idx = np.digitize(past_tau, edges[1:-1], right=True)

        # Event rate per bin
        pr = np.full(len(edges) - 1, np.nan)
        for k in range(len(pr)):
            sel = (idx == k)
            if np.count_nonzero(sel) >= min_per_bin:
                pr[k] = np.nanmean(past_y[sel])

        # Map current τ_c[t] to its bin and assign P(Y=1)
        kt = np.digitize([tau[t]], edges[1:-1], right=True)[0]
        if 0 <= kt < len(pr) and np.isfinite(pr[kt]):
            p_tau[t + h] = pr[kt]

    return p_tau


def reliability_curve(y_true, p_pred, n_bins=10):
    """
    Compute reliability (calibration) curve points:
      - Bin predicted probabilities into quantile bins
      - For each bin: x = mean(predicted), y = mean(observed)
    """
    y_true = np.asarray(y_true, float)
    p_pred = np.asarray(p_pred, float)
    m = np.isfinite(y_true) & np.isfinite(p_pred)
    y_true = y_true[m]
    p_pred = p_pred[m]
    if len(y_true) < max(30, n_bins * 3):  # rough guardrail
        return np.array([]), np.array([])

    q = np.quantile(p_pred, np.linspace(0, 1, n_bins + 1))
    q = np.unique(q)
    if len(q) <= 2:
        return np.array([]), np.array([])

    idx = np.digitize(p_pred, q[1:-1], right=True)
    xs, ys = [], []
    for k in range(len(q) - 1):
        sel = (idx == k)
        if np.count_nonzero(sel) >= 5:
            xs.append(np.mean(p_pred[sel]))
            ys.append(np.mean(y_true[sel]))
    return np.array(xs, float), np.array(ys, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", default="artifacts/tau_c_per_window_dense.csv",
                    help="CSV with a 'tau_c' column.")
    ap.add_argument("--labels", default="artifacts/labels_softdtw_K2.csv",
                    help="CSV with a binary 'label' column (0/1).")
    ap.add_argument("--h", type=int, default=1, help="Forecast horizon (steps ahead).")
    ap.add_argument("--lb", type=int, default=78, help="Look-back window for on-the-fly calibration.")
    ap.add_argument("--bins", type=int, default=10, help="Number of quantile bins for calibration.")
    ap.add_argument("--min-per-bin", type=int, default=5, help="Minimum samples per past bin to estimate a rate.")
    ap.add_argument("--out", default="artifacts/fig_5_11_brier.png",
                    help="Bar chart output path.")
    ap.add_argument("--reliability-out", default="",
                    help="Optional: path to save a reliability (calibration) curve for τc model.")
    args = ap.parse_args()

    # --- Load
    T = pd.read_csv(args.tau)
    if "tau_c" not in T.columns:
        # try to guess a tau column
        tau_col = next((c for c in T.columns if str(c).lower().startswith("tau")), None)
        if not tau_col:
            raise SystemExit("No 'tau_c' column found in tau file.")
        T = T.rename(columns={tau_col: "tau_c"})

    L = pd.read_csv(args.labels)
    lab_col = find_label_column(L)
    y = pd.to_numeric(L[lab_col], errors="coerce").astype(float).to_numpy()
    if set(np.unique(y[~np.isnan(y)])).difference({0.0, 1.0}):
        raise SystemExit(f"Label column '{lab_col}' must be binary (0/1).")

    tau = pd.to_numeric(T["tau_c"], errors="coerce").to_numpy()

    n = min(len(y), len(tau))
    if len(y) != len(tau):
        print(f"⚠️  Length mismatch: labels={len(y)}, tau={len(tau)} → truncating to n={n}")
    y = y[:n]
    tau = tau[:n]

    # --- Calibrated probabilities from τc
    p_tau = rolling_calibrated_probs(
        tau=tau, y=y, h=args.h, lb=args.lb, bins=args.bins, min_per_bin=args.min_per_bin
    )

    # --- Baselines
    # Unconditional base rate estimated on the first lb observations (climatology)
    if args.lb >= n:
        raise SystemExit("--lb must be smaller than the sample length.")
    p_uncond = np.nanmean(y[:args.lb]) * np.ones(n)
    # 1-step persistence (copy last observed label)
    p_persist = np.r_[np.nan, y[:-1]]

    # --- Brier scores (only where p_tau is defined)
    mask = np.isfinite(p_tau) & np.isfinite(y)
    if not mask.any():
        raise SystemExit("No valid evaluation region (p_tau is all NaN). Increase data or relax settings.")

    b_tau = brier(y[mask], p_tau[mask])
    b_un = brier(y[mask], p_uncond[mask])
    b_per = brier(y[mask], p_persist[mask])

    # --- Plot Brier comparison
    vals = [b_tau, b_un, b_per]
    names = ["τc (calibrated)", "unconditional", "persistence"]

    plt.figure(figsize=(5.4, 4.2))
    plt.bar(range(3), vals)
    plt.xticks(range(3), names, rotation=10)
    plt.ylabel("Brier score (lower is better)")
    plt.title(f"Brier comparison (h={args.h}, look-back={args.lb}, bins={args.bins})")
    for i, v in enumerate(vals):
        if np.isfinite(v):
            plt.text(i, v + 0.001, f"{v:.3f}", ha="center", va="bottom")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    print("Saved:", args.out)

    # --- Optional reliability diagram for τc model
    if args.reliability_out:
        x, y_obs = reliability_curve(y[mask], p_tau[mask], n_bins=max(6, args.bins))
        if len(x):
            plt.figure(figsize=(4.6, 4.2))
            plt.plot([0, 1], [0, 1], "--", linewidth=1, label="perfect")
            plt.plot(x, y_obs, "o-", label="τc (calibrated)")
            plt.xlabel("Predicted probability")
            plt.ylabel("Observed event rate")
            plt.title("Reliability curve")
            plt.legend(frameon=False)
            Path(args.reliability_out).parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(args.reliability_out, dpi=180)
            print("Saved:", args.reliability_out)
        else:
            print("Not enough variability for a reliability curve (skipped).")


if __name__ == "__main__":
    main()
