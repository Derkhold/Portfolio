# scripts/03f_plot_benchmark.py
# Plot metric benchmarking results (bar chart, silhouette vs K, LaTeX table)

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ART = Path("artifacts")


# --- Helpers -----------------------------------------------------------------

def load_csv(name: str) -> pd.DataFrame:
    p = ART / name
    if not p.exists():
        raise SystemExit(f"File not found: {p}. Run scripts/03c_compare_metrics.py first.")
    return pd.read_csv(p)


# --- 1) Bar chart: silhouette by metric -------------------------------------

df = load_csv("metric_benchmark.csv").sort_values("silhouette", ascending=False)

plt.figure(figsize=(6, 3.4))
plt.bar(df["metric"], df["silhouette"])
plt.ylabel("Silhouette (higher is better)")
plt.title("Metric comparison (spectral clustering)")
for i, v in enumerate(df["silhouette"]):
    plt.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
plt.ylim(0, max(0.55, df["silhouette"].max() + 0.08))
plt.tight_layout()
out1 = ART / "metric_benchmark_bar.png"
plt.savefig(out1, dpi=180)
plt.close()
print("Saved:", out1)


# --- 2) Curves: silhouette vs K for each metric -----------------------------

dfd = load_csv("metric_benchmark_detail.csv")

order = df["metric"].tolist()
plt.figure(figsize=(7.5, 4.2))
for m in order:
    sub = dfd[dfd["metric"] == m].sort_values("k")
    if sub.empty:
        continue
    plt.plot(sub["k"], sub["silhouette"], marker="o", label=m, linewidth=1.8)
plt.xlabel("Number of clusters K")
plt.ylabel("Silhouette")
plt.title("Silhouette as a function of K (by metric)")
plt.legend(title="Metrics", ncol=min(len(order), 3))
plt.grid(True, alpha=0.25)
plt.tight_layout()
out2 = ART / "metric_benchmark_sil_vs_k.png"
plt.savefig(out2, dpi=180)
plt.close()
print("Saved:", out2)


# --- 3) LaTeX table (optional) ----------------------------------------------

tex_lines = [
    r"\begin{tabular}{lrr}",
    r"\toprule",
    r"Metrics & $K^\star$ & Silhouette \\",
    r"\midrule",
]
for _, r in df.iterrows():
    tex_lines.append(f"{r['metric']} & {int(r['best_k'])} & {r['silhouette']:.3f} \\\\")
tex_lines += [r"\bottomrule", r"\end{tabular}"]

out3 = ART / "metric_benchmark_table.tex"
out3.write_text("\n".join(tex_lines), encoding="utf-8")
print("Wrote LaTeX table:", out3)
