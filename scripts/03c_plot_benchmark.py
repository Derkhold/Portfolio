# --- python -m scripts.03c_plot_benchmark ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ART = Path("artifacts")

def load_csv(name):
    p = ART / name
    if not p.exists():
        raise SystemExit(f"Fichier introuvable: {p}. Lance d'abord scripts/03c_compare_metrics.")
    return pd.read_csv(p)

# ---------- 1) Bar chart: silhouette par métrique ----------
df = load_csv("metric_benchmark.csv").sort_values("silhouette", ascending=False)

plt.figure(figsize=(6, 3.4))
plt.bar(df["metric"], df["silhouette"])
plt.ylabel("Silhouette (↑ meilleur)")
plt.title("Comparatif de métriques (spectral clustering)")
for i, v in enumerate(df["silhouette"]):
    plt.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
plt.ylim(0, max(0.55, df["silhouette"].max() + 0.08))
plt.tight_layout()
plt.savefig(ART / "metric_benchmark_bar.png", dpi=180)
print("📈 saved:", ART / "metric_benchmark_bar.png")

# ---------- 2) Courbes silhouette vs K pour chaque métrique ----------
dfd = load_csv("metric_benchmark_detail.csv")

# ordre de tri des métriques (même que le bar chart)
order = df["metric"].tolist()
colors = {m: c for m, c in zip(order, plt.rcParams['axes.prop_cycle'].by_key()['color'])}

plt.figure(figsize=(7.5, 4.2))
for m in order:
    sub = dfd[dfd["metric"] == m].sort_values("k")
    if sub.empty:
        continue
    plt.plot(sub["k"], sub["silhouette"], marker="o", label=m, linewidth=1.8)
plt.xlabel("Number of clusters K")
plt.ylabel("Silhouette")
plt.title("Silhouette en fonction de K (par métrique)")
plt.legend(title="Metrics", ncol=min(len(order), 3))
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(ART / "metric_benchmark_sil_vs_k.png", dpi=180)
print("📈 saved:", ART / "metric_benchmark_sil_vs_k.png")

# ---------- 3) Tableau LaTeX (optionnel) ----------
tex_lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Metrics & $K^\star$ & Silhouette \\",
             r"\midrule"]
for _, r in df.iterrows():
    tex_lines.append(f"{r['metric']} & {int(r['best_k'])} & {r['silhouette']:.3f} \\\\")
tex_lines += [r"\bottomrule", r"\end{tabular}"]
(Path(ART / "metric_benchmark_table.tex")).write_text("\n".join(tex_lines), encoding="utf-8")
print("📝 LaTeX table:", ART / "metric_benchmark_table.tex")
