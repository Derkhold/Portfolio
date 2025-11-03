# Morphology and Percolation in Intraday Markets

## Elastic Shape Similarity • Graph Cohesion • Early-Warning Diagnostics

This project explores the use of morphological analysis and graph percolation to identify structural regimes in intraday financial markets and assess their predictive power for liquidity stress.

Rather than focusing on price forecasting, the goal is to study how the shape of price trajectories encodes information about market microstructure, cohesion, and stress propagation.
The entire analysis is implemented from scratch, using a modular pipeline of Python scripts designed for reproducibility, interpretability, and quantitative rigor.


## Project Overview
The project integrates three complementary layers:
	1.	Shape-Based Regime Detection
Intraday prices are segmented into normalized windows and compared using soft-DTW (Dynamic Time Warping).
A mutual k-NN graph is built from these distances, and morphological regimes are extracted via HDBSCAN clustering.
	2.	Graph Cohesion & Percolation Threshold
A cohesion measure, τ₍c₎, is computed as the critical threshold at which the graph becomes fully connected.
τ₍c₎ captures the morphological density of the market and serves as a diagnostic of structural stability.
	3.	Liquidity & Early-Warning Analysis
Several microstructure proxies (Roll spread, Corwin–Schultz, Garman–Klass, Parkinson) are computed from intraday OHLC data.
Lead–lag and quantile-based analyses test whether changes in τ₍c₎ anticipate liquidity deterioration, supported by statistical validation (Fisher tests, odds ratios, Holm–Bonferroni correction, block bootstrap CIs).


## Objective
The objective is not to build a trading model but to develop a quantitative early-warning framework that:
	•	Detects morphological regimes in intraday price dynamics
	•	Quantifies graph cohesion as a proxy for market stability
	•	Tests whether τ₍c₎ provides leading indicators of liquidity stress
	•	Validates statistical and probabilistic robustness across rolling horizons
	•	Provides a transparent, fully reproducible research pipeline

⸻

## Project Structure

```text
morphology_percolation_project/
│
├── cmorph/                      # Core morphology library (DTW, graph, clustering)
│
├── scripts/                     # Reproducible analytical modules
│   ├── 05d_cluster_vs_features.py       # Statistical tests on roughness vs. clusters
│   ├── 06_liquidity_proxies.py          # Computes liquidity metrics from OHLC data
│   ├── 06c_leadlag_tau_vs_proxies.py    # Aligns τ₍c₎ and proxies, computes cross-corr
│   └── 07a_early_warning.py             # Quantile-based early-warning tests 
│   
├── artifacts/                   # Generated results (CSV, Markdown, Figures)
│   ├── ew_*.csv                 # Early-warning metrics per configuration
│   ├── ew_heatmap_*.png         # Odds ratio heatmaps
│   ├── cluster_features.csv     # Cluster-level morphological statistics
│   └── fig_5_11_brier.png       # Brier comparison plot
│
├── data/                        # Local data (not published)
│  
│
└── README.md                    # Project documentation (this file)

```
Note: Raw OHLC data is excluded from version control for confidentiality.
All generated figures, summaries, and metrics are reproducible using the scripts above.


## Key Results
	•	τ₍c₎ (graph cohesion) acts as a morphological measure of market stability.
	•	Morphological regimes are statistically distinct in roughness and volatility structure.
	•	Decreasing τ₍c₎ systematically precedes liquidity stress, especially in Roll and Corwin–Schultz proxies.
	•	Early-warning relationships remain robust under expanding quantile scopes and bootstrap confidence intervals.
	•	τ₍c₎-based probabilistic forecasts outperform baselines in Brier score tests.


## Summary
This project bridges market microstructure, shape analysis, and graph theory into a coherent framework for quantitative regime diagnostics.
It demonstrates how morphological features of intraday price paths can be systematically linked to liquidity conditions, offering both scientific insight and practical stress indicators.


