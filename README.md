# Quantitative Research & System Engineering Portfolio

## Introduction

This repository serves as the central directory for my continuous exploration of quantitative finance, market microstructure, and computational system design through hands-on, code-driven experimentation.

Rather than relying on pre-packaged solutions or purely academic exercises, I choose to build analytical frameworks and trading infrastructure from scratch. This approach forces a deep understanding of how mathematical logic can be structured, rigorously tested, and scaled within realistic environments. 

Each project detailed below represents a distinct facet of quantitative development: from architecting modular execution systems and detecting topological structures in market dynamics, to deploying deep sequential learning for physical thermodynamics. 

Ultimately, this portfolio is not a static showcase, but a living technical notebook reflecting the dual dimensions of my work: engineering robust infrastructure and conducting rigorous data-driven research.

---

## Projects Directory

Each project links to its own dedicated repository containing the source code, research notebooks, and methodological documentation.

### 1. Modular Algorithmic Trading Architecture
🔗 **[TradingArchitecture](https://github.com/Derkhold/TradingArchitecture)**

A Python-based trading architecture designed to simulate and evaluate quantitative strategies within a realistic, event-driven environment. Rather than focusing solely on alpha generation, the objective was to engineer the underlying infrastructure required for a full trading lifecycle.

* **Core Mechanics:** Strict separation of concerns with dedicated managers for orders, positions, risk, and real-time trade monitoring.
* **Data Integration:** Clean API integration with Alpaca market data, featuring optimized local caching.
* **Analytics:** Custom technical indicators implementation (TA-Lib) and a highly configurable backtest engine with granular CSV exports.
* **Key Takeaway:** Mastering the structural design of a complete algorithmic workflow, demonstrating how modular object-oriented programming (OOP) brings absolute control and transparency to quantitative research.

### 2. Morphology and Percolation in Intraday Markets
🔗 **[MarketMorphology](https://github.com/Derkhold/MarketMorphology)**

An analytical research pipeline investigating how the geometry of intraday price trajectories can reveal early warning signs of liquidity stress. This project moves beyond standard volatility metrics to capture the actual "shape" of the market.

* **Core Mechanics:** Elastic time-series comparison utilizing soft-DTW (Dynamic Time Warping) on normalized intraday windows.
* **Graph Theory:** Development of a graph-based cohesion metric derived from mutual k-NN percolation.
* **Microstructure:** Integration of advanced liquidity proxies (Roll, Corwin–Schultz, Garman–Klass, Parkinson).
* **Key Takeaway:** Translating abstract theoretical concepts into a reproducible, statistically validated empirical workflow, successfully bridging time-series geometry with graph dynamics.

### 3. PMSM Thermal Soft-Sensor: Deep Sequence Modeling
🔗 **[Pmsm_soft_sensor](https://github.com/Derkhold/Pmsm_soft_sensor)**

A physics-informed Machine Learning project aimed at building a virtual temperature sensor for Permanent Magnet Synchronous Motors (PMSM) using highly dynamic telemetry data.

* **Core Mechanics:** Implementation of Deep Sequential Learning architectures (GRU) using PyTorch to natively learn thermal inertia and latent heat memory.
* **Defensive Engineering:** Strict enforcement of group-aware validation (GroupKFold) to prevent temporal autocorrelation and data leakage across causally independent driving profiles.
* **Data Engineering:** High-performance data ingestion utilizing columnar Parquet storage and Polars.
* **Key Takeaway:** Managing massive industrial datasets, designing recurrent neural networks as numerical integrators, and rigorously auditing "black-box" models using Explainable AI (Permutation Importance).

---

## Methodology & Technical Philosophy

Across all my projects, my primary focus is ensuring that systems are functional, maintainable, and logically impenetrable. I view programming not just as an implementation tool, but as a structured framework to deconstruct complex environments.

I consistently apply the following guiding principles:
* **Separation of Concerns:** Strictly isolating data ingestion, model logic, risk control, and performance measurement.
* **Defensive Engineering:** Enforcing rigorous validation protocols to prevent look-ahead bias, data leakage, and overfitting in time-series environments.
* **Transparency:** Designing reproducible pipelines with detailed logging, deterministic execution, and clear artifact serialization.

Rather than depending on black-box third-party frameworks, I choose to build core architectures myself to gain a structural understanding of how every computational layer interacts.

---

## Ongoing Work: C++ Options Pricing Engine

I am currently rebuilding a Volatility Surface and Option Pricing Engine in **C++**, marking my transition toward lower-level, high-performance quantitative development. 

The original Python version implemented Black–Scholes and SABR models, complete with volatility surface calibration, Greeks decomposition, and earnings-scenario analysis. Re-engineering this framework in C++ is a deliberate exercise to master memory management, numerical stability, and computational optimization—the foundational pillars of professional-grade pricing infrastructure.
