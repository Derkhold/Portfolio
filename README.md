## Introduction

This portfolio documents my ongoing exploration of quantitative trading and market system design through hands-on, code-driven experimentation.

Rather than relying on pre-packaged solutions or purely academic exercises, I’ve chosen to build things from scratch, to understand how trading logic can be structured, tested, and scaled in realistic environments. Each project represents a step in that learning process: from writing clean, modular code to developing full research pipelines that connect theory and data.

Each project here represents a step in that journey: starting from modular system design and moving toward the analytical side of market morphology. Together, they reflect both the engineering and the research dimensions of my work : how to make trading systems robust, and how to detect structure in market dynamics.

This portfolio isn’t meant as a final product. It’s a snapshot of my progress, a set of systems, experiments, and frameworks that I continue to refine as I learn.

---

## About Me

I’m currently studying Financial Markets at NEOMA Business School, where my coursework covers market theory, products, and risk management. Alongside this, I’ve developed a strong interest in the technical side of finance : how models and trading ideas can be implemented, tested, and turned into working systems.

I'm a self-motivated learner. I started by writing basic scripts to analyze price data, then progressively became more interested in building real components: backtest engines, order and risk managers, reporting tools.
That process made me appreciate the difference between writing code and designing systems, between tools that just work, and tools that can evolve and be trusted.

My approach is iterative: I prototype, test, and rebuild. Each iteration brings me closer to understanding not just how to model markets, but how to engineer robust frameworks around them.

---

## Projects Included

This portfolio currently includes the following projects, each of which is structured in its own directory with supporting code, documentation, and results.

### 1. Modular Algorithmic Trading Architecture

[`/trading_architecture_project`](./trading_architecture_project)

This project is my most complete system to date. It’s a Python-based trading architecture designed to simulate and evaluate strategies in a realistic environment.
Rather than focusing on a single strategy, the goal here was to build the underlying components of a trading system from scratch:

- Modular structure: separate managers for orders, positions, risk, and trade monitoring
- Clean integration with Alpaca market data (including caching)
- Custom indicators using TA-Lib
- Configurable backtest engine with exports (CSV)
- Strategy templates for trend-following, reversal, and exhaustion patterns

What I learned: how to structure a full trading workflow — from signal generation to execution — and how modular design brings both control and transparency to quantitative research.

### 2. Morphology and Percolation in Intraday Markets

[`/morphology_percolation_project`](./MarketMorphology)

This research project takes a more analytical turn. It investigates how intraday price morphology can reveal early warning signs of liquidity stress.

The idea was to move beyond price and volatility, and look at shape, to capture the geometry of market trajectories rather than just their magnitude.

The project combines elastic time-series comparison, graph percolation, and microstructure-based liquidity proxies into one coherent analytical pipeline.
It’s a full research pipeline built from scratch to link shape-based time series analysis with graph percolation and market microstructure diagnostics.

    •	Shape similarity computed with soft-DTW on normalized intraday windows
    •	Graph-based cohesion metric (τ₍c₎) derived from mutual k-NN percolation
    •	Liquidity proxies computed from OHLC data (Roll, Corwin–Schultz, Garman–Klass, Parkinson)
    •	Lead–lag and quantile-based early-warning tests with statistical validation
    •	Reproducible scripts for calibration, visualization, and probabilistic evaluation (Brier score)

What I learned: how to translate a theoretical idea “market shape encodes stress” (with shape analysis, graph dynamics) into a reproducible empirical workflow, combining data engineering, graph theory, and statistical testing.

---

## Methodology & Technical Stack

My main focus when building these projects has been to understand what makes a trading system functional, maintainable, and logically sound. I try to approach each project as more than just a coding exercise, it’s a way to explore how financial logic can be broken down into clear, testable components.

Early on, I realized that the difference between a simple script and a usable system lies in structure: isolating responsibilities, controlling data flow, and ensuring that each part of the process — from strategy execution to trade tracking — is handled cleanly.

Across projects, I’ve consistently applied a few guiding principles:

    •	Separation of concerns between data, model logic, risk control, and performance measurement
    •	Modular design for clarity, reusability, and extensibility
    •	Transparency at every stage, with reproducible outputs and detailed logs
    •	Iterative improvement : each build cycle is an opportunity to refactor and understand systems more deeply

Rather than depending on complex third-party tools or automated pipelines, I’ve chosen to write most of the architecture myself — not to reinvent the wheel, but to gain a deeper understanding of how each layer fits together.

Each project in this portfolio reflects a balance between building just enough to simulate realistic behavior, and keeping things simple and transparent enough to learn from.

---

## Ongoing Work

I’m currently rebuilding one of my previous projects, a Volatility Surface and Option Pricing Engine, in C++, as part of my transition toward lower-level, performance-oriented quantitative development.

The original version, built in Python, implemented Black–Scholes and SABR models with volatility surface calibration, Greeks decomposition, and scenario analysis for earnings-related volatility events.
Rewriting it in C++ allows me to better understand memory management, numerical stability, and optimization ; essential aspects of professional-grade pricing systems.

This process isn’t about rewriting for its own sake, but about learning how quantitative infrastructure is built at scale: how to balance precision, speed, and design clarity in the core of a pricing engine.

---

## Contact

If you'd like to discuss these projects, ask questions about the architecture, or talk about opportunities related to trading systems or quantitative research, feel free to reach out.
Email: derkhold@gmail.com

Thank you for visiting this portfolio. Feedback, technical suggestions, and constructive questions are always welcome.
