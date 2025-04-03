## Introduction

This portfolio documents my ongoing exploration of quantitative trading and system design through hands-on, code-driven experimentation.

Rather than focusing on pre-packaged solutions or academic exercises, I’ve chosen to build things from scratch — to understand how trading logic can be structured, tested, and adapted in realistic environments. Each project here reflects a stage in that learning process: from writing clean, modular code to simulating executions and designing reusable components.

My goal is to go beyond isolated scripts and move toward a more engineering-minded approach to finance. I’m interested in how trading systems are actually built — how strategies interface with risk controls, how code architecture supports clarity and flexibility, and how outputs can be made transparent and audit-ready.

This portfolio isn’t intended as a final product. It’s a snapshot of what I’ve built so far, and a foundation I plan to improve and expand as I continue learning.

---

## About Me

I'm currently studying Financial Markets at NEOMA Business School. While my academic program focuses on market theory, products, and macro structure, I've gradually built a strong interest in the technical side of finance — especially how trading ideas can be turned into working systems.

I'm a self-motivated learner. I started by writing basic scripts to analyze price data, then progressively became more interested in building real components: backtest engines, order and risk managers, reporting tools.  
That process made me appreciate the difference between writing code and designing systems — and gave me a better sense of what’s needed to build something maintainable, transparent, and usable.

I don’t aim to replicate complex models yet — I prefer to understand the foundations first. My approach is iterative: I try something, break it, improve it, and document it. This portfolio reflects that mindset more than any finished product.


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
- Configurable backtest engine with exports (PDF, CSV, JSON)
- Strategy templates for trend-following, reversal, and exhaustion patterns

What I learned: how to structure a real system, how different components interact, and how to track trades and performance consistently across tests.

---
## Methodology & Technical Stack

My main focus when building these projects has been to understand what makes a trading system functional, maintainable, and logically sound. I try to approach each project as more than just a coding exercise — it’s a way to explore how financial logic can be broken down into clear, testable components.

Early on, I realized that the difference between a simple script and a usable system lies in structure: isolating responsibilities, controlling data flow, and ensuring that each part of the process — from strategy execution to trade tracking — is handled cleanly.

Across projects, I’ve consistently applied a few guiding principles:

- **Separation of concerns** between strategy logic, execution, risk management, and performance tracking
- **Modular design** to make each component understandable, replaceable, and reusable
- **Transparency** in all stages of the process, including order logs, performance outputs, and position history
- **Replicability** through the use of consistent inputs, exports, and scenario controls
- **Iterative thinking** — improving the structure as I learn, refactor, and test ideas over time

Rather than depending on complex third-party tools or automated pipelines, I’ve chosen to write most of the architecture myself — not to reinvent the wheel, but to gain a deeper understanding of how each layer fits together.

Each project in this portfolio reflects a balance between building just enough to simulate realistic behavior, and keeping things simple and transparent enough to learn from.

---

## Roadmap & Next Steps

This portfolio is a continuous work in progress. Each project helps me identify new ideas to explore, missing components to build, or aspects of the system that could be made more robust.

Over the coming months, I aim to focus on the following directions:

- **Improve strategy modularity**  
  Refactor the current system to make strategy definition and parameterization more flexible and self-contained.

- **Multi-asset and multi-timeframe capabilities**  
  Enable backtests that support portfolio-level analysis or cross-timeframe validation.

- **Risk-adjusted position sizing**  
  Integrate logic for sizing trades dynamically based on volatility or recent performance.

- **Live data simulation and paper trading**  
  Move beyond historical backtests by building a workflow that supports real-time or simulated execution using live market data.

- **Logging and error handling improvements**  
  Strengthen the system’s robustness by implementing cleaner logs, custom exceptions, and failure handling.

- **Performance diagnostics**  
  Add deeper analytics on strategy behavior: drawdown events, trade durations, PnL attribution, etc.

This roadmap will likely evolve as I progress, but it reflects my current focus: not just building more, but building better — with more clarity, control, and reusability in each step.

---

## Contact

If you'd like to discuss any of these projects, ask questions about the architecture, or talk about opportunities related to trading systems or quantitative research, feel free to reach out.
Email: derkhold@gmail.com



Thank you for visiting this portfolio. Feedback, technical suggestions, and constructive questions are always welcome.


