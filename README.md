# Monte Carlo Option Pricing Engine

A Monte Carlo simulation framework for pricing European options under the **Black–Scholes model**.  
Built from scratch in **Python** (with NumPy + C++ implementations), this project explores numerical finance techniques, performance tradeoffs, and convergence behavior.

---

## Features

- **Naive Monte Carlo (Python loops)**  
  - Path-by-path simulation of GBM (geometric Brownian motion) stock paths.  
  - Validated against closed-form Black–Scholes prices.  

- **Vectorized NumPy Monte Carlo**  
  - End-to-end simulation using array operations.  
  - Achieves ~20–50× speedups vs naive Python.  
  - Includes both **path-based** and **terminal-only** variants.

- **Convergence Testing**  
  - Confirms Monte Carlo error shrinks at ~`1/√N`.  
  - Automated Pytest checks (SE scaling and accuracy vs Black–Scholes).  
  - Benchmark sweeps with CSV + plotting utilities.

- **Reproducibility**  
  - Fully isolated Python virtual environment.  
  - Dependencies tracked in `requirements.txt`.  
  - Results saved to `results/*.csv` and visualizations/plots in `results/*.png`.

---
## Setup

1. Clone repo and create venv:
   ```bash
   git clone https://github.com/<your-username>/MC-optionspricing.git
   cd MC-optionspricing
   python -m venv .venv
   source .venv/bin/activate

2. Install dependencies:
   ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
3. Run Tests:
   ```bash
   PYTHONPATH=$(pwd) python -m pytest -q
