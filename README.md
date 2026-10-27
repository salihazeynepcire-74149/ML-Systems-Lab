# Machine Learning Systems: Experimental Laboratory Portfolio

**Student:** Saliha Zeynep Cire  
**Student ID:** 74149  
**Institution:** Vistula University — Machine Learning Systems (2025/2026)  

---

## Coursework Overview

This portfolio presents a unified suite of four laboratory investigations centered on the computational, mathematical, and data-engineering aspects of modern machine learning systems:

### Module Directory
- **`lab01/` (Systems Profiling & Runtime Baseline):** Rigorous telemetry of parametric vs ensemble models on tabular diagnostic data, recording memory usage, execution times, and target hardware tiers.
- **`lab02/` (Neural Network from First Principles):** Object-oriented, vectorized neural network in NumPy featuring exact analytical matrix gradients, two-sided finite difference checking, and hyperparameter grid dynamics.
- **`lab03/` (Architectural Topologies and Computational Budgets):** Multi-model comparison (MLP, CNN, Attention) on Fashion-MNIST analyzing parameter counts, theoretical FLOPs, profiling MACs, and inference speed across batch regimes.
- **`lab04/` (Production Data Engineering & Validation):** Zero-leakage ETL pipeline incorporating domain boundary validation, mode/median imputers fitted strictly on training data, feature synthesis, and duplicate label noise audits.

## Reproducibility Protocol
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
