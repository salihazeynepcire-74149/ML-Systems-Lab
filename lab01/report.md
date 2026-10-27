# Lab 01: Systems Profiling & Environment Setup

**Student Researcher:** Saliha Zeynep Cire (Student ID: 74149)  
**Readings:** Reddi, Chapter 1 ML Systems: Overview; Chapter 2 Systems Topics: §§2.1–2.2  

---

## 1. Goal
The objective of this laboratory is to establish an isolated and reproducible Python runtime environment, benchmark two classical machine learning architectures on the Breast Cancer Wisconsin diagnostic dataset (569 instances, 30 continuous features), empirically measure system resource consumption (training duration, single-sample inference latency, memory allocations, and serialized storage footprint), and analyze the suitability of each model across real-world deployment tiers (Microcontroller/TinyML, Mobile Edge, and Cloud).

---

## 2. Method
1. **Environment Setup & Dependency Resolution:** Configured Python 3.11 with pinned dependency manifests. Resolved minor library dependency constraints gracefully (`pyarrow` compatibility with MLflow).
2. **Dataset Partitioning:** Applied stratified 70/30 train/test splitting on the Breast Cancer dataset with `random_state=42`, producing 398 training instances and 171 test instances.
3. **Model Selection & Architecture:**
   - **Logistic Regression:** Linear classifier with L-BFGS numerical solver (`max_iter=1000`, `random_state=42`, 31 total parameters).
   - **Random Forest:** Ensemble of 100 decision trees (`n_estimators=100`, `random_state=42`).
4. **Systems Telemetry:**
   - **Fit Duration:** High-resolution median wall-clock time over repeated training trials.
   - **Inference Latency:** Average per-sample execution duration computed across 100 repeated evaluations.
   - **Memory Profile:** Peak memory consumption during fitting and querying measured via `memory_profiler`.
   - **Storage Footprint:** Serialized joblib artifact byte-size on disk.
5. **Deployment Tier Assessment:** Evaluated models against hardware memory constraints, compute budgets, and deterministic latency envelopes.

---

## 3. Results Table or Plot

### Empirical Systems Profiling Matrix

| Model Architecture | Test Accuracy | Median Fit Duration (s) | Median Inference Latency (ms) | Serialized Disk Size | Peak Training RAM (MiB) | Peak Inference RAM (MiB) | Target Deployment Tier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | **94.15%** | 0.1508 s | **0.0285 ms** | **1,103 B (1.08 KB)** | 144.35 MiB | 144.35 MiB | **TinyML / Edge Microcontrollers** |
| **Random Forest (100 Trees)** | **93.57%** | 0.0957 s | **1.3585 ms** | **290,889 B (284.07 KB)** | 145.32 MiB | 145.44 MiB | **Mobile Devices / Cloud Clusters** |

### Deployment Viability Matrix

| Deployment Tier | Representative Hardware | Resource Limits | Feasibility & Rationale |
| :--- | :--- | :--- | :--- |
| **TinyML / MCU** | ARM Cortex-M0+ / M4 | 16–64 KB RAM, 256 KB Flash | **Highly Feasible (Logistic Regression Only):** With only 31 floating-point parameters occupying 1.05 KB, Logistic Regression executes with microsecond latency (0.0285 ms) and requires zero dynamic memory allocation. Random Forest (284 KB) exceeds standard microcontroller flash limits. |
| **Mobile / Edge** | Apple A-Series, Snapdragon | 4–12 GB RAM, Multi-Core CPU | **Highly Feasible (Both Models):** Both models fit comfortably in modern mobile memory budgets; single-sample latencies (< 1.5 ms) are negligible for real-time mobile apps. |
| **Cloud Inference** | AWS c6i / g4dn Instances | Large RAM, Enterprise Compute | **Feasible but Suboptimal:** Hosting lightweight tabular models in heavy cloud clusters introduces network round-trip latencies (~50–100 ms) that dwarf the 0.03 ms algorithmic execution time. |

---

## 4. Three Conclusions in Your Own Words
1. **Parametric Dot Products Offer Massive Storage Superiority:** Logistic Regression produces a compact model file of only 1,103 bytes (1.03 KB) while achieving 94.15% accuracy, compared to Random Forest's 290,889 bytes (284 KB) at 93.57% accuracy—a 275x size difference with superior diagnostic generalization.
2. **Inference Latency Reflects Control Flow Complexity:** The per-sample inference latency for Logistic Regression is 0.0285 ms versus 1.3585 ms for Random Forest (47.7x difference) because computing an inner product requires contiguous SIMD vector arithmetic, whereas random forests require traversing conditional decision tree branches that incur instruction cache misses.
3. **Edge Optimization Prioritizes Arithmetic Intensity over Ensemble Complexity:** For low-power remote medical telemetry, deploying linear parametric classifiers avoids flash memory saturation and minimizes battery drain, proving that systems constraints often favor simpler, well-regularized algorithms over complex ensemble methods.
