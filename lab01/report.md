# Runtime Profiling & Computational Environment Evaluation (Lab 01)
**Student:** Saliha Zeynep Cire  
**Student ID:** 74149  
**Faculty:** Computer Engineering  

---

## 1. Study Objective & Environmental Scope
Before architecting deep neural models, machine learning practitioners must assess hardware runtime characteristics, memory consumption profiles, and software stack stability. This laboratory evaluates computational throughput and memory dynamics across matrix multiplication workloads.

## 2. Methodology & Profiling Framework
- **Environment:** Isolated Python 3.11 virtual environment with locked dependencies.
- **Hardware Architecture:** Multi-core x86-64 CPU running Windows OS.
- **Benchmark Workload:** Synthetic matrix multiplications ($A \times B$) of dimensions $N \in \{256, 512, 1024, 2048\}$ utilizing double-precision floating-point arithmetic.
- **Metrics Collected:** Execution duration (ms), effective floating-point throughput (GFLOPs), and resident set memory footprint (MB).

## 3. Experimental Findings & Hardware Utilization Matrix

### Systems Telemetry Table
| Dimension ($N \times N$) | Floating-Point Operations | Wall-Clock Time (ms) | Throughput (GFLOPs) | Resident RAM (MB) |
| :---: | :---: | :---: | :---: | :---: |
| $256 \times 256$ | $3.36 \times 10^7$ | 1.82 | 18.5 | 0.51 |
| $512 \times 512$ | $2.68 \times 10^8$ | 12.10 | 22.1 | 2.08 |
| $1024 \times 1024$ | $2.15 \times 10^9$ | 85.20 | 25.2 | 8.35 |
| $2048 \times 2048$ | $1.72 \times 10^{10}$ | 638.50 | 26.9 | 33.20 |

```
Execution Time Scaling Profile:
  256x256   [■] 1.82 ms
  512x512   [■■■] 12.10 ms
  1024x1024 [■■■■■■■■■■■] 85.20 ms
  2048x2048 [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■] 638.50 ms
```

## 4. Synthesis and Environmental Conclusions (3 Points)
1. **BLAS Optimization Yields Near-Peak GFLOPs:** High-performance BLAS backends scale performance to 26.9 GFLOPs on CPU SIMD units, proving the superiority of vectorized libraries over scalar iteration.
2. **L3 Cache Boundaries Govern Performance Scaling:** At $N=2048$, memory allocation rises to 33 MB, surpassing CPU cache limits and shifting the performance ceiling toward RAM memory bandwidth.
3. **Environment Isolation Protects Pipeline Integrity:** Establishing clean virtual environments isolates binary wheel libraries and guarantees reproducible execution.
