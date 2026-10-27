# Lab 03 Systems Study: Trade-Off Analysis Across Vision Topologies
**Student:** Saliha Zeynep Cire  
**Student ID:** 74149  
**Faculty:** Computer Engineering  
**Subject:** Deep Learning Systems Engineering  

---

## 1. Purpose and Systems Motivation
When architecting neural networks for production pipelines, computational footprint is as vital a metric as classification accuracy. A model with high predictive power is useless if its latency or memory footprint breaches hardware boundaries. This investigation studies the empirical Pareto boundary formed by three distinct structural archetypes on Fashion-MNIST:

1. **Multilayer Perceptron (`SalihaMLP`)**: Dense vector operations with ELU activation functions.
2. **Convolutional Network (`SalihaCNN`)**: 2-stage hierarchical representation exploiting spatial locality and parameter reuse.
3. **Vision Transformer (`SalihaViT`)**: Dedicated learnable `[CLS]` token architecture with 16 visual patches and multi-head self-attention.

The core systems objective is to understand how architectural inductive biases shape computational intensity, cache locality, and runtime execution efficiency.

---

## 2. Model Formulations and Benchmarking Protocol

### Topology Specifications

- **ELU Perceptron (`SalihaMLP`):**  
  Transforms flattened $28 \times 28$ inputs across two dense hidden layers ($784 \rightarrow 200 \rightarrow 100 \rightarrow 10$). Employs Exponential Linear Units (ELU) to smooth gradient transitions and prevent dead activation states.
  
- **Deep 2-Stage CNN (`SalihaCNN`):**  
  Implements two convolutional blocks: Block 1 maps $1 \rightarrow 24$ channels ($3 \times 3$), followed by $2 \times 2$ Max-Pooling. Block 2 expands to 48 channels. A dropout-regularized dense linear head classifies the flattened $48 \times 7 \times 7$ tensor.

- **CLS Vision Transformer (`SalihaViT`):**  
  Splits images into $16$ non-overlapping patches ($7 \times 7$ pixels). An extra learnable class token `[CLS]` is prepended to the sequence (yielding $17$ total tokens). Two TransformerEncoder layers process token interactions, and the classification decision is extracted strictly from the final `[CLS]` state.

### Execution Environment
All benchmark runs were executed on identical hardware under CPU thread isolation. Each inference latency figure reflects the median over 75 timed passes after 15 warmup passes to stabilize OS scheduling.

---

## 3. Empirical Cost-Accuracy Trade-Offs

### Vision Topology Benchmarks

| Architecture | Parameter Count | FLOPs (M) | Mean Latency (ms) | Peak RAM (MB) | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ELU Perceptron (`SalihaMLP`)** | 178,110 | 0.36 | 0.34 | 1.65 | 88.58% |
| **Deep 2-Stage CNN (`SalihaCNN`)** | 313,290 | 5.82 | 1.15 | 3.10 | 91.48% |
| **CLS Vision Transformer (`SalihaViT`)** | 71,946 | 2.15 | 1.48 | 5.20 | 85.32% |

### Comparative Computational Bottlenecks
- **GEMM Vector Operations (MLP):** The MLP executes rapidly (**0.34 ms**) because large dense matrix multiplications map directly to highly optimized BLAS libraries. However, because it discards spatial 2D relationships, it requires significantly more parameters per unit of accuracy.
- **Hierarchical Spatial Convolutions (CNN):** Although the CNN incurs the highest raw FLOP count (**5.82 MFLOPs**), its regular memory-access stride enables near-peak SIMD throughput, delivering the highest accuracy (**91.48%**) at a reasonable latency of **1.15 ms**.
- **Quadratic Self-Attention Overhead (ViT):** The Vision Transformer has only 72k parameters, but its latency is the highest (**1.48 ms**). The combination of token prepending, dynamic attention matrix calculation ($Q K^T$), and LayerNorm operations results in fragmented memory access and cache misses on small-scale tensors.

### Profiler Reconciliation (FLOPs vs. MACs)
In empirical profiling tools like `thop`, each Multiply-Accumulate operation represents two floating-point operations:
$$\text{FLOPs} = 2 \times \text{MACs}$$
For example, `SalihaCNN` registers approximately $2.91\times 10^6$ MACs, producing the calculated $5.82$ MFLOPs.

---

## 4. Conclusions and Practical Deployment Rules (3 Synthesis Points)

1. **Convolution Remains the Gold Standard for Edge Visual Tasks:**  
   The CNN achieved the highest accuracy (**91.48%**) by a substantial margin. Its localized inductive bias allows it to learn high-fidelity visual representations from modest dataset sizes without requiring hundreds of thousands of pre-training examples.

2. **[CLS] Token Attention Introduces Noticeable Latency Friction:**  
   Prepend-token transformer architectures introduce runtime latency penalties when scaled down to small patch spaces. The dynamic $17 \times 17$ attention matrix calculation produces multiple memory roundtrips that outweigh the reduction in parameter count.

3. **Inference Latency is Governed by Memory Architecture, Not Just FLOPs:**  
   Evaluating models purely on theoretical FLOPs provides a distorted picture. The CNN executes $2.7\times$ more FLOPs than the ViT (5.82 M vs 2.15 M), yet runs **22.3% faster** in wall-clock time due to superior hardware cache utilization and vector unit saturation.
