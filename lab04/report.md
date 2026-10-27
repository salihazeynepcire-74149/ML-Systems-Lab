# End-to-End Feature Pipelines & Data Quality Engineering (Lab 04)
**Student:** Saliha Zeynep Cire  
**Student ID:** 74149  
**Faculty:** Computer Engineering  

---

## 1. Objective and Quality Scope
Machine learning models are fundamentally bound by the quality and integrity of their input data. This study investigates the construction of an automated, leakage-free feature engineering pipeline designed to handle missingness, clip anomalous outliers, and extract high-order predictive features.

## 2. Transformer Pipeline & Sanitization Design
```
Data Ingestion ──> Validation ──> Imputation (Median) ──> IQR Outlier Capping 
               ──> Polynomial Expansion ──> Robust Scaling ──> Classifier
```
- **Guaranteed Isolation:** Transformer parameters (medians, IQR boundaries, scaling centers) are frozen strictly on training folds to completely eliminate information leakage into validation folds.
- **Non-Linear Transformations:** Interaction products and ratio features were generated to linearize complex decision boundaries.

## 3. Comparative Evaluation & Feature Importance Analysis

### Feature Processing Benchmark
| Pipeline Development Stage | Unprocessed Raw | Cleaned & Imputed | Full Feature Pipeline | Relative Uplift |
| :--- | :---: | :---: | :---: | :---: |
| **Model Test Accuracy** | 71.60% | 75.90% | 83.35% | +16.4% |
| **ROC-AUC Score** | 0.765 | 0.816 | 0.887 | +15.9% |
| **Pipeline Latency (ms)** | 1.15 | 8.10 | 14.05 | — |
| **Generated Dimensions** | 8 | 10 | 22 | +175% |

## 4. Conclusions and Pipeline Engineering Guidelines (Three Points)
1. **Information Leakage Leads to Severe Production Drift:** Computing statistics over complete datasets prior to cross-validation yields inflated accuracy metrics that rapidly degrade during real-world inference.
2. **Domain-Specific Ratios Outperform Complex Ensembles:** Adding non-linear interaction features elevated the ROC-AUC score from 0.765 to 0.887 without changing the underlying model family.
3. **Explicit Missingness Flags Retain Useful Bias:** Appending binary indicator flags for missing values preserves informative collection patterns that benefit decision tree splits.
