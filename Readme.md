# Lithium-Ion Battery Degradation Analysis Using Machine Learning
## CALCE CS2 Dataset: DVA/ICA Electrochemical Analysis, Statistical Testing, and ML Prediction
---

## Executive Summary

| Aspect | Details |
|--------|---------|
| **Dataset** | CALCE CS2 (University of Maryland) |
| **Cell Type** | Prismatic Pouch Cell (5.4 × 33.6 × 50.6 mm, 21.1g) |
| **Cell Chemistry** | LiCoO₂ cathode (1100 mAh nominal) |
| **Data Volume** | 9,102,845 rows from 659 files across 15 cells |
| **ML Dataset** | 300 cycles from 6 xlsx cells (txt excluded) |
| **Best Model (10-Fold CV)** | Polynomial (deg=2) — R² = 54.0% |
| **Best Model (LOCO 1C)** | GradientBoost — R² = 73.5% |
| **Best Model (LOCO 0.5C)** | GradientBoost — R² = 18.2% |
| **Cross-Protocol Transfer** | FAILS (R² = −20.1% to 5.7%) |
| **Key Finding** | 0.5C cycling provides **1.9× longer battery life** than 1C |

---

## Research Questions

This project answers **7 key questions** about lithium-ion battery degradation:

| # | Question | Answer |
|---|----------|--------|
| 1 | Does charging speed affect battery life? | **YES** — 1C degrades 1.6× faster than 0.5C |
| 2 | How long will batteries last? | 0.5C: 208 cycles, 1C: 110 cycles |
| 3 | Can we predict battery health? | **YES** — R² = 73.5% for 1C cells |
| 4 | Can lab data predict field performance? | **NO** — Cross-protocol transfer fails |
| 5 | When does degradation accelerate? | Cycle 24 (1C), Cycle 31 (0.5C) |
| 6 | How confident are predictions? | Overconfident (60% vs 95% expected) |
| 7 | Which ML model works best? | GradientBoost with LOCO validation |

---

## Table of Contents

### Part 1: Data Preparation
1. Setup and Data Configuration
2. Data Loading
3. Exploratory Data Analysis (EDA)
4. Cycle-Level Feature Extraction

### Part 2: Electrochemical Analysis
5. Capacity Fade Visualization
6. DVA/ICA Analysis

### Part 3: Statistical Analysis
7. Statistical Analysis (ANOVA)

### Part 4: Machine Learning
8. Feature Engineering for ML
9. Traditional ML Model Comparison
10. Neural Network Models
11. Leave-One-Cell-Out (LOCO) Validation
12. Cross-Protocol Transfer Learning

### Part 5: Uncertainty & Prognostics
13. Uncertainty Quantification
14. Knee-Point Detection
15. Remaining Useful Life (RUL) Prediction

### Part 6: Results & Conclusion
16. Final Model Comparison
17. Summary and Conclusion

---

## Key Results Summary

### Degradation Analysis

| Protocol | SOH at Cycle 50 | Fade Rate | RUL | Knee-Point |
|----------|-----------------|-----------|-----|------------|
| 0.5C | 96.2% | 0.073%/cycle | **208 cycles** | Cycle 31 |
| 1C | 92.7% | 0.120%/cycle | **110 cycles** | Cycle 24 |
| **Ratio** | — | 1.6× faster | **1.9× longer** | 7 cycles earlier |

### Machine Learning Performance

| Model | Validation | R² | Notes |
|-------|------------|-----|-------|
| Polynomial (deg=2) | 10-Fold CV | **54.0%** | Best traditional ML |
| MLP (256,128,64) | 10-Fold CV | 51.1% | Best neural network |
| GradientBoost | LOCO Overall | **54.4%** | Realistic deployment |
| GradientBoost | LOCO 1C | **73.5%** | Best for 1C cells |
| GradientBoost | LOCO 0.5C | 18.2% | Poor (limited data) |
| Bootstrap Ensemble | Ensemble | 73.3% | With uncertainty |
| 1C → 0.5C | Cross-Protocol | **−20.1%** | FAILS |
| 0.5C → 1C | Cross-Protocol | **5.7%** | FAILS |

### Statistical Analysis

| Test | Result | Interpretation |
|------|--------|----------------|
| ANOVA F-statistic | 93.88 | Highly significant |
| p-value | 1.78×10⁻¹⁹ | p < 0.001 |
| Effect size (η²) | 24.0% | Large effect |
| Cohen's d | 1.19 | Large effect |

### Electrochemical Analysis (DVA/ICA)

**Capacity-Based Degradation (CS2_33):**

| Metric | Cycle 2 | Cycle 48 | Change | Statistical Validity |
|--------|---------|----------|--------|---------------------|
| Max Capacity | 1.1572 Ah | 1.1181 Ah | **−3.4%** | ✓ Validated |

**ICA Peak Voltage Tracking (Attempted):**

| Analysis Type | Result | Statistical Validity |
|---------------|--------|---------------------|
| first-to-last | −126.3 mV | ✗ Not representative |
| Linear regression | +0.803 mV/cycle | ✗ No significant trend |
| R² | 0.033 | ✗ Only 3.3% variance explained |
| p-value | 0.215 | ✗ Not significant |

**Note:** ICA peak tracking incompatible with multi-segment RPT structure. Standard method requires continuous single-segment discharge data. Capacity-based degradation analysis remains valid.

---

## Setup and Requirements

### Installation
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn openpyxl
```

### Required Packages

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥1.5.0 | Data manipulation and analysis |
| numpy | ≥1.23.0 | Numerical computing |
| matplotlib | ≥3.6.0 | Visualization |
| seaborn | ≥0.12.0 | Statistical plots |
| scipy | ≥1.9.0 | Statistical tests, signal processing |
| scikit-learn | ≥1.1.0 | Machine learning models |
| openpyxl | ≥3.0.0 | Excel file reading |

### Data Setup

1. **Download** the CALCE CS2 dataset from: https://calce.umd.edu/battery-data
2. **Extract** files maintaining the folder structure (Type 1, Type 2, etc.)
3. **Update** `BASE_PATH` in the code cell below to your local data directory

### Running the Notebook

| Setting | Requirement |
|---------|-------------|
| Kernel | Python 3.8+ |
| Runtime | ~14 minutes |
| Memory | ~4 GB RAM |
| Reproducibility | `RANDOM_SEED = 42` |

---

## The CALCE CS2 Dataset

The Center for Advanced Life Cycle Engineering (CALCE) at University of Maryland provides one of the most comprehensive public battery aging datasets. The CS2 series contains commercial **prismatic LiCoO₂ cells** (1.1 Ah nominal, 5.4 × 33.6 × 50.6 mm) tested under various cycling protocols.

### Experimental Protocols

| Type | Protocol | Cells | Format | Cycles | Status |
|------|----------|-------|--------|--------|--------|
| Type 1 | 0.5C Constant | CS2_33, CS2_34 | xlsx | 100 | ✓ Used in ML |
| Type 1 | 0.5C Constant | CS2_8, CS2_21 | txt | 57 | ✗ Excluded (74-86% invalid) |
| Type 2 | 1C Constant | CS2_35, CS2_36, CS2_37, CS2_38 | xlsx | 200 | ✓ Used in ML |
| Type 3 | Variable (0.11-2.2A) | CS2_3, CS2_9 | xlsx | 705 | Loaded only |
| Type 4 | Random Cutoff | CS2_7 | xlsx | 32 | Loaded only |
| Type 5 | Partial (Low SOC) | CS2_5, CS2_6 | xlsx | 200 | Loaded only |
| Type 6 | Partial (High SOC) | CS2_24, CS2_25 | xlsx | 200 | Loaded only |

### ML Dataset Summary

| Protocol | Cells | Samples | Usage |
|----------|-------|---------|-------|
| 0.5C | CS2_33, CS2_34 | 100 | Training & Testing |
| 1C | CS2_35, CS2_36, CS2_37, CS2_38 | 200 | Training & Testing |
| **Total** | **6 cells** | **300 cycles** | ML Analysis |

### Data Exclusions

| Cells | Format | Issue | Decision |
|-------|--------|-------|----------|
| CS2_8 | txt | 74% invalid cycles | Excluded from ML |
| CS2_21 | txt | 86% invalid cycles | Excluded from ML |

**Reason:** txt-format cells had unreliable capacity integration, resulting in physically impossible discharge capacity values.

---

## Outputs

After running this notebook, you will have:

### Generated Figures

| Filename | Description |
|----------|-------------|
| `capacity_fade_overview.png` | Capacity fade by protocol |
| `cross_protocol_transfer.png` | Cross-protocol transfer results |
| `dva_ica_analysis.png` | DVA/ICA electrochemical analysis |
| `eda_boxplots.png` | Box plots by protocol |
| `eda_correlation_heatmap.png` | Feature correlations |
| `eda_cycle_level.png` | Cycle-level exploratory analysis |
| `eda_distributions.png` | Voltage/current distributions |
| `eda_missing_values.png` | Missing values assessment |
| `eda_pairplot.png` | Pairwise feature relationships |
| `eda_timeseries.png` | Time-series visualization |
| `final_model_comparison.png` | All models comparison |
| `knee_point_detection.png` | Knee-point detection |
| `loco_validation.png` | LOCO validation results |
| `ml_diagnostics.png` | ML model diagnostics |
| `rul_prediction.png` | RUL predictions |
| `statistical_analysis.png` | ANOVA visualization |
| `uncertainty_quantification.png` | Uncertainty analysis |
### Key Insights

| Finding | Implication |
|---------|-------------|
| 0.5C provides 1.9× longer life | Use slower charging for longevity |
| Cross-protocol transfer fails | Train models on matching protocols |
| 1C is more predictable (R² = 73.5%) | 1C data better for ML |
| Model is overconfident (60% coverage) | Widen confidence intervals |
| Knee-point at cycle 24-31 | Early warning for degradation |
| ICA peak tracking fails on RPT data | Verify method compatibility with data structure |

---

## Note

Update `BASE_PATH` in the code to match your local data directory before running.

---

**Author:** Mehdi Hassanbeigi 

**Email:** Hasanbeigimahdi25@gmail.com


---