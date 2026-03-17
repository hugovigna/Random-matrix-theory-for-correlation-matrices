# Random Matrix Theory for Correlation Matrix Denoising

Comparison of covariance matrix denoising methods using Random Matrix Theory (RMT) on high-dimensional financial data.

> Projet de recherche — Hugo Vigna, CentraleSupélec

## Overview

In high-dimensional settings (large number of assets, limited observations), sample correlation matrices are noisy and lead to poor portfolio optimization. This project compares five denoising methods based on Random Matrix Theory to clean correlation matrices and improve mean-variance portfolio performance.

## Methods Compared

| # | Method | Principle |
|---|--------|-----------|
| 1 | **Linear Shrinkage** (Ledoit-Wolf) | Shrinks toward structured estimator (constant correlation) |
| 2 | **Analytical Shrinkage** (Ledoit-Wolf) | Non-linear shrinkage via RMT analytical formulas |
| 3 | **Average Oracle** (Bongiorno-Challet) | Averages over potential oracle estimators |
| 4 | **MP Clipping** (Theory) | Clips eigenvalues outside theoretical Marchenko-Pastur bounds |
| 5 | **MP Clipping** (99th Percentile) | Clips using empirical 99th percentile as upper bound |

## Data

- **Synthetic data**: Simulated returns with controlled covariance structure and known ground truth
- **Real data**: Financial asset returns in the high-dimensional regime (p/n >= 0.1)

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MV Loss** | Portfolio performance degradation: (w_est - w_true)' Sigma_true (w_est - w_true) |
| **Frobenius Norm** | Distance between estimated and true covariance matrices |
| **Efficient Frontier** | Visual comparison of oracle, sample, and denoised frontiers |
| **KS Distance** | Eigenvalue distribution vs. theoretical Marchenko-Pastur |
| **Rejection Rate** | Consistency test of returns from cleaned covariance (5% significance) |

## Project Structure

```
├── README.md
├── global_report.pdf               # Full research report
├── code/
│   ├── methods.py                  # All denoising implementations
│   ├── synthetic_data.ipynb        # Synthetic data analysis
│   └── real_data.ipynb             # Real financial data analysis
└── results/
    ├── synthetic_data/             # Figures: MV loss, KS distance, frontiers, etc.
    └── real_data/                  # Figures: spectra, eigenvalue effects, frontiers
```

## Usage

### Installation

```bash
git clone https://github.com/hugovigna/Random-matrix-theory-for-correlation-matrices.git
cd Random-matrix-theory-for-correlation-matrices
pip install numpy scipy pandas matplotlib seaborn scikit-learn
```

### Run the analysis

Open the Jupyter notebooks in `code/`:
- `synthetic_data.ipynb` — Synthetic data experiments with ground truth
- `real_data.ipynb` — Real financial data analysis

All denoising methods are implemented in `code/methods.py`.

## Sample Results

### Synthetic Data — Efficient Frontiers
The denoised frontiers closely track the oracle frontier, while the sample covariance frontier deviates significantly.

### Real Data — Eigenvalue Effects
Cleaning small eigenvalues (noise in the bulk) has a larger impact on portfolio weights than correcting the largest eigenvalues (market factor).

See `global_report.pdf` for the complete analysis and numerical results.

## References

1. Ledoit, O. & Wolf, M. (2004). *Honey, I Shrunk the Sample Covariance Matrix*. Journal of Portfolio Management.
2. Ledoit, O. & Wolf, M. (2020). *Analytical Nonlinear Shrinkage of Large-Dimensional Covariance Matrices*. Annals of Statistics.
3. Bongiorno, C. & Challet, D. (2020). *Covariance Matrix Filtering with Bootstrapped Hierarchies*. PloS One.
4. Marcenko, V. A. & Pastur, L. A. (1967). *Distribution of Eigenvalues for Some Sets of Random Matrices*. Mathematics of the USSR-Sbornik.
5. Laloux, L., Cizeau, P., Bouchaud, J.-P. & Potters, M. (1999). *Noise Dressing of Financial Correlation Matrices*. Physical Review Letters.

## Author

Hugo Vigna — [@hugovigna](https://github.com/hugovigna)
