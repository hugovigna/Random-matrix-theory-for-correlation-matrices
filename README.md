# Random Matrix Theory for Correlation Matrix Denoising

Comparison of covariance matrix denoising methods using Random Matrix Theory (RMT) on high-dimensional financial data.

## Overview

In high-dimensional settings (large number of assets, limited observations), sample correlation matrices are noisy and lead to poor portfolio optimization. This project compares five denoising methods based on Random Matrix Theory to clean correlation matrices and improve mean-variance portfolio performance.

## Methods Compared

### 1. Linear Shrinkage (Ledoit-Wolf)
Shrinks the sample covariance matrix toward a structured estimator (typically constant correlation).

### 2. Analytical Shrinkage (Ledoit-Wolf)
Non-linear shrinkage using analytical formulas derived from RMT, targeting the optimal oracle estimator.

### 3. Average Oracle (Bongiorno & Chalet)
Oracle-based approach that averages over potential oracle estimators to reduce estimation error.

### 4. Marchenko-Pastur Clipping (Theory)
Clips eigenvalues outside the Marchenko-Pastur bounds. Uses the theoretical upper bound λ₊ based on the aspect ratio q = p/n.

### 5. Marchenko-Pastur Clipping (99th Percentile)
Clips eigenvalues using the empirical 99th percentile as the upper bound instead of the theoretical limit.

## Data

### Synthetic Data
- Simulated returns with controlled covariance structure
- Known ground truth for validation
- Varying dimensions (p) and sample sizes (n)

### Real Data
- Financial asset returns (stocks, indices, etc.)
- High-dimensional regime (p/n ≥ 0.1)

## Evaluation Metrics

### 1. Mean-Variance (MV) Loss
Measures portfolio performance degradation due to estimation error:
```
MV Loss = (w_estimated - w_true)ᵀ Σ_true (w_estimated - w_true)
```
Lower is better.

### 2. Frobenius Norm
Distance between estimated and true covariance matrices:
```
||Σ_estimated - Σ_true||_F
```

### 3. Markowitz Efficient Frontier
Visual comparison of efficient frontiers:
- True frontier (oracle)
- Sample covariance frontier
- Denoised covariance frontiers

### 4. Kolmogorov-Smirnov Distance
Compares eigenvalue distributions of the cleaned matrix to the theoretical Marchenko-Pastur distribution.

### 5. Rejection Rate
Tests if returns generated from the cleaned covariance matrix are statistically consistent with the sample:
- Generate synthetic returns from Σ_cleaned
- Compare distribution to observed returns
- Report rejection rate at 5% significance level

## Project Structure

```
├── README.md
├── notebooks/
│   ├── 01_synthetic_data_analysis.ipynb
│   ├── 02_real_data_analysis.ipynb
│   └── 03_method_comparison.ipynb
├── src/
│   ├── denoising/
│   │   ├── linear_shrinkage.py
│   │   ├── analytical_shrinkage.py
│   │   ├── average_oracle.py
│   │   └── marchenko_pastur.py
│   ├── metrics.py
│   └── utils.py
├── data/
│   ├── synthetic/
│   └── real/
└── results/
    ├── figures/
    └── metrics/
```

## Key Results

| Method | MV Loss ↓ | Frobenius ↓ | KS Distance ↓ |
|--------|-----------|-------------|---------------|
| Sample (baseline) | 1.000 | 1.000 | 0.XXX |
| Linear Shrinkage | 0.XXX | 0.XXX | 0.XXX |
| Analytical Shrinkage | 0.XXX | 0.XXX | 0.XXX |
| Average Oracle | 0.XXX | 0.XXX | 0.XXX |
| MP Clipping (Theory) | 0.XXX | 0.XXX | 0.XXX |
| MP Clipping (99th %) | 0.XXX | 0.XXX | 0.XXX |

*(Lower values indicate better performance)*

## Installation

```bash
git clone https://github.com/yourusername/rmt-covariance-denoising.git
cd rmt-covariance-denoising

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Denoise a Correlation Matrix

```python
from src.denoising import AnalyticalShrinkage, MarchenkoPastur

# Load your returns data (n_samples x n_assets)
returns = load_returns_data()

# Method 1: Analytical Shrinkage
denoiser = AnalyticalShrinkage()
Sigma_clean = denoiser.fit_transform(returns)

# Method 2: Marchenko-Pastur Clipping
mp_denoiser = MarchenkoPastur(method='theory')  # or 'percentile'
Sigma_clean = mp_denoiser.fit_transform(returns)
```

### Compute Metrics

```python
from src.metrics import mv_loss, frobenius_distance

# Compare to true covariance (synthetic data)
mv = mv_loss(Sigma_clean, Sigma_true, returns)
frob = frobenius_distance(Sigma_clean, Sigma_true)

print(f"MV Loss: {mv:.4f}")
print(f"Frobenius Distance: {frob:.4f}")
```

### Plot Efficient Frontier

```python
from src.utils import plot_efficient_frontier

plot_efficient_frontier(
    Sigma_sample=sample_cov,
    Sigma_denoised=Sigma_clean,
    Sigma_true=Sigma_true,  # Optional
    mu=expected_returns
)
```

## Requirements

- Python 3.8+
- NumPy, SciPy
- pandas
- matplotlib, seaborn
- scikit-learn

See `requirements.txt` for complete list.

## References

### Ledoit-Wolf Shrinkage
1. Ledoit, O., & Wolf, M. (2004). *Honey, I Shrunk the Sample Covariance Matrix*. Journal of Portfolio Management.
2. Ledoit, O., & Wolf, M. (2020). *Analytical Nonlinear Shrinkage of Large-Dimensional Covariance Matrices*. Annals of Statistics.

### Average Oracle
3. Bongiorno, C., & Challet, D. (2020). *Covariance Matrix Filtering with Bootstrapped Hierarchies*. PloS One.

### Marchenko-Pastur
4. Marčenko, V. A., & Pastur, L. A. (1967). *Distribution of Eigenvalues for Some Sets of Random Matrices*. Mathematics of the USSR-Sbornik.
5. Laloux, L., Cizeau, P., Bouchaud, J. P., & Potters, M. (1999). *Noise Dressing of Financial Correlation Matrices*. Physical Review Letters.

## License

MIT License - see LICENSE file for details.

## Author

Hugo Vigna  
GitHub: [@hugovigna](https://github.com/hugovigna)

## Acknowledgments

This work builds on the pioneering research in Random Matrix Theory applied to finance by Ledoit, Wolf, Bouchaud, Potters, and others.
