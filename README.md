# Machine Learning Fundamentals

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An in-depth, hands-on laboratory curriculum exploring classical and modern machine learning foundations. Each laboratory is implemented in an end-to-end Jupyter Notebook with comprehensive exploratory data analysis, feature engineering, mathematical intuition, model evaluation, and practical real-world problem solving.

Managed with [**uv**](https://github.com/astral-sh/uv), the blazingly fast Python package and environment manager.

## Overview

<details open>
<summary><b>Lab 1: Linear Models & Robust Regression</b></summary>

- **Focus**: End-to-end regression pipelines, data preprocessing, and handling outliers on the Ames Housing dataset.
- **Key Concepts**:
  - Exploratory data analysis, missing value imputation, categorical encoding, and power transformations (Box-Cox, Yeo-Johnson).
  - Regularization dynamics comparing OLS, Ridge ($L_2$), and LASSO ($L_1$).
  - Robust regression comparing Least Absolute Deviations (LAD / `QuantileRegressor`) with `HuberRegressor` for outlier resilience.
  - Scikit-learn Pipeline construction, evaluation metrics, and feature importance inspection via model coefficients and SHAP.
- **Dataset**: [Ames Housing](./lab1/ames_data.parquet) ([Field Descriptions](./lab1/ames_description.txt)).
- **Notebook**: [`lab1/notebook.ipynb`](./lab1/notebook.ipynb)
</details>

<details>
<summary><b>Lab 2: Neighborhood-Based Methods & Approximate Nearest Neighbors</b></summary>

- **Focus**: Distance metrics, the curse of dimensionality, and scalable nearest neighbor search.
- **Key Concepts**:
  - Distance functions (Euclidean, Manhattan, Cosine) and `KNeighborsClassifier` on numerical data (Codon Usage).
  - High-dimensional scaling issues and Approximate Nearest Neighbors (ANN) using **PyNNDescent**.
  - Molecular property prediction using Morgan / ECFP chemical fingerprints on MoleculeNet HIV.
  - Semantic nearest neighbor image retrieval with neural representations (PyTorch & Sentence-Transformers).
- **Datasets**: [Codon Usage](./lab2/codon_usage.csv), [MoleculeNet HIV](./lab2/), and [Image Embeddings](./lab2/images.csv).
- **Notebook**: [`lab2/notebook.ipynb`](./lab2/notebook.ipynb)
</details>

<details>
<summary><b>Lab 3: Kernel Methods & Support Vector Machines</b></summary>

- **Focus**: Dual formulation, max-margin classifiers, the kernel trick, and large-scale kernel approximations.
- **Key Concepts**:
  - Linear SVM (`LinearSVC`) vs. Kernel SVM (`SVC` with RBF, Sigmoid, and Polynomial kernels).
  - Hyperparameter sensitivity: penalty $C$, margin slack, and kernel bandwidth $\gamma$.
  - Scaling kernel methods to large datasets (`COD-RNA`) using the **Nyström** method and **Random Fourier Features** (`RBFSampler`).
  - Performance and decision boundary comparisons across small and large sample spaces.
- **Datasets**: [COD-RNA](./lab3/cod_rna_train.parquet) and [Customer Churn](./lab3/customer_churn_data.csv).
- **Notebook**: [`lab3/notebook.ipynb`](./lab3/notebook.ipynb)
</details>

<details>
<summary><b>Lab 4: Probabilistic Methods & Calibration</b></summary>

- **Focus**: Generative vs. discriminative classifiers, Bayes' theorem, and well-calibrated probabilities.
- **Key Concepts**:
  - Naive Bayes taxonomy: Gaussian, Multinomial, and Bernoulli Naive Bayes.
  - Text feature extraction with TF-IDF, tokenization, and spam detection on Enron emails.
  - High-dimensional, low-sample-size (HDLSS) genomic classification on microarray data.
  - Probability calibration with `CalibratedClassifierCV` (Isotonic regression and Sigmoid/Platt scaling), calibration curves, Expected Calibration Error (ECE), and log-loss evaluation on credit scoring.
- **Datasets**: [Give Me Some Credit](./lab4/give_me_some_credit_data.parquet), [Enron Spam](./lab4/enron_spam_data.csv), and [GSE2034 Microarray](./lab4/).
- **Notebook**: [`lab4/notebook.ipynb`](./lab4/notebook.ipynb)
</details>

<details>
<summary><b>Lab 5: Imbalanced Learning & Anomaly Detection</b></summary>

- **Focus**: Solving severe class skew, asymmetric error costs, and detecting rare anomalies.
- **Key Concepts**:
  - Threshold moving, precision-recall curve analysis, and custom cost matrix optimization for bankruptcy prediction.
  - Resampling algorithms: Synthetic Minority Over-sampling Technique (**SMOTE**) and `RandomUnderSampler`.
  - Extreme imbalance and anomaly detection on credit card fraud using **PyOD**: k-Nearest Neighbors (`KNN`), Isolation Forest (`IForest`), and semi-supervised `XGBOD`.
- **Datasets**: [Polish Companies Bankruptcy (3-year)](./lab5/polish_companies_bankruptcy_3_year_data.arff) and [Credit Card Fraud](./lab5/credit_card_fraud_data.parquet).
- **Notebook**: [`lab5/notebook.ipynb`](./lab5/notebook.ipynb)
</details>

<details>
<summary><b>Lab 6: Feature Selection & Dimensionality Reduction</b></summary>

- **Focus**: Selecting informative signals, combating multicollinearity, and compact subspace projection.
- **Key Concepts**:
  - **Filter methods**: `VarianceThreshold`, Pearson correlation filtering, and Mutual Information (`mutual_info_regression`).
  - **Wrapper methods**: Recursive Feature Elimination (`RFE`) and Cross-Validated RFE (`RFECV`).
  - **Embedded methods**: LASSO ($L_1$) penalization and tree-based feature importances.
  - **Dimensionality reduction**: Principal Component Analysis (`PCA`), Truncated SVD, and non-linear manifold projection via **UMAP**.
- **Dataset**: [Ames Housing](./lab6/ames_data.csv).
- **Notebook**: [`lab6/notebook.ipynb`](./lab6/notebook.ipynb)
</details>

<details>
<summary><b>Lab 7: Time Series Forecasting</b></summary>

- **Focus**: Statistical time series decomposition, classical econometric modeling, and automated forecasting with `sktime`.
- **Key Concepts**:
  - Differencing, stationarity diagnostics, and STL decomposition (`STLTransformer`).
  - Automated classical forecasting with `sktime`: `StatsForecastAutoARIMA`, `StatsForecastAutoETS`, and `NaiveForecaster`.
  - Seasonal modeling (AutoSARIMA) and pipeline transformations (`LogTransformer`, `BoxCoxTransformer`, `SqrtTransformer`).
  - Exogenous feature integration (ARIMAX, SARIMAX) with calendar/holiday features (`HolidayFeatures` and `holidays` library) on Polish CPI inflation, Wikipedia web traffic, and Italian pasta sales.
- **Datasets**: [Polish Inflation CPI](./lab7/polish_inflation.csv), [Wikipedia Web Traffic](./lab7/wikipedia_traffic.parquet), and [Italian Pasta Sales](./lab7/italian_pasta.csv).
- **Notebook**: [`lab7/notebook.ipynb`](./lab7/notebook.ipynb)
</details>

## Getting Started

Ensure you have [uv](https://docs.astral.sh/uv/) installed.

### 1. Clone & Sync Environment

Clone the repository and run `uv sync`. This automatically provisions Python 3.12 (if not already installed), creates a local `.venv`, and installs all locked dependencies:

```bash
git clone https://github.com/barhanc/machine-learning-fundamentals.git
cd machine-learning-fundamentals

# Synchronize the virtual environment with uv.lock in seconds
uv sync
```

### 2. Launch Jupyter

Run JupyterLab or Jupyter Notebook directly through `uv`:

```bash
# Launch JupyterLab
uv run jupyter lab

# Or classic Jupyter Notebook
uv run jupyter notebook
```
