<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">


# SVM-SOCP-L2-SOLVERS

<em>Empowering Smarter Decisions Through Advanced Optimization</em>

<!-- BADGES -->
<img src="https://img.shields.io/github/last-commit/mmatthieu1290/svm-socp-lp-solvers?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
<img src="https://img.shields.io/github/languages/top/mmatthieu1290/svm-socp-lp-solvers?style=flat&color=0080ff" alt="repo-top-language">
<img src="https://img.shields.io/github/languages/count/mmatthieu1290/svm-socp-lp-solvers?style=flat&color=0080ff" alt="repo-language-count">

<em>Built with the tools and technologies:</em>

<img src="https://img.shields.io/badge/Markdown-000000.svg?style=flat&logo=Markdown&logoColor=white" alt="Markdown">
<img src="https://img.shields.io/badge/TOML-9C4121.svg?style=flat&logo=TOML&logoColor=white" alt="TOML">
<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white" alt="scikitlearn">
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat&logo=NumPy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">

</div>
<br>

---

## Table of Contents

- [Overview](#overview)
- [Mathematical Model](#mathematical-model)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Testing](#testing)
- [Examples](#examples)

---

## Overview

svm-socp-lp-solvers is a Python package that provides advanced solvers for Support Vector Machine (SVM) optimization problems using Second-Order Cone Programming (SOCP) and Linear Programming (LP). Designed for high-dimensional and large-scale data, it enables efficient, robust, and interpretable machine learning models.

**Authors:** Miguel Carrasco, Julio Lopez, Matthieu Marechal

**Why svm-socp-lp-solvers?**

This project aims to simplify and accelerate the development of SVM-based solutions with cutting-edge convex optimization techniques. The core features include:

- 🧩 **🔧 Modular Architecture:** Seamlessly integrates with existing ML workflows and scales across modules.
- 🚀 **⚙️ High-Performance Solvers:** Efficiently handles large-scale SOCP and LP problems for support vector machines.
- 📊 **📈 Utility Functions:** Provides tools for model inference, including predictions and probability estimates.
- 🧮 **🔍 Focused on Convex Optimization:** Leverages cvxpy and other libraries for robust, reliable solutions.
- 🧠 **🤖 Support for Sparse, Robust Models:** Facilitates feature selection and high-dimensional data handling.

---

## Mathematical model 

### SVMLp

 This estimator solves the following optimization problem:

$$
\min_{w,b,\xi}\ \sum_{j=1}^n (|w_j|+\varepsilon)^p + C\sum_{i=1}^m \xi_i
\quad \mathrm{s.t.}\quad
y_i (w^\top x_i + b) \ge 1 - \xi_i,\ \xi_i \ge 0,\ i=1,\dots,m.
$$

The smoothing parameter $\varepsilon>0$ makes the objective locally
Lipschitz and avoids singular behavior at $w_j=0$. The vector $x_i$ contains the features of $i$-th observation and $y_i=\pm1$ is its class.  

### SOCPLp

This estimator solves the following optimization problem:

$$
\min_{w,b,\xi}\ \sum_{j=1}^n (|w_j|+\varepsilon)^p + C \sum_{i=1}^2 \xi_i
\quad \mathrm{s.t.}\quad
\begin{aligned}
& (w,b,\xi) \in \mathbb{R}^{n+3} \\
& w^\top \mu_1 + b \ge 1 - \xi_1 + \kappa(\alpha_1)\|S_1^\top w\|, \\
& -(w^\top \mu_2 + b) \ge 1 - \xi_2 + \kappa(\alpha_2)\|S_2^\top w\|, \\
& \xi \ge 0.
\end{aligned}
$$

where $\kappa(\alpha_i)=\sqrt{\frac{\alpha_i}{1-\alpha_i}}$. The vector $\mu_1$ (resp. $\mu_2$) is the mean feature vector associated with the
positive (resp. negative) class.

The matrix $S_j \in \mathbb{R}^{n \times m_j}$, with $j \in \{1,2\}$, satisfies
$\Sigma_j = S_j S_j^\top$, where $\Sigma_1$ (resp. $\Sigma_2$) is the covariance
matrix of the features associated with the positive (resp. negative) class.

The constraint set above is a reformulation of the following probabilistic
constraint using the multivariate Chebyshev inequality:

$$
\inf_{\tilde{x}_j \sim (\mu_j,\Sigma_j)}
\Pr\left\( (-1)^{j+1}(w^\top \tilde{x}_j + b) \ge 0 \right\)
\ge \alpha_j, \quad j = 1,2.
$$

The notation $\tilde{x}_j \sim (\mu_j,\Sigma_j)$ indicates that the random vectors
$\tilde{x}_j$ have mean $\mu_j$ and covariance matrix $\Sigma_j$.

This model can be interpreted as a robust version of SVM_Lp.

The smoothing parameter $\varepsilon > 0$ makes the objective locally Lipschitz
and avoids singular behavior at $w_j = 0$.

## Getting Started

### Prerequisites

This project requires the following dependencies:

- **Programming Language:** Python
- **Package Manager:** Conda

### Installation

1. **Install the library:**

    ```sh
    pip install git+https://github.com/mmatthieu1290/svm-socp-lp-solvers.git
    ```
2. **Import the solvers:**

    ```sh
    from svm_socp_l2_solvers import SVMLp,SOCPLp
    ```
## Examples

### SVMLp

    from svm_socp_l2_solvers import SVMLp
    import pandas as pd

    url = "https://raw.githubusercontent.com/mmatthieu1290/svm-socp-lp-solvers/main/datos_Titanic.xlsx"
    df = pd.read_excel(url, engine="openpyxl")
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    svm = SVM_Lp(C = 1e7,eps = 1e-4,tol_select_features = 1e-3)
    svm.fit(X,y)

    print("Coefs : ",svm.coef_)
    print("Selected features : ",svm.selected_feature_names_)

### SOCPLp 

    from svm_socp_l2_solvers import SOCPLp
    import pandas as pd
    
    url = "https://raw.githubusercontent.com/mmatthieu1290/svm-socp-lp-solvers/main/datos_Titanic.xlsx"
    df = pd.read_excel(url, engine="openpyxl")
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    socp = SOCPLp(p=0.1,alpha_1=0.2,alpha_2=0.2)
    socp.fit(X,y)

    print("Coefs : ",socp.coef_)
    print("Selected features : ",socp.selected_feature_names_)






