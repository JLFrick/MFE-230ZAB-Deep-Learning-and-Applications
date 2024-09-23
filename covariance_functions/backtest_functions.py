from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from covariance_functions.regularization_functions import em_regularize_covariance
from covariance_functions.regularization_functions import regularize_covariance
from covariance_functions.ewma_functions  import ewma

def MSE(returns, covariances):
    returns_shifted = returns.shift(-1)

    MSEs = []
    for time, cov in covariances.items():
        realized_cov = returns_shifted.loc[time].values.reshape(
            -1, 1
        ) @ returns_shifted.loc[time].values.reshape(1, -1)
        MSEs.append(np.linalg.norm(cov - realized_cov) ** 2)

    return pd.Series(MSEs, index=covariances.keys())

def log_likelihood(returns, Sigmas, means=None, scale=1):
    """
    Computes the log likelihhod assuming Gaussian returns with covariance matrix
    Sigmas and mean vector means

    param returns: numpy array where rows are vector of asset returns
    param Sigmas: numpy array of covariance matrix
    param means: numpy array of mean vector; if None, assumes zero mean
    """
    if means is None:
        means = np.zeros_like(returns)

    T, n = returns.shape

    returns = returns.reshape(T, n, 1)
    means = means.reshape(T, n, 1)

    returns = returns * scale
    means = means * scale
    Sigmas = Sigmas * scale**2

    dets = np.linalg.det(Sigmas).reshape(len(Sigmas), 1, 1)
    Sigma_invs = np.linalg.inv(Sigmas)

    return (
        -n / 2 * np.log(2 * np.pi)
        - 1 / 2 * np.log(dets)
        - 1
        / 2
        * np.transpose(returns - means, axes=(0, 2, 1))
        @ Sigma_invs
        @ (returns - means)
    ).flatten()

def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n + 1) / n
    return x, y