from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd
from pandas._typing import TimedeltaConvertibleTypes

def rolling_window(returns, memory, min_periods=20):
    min_periods = max(min_periods, 1)

    times = returns.index
    assets = returns.columns

    returns = returns.values

    Sigmas = np.zeros((returns.shape[0], returns.shape[1], returns.shape[1]))
    Sigmas[0] = np.outer(returns[0], returns[0])

    for t in range(1, returns.shape[0]):
        alpha_old = 1 / min(t + 1, memory)
        alpha_new = 1 / min(t + 2, memory)

        if t >= memory:
            Sigmas[t] = alpha_new / alpha_old * Sigmas[t - 1] + alpha_new * (
                np.outer(returns[t], returns[t])
                - np.outer(returns[t - memory], returns[t - memory])
            )
        else:
            Sigmas[t] = alpha_new / alpha_old * Sigmas[t - 1] + alpha_new * (
                np.outer(returns[t], returns[t])
            )

    Sigmas = Sigmas[min_periods - 1 :]
    times = times[min_periods - 1 :]

    return {
        times[t]: pd.DataFrame(Sigmas[t], index=assets, columns=assets)
        for t in range(len(times))
    }

def add_to_diagonal(Sigmas, lamda):
    """
    Adds lamda*diag(Sigma) to each covariance (Sigma) matrix in Sigmas

    param Sigmas: dictionary of covariance matrices
    param lamda: scalar
    """
    for key in Sigmas.keys():
        Sigmas[key] = Sigmas[key] + lamda * np.diag(np.diag(Sigmas[key]))

    return Sigmas

def from_row_matrix_to_covariance(M, n):
    """
    Convert Tx(n(n+1)/2) matrix of upper diagonal parts of covariance matrices to a Txnxn matrix of covariance matrices
    """
    Sigmas = []
    T = M.shape[0]
    for t in range(T):
        Sigmas.append(from_row_to_covariance(M[t], n))
    return np.array(Sigmas)