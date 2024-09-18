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

def _ewma_cov(data, halflife, min_periods=0):
    for t, ewma in _general(
        data.values,
        times=data.index,
        halflife=halflife,
        fct=lambda x: np.outer(x, x),
        min_periods=min_periods,
    ):
        if not np.isnan(ewma).all():
            yield t, pd.DataFrame(index=data.columns, columns=data.columns, data=ewma)


def iterated_ewma(
    returns,
    vola_halflife,
    cov_halflife,
    min_periods_vola=20,
    min_periods_cov=20,
    mean=False,
    mu_halflife1=None,
    mu_halflife2=None,
    clip_at=None,
    nan_to_num=True,
):
    mu_halflife1 = mu_halflife1 or vola_halflife
    mu_halflife2 = mu_halflife2 or cov_halflife

def MSE(returns, covariances):
    returns_shifted = returns.shift(-1)

    MSEs = []
    for time, cov in covariances.items():
        realized_cov = returns_shifted.loc[time].values.reshape(
            -1, 1
        ) @ returns_shifted.loc[time].values.reshape(1, -1)
        MSEs.append(np.linalg.norm(cov - realized_cov) ** 2)

    return pd.Series(MSEs, index=covariances.keys())

def log_likelihood_low_rank(returns, Sigmas, means=None):
    """
    Computes the log-likelihoods

    param returns: pandas DataFrame of returns
    param Sigmas: dictionary of covariance matrices where each covariance matrix
                  is a namedtuple with fields "F" and "d"
    param means: pandas DataFrame of means

    Note: Sigmas[time] is covariance prediction for returns[time+1]
        same for means.loc[time]
    """
    returns = returns.shift(-1)

    ll = []
    m = np.zeros_like(returns.iloc[0].values).reshape(-1, 1)

    times = []

    for time, low_rank in Sigmas.items():
        # TODO: forming the covariance matrix is bad...
        cov = low_rank.F @ (low_rank.F).T + np.diag(low_rank.d)

        if not returns.loc[time].isna()[0]:
            if means is not None:
                m = means.loc[time].values.reshape(-1, 1)
            ll.append(
                _single_log_likelihood(
                    returns.loc[time].values.reshape(-1, 1), cov.values, m
                )
            )
            times.append(time)

    return pd.Series(ll, index=times).astype(float)

def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n + 1) / n
    return x, y

def add_to_diagonal(Sigmas, lamda):
    """
    Adds lamda*diag(Sigma) to each covariance (Sigma) matrix in Sigmas

    param Sigmas: dictionary of covariance matrices
    param lamda: scalar
    """
    for key in Sigmas.keys():
        Sigmas[key] = Sigmas[key] + lamda * np.diag(np.diag(Sigmas[key]))

    return Sigmas

def from_row_to_covariance(row, n):
    """
    Convert upper diagonal part of covariance matrix to a covariance matrix
    """
    Sigma = np.zeros((n, n))

    # set upper triangular part
    upper_mask = np.triu(np.ones((n, n)), k=0).astype(bool)
    Sigma[upper_mask] = row

    # set lower triangular part
    lower_mask = np.tril(np.ones((n, n)), k=0).astype(bool)
    Sigma[lower_mask] = Sigma.T[lower_mask]
    return Sigma