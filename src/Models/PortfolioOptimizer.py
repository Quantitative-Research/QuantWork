"""
portfolio_optimizer.py
- compute log returns from price DataFrames
- compute covariance matrices
- compute Markowitz optimal weights (min variance or target return)
"""

from typing import Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize


# --------------------------------------------------
# Returns & moments
# --------------------------------------------------

def daily_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns for a DataFrame of prices.
    price_df: rows = dates, columns = tickers
    """
    return np.log(price_df / price_df.shift(1)).dropna(how="all")


def covariance_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Return sample covariance matrix of returns (pandas DataFrame)."""
    return returns_df.cov()


def annualize_returns(
    daily_returns: pd.DataFrame,
    trading_days: int = 252
) -> pd.Series:
    """
    Convert daily mean returns to annualized arithmetic returns.
    """
    return daily_returns.mean() * trading_days


def annualize_cov(
    cov_daily: pd.DataFrame,
    trading_days: int = 252
) -> pd.DataFrame:
    """
    Scale daily covariance matrix to annual.
    """
    return cov_daily * trading_days


# --------------------------------------------------
# Portfolio optimization
# --------------------------------------------------

def min_variance_weights(
    cov: pd.DataFrame,
    allow_short: bool = False
) -> pd.Series:
    """
    Compute minimum variance portfolio weights subject to sum(weights) = 1.
    """

    n = cov.shape[0]

    # Trivial 1-asset case (safety)
    if n == 1:
        return pd.Series([1.0], index=cov.index)

    cov_mat = cov.values
    x0 = np.ones(n) / n

    bounds = None if allow_short else tuple((0.0, 1.0) for _ in range(n))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)

    def objective(w):
        return float(w.T @ cov_mat @ w)

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not res.success:
        raise RuntimeError("Min-variance optimization failed: " + str(res.message))

    return pd.Series(res.x, index=cov.index)


def markowitz_weights(
    cov: pd.DataFrame,
    exp_returns: Optional[pd.Series] = None,
    target_return: Optional[float] = None,
    allow_short: bool = False
) -> pd.Series:
    """
    Markowitz portfolio optimization.

    - If target_return is None -> minimum variance portfolio
    - If target_return specified -> minimal variance with expected return constraint
    """

    n = cov.shape[0]

    # Safety: 1-asset case
    if n == 1:
        return pd.Series([1.0], index=cov.index)

    # Fallback to min-variance if no target specified
    if target_return is None or exp_returns is None:
        return min_variance_weights(cov, allow_short=allow_short)

    cov_mat = cov.values
    mu = exp_returns.loc[cov.index].values

    x0 = np.ones(n) / n
    bounds = None if allow_short else tuple((0.0, 1.0) for _ in range(n))

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'eq', 'fun': lambda w: float(np.dot(w, mu) - target_return)}
    ]

    def objective(w):
        return float(w.T @ cov_mat @ w)

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not res.success:
        raise RuntimeError("Markowitz optimization failed: " + str(res.message))

    return pd.Series(res.x, index=cov.index)
