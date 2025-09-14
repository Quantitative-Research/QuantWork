"""
portfolio_optimizer.py
- compute log returns from price DataFrames
- compute covariance matrices
- compute Markowitz optimal weights (min variance or target return)
"""

from typing import Dict, Optional, List
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def daily_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns for a DataFrame of prices.
    price_df: rows = dates, columns = tickers
    """
    return np.log(price_df / price_df.shift(1)).dropna(how='all')


def covariance_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Return sample covariance matrix of returns (pandas DataFrame)."""
    return returns_df.cov()


def min_variance_weights(cov: pd.DataFrame, allow_short: bool = False) -> pd.Series:
    """
    Compute minimum variance portfolio weights subject to sum(weights) = 1.
    If allow_short is False, weights constrained to >= 0.
    """
    n = cov.shape[0]
    cov_mat = cov.values

    x0 = np.ones(n) / n
    bounds = None if allow_short else tuple((0.0, 1.0) for _ in range(n))
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)

    def obj(w):
        return w.T @ cov_mat @ w

    res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons)
    if not res.success:
        raise RuntimeError("Optimization failed: " + str(res.message))
    return pd.Series(res.x, index=cov.index)


def markowitz_weights(cov: pd.DataFrame,
                      exp_returns: Optional[pd.Series] = None,
                      target_return: Optional[float] = None,
                      allow_short: bool = False) -> pd.Series:
    """
    If target_return is None -> return minimum variance portfolio.
    If target_return specified -> solve for minimal variance with expected return == target_return.
    exp_returns: pandas Series indexed as cov.index (annualized or same units as target_return).
    target_return: scalar (same units as exp_returns). If exp_returns is daily, use daily target.
    """
    if target_return is None or exp_returns is None:
        return min_variance_weights(cov, allow_short=allow_short)

    n = cov.shape[0]
    cov_mat = cov.values
    mu = exp_returns.loc[cov.index].values

    x0 = np.ones(n) / n
    bounds = None if allow_short else tuple((0.0, 1.0) for _ in range(n))

    cons = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'eq', 'fun': lambda w: float(np.dot(w, mu) - target_return)}
    ]

    def obj(w):
        return float(w.T @ cov_mat @ w)

    res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons)
    if not res.success:
        raise RuntimeError("Optimization failed: " + str(res.message))
    return pd.Series(res.x, index=cov.index)


def annualize_returns(daily_returns: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    """Convert daily mean returns to annualized arithmetic returns (approx): mean * trading_days"""
    return daily_returns.mean() * trading_days


def annualize_cov(cov_daily: pd.DataFrame, trading_days: int = 252) -> pd.DataFrame:
    """Scale daily covariance matrix to annual using trading_days"""
    return cov_daily * trading_days
