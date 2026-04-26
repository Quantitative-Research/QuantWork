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
    log_returns = np.log(price_df / price_df.shift(1))
    return log_returns.dropna(how="any")  # type: ignore


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
        {'type': 'eq', 'fun': lambda w: float(np.dot(w, mu) - target_return)}  # type: ignore
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


# --------------------------------------------------
# Portfolio analysis
# --------------------------------------------------

def portfolio_returns(
    returns_df: pd.DataFrame,
    weights: pd.Series
) -> pd.Series:
    """
    Compute daily portfolio returns given asset returns and weights.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns, shape (n_dates, n_assets)
    weights : pd.Series
        Portfolio weights, indexed by asset tickers
        
    Returns
    -------
    pd.Series
        Daily portfolio returns
    """
    # Align weights with returns columns
    aligned_weights = weights.loc[returns_df.columns]
    return (returns_df * aligned_weights).sum(axis=1)


def portfolio_volatility(
    returns_df: pd.DataFrame,
    weights: pd.Series,
    trading_days: int = 252
) -> float:
    """
    Compute annualized portfolio volatility.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns, shape (n_dates, n_assets)
    weights : pd.Series
        Portfolio weights, indexed by asset tickers
    trading_days : int
        Number of trading days per year (default 252)
        
    Returns
    -------
    float
        Annualized volatility of the portfolio
    """
    cov = covariance_matrix(returns_df)
    cov_annual = annualize_cov(cov, trading_days=trading_days)
    
    # Align weights with covariance matrix
    aligned_weights = weights.loc[cov_annual.index]
    portfolio_var = float(aligned_weights.T @ cov_annual @ aligned_weights)  # type: ignore
    return np.sqrt(portfolio_var)


def portfolio_beta(
    returns_df: pd.DataFrame,
    weights: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Compute portfolio beta relative to a benchmark.
    
    Portfolio returns must be aligned with benchmark returns (same index).
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns of assets, shape (n_dates, n_assets)
    weights : pd.Series
        Portfolio weights, indexed by asset tickers
    benchmark_returns : pd.Series
        Daily benchmark returns, same date index as returns_df
        
    Returns
    -------
    float
        Beta of the portfolio relative to the benchmark
    """
    port_ret = portfolio_returns(returns_df, weights)
    
    # Align dates
    common_dates = port_ret.index.intersection(benchmark_returns.index)
    port_ret_aligned = port_ret.loc[common_dates]
    bench_ret_aligned = benchmark_returns.loc[common_dates]
    
    # Compute covariance and benchmark variance
    cov = np.cov(port_ret_aligned, bench_ret_aligned)[0, 1]
    bench_var = np.var(bench_ret_aligned, ddof=1)
    
    if bench_var == 0:
        return 0.0
    
    return cov / bench_var
