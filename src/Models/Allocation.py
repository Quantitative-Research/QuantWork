# src/Models/Allocation.py

from typing import List
import pandas as pd

from src.MarketDataLoader.HistoricalPricesLoader import load_prices
from src.Models import PortfolioOptimizer as po


def OptimalAllocation(
    tickers: List[str],
    measure: str = "Close",
    method: str = "Markowitz",
    allow_short: bool = False,
    target_return: float | None = None,
    period_days: int = 730
) -> pd.Series:
    """
    High-level wrapper: fetch historical price data, compute returns,
    and return optimal portfolio weights.
    
    Parameters
    ----------
    tickers : list[str]
        List of tickers to include in the portfolio
    measure : str
        Price measure to use ('Close', 'High', or 'Low')
    method : str
        Optimization method ('Markowitz')
    allow_short : bool
        Whether to allow short positions
    target_return : float or None
        Target annual return (None for minimum variance)
    period_days : int
        Number of days of historical data to use for calibration (default 730 ≈ 2 years)
    """

    # --------------------------------------------------
    # Load prices
    # --------------------------------------------------
    prices = load_prices(tickers, period_days=period_days)

    if prices.keys() == []:
        raise ValueError("No price data returned.")

    if measure not in prices:
        raise ValueError(f"Price measure '{measure}' not available.")

    df = prices[measure]

    # --------------------------------------------------
    # Compute returns
    # --------------------------------------------------
    ret = po.daily_log_returns(df)
    ret = ret.dropna(axis=1)

    n_assets = ret.shape[1]

    if n_assets == 0:
        raise ValueError("No valid assets after data cleaning.")

    # --------------------------------------------------
    # 1-asset case → 100%
    # --------------------------------------------------
    if n_assets == 1:
        ticker = ret.columns[0]
        return pd.Series([1.0], index=[ticker])

    # --------------------------------------------------
    # Moments
    # --------------------------------------------------
    cov = po.annualize_cov(po.covariance_matrix(ret))
    exp_ret = po.annualize_returns(ret)

    # --------------------------------------------------
    # Optimization
    # --------------------------------------------------
    method = method.lower()

    if method == "markowitz":
        weights = po.markowitz_weights(
            cov=cov,
            exp_returns=exp_ret if target_return is not None else None,
            target_return=target_return,
            allow_short=allow_short
        )
        return weights

    else:
        raise NotImplementedError(f"Method '{method}' not implemented.")
