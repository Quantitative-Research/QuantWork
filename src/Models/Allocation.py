# src/Models/Allocation.py

from typing import List, Dict
import pandas as pd
from MarketDataLoader.HistoricalPricesLoader import load_prices
from Models import PortfolioOptimizer as po


def OptimalAllocation(
    tickers: List[str],
    measure: str = "Close",
    method: str = "Markovitz",
    allow_short: bool = False,
    target_return: float = None
) -> pd.Series:
    """
    High-level wrapper: fetch 2 years of price data, compute returns,
    and return optimal portfolio weights.

    Parameters
    ----------
    tickers : list[str]
        Stock tickers (e.g. ["AAPL","MSFT","GOOG"]).
    measure : {"Close","High","Low"}
        Which price measure to use (default: Close).
    method : str
        Optimization method ("Markovitz" currently supported).
    allow_short : bool
        Whether short positions are allowed.
    target_return : float, optional
        If set, compute Markowitz portfolio with this target (annualized).
        If None, compute min variance portfolio.

    Returns
    -------
    pd.Series
        Optimal weights, indexed by ticker.
    """
    prices = load_prices(tickers, period_days=730)
    df = prices[measure]

    # daily log returns
    ret = po.daily_log_returns(df)
    cov = po.annualize_cov(po.covariance_matrix(ret))
    exp_ret = po.annualize_returns(ret)

    if method.lower() == "markovitz":
        weights = po.markowitz_weights(
            cov,
            exp_returns=exp_ret if target_return else None,
            target_return=target_return,
            allow_short=allow_short
        )
        return weights
    else:
        raise NotImplementedError(f"Method {method} not implemented")
