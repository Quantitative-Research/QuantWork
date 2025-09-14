"""
PriceLoader.py
- Downloads historical High/Low/Close from Yahoo Finance (2 years daily by default).
- Returns a dict of DataFrames keyed by measure: 'Close','High','Low'.
"""

from typing import List, Dict
import pandas as pd
import datetime
import yfinance as yf


def load_prices(tickers: List[str], period_days: int = 730) -> Dict[str, pd.DataFrame]:
    """
    Download daily historical prices for given tickers from Yahoo Finance.

    Parameters
    ----------
    tickers : list[str]
        List of tickers (e.g. ["AAPL", "MSFT"]).
    period_days : int
        Number of days to look back (default 730 ≈ 2 years).

    Returns
    -------
    dict
        { 'Close': DataFrame, 'High': DataFrame, 'Low': DataFrame }
        Each DataFrame has index = date, columns = tickers.
    """
    end = datetime.date.today()
    start = end - datetime.timedelta(days=period_days)
    # yfinance download: multi-ticker => columns are multiindex (field, ticker)
    df = yf.download(tickers, start=start.isoformat(), end=end.isoformat(), progress=False, group_by='ticker', threads=True)

    # If single ticker, yfinance returns normal columns; normalize to multiindex
    # We'll create per-measure DataFrames
    measures = ['Close', 'High', 'Low']
    out = {}
    # Try two possible df layouts: (1) top-level columns are fields when single ticker, or
    # (2) MultiIndex where first level field, second level ticker or vice versa.
    # Simpler robust approach: for each ticker, request history individually and join.
    per_ticker = {}
    if isinstance(tickers, str):
        tickers = [tickers]
    # Use single requests per ticker to avoid ambiguous shapes
    for t in tickers:
        hist = yf.Ticker(t).history(start=start.isoformat(), end=end.isoformat(), interval="1d", actions=False)
        # history() may include NaNs for some days; keep Date index and select fields
        per_ticker[t] = hist[['High', 'Low', 'Close']]

    # Combine into per-measure DataFrames
    for measure in measures:
        frames = []
        for t in tickers:
            s = per_ticker[t][measure].rename(t)
            frames.append(s)
        out[measure] = pd.concat(frames, axis=1).sort_index()

    return out
