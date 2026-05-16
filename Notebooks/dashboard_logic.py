from Models.Allocation import OptimalAllocation
from Models import PortfolioOptimizer as po
from MarketDataLoader.HistoricalPricesLoader import load_prices
from utils.TickerResolver import resolve_ticker

def compute_portfolio_metrics(resolved_tickers, weights, hist_window, ref_ticker):
    prices = load_prices(resolved_tickers, period_days=int(hist_window))
    returns = po.daily_log_returns(prices["Close"]).dropna(axis=1)
    port_vol = po.portfolio_volatility(returns, weights)
    beta = None
    if ref_ticker and ref_ticker.strip():
        ref_prices = load_prices([ref_ticker.strip()], period_days=int(hist_window))
        ref_returns = po.daily_log_returns(ref_prices["Close"])
        ref_ticker_col = ref_ticker.strip()
        if ref_ticker_col in ref_returns.columns:
            common_idx = returns.index.intersection(ref_returns.index)
            aligned_portfolio = returns.loc[common_idx]
            aligned_reference = ref_returns.loc[common_idx, ref_ticker_col]
            beta = po.portfolio_beta(aligned_portfolio, weights, aligned_reference)
    return port_vol, beta