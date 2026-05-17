from usage.Markowitz.dashboard_logic import compute_portfolio_metrics
import pandas as pd

def test_compute_portfolio_metrics_US():
    resolved_tickers = ["AAPL", "MSFT", "GOOGL"]
    weights = pd.Series([0.5, 0.3, 0.2], index=resolved_tickers)
    hist_window = 365
    ref_ticker = "SPY"
    port_vol, beta = compute_portfolio_metrics(resolved_tickers, weights, hist_window, ref_ticker)
    assert port_vol > 0, "Portfolio volatility should be positive"
    assert beta is not None, "Beta should be computed when reference ticker is provided"

def test_compute_portfolio_metrics_EUR():
    resolved_tickers = ["AIR.PA", "OR.PA", "SAN.PA"]
    weights = pd.Series([0.4, 0.4, 0.2], index=resolved_tickers)
    hist_window = 365
    ref_ticker = "SPY"
    port_vol, beta = compute_portfolio_metrics(resolved_tickers, weights, hist_window, ref_ticker)
    assert port_vol > 0, "Portfolio volatility should be positive"
    assert beta is not None, "Beta should be computed when reference ticker is provided"