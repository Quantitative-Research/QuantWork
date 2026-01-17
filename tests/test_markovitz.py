import pytest
from Models.Allocation import OptimalAllocation

def test_optimal_allocation_us_stocks():
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
    weights = OptimalAllocation(tickers, method="Markovitz", allow_short=False)
    assert len(weights) == len(tickers)
    assert pytest.approx(weights.sum(), abs=1e-6) == 1.0
    assert all(0 <= w <= 1 for w in weights)

def test_optimal_allocation_french_stocks():
    tickers = ["ENGI.PA", "TTE.PA", "SAN.PA", "ACA.PA"]
    weights = OptimalAllocation(tickers, method="Markovitz", allow_short=False)
    assert len(weights) == len(tickers)
    assert pytest.approx(weights.sum(), abs=1e-6) == 1.0
    assert all(0 <= w <= 1 for w in weights)

def test_optimal_allocation_french_stocks_with_target_return():
    tickers = ["ENGI.PA", "TTE.PA", "SAN.PA", "ACA.PA"]
    target_return = 0.10
    weights = OptimalAllocation(
        tickers,
        method="Markovitz",
        allow_short=False,
        target_return=target_return
    )
    assert len(weights) == len(tickers)
    assert pytest.approx(weights.sum(), abs=1e-6) == 1.0
    assert all(0 <= w <= 1 for w in weights)