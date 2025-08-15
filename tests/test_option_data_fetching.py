from MarketDataLoader.OptionDataFetcher import OptionDataFetcher,OptionMaturity

def test_collect_data():
    # Define the ticker symbol (e.g., Apple Inc.)
    ticker1, ticker2 = "AAPL", "^VIX"
    AppleData, VIXData = OptionDataFetcher(ticker1), OptionDataFetcher(ticker2)
    assert 1==1

def test_build_markets():
    # Define the ticker symbol (e.g., Apple Inc.)
    ticker1, ticker2 = "AAPL", "^VIX"
    AppleData, VIXData = OptionDataFetcher(ticker1), OptionDataFetcher(ticker2)
    AppleData.build_market()
    VIXData.build_market()
    assert 1==1