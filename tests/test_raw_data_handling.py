import pytest
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend for CI

from MarketDataLoader.OptionDataFetcher import OptionDataFetcher, OptionMaturity

@pytest.fixture(scope="module")
def apple_data():
    ticker = "AAPL"
    data = OptionDataFetcher(ticker)
    data.build_market()
    return data

def test_forward_curve_builds(apple_data):
    # Check we have maturities
    maturities = apple_data.get_maturities()
    assert len(maturities) > 0

    # Check forward interpolation works for a known expiry
    expiry = maturities[0]
    forward_price = apple_data.get_forward(expiry)
    assert forward_price > 0

def test_smile_cleaning_runs(apple_data):
    maturities = apple_data.get_maturities()
    maturity = maturities[3]
    options = apple_data.get_maturity_data(maturity)
    m = OptionMaturity(expiry=maturity, calls=options.calls, puts=options.puts, spot_price=options.spot_price)

    # Generate cleaned smile
    _ = m.cleaned_smile_hat  # should compute without errors
    assert m.cleaned_smile_hat is not None
    assert not m.cleaned_smile_hat.empty

def test_plotting_no_error(apple_data, tmp_path):
    """Plots should run without throwing errors."""
    maturities = apple_data.get_maturities()
    maturity = maturities[3]
    options = apple_data.get_maturity_data(maturity)
    m = OptionMaturity(expiry=maturity, calls=options.calls, puts=options.puts, spot_price=options.spot_price)

    # Save plots to files (so pytest does not try to display them)
    import matplotlib.pyplot as plt

    m.plot_smile(option_type="call", x_axis="moneyness")
    plt.savefig(tmp_path / "smile_call.png")
    plt.close()

    m.plot_smile_clean()
    plt.savefig(tmp_path / "smile_clean.png")
    plt.close()

    apple_data.plot_forward_curve()
    plt.savefig(tmp_path / "forward_curve.png")
    plt.close()
