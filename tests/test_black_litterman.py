import pandas as pd
import numpy as np

from BlackLitterman.black_litterman import black_litterman_weights


def _sample_returns():
    return pd.DataFrame(
        {
            "AAPL": [0.01, 0.02, -0.01, 0.04, 0.02, 0.00],
            "MSFT": [0.015, 0.01, 0.00, 0.03, 0.01, -0.02],
            "AMZN": [0.008, 0.018, -0.005, 0.02, 0.03, 0.01],
        },
        index=pd.date_range("2024-01-01", periods=6, freq="MS"),
    )


def test_black_litterman_weights_are_valid():
    returns = _sample_returns()
    weights = black_litterman_weights(
        returns=returns,
        market_weights=[0.4, 0.35, 0.25],
        views=[0.12, 0.10, 0.08],
        view_confidence=[0.7, 0.6, 0.5],
        allow_short=False,
    )

    assert weights.shape[0] == 3
    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= -1e-12).all()


def test_black_litterman_without_views_keeps_market_weights():
    returns = _sample_returns()
    weights = black_litterman_weights(
        returns=returns,
        market_weights=[0.5, 0.3, 0.2],
        allow_short=False,
    )

    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0).all()
