import datetime
import pytest
import numpy as np

from MarketDataLoader.MarketFromExcel import Market
from Models.BlackModel import BlackModel
from Priceable.CustomEuropeanPriceable import CustomEuropeanPriceable
from Products.VixLike import VixLike
from utils.paths import data_path


# ============================================================
# Fixtures (loaded once per test session)
# ============================================================

@pytest.fixture(scope="session")
def market() -> Market:
    xl_file = data_path("spx_1_nov_24.xlsx")
    pricingdate = datetime.datetime(2024, 11, 1)
    return Market(xl_file, pricingdate)


@pytest.fixture(scope="session")
def black_model(market) -> BlackModel:
    return BlackModel(market)


@pytest.fixture
def priceable_factory(black_model):
    def _factory(option):
        return CustomEuropeanPriceable(black_model, option)
    return _factory


def test_price_vix_like(market, priceable_factory):
    forward_level = market.get_forward(30/365)
    option = VixLike(T=30/365, forward_level=forward_level, K_min=0.8, K_max=1.2, number_of_strikes=20, quantity=1)
    option_finer_grid = VixLike(T=30/365, forward_level=forward_level, K_min=0.5, K_max=1.5, number_of_strikes=1000, quantity=1) # type: ignore
    price = priceable_factory(option).price()
    price_finer_grid = priceable_factory(option_finer_grid).price()
    assert price > 0, "VIX-like product price should be positive"
    VIX_value = np.sqrt(price) * 100
    VIX_value_finer_grid = np.sqrt(price_finer_grid) * 100
    assert VIX_value < 100, "VIX-like product implied volatility should be reasonable"
    assert abs(VIX_value - VIX_value_finer_grid) < 1e-1, "VIX-like product price should be stable across different strike grids"