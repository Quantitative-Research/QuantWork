import datetime
import pytest

from MarketDataLoader.MarketFromExcel import Market
from Models.BlackModel import BlackModel
from Priceable.CustomEuropeanPriceable import CustomEuropeanPriceable
from Products.CommonOption import Straddle, Butterfly, Digital, RiskReversal
from Products.Option import VanillaOption
from Products.Enums import OptionType
from utils.paths import data_path


# ============================================================
# Fixtures (loaded once per test session)
# ============================================================

@pytest.fixture(scope="session")
def market():
    xl_file = data_path("spx_1_nov_24.xlsx")
    pricingdate = datetime.datetime(2024, 11, 1)
    return Market(xl_file, pricingdate)


@pytest.fixture(scope="session")
def black_model(market):
    return BlackModel(market)


@pytest.fixture
def priceable_factory(black_model):
    def _factory(option):
        return CustomEuropeanPriceable(black_model, option)
    return _factory


# ============================================================
# Straddle tests
# ============================================================

def test_price_straddle(market, priceable_factory):
    option = Straddle(market.spot, T=1)
    price = priceable_factory(option).price()
    assert price > 0


def test_consistency_straddle(market, black_model, priceable_factory):
    option = Straddle(market.spot, T=1)
    straddle_price = priceable_factory(option).price()

    call = VanillaOption(K=market.spot, T=1, option_type=OptionType.CALL)
    put = VanillaOption(K=market.spot, T=1, option_type=OptionType.PUT)

    expected = (
        black_model.PriceVanillaOption(call)
        + black_model.PriceVanillaOption(put)
    )

    assert straddle_price == pytest.approx(expected, abs=1e-10)


def test_price_and_greeks_straddle(market, priceable_factory):
    option = Straddle(market.spot, T=1)
    priceable = priceable_factory(option)

    result = priceable.price_and_greeks()
    price = priceable.price()

    assert result.price == pytest.approx(price, abs=1e-10)
    assert result.delta is not None
    assert result.gamma is not None
    assert result.vega is not None
    assert result.theta is not None


# ============================================================
# Butterfly tests
# ============================================================

def test_price_butterfly(market, priceable_factory):
    option = Butterfly(market.spot, T=1)
    price = priceable_factory(option).price()
    assert price > 0


def test_consistency_butterfly(market, black_model, priceable_factory):
    option = Butterfly(market.spot, T=1)
    butterfly_price = priceable_factory(option).price()

    epsilon = 0.05
    deltaK = market.spot * epsilon

    call_low = VanillaOption(K=market.spot - deltaK, T=1, option_type=OptionType.CALL)
    call_mid = VanillaOption(K=market.spot, T=1, option_type=OptionType.CALL)
    call_high = VanillaOption(K=market.spot + deltaK, T=1, option_type=OptionType.CALL)

    price_low = black_model.PriceVanillaOption(call_low)
    price_mid = black_model.PriceVanillaOption(call_mid)
    price_high = black_model.PriceVanillaOption(call_high)

    finite_diff = (price_high - 2 * price_mid + price_low) / (deltaK ** 2)

    assert butterfly_price == pytest.approx(finite_diff, abs=1e-4)


def test_price_and_greeks_butterfly(market, priceable_factory):
    option = Butterfly(market.spot, T=1)
    priceable = priceable_factory(option)

    result = priceable.price_and_greeks()
    price = priceable.price()

    assert result.price == pytest.approx(price, abs=1e-10)
    assert result.delta is not None
    assert result.gamma is not None
    assert result.vega is not None
    assert result.theta is not None


# ============================================================
# Digital option tests
# ============================================================

def test_price_digital(market, priceable_factory):
    call = Digital(market.spot, T=1)
    put = Digital(market.spot, T=1, optiontype=OptionType.PUT)

    call_price = priceable_factory(call).price()
    put_price = priceable_factory(put).price()

    assert call_price > 0
    assert put_price > 0
    assert call_price + put_price == pytest.approx(1.0, abs=1e-10)


def test_consistency_digital(market, black_model, priceable_factory):
    digital_call = Digital(market.spot, T=1)
    digital_price = priceable_factory(digital_call).price()

    shift = 1.0
    call_plus = VanillaOption(K=market.spot + shift, T=1, option_type=OptionType.CALL)
    call_minus = VanillaOption(K=market.spot - shift, T=1, option_type=OptionType.CALL)

    price_plus = black_model.PriceVanillaOption(call_plus)
    price_minus = black_model.PriceVanillaOption(call_minus)

    finite_diff = -(price_plus - price_minus) / (2 * shift)

    assert digital_price == pytest.approx(finite_diff, abs=1e-3)


def test_price_and_greeks_digital(market, priceable_factory):
    option = Digital(market.spot, T=1)
    priceable = priceable_factory(option)

    result = priceable.price_and_greeks()
    price = priceable.price()

    assert result.price == pytest.approx(price, abs=1e-10)
    assert result.delta is not None
    assert result.gamma is not None
    assert result.vega is not None
    assert result.theta is not None


# ============================================================
# Risk reversal tests
# ============================================================

def test_price_risk_reversal(market, priceable_factory):
    K_low = market.spot * 0.9
    K_high = market.spot * 1.1

    option = RiskReversal(K_low, K_high, T=1)
    price = priceable_factory(option).price()

    assert isinstance(price, float)


def test_consistency_risk_reversal(market, black_model, priceable_factory):
    K_low = market.spot * 0.9
    K_high = market.spot * 1.1

    option = RiskReversal(K_low, K_high, T=1)
    rr_price = priceable_factory(option).price()

    put = VanillaOption(K=K_low, T=1, option_type=OptionType.PUT)
    call = VanillaOption(K=K_high, T=1, option_type=OptionType.CALL)

    expected = (
        black_model.PriceVanillaOption(call)
        - black_model.PriceVanillaOption(put)
    )

    assert rr_price == pytest.approx(expected, abs=1e-4)


def test_price_and_greeks_risk_reversal(market, priceable_factory):
    K_low = market.spot * 0.9
    K_high = market.spot * 1.1

    option = RiskReversal(K_low, K_high, T=1)
    priceable = priceable_factory(option)

    result = priceable.price_and_greeks()
    price = priceable.price()

    assert result.price == pytest.approx(price, abs=1e-10)
    assert result.delta is not None
    assert result.gamma is not None
    assert result.vega is not None
    assert result.theta is not None
