from MarketDataLoader.MarketFromExcel import Market
from utils.paths import data_path
import datetime
from Priceable.CustomEuropeanPriceable import CustomEuropeanPriceable
from Models.BlackModel import BlackModel
from Products.CommonOption import Straddle, Butterfly, Digital, RiskReversal
from Products.Option import VanillaOption
from Products.Enums import OptionType

def test_price_straddle():
    xl_file = data_path('spx_1_nov_24.xlsx')
    pricingdate = datetime.datetime(2024, 11,1)
    market = Market(xl_file, pricingdate)
    black_model = BlackModel(market)
    straddle_option = Straddle(market.spot, T = 1)
    priceable = CustomEuropeanPriceable(black_model, straddle_option)
    price = priceable.price()
    assert price>0

def test_consistency_straddle():
    xl_file = data_path('spx_1_nov_24.xlsx')
    pricingdate = datetime.datetime(2024, 11,1)
    market = Market(xl_file, pricingdate)
    black_model = BlackModel(market)
    straddle_option = Straddle(market.spot, T = 1)
    priceable = CustomEuropeanPriceable(black_model, straddle_option)
    straddle_price = priceable.price()
    call_option, put_option = VanillaOption(K = market.spot, T = 1,option_type= OptionType.CALL), VanillaOption(K = market.spot, T = 1, option_type=OptionType.PUT)
    call_price = black_model.PriceVanillaOption(call_option)
    put_price = black_model.PriceVanillaOption(put_option)
    assert straddle_price == (call_price+put_price)

def test_price_and_greeks_straddle():
    xl_file = data_path('spx_1_nov_24.xlsx')
    pricingdate = datetime.datetime(2024, 11,1)
    market = Market(xl_file, pricingdate)
    black_model = BlackModel(market)
    straddle_option = Straddle(market.spot, T = 1)
    priceable = CustomEuropeanPriceable(black_model, straddle_option)
    result = priceable.price_and_greeks()
    price = priceable.price()
    assert result.price>0
    assert result.price - price < 1e-10
    assert result.delta is not None
    assert result.gamma is not None
    assert result.vega is not None
    assert result.theta is not None

def test_price_butterfly():
    xl_file = data_path('spx_1_nov_24.xlsx')
    pricingdate = datetime.datetime(2024, 11,1)
    market = Market(xl_file, pricingdate)
    black_model = BlackModel(market)
    butterfly_option = Butterfly(market.spot, T = 1)
    priceable = CustomEuropeanPriceable(black_model, butterfly_option)
    price = priceable.price()
    assert price>0

def test_consistency_butterfly():
    xl_file = data_path('spx_1_nov_24.xlsx')
    pricingdate = datetime.datetime(2024, 11,1)
    market = Market(xl_file, pricingdate)
    black_model = BlackModel(market)
    butterfly_option = Butterfly(market.spot, T = 1)
    priceable = CustomEuropeanPriceable(black_model, butterfly_option)
    butterfly_price = priceable.price()
    epsilon = 0.05
    deltaK = market.spot * epsilon
    call_low = VanillaOption(K = market.spot - deltaK, T =1, option_type= OptionType.CALL)
    call_mid = VanillaOption(K = market.spot, T =1, option_type= OptionType.CALL)
    call_high = VanillaOption(K = market.spot + deltaK, T =1, option_type= OptionType.CALL)
    call_low_price = black_model.PriceVanillaOption(call_low)
    call_mid_price = black_model.PriceVanillaOption(call_mid)
    call_high_price = black_model.PriceVanillaOption(call_high)
    finite_diff_butterfly_price = (call_high_price - 2 * call_mid_price + call_low_price) / (deltaK **2)
    assert abs(butterfly_price - finite_diff_butterfly_price) < 1e-4

def test_price_and_greeks_butterfly():
    xl_file = data_path('spx_1_nov_24.xlsx')
    pricingdate = datetime.datetime(2024, 11,1)
    market = Market(xl_file, pricingdate)
    black_model = BlackModel(market)
    butterfly_option = Butterfly(market.spot, T = 1)
    priceable = CustomEuropeanPriceable(black_model, butterfly_option)
    result = priceable.price_and_greeks()
    price = priceable.price()
    assert result.price>0
    assert result.price - price < 1e-10
    assert result.delta is not None
    assert result.gamma is not None
    assert result.vega is not None
    assert result.theta is not None

def test_price_digital():
    xl_file = data_path('spx_1_nov_24.xlsx')
    pricingdate = datetime.datetime(2024, 11,1)
    market = Market(xl_file, pricingdate)
    black_model = BlackModel(market)
    digital_call = Digital(market.spot, T = 1)
    digital_put = Digital(market.spot, T = 1, optiontype = OptionType.PUT)
    priceable_call = CustomEuropeanPriceable(black_model, digital_call)
    priceable_put = CustomEuropeanPriceable(black_model, digital_put)
    price_call = priceable_call.price()
    price_put = priceable_put.price()
    assert price_call>0
    assert price_put>0
    assert (price_call + price_put -1) < 1e-10

def test_consistency_digital():
    xl_file = data_path('spx_1_nov_24.xlsx')
    pricingdate = datetime.datetime(2024, 11,1)
    market = Market(xl_file, pricingdate)
    black_model = BlackModel(market)
    digital_call = Digital(market.spot, T = 1)
    priceable_call = CustomEuropeanPriceable(black_model, digital_call)
    digital_call_price = priceable_call.price()
    call_option_plus = VanillaOption(K = market.spot + 1, T = 1, option_type= OptionType.CALL)
    call_option_minus = VanillaOption(K = market.spot - 1, T = 1, option_type= OptionType.CALL)
    call_price_plus = black_model.PriceVanillaOption(call_option_plus)
    call_price_minus = black_model.PriceVanillaOption(call_option_minus)
    finite_diff_digital_call_price = -(call_price_plus - call_price_minus) / (2 * 1)
    assert abs(digital_call_price - finite_diff_digital_call_price) < 1e-3

def test_price_and_greeks_digital():
    xl_file = data_path('spx_1_nov_24.xlsx')
    pricingdate = datetime.datetime(2024, 11,1)
    market = Market(xl_file, pricingdate)
    black_model = BlackModel(market)
    digital_call = Digital(market.spot, T = 1)
    priceable_call = CustomEuropeanPriceable(black_model, digital_call)
    result = priceable_call.price_and_greeks()
    price = priceable_call.price()
    assert result.price>0
    assert result.price - price < 1e-10
    assert result.delta is not None
    assert result.gamma is not None
    assert result.vega is not None
    assert result.theta is not None

def test_price_risk_reversal():
    xl_file = data_path('spx_1_nov_24.xlsx')
    pricingdate = datetime.datetime(2024, 11,1)
    market = Market(xl_file, pricingdate)
    black_model = BlackModel(market)
    K_low = market.spot * 0.9
    K_high = market.spot * 1.1
    risk_reversal_option = RiskReversal(K_low, K_high, T = 1)
    priceable = CustomEuropeanPriceable(black_model, risk_reversal_option)
    price = priceable.price()
    assert isinstance(price, float)

def test_consistency_risk_reversal():
    xl_file = data_path('spx_1_nov_24.xlsx')
    pricingdate = datetime.datetime(2024, 11,1)
    market = Market(xl_file, pricingdate)
    black_model = BlackModel(market)
    K_low = market.spot * 0.9
    K_high = market.spot * 1.1
    risk_reversal_option = RiskReversal(K_low, K_high, T = 1)
    priceable = CustomEuropeanPriceable(black_model, risk_reversal_option)
    rr_price = priceable.price()
    put_option = VanillaOption(K = K_low, T = 1, option_type= OptionType.PUT)
    call_option = VanillaOption(K = K_high, T = 1, option_type= OptionType.CALL)
    put_price = black_model.PriceVanillaOption(put_option)
    call_price = black_model.PriceVanillaOption(call_option)
    finite_diff_rr_price = call_price - put_price
    assert abs(rr_price - finite_diff_rr_price) < 1e-4

def test_price_and_greeks_risk_reversal():
    xl_file = data_path('spx_1_nov_24.xlsx')
    pricingdate = datetime.datetime(2024, 11,1)
    market = Market(xl_file, pricingdate)
    black_model = BlackModel(market)
    K_low = market.spot * 0.9
    K_high = market.spot * 1.1
    risk_reversal_option = RiskReversal(K_low, K_high, T = 1)
    priceable = CustomEuropeanPriceable(black_model, risk_reversal_option)
    result = priceable.price_and_greeks()
    price = priceable.price()
    assert result.price>0
    assert result.price - price < 1e-10
    assert result.delta is not None
    assert result.gamma is not None
    assert result.vega is not None
    assert result.theta is not None