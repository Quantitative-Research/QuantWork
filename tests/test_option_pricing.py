from MarketDataLoader.MarketFromExcel import Market
from utils.paths import data_path
import datetime
from Priceable.CustomEuropeanPriceable import CustomEuropeanPriceable
from Models.BlackModel import BlackModel
from Products.Option import VanillaOption
from Products.EuropeanCustomOption import EuropeanCustomOption
from Products.Enums import OptionType

def test_VanillaOptionPricing():
    xl_file = data_path('spx_1_nov_24.xlsx')
    pricingdate = datetime.datetime(2024, 11, 1)
    mkt = Market(xl_file, pricingdate)
    black_model = BlackModel(mkt)

    call = VanillaOption(K=mkt.spot, T=1)
    price = black_model.PriceVanillaOption(call)

    assert price > 0, "Vanilla call option price should be positive"


def test_custom_european_option():
    xl_file = data_path('spx_1_nov_24.xlsx')
    pricingdate = datetime.datetime(2024, 11, 1)
    market = Market(xl_file, pricingdate)

    call1 = (market.strikes[2], market.maturities[1])
    put1 = (market.strikes[1], market.maturities[1])

    booked = {
        OptionType.CALL: {call1: 2.0},
        OptionType.PUT: {put1: 1.5}
    }

    custom_option = EuropeanCustomOption(booked)

    assert len(custom_option.list_calls) == 1
    assert len(custom_option.list_puts) == 1
    assert custom_option.list_calls[0].quantity == 2.0
    assert custom_option.list_puts[0].quantity == 1.5


def test_priceable_european_custom_option():
    xl_file = data_path('spx_1_nov_24.xlsx')
    pricingdate = datetime.datetime(2024, 11,1)
    market = Market(xl_file, pricingdate)
    black_model = BlackModel(market)

    # Build a small portfolio from valid (K, T) grid points
    call1 = (market.strikes[2], market.maturities[1])
    call2 = (market.strikes[3], market.maturities[1])
    put1 = (market.strikes[1], market.maturities[1])

    booked = {
        OptionType.CALL: {
            call1: 2.0,  # quantity
            call2: 1.0
        },
        OptionType.PUT: {
            put1: 1.5
        }
    }

    custom_option = EuropeanCustomOption(booked)
    priceable = CustomEuropeanPriceable(black_model, custom_option)
    assert 1 == 1  

def test_price_european_custom_option():
    xl_file = data_path('spx_1_nov_24.xlsx')
    pricingdate = datetime.datetime(2024, 11, 1)
    market = Market(xl_file, pricingdate)
    black_model = BlackModel(market)

    call1 = (market.strikes[2], market.maturities[1])
    call2 = (market.strikes[3], market.maturities[1])
    put1 = (market.strikes[1], market.maturities[1])

    booked = {
        OptionType.CALL: {call1: 2.0, call2: 1.0},
        OptionType.PUT: {put1: 1.5}
    }

    custom_option = EuropeanCustomOption(booked)
    priceable = CustomEuropeanPriceable(black_model, custom_option)
    price = priceable.price()

    assert price > 0, f"Expected positive price, got {price}"


