from Models.Numericals.Black76 import Black76
from Products.Enums import OptionType

def test_CallPrice():
    model = Black76(F=100, T=1, discount_rate=0.05)
    price = model.call_price(K=100, sigma=0.2)
    assert isinstance(price, float)
    assert price > 0

def test_PutPrice():
    model = Black76(F=100, T=1, discount_rate=0.05)
    price = model.put_price(K=100, sigma=0.2)
    assert isinstance(price, float)
    assert price > 0

def test_DeltaCall():
    model = Black76(F=100, T=1, discount_rate=0.05)
    delta = model.delta(K=100, sigma=0.2, option=OptionType.CALL)
    assert 0 < delta < 1

def test_DeltaPut():
    model = Black76(F=100, T=1, discount_rate=0.05)
    delta = model.delta(K=100, sigma=0.2, option=OptionType.PUT)
    assert -1 < delta < 0

def test_DeltaCloseFormulaVsFiniteDiff():
    model = Black76(F=100, T=1, discount_rate=0.05)
    h = 1
    model_plus = Black76(F=100+h, T=1, discount_rate=0.05)
    model_minus = Black76(F=100-h, T=1, discount_rate=0.05)
    price_plus = model_plus.call_price(K=100, sigma=0.2)
    price_minus = model_minus.call_price(K=100, sigma=0.2)
    finite_diff_delta = (price_plus - price_minus) / (2 * h)
    analytic_delta = model.delta(K=100, sigma=0.2, option=OptionType.CALL)
    assert abs(finite_diff_delta - analytic_delta) < 1e-4