from .BlackScholes import BlackScholes
from Products.Enums import OptionType
import math

class Black76:
    """
    Diffusion of the type :
    dF_t/F_t = vol dW_t
    By default => gives undiscounted Prices and Greeks
    Can handle discounting if need be :)
    """
    def __init__(self, F : float , T : float, discount_rate : float = 0.0):
        self.F = F
        self.T = T
        self.discount_rate = discount_rate
        self.black = BlackScholes(S=F, r=0, q=0)

    def call_price(self, K : float, sigma : float) -> float:
        undiscounted = self.black.call_price(K, self.T, sigma)
        return math.exp(-self.discount_rate * self.T) * undiscounted

    def put_price(self, K : float, sigma : float) -> float:
        undiscounted = self.black.put_price(K, self.T, sigma)
        return math.exp(-self.discount_rate * self.T) * undiscounted

    def price(self, K : float, sigma : float, option : OptionType = OptionType.CALL) -> float:
        if option == OptionType.CALL:
            return self.call_price(K, sigma)
        elif option == OptionType.PUT:
            return self.put_price(K, sigma)
        else:
            raise ValueError("Invalid option type")

    def delta(self, K : float, sigma : float, option : OptionType = OptionType.CALL) -> float:
        raw = self.black.delta(K, self.T, sigma, option)
        return math.exp(-self.discount_rate * self.T) * raw

    def gamma(self, K : float, sigma : float) -> float:
        raw = self.black.gamma(K, self.T, sigma)
        return math.exp(-self.discount_rate * self.T) * raw

    def vega(self, K : float, sigma : float) -> float:
        raw = self.black.vega(K, self.T, sigma)
        return math.exp(-self.discount_rate * self.T) * raw

    def theta(self, K : float, sigma : float, option : OptionType = OptionType.CALL) -> float:
        raw = self.black.theta(K, self.T, sigma, option)
        return math.exp(-self.discount_rate * self.T) * raw

    def rho(self, K : float, sigma : float, option : OptionType = OptionType.CALL) -> float:
        raw = self.black.rho(K, self.T, sigma, option)
        return math.exp(-self.discount_rate * self.T) * raw
