import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod
from Products.Enums import OptionType

# === Black-Scholes model ===

class BlackScholes:
    def __init__(self, S, r, q):
        self._S = S
        self._r = r
        self._q = q

    @property
    def S(self):
        return self._S

    @S.setter
    def S(self, value):
        self._S = value

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, value):
        self._r = value

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        self._q = value

    def _d1(self, K, T, sigma):
        return (np.log(self.S / K) + (self.r - self.q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    def _d2(self, K, T, sigma):
        return self._d1(K, T, sigma) - sigma * np.sqrt(T)

    def call_price(self, K, T, sigma) -> float:
        d1 = self._d1(K, T, sigma)
        d2 = self._d2(K, T, sigma)
        return self.S * np.exp(-self.q * T) * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)

    def put_price(self, K, T, sigma) -> float:
        d1 = self._d1(K, T, sigma)
        d2 = self._d2(K, T, sigma)
        return K * np.exp(-self.r * T) * norm.cdf(-d2) - self.S * np.exp(-self.q * T) * norm.cdf(-d1)

    def delta(self, K : float, T : float, sigma : float, option : OptionType = OptionType.CALL) -> float:
        d1 = self._d1(K, T, sigma)
        if option == OptionType.CALL:
            return np.exp(-self.q * T) * norm.cdf(d1)
        elif option == OptionType.PUT:
            return np.exp(-self.q * T) * (norm.cdf(d1) - 1)
        else:
            raise ValueError("Invalid option type")

    def gamma(self, K, T, sigma) -> float:
        d1 = self._d1(K, T, sigma)
        return (np.exp(-self.q * T) * norm.pdf(d1)) / (self.S * sigma * np.sqrt(T))

    def vega(self, K, T, sigma) -> float:
        d1 = self._d1(K, T, sigma)
        return self.S * np.exp(-self.q * T) * norm.pdf(d1) * np.sqrt(T)

    def theta(self, K, T, sigma, option=OptionType.CALL) -> float:
        d1 = self._d1(K, T, sigma)
        d2 = self._d2(K, T, sigma)
        term1 = - (self.S * sigma * np.exp(-self.q * T) * norm.pdf(d1)) / (2 * np.sqrt(T))
        if option == OptionType.CALL:
            term2 = self.q * self.S * np.exp(-self.q * T) * norm.cdf(d1)
            term3 = -self.r * K * np.exp(-self.r * T) * norm.cdf(d2)
            return term1 + term2 + term3
        elif option == OptionType.PUT:
            term2 = -self.q * self.S * np.exp(-self.q * T) * norm.cdf(-d1)
            term3 = self.r * K * np.exp(-self.r * T) * norm.cdf(-d2)
            return term1 + term2 + term3
        else:
            raise ValueError("Invalid option type")

    def rho(self, K, T, sigma, option=OptionType.CALL) -> float:
        d2 = self._d2(K, T, sigma)
        if option == OptionType.CALL:
            return K * T * np.exp(-self.r * T) * norm.cdf(d2)
        elif option == OptionType.PUT:
            return -K * T * np.exp(-self.r * T) * norm.cdf(-d2)
        else:
            raise ValueError("Invalid option type")
        
