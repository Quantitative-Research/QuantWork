from Models.PricingModel import PricingModel
from Products.Option import Option
from Products.EuropeanCustomOption import EuropeanCustomOption
from typing import Union
from PricingResult.PricingResult import PricingResult
from abc import ABC,abstractmethod

class Priceable(ABC):
    """
    A priceable is a virtual object reprenting how pricing is handled,
    PRICEABLE = MODEL + PRODUCT
    """
    def __init__(self, model: PricingModel, option: Union[Option, EuropeanCustomOption]):
        self._model = model
        self._option = option
    
    @abstractmethod
    def price(self)-> float:
        """ Handles the pricing"""
        pass

    @abstractmethod
    def price_and_greeks(self)-> PricingResult:
        """ Handles the pricing and greeks calculation"""
        pass

    @property
    def model(self) -> PricingModel:
        """Returns the pricing model."""
        return self._model

    @property
    def option(self) -> Union[Option, EuropeanCustomOption]:
        """Returns the financial option."""
        return self._option
    
