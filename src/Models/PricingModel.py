from abc import ABC, abstractmethod
from MarketDataLoader.MarketFromExcel import Market
from Products.Option import VanillaOption
from PricingResult.PricingResult import PricingResult

class PricingModel(ABC):
    def __init__(self, market: Market):
        self.market = market
        self.spot = market.spot
        self._volKT = market.vol_surface

    @abstractmethod
    def GeneratePaths(self, number_paths : int, T : float, frequency):
        """
        Abstract method to generate Monte Carlo path
        """
        pass

    @abstractmethod
    def AvailablePricingMethod(self):
        pass
    
    @abstractmethod
    def PriceVanillaOption(self, vanilla : VanillaOption) -> float:
        pass
    
    @abstractmethod
    def PriceAndGreeksVanillaOption(self, vanilla : VanillaOption) -> PricingResult:
        pass