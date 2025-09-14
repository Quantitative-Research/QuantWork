import numpy as np
from .Numericals.Black76 import Black76
from .PricingModel import PricingModel
from MarketDataLoader.MarketFromExcel import Market
from Products.Enums import OptionType
from Products.Option import VanillaOption

class BlackModel(PricingModel):
    def __init__(self, market: Market):
        super().__init__(market)

    def GeneratePaths(self, number_paths : int, T : float, frequency): # override
        pass
    
    def AvailablePricingMethod(self): # override
        return super().AvailablePricingMethod()
    
    def _PriceVanillaOption(self, K, T, OptionType : OptionType = OptionType.CALL):
        
        vol = self.market.get_volatility(K, T)
        F = self.market.get_forward(T)
        DiscountFactor = self.market.get_DiscountFactor(T)
        if DiscountFactor is None:
            DiscountFactor = 1
        if OptionType == OptionType.CALL:
            return DiscountFactor * Black76(F = F, T = T).call_price(K = K, sigma = vol)
        elif OptionType == OptionType.PUT:
            return DiscountFactor * Black76(F = F, T = T).put_price(K = K, sigma = vol)
        else:
            raise NotImplementedError(f"Can not price {OptionType} with current method")

    def PriceVanillaOption(self, vanilla : VanillaOption):
        K, T, option_type, quantity = vanilla.K, vanilla.T, vanilla.option_type, vanilla.quantity
        return self._PriceVanillaOption(K, T, option_type) * quantity

    def PriceAndGreeksVanillaOption(self, vanilla : VanillaOption):
        K, T, option_type, quantity = vanilla.K, vanilla.T, vanilla.option_type, vanilla.quantity
        pass