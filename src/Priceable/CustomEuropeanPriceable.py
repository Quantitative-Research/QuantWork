from Products.EuropeanCustom import EuropeanCustomOption
from Models.BlackModel import BlackModel
from Priceable.Priceable import Priceable  
class CustomEuropeanPriceable(Priceable):
    def __init__(self, pricing_model: BlackModel, european_custom_option: EuropeanCustomOption):
        super().__init__(pricing_model, european_custom_option)

    def price(self) -> float:
        total_price = 0.0
        for vanilla_option in self.option.list_calls:
            total_price += self.model.PriceVanillaOption(vanilla_option) 
        for vanilla_option in self.option.list_puts:
            total_price += self.model.PriceVanillaOption(vanilla_option)
        return total_price
    
    def price_and_greeks(self):
        pass
