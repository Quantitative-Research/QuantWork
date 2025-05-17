from abc import ABC, abstractmethod
from .Enums import OptionType
from Models.ModelType import ModelType
# === Abstract Option class ===

class Option(ABC):
    def __init__(self, K, T, option_type=OptionType.CALL, quantity = 1):
        self.K = K
        self.T = T
        self.option_type = option_type
        self.quantity = quantity

# === Vanilla Option class ===

class VanillaOption(Option):
    def __init__(self, K, T, option_type=OptionType.CALL, quantity = 1):
        super().__init__(K, T, option_type, quantity)



    
