from Products.Enums import OptionType
from Products.EuropeanCustomOption import EuropeanCustomOption

class Straddle(EuropeanCustomOption):
    def __init__(self, K, T, quantity=1):
        booked = {
            OptionType.CALL: {
                (K, T): quantity
            },
            OptionType.PUT: {
                (K, T): quantity
            }
        }
        super().__init__(booked)

class Butterfly(EuropeanCustomOption):
    def __init__(self, K, T, quantity=1, epsilon = 0.05):
        deltaK = K * epsilon 
        booked = {
            OptionType.CALL: {
                (K, T): (-2 * quantity)/ deltaK**2,
                (K + deltaK , T): quantity / deltaK**2,
                (K - deltaK , T): quantity / deltaK**2
            }
        }
        super().__init__(booked)       

class Digital(EuropeanCustomOption):
    def __init__(self, K, T, quantity=1, epsilon = 0.01, optiontype = OptionType.CALL):
        dealway = 1 if optiontype is OptionType.CALL else -1
        deltaK = K * epsilon
        booked = {
            optiontype: {
                (K + deltaK, T): - dealway * quantity / (2 * deltaK),
                (K - deltaK, T): dealway * quantity / (2 * deltaK)
            }
        }
        super().__init__(booked)