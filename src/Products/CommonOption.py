from Products.Enums import OptionType
from Products.EuropeanCustomOption import EuropeanCustomOption

class Straddle(EuropeanCustomOption):
    def __init__(self, K, T, quantity=1):
        booked = {
            OptionType.CALL.value: {
                (K, T): quantity
            },
            OptionType.PUT.value: {
                (K, T): quantity
            }
        }
        super().__init__(booked)

    def __repr__(self) -> str:
        return f"Straddle with K={self.list_calls[0].K}, T={self.list_calls[0].T}, quantity={self.list_calls[0].quantity}"

class Butterfly(EuropeanCustomOption):
    def __init__(self, K, T, quantity=1, epsilon = 0.05):
        deltaK = K * epsilon 
        booked = {
            OptionType.CALL.value: {
                (K, T): (-2 * quantity)/ deltaK**2,
                (K + deltaK , T): quantity / deltaK**2,
                (K - deltaK , T): quantity / deltaK**2
            }
        }
        super().__init__(booked) 

    def __repr__(self) -> str:
        return f"Butterfly with K={self.list_calls[0].K}, T={self.list_calls[0].T}, quantity={self.list_calls[0].quantity}"      

class Digital(EuropeanCustomOption):
    def __init__(self, K, T, quantity=1, epsilon = 0.01, optiontype = OptionType.CALL):
        dealway = 1 if optiontype.value == OptionType.CALL.value else -1
        deltaK = K * epsilon
        booked = {
            optiontype.value: {
                (K + deltaK, T): - dealway * quantity / (2 * deltaK),
                (K - deltaK, T): dealway * quantity / (2 * deltaK)
            }
        }
        super().__init__(booked)

    def __repr__(self) -> str:
        return f"Digital with K={self.list_calls[0].K if self.list_calls else self.list_puts[0].K}, T={self.list_calls[0].T if self.list_calls else self.list_puts[0].T}, quantity={self.list_calls[0].quantity if self.list_calls else self.list_puts[0].quantity}"

class RiskReversal(EuropeanCustomOption):
    def __init__(self, K1, K2, T, quantity=1):
        booked = {
            OptionType.CALL.value: {
                (K2, T): quantity
            },
            OptionType.PUT.value: {
                (K1, T): -quantity
            }
        }
        super().__init__(booked)
    
    def __repr__(self) -> str:
        return f"RiskReversal with K1={self.list_puts[0].K}, K2={self.list_calls[0].K}, T={self.list_calls[0].T}, quantity={self.list_calls[0].quantity}"