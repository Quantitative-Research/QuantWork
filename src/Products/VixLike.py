from Products.EuropeanCustomOption import EuropeanCustomOption
from Products.Enums import OptionType

class VixLike(EuropeanCustomOption):
    def __init__(self, T: float = 30/365, forward_level : float = 100.0, K_min: float = 0.8, K_max: float = 1.2, number_of_strikes: int = 20, quantity: int =1):
        """
        Create a VIX like product by booking a grid of vanilla options
        """

        n_call = int(number_of_strikes / 2)
        n_put = number_of_strikes - n_call

        # Create a strike grid around the forward level for OTM options
        put_grid = [forward_level * (K_min + i * (1 - K_min) / (n_put - 1)) for i in range(n_put)]
        call_grid = [forward_level * (K_max - i * (K_max - 1) / (n_call - 1)) for i in range(n_call)]

        interstrike = call_grid[0] - put_grid[-1]
        strike_grid = put_grid + call_grid

        booked = {
            OptionType.CALL: {
                (K, T): 2 * quantity * interstrike/ (K * K * T ) for K in call_grid
            },
            OptionType.PUT: {
                (K, T): 2 * quantity * interstrike / (K * K * T ) for K in put_grid
            }
        }

        super().__init__(booked)