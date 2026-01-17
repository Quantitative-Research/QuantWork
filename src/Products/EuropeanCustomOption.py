from Products.Enums import OptionType
from Products.Option import VanillaOption
class EuropeanCustomOption:
    def __init__(self, dict_booked_option: dict):
        calls = dict_booked_option.get(OptionType.CALL, {})
        puts = dict_booked_option.get(OptionType.PUT, {})

        self.list_calls = [
            VanillaOption(K=K, T=T, option_type=OptionType.CALL, quantity=qty)
            for (K, T), qty in calls.items()
        ]
        self.list_puts = [
            VanillaOption(K=K, T=T, option_type=OptionType.PUT, quantity=qty)
            for (K, T), qty in puts.items()
        ]
