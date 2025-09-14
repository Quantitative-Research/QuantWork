class PricingResult:
    def __init__(self, price, delta, vega, theta, gamma, vanna, volga):
        self.price = price
        self.delta = delta
        self.vega = vega
        self.theta = theta
        self.gamma = gamma
        self.vanna = vanna
        self.volga = volga

    def __str__(self):
        return (f"PricingResult(price={self.price}, delta={self.delta}, vega={self.vega}, "
                f"theta={self.theta}, gamma={self.gamma}, vanna={self.vanna}, volga={self.volga})")

    def to_dict(self):
        return {
            "price": self.price,
            "delta": self.delta,
            "vega": self.vega,
            "theta": self.theta,
            "gamma": self.gamma,
            "vanna": self.vanna,
            "volga": self.volga
        }
