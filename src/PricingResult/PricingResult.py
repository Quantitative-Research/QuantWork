from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PricingResult:
    price: float
    delta: float
    vega: float
    theta: float
    gamma: float
    vanna: float
    volga: float

    # ---------- algebra ----------

    def __add__(self, other: PricingResult) -> PricingResult:
        if not isinstance(other, PricingResult):
            return NotImplemented

        return PricingResult(
            price=self.price + other.price,
            delta=self.delta + other.delta,
            vega=self.vega + other.vega,
            theta=self.theta + other.theta,
            gamma=self.gamma + other.gamma,
            vanna=self.vanna + other.vanna,
            volga=self.volga + other.volga,
        )

    def __mul__(self, factor: float) -> PricingResult:
        if not isinstance(factor, (int, float)):
            return NotImplemented

        return PricingResult(
            price=self.price * factor,
            delta=self.delta * factor,
            vega=self.vega * factor,
            theta=self.theta * factor,
            gamma=self.gamma * factor,
            vanna=self.vanna * factor,
            volga=self.volga * factor,
        )

    __rmul__ = __mul__

    # ---------- helpers ----------

    def to_dict(self) -> Dict[str, float]:
        return {
            "price": self.price,
            "delta": self.delta,
            "vega": self.vega,
            "theta": self.theta,
            "gamma": self.gamma,
            "vanna": self.vanna,
            "volga": self.volga,
        }
