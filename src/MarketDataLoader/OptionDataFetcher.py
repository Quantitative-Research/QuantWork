from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd
import numpy as np
import yfinance as yf
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from .plot_helpers import show_plot

from .OptionMaturity import OptionMaturity  

logger = logging.getLogger(__name__)

class OptionDataFetcher:
    """
    Fetches and processes option market data for a given ticker.
    Builds forward curve, zero-coupon curve, and convex-projected volatility surface.
    """

    ticker: str
    spot_price: Optional[float]
    option_maturities: Dict[pd.Timestamp, OptionMaturity]
    vol_surface: Optional[pd.DataFrame]
    forward_curve: Optional[pd.DataFrame]
    zc_curve: Optional[pd.DataFrame]
    forward_interpolator: Optional[interp1d]

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.option_maturities = {}
        self.spot_price = None
        self.forward_curve = None
        self.zc_curve = None
        self.vol_surface = None
        self.forward_interpolator = None

    # ----------------------
    # Public API
    # ----------------------
    def build_market(self, fetch: bool = True) -> None:
        """
        Full market build: fetch data, build forward curve, vol surface, and ZC curve.

        Args:
            fetch: If True, re-fetch options from Yahoo Finance, else use existing data.
        """
        if fetch:
            self.fetch_options()
        self.build_forward_curve()
        self.build_vol_surface()
        self.build_ZC_curve()

    def get_maturities(self) -> list[pd.Timestamp]:
        """Return list of available maturities."""
        return list(self.option_maturities.keys())  

    def get_forward(self, T: pd.Timestamp) -> float:
        """
        Get forward price for a given maturity T.
        Uses linear interpolation if T is not directly available.
        """
        if self.forward_interpolator is None:
            raise RuntimeError("Forward curve not built. Call build_market() first.")
        
        days_to_expiry = (T - pd.Timestamp.today()).days
        return float(self.forward_interpolator(days_to_expiry))

    def get_volatility(self, K: float, T: pd.Timestamp) -> float:
        """
        Get implied volatility for a given strike K and maturity T.
        Uses linear interpolation on the volatility surface.
        """
        if self.vol_surface is None:
            raise RuntimeError("Volatility surface not built. Call build_market() first.")
        
        if T not in self.vol_surface.columns:
            raise ValueError(f"No volatility data for maturity {T.date()}")

        vol_series = self.vol_surface[T]
        return float(np.interp(K, vol_series.index, vol_series.values))  
           
    def fetch_options(self) -> None:
        """Fetch options chain for all expiries from Yahoo Finance."""
        logger.info(f"Fetching option data for {self.ticker}...")
        stock = yf.Ticker(self.ticker)

        # Spot price: last close
        self.spot_price = stock.history(period='1d')['Close'].iloc[-1]

        expirations = stock.options
        for expiry_str in expirations:
            expiry = pd.to_datetime(expiry_str)
            opt_chain = stock.option_chain(expiry_str)
            maturity = OptionMaturity(
                expiry=expiry,
                calls=opt_chain.calls,
                puts=opt_chain.puts,
                spot_price=self.spot_price
            )
            self.option_maturities[expiry] = maturity

# --- in OptionDataFetcher ---------------------------------------------
    def build_forward_curve(self) -> None:
        """
        Build forward curve using ONLY the C-P=0 zero-crossing.
        Drop any expiry where no sign change occurs across common strikes.
        """
        logger.info("Building forward curve (zero-crossing only)...")
        rows = []
        to_drop = []

        # iterate over a snapshot of keys since we may delete
        for expiry in list(self.option_maturities.keys()):
            maturity = self.option_maturities[expiry]
            try:
                fwd = maturity.estimate_forward()
                if np.isfinite(fwd):
                    rows.append({"expiry": expiry, "forward": fwd})
                else:
                    logger.warning(f"No C-P sign change for {expiry.date()} — removing this maturity.")
                    to_drop.append(expiry)
            except Exception as e:
                logger.warning(f"Forward failed for {expiry.date()}: {e}")
                to_drop.append(expiry)

        # remove entire slices with no sign change
        for expiry in to_drop:
            self.option_maturities.pop(expiry, None)

        if not rows:
            logger.error("No valid forwards from zero-crossing. Forward curve not built.")
            self.forward_curve = None
            self.forward_interpolator = None
            return

        df = pd.DataFrame(rows).sort_values("expiry").set_index("expiry")
        self.forward_curve = df

        # Interpolator on calendar days
        today = pd.Timestamp.today().normalize()
        expiry_days = (df.index - today).days.values
        self.forward_interpolator = interp1d(
            expiry_days, df["forward"].values, kind='linear', fill_value="extrapolate"
        )

    def build_vol_surface(self) -> Optional[pd.DataFrame]:
        """
        Build a volatility surface matrix from convex-projected smiles.

        Returns:
            DataFrame: vol surface indexed by strikes, columns = maturities
        """
        logger.info("Building volatility surface from convex smiles...")

        if not self.option_maturities:
            logger.error("No maturities found. Run fetch_options() first.")
            return None

        # Gather all strikes across maturities
        all_strikes = sorted({
            strike
            for maturity in self.option_maturities.values()
            if maturity.cleaned_smile_hat is not None
            for strike in maturity.cleaned_smile_hat['strike'].values
        })

        if not all_strikes:
            logger.warning("No convex smiles available for vol surface.")
            return None

        vol_surface = pd.DataFrame(index=all_strikes)

        for expiry, maturity in self.option_maturities.items():
            if maturity.cleaned_smile_hat is None:
                continue
            smile = maturity.cleaned_smile_hat
            vol_series = pd.Series(
                data=smile['impliedVolatility'].values,
                index=smile['strike'].values
            )
            vol_surface[expiry] = vol_series.reindex(all_strikes)

        self.vol_surface = vol_surface.sort_index(axis=1)
        return self.vol_surface

    def build_ZC_curve(self) -> None:
        """
        Placeholder for building zero-coupon curve.
        Currently does not implement any logic.
        """
        logger.info("Building zero-coupon curve (not implemented).")
        # Implement ZC curve logic here if needed
        self.zc_curve = None    

    # ----------------------
    # Plotting
    # ----------------------
    def plot_vol_surface(self) -> None:
        """3D surface plot of volatility surface."""
        if self.vol_surface is None:
            logger.error("Vol surface not built.")
            return

        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(
            [exp.date() for exp in self.vol_surface.columns],
            self.vol_surface.index
        )
        Z = self.vol_surface.values

        ax.plot_surface(X, Y, Z, cmap="viridis")
        ax.set_xlabel("Maturity")
        ax.set_ylabel("Strike")
        ax.set_zlabel("Implied Volatility")
        ax.set_title(f"Volatility Surface for {self.ticker}")

        show_plot()

    def plot_forward_curve(self) -> None:
        """Plot the forward curve."""
        if self.forward_curve is None:
            logger.error("Forward curve not built.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.forward_curve.index, self.forward_curve['forward'], marker='o')
        plt.title(f"Forward Curve for {self.ticker}")
        plt.xlabel("Expiry Date")
        plt.ylabel("Forward Price")
        plt.grid()
        show_plot()

    def plot_ZC_curve(self) -> None:
        """Placeholder for ZC curve plotting."""
        if self.zc_curve is None:
            logger.error("Zero-coupon curve not built.")
            return

        # Implement ZC curve plotting logic here
        pass
