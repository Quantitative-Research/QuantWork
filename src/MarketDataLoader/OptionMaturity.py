from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize, Bounds

from .convex_projection import project_convex, convex_monotone_projection
from .plot_helpers import show_plot

logger = logging.getLogger(__name__)


@dataclass
class OutlierConfig:
    iv_min: float = 1e-3
    iv_max: float = 5.0
    min_open_interest: int = 0     # set >0 if you want a liquidity floor
    min_volume: int = 0            # set >0 if you want a liquidity floor
    roll_window: int = 5           # odd integer, median/MAD window (in index space)
    mad_thresh: float = 3.0        # tau in |x - med| > tau*MAD
    max_local_slope: float = 5.0   # cap on |d sigma / d k| to prune jagged jumps


class OptionMaturity:
    """
    Container for a single expiry's option data (calls/puts) with cleaning,
    convex regularization (IV and prices), and forward/discount extraction.

    Attributes
    ----------
    expiry : pd.Timestamp
        Expiry date.
    spot_price : float
        Underlying spot used for OTM split and moneyness.
    calls, puts : pd.DataFrame
        Raw option tables from yfinance (copied on init).
    cleaned_smile : Optional[pd.DataFrame]
        Cleaned OTM IV slice (strike, impliedVolatility).
    cleaned_smile_hat : Optional[pd.DataFrame]
        Convex-projected IV slice (strike, impliedVolatility).
    calls_convex, puts_convex : Optional[pd.DataFrame]
        Convex + monotone regularized price curves with column 'price_convex'.
    """

    expiry: pd.Timestamp
    spot_price: float
    calls: pd.DataFrame
    puts: pd.DataFrame
    cleaned_smile: Optional[pd.DataFrame]
    cleaned_smile_hat: Optional[pd.DataFrame]
    calls_convex: Optional[pd.DataFrame]
    puts_convex: Optional[pd.DataFrame]

    def __init__(
        self,
        expiry: pd.Timestamp | str,
        calls: pd.DataFrame,
        puts: pd.DataFrame,
        spot_price: float,
        convex_iv_method: str = "slopes",
        max_iv_jump: float = 5.0,
    ) -> None:
        self.expiry = pd.to_datetime(expiry)
        self.calls = calls.copy()
        self.puts = puts.copy()
        self.spot_price = float(spot_price)

        self.cleaned_smile = None
        self.cleaned_smile_hat = None
        self.calls_convex = None
        self.puts_convex = None

        self._ensure_numeric_columns()
        self._clean_option_tables()

        # Build IV smile (clean + convex projection)
        self._build_cleaned_smile(max_iv_jump=max_iv_jump)
        self._project_convex_smile(method=convex_iv_method)

        # Build convex price curves (for ZC/forward via parity)
        self.build_convex_prices()

    # ------------------------------------------------------------------
    # Internal cleaning helpers
    # ------------------------------------------------------------------
# --- inside OptionMaturity -------------------------------------------
    def _otm_iv_with_liquidity(self) -> pd.DataFrame:
        """
        OTM IV slice with optional liquidity columns if present:
        returns columns: ['strike','impliedVolatility','k','openInterest','volume']
        """
        df = self.otm_iv_df.copy()
        if df.empty:
            return df

        # attach OI/volume if they exist on both sides; default 0
        def pick_cols(src, name):
            keep = ["strike", "impliedVolatility", "openInterest", "volume"]
            cols = [c for c in keep if c in src.columns]
            out = src[cols].copy()
            out.rename(columns={"impliedVolatility": f"iv_{name}"}, inplace=True)
            return out

        calls = self.calls[self.calls["strike"] > self.spot_price]
        puts  = self.puts[self.puts["strike"] < self.spot_price]

        # merge liquidity by strike (outer, then fillna)
        liq = None
        if not calls.empty:
            liq = pick_cols(calls, "call")
        if not puts.empty:
            liq_put = pick_cols(puts, "put")
            liq = liq_put if liq is None else pd.merge(liq, liq_put, on="strike", how="outer")

        # base OTM IV
        df = df[["strike", "impliedVolatility"]].copy()
        if liq is not None:
            df = pd.merge(df, liq, on="strike", how="left")

        # consolidate OI/volume as max of available (call/put) on each strike
        for col in ("openInterest", "volume"):
            c1 = f"{col}_call"
            c2 = f"{col}_put"
            if c1 in df.columns or c2 in df.columns:
                df[col] = df[[c for c in (c1, c2) if c in df.columns]].max(axis=1, skipna=True)
            else:
                df[col] = 0

        df["k"] = np.log(df["strike"].values / self.spot_price)
        return df[["strike", "impliedVolatility", "k", "openInterest", "volume"]].dropna()


    def _ensure_numeric_columns(self) -> None:
        """Coerce relevant numeric columns if present."""
        for df in (self.calls, self.puts):
            for col in ("strike", "lastPrice", "impliedVolatility"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

    def _clean_option_tables(self) -> None:
        """
        Basic cleaning: drop NaNs in strikes, keep positive prices/IV when present.
        """
        for name, df in (("calls", self.calls), ("puts", self.puts)):
            before = len(df)
            df.dropna(subset=["strike"], inplace=True)
            # Keep rows where at least one of price/IV is present
            if "lastPrice" in df.columns:
                df = df[df["lastPrice"].notna()]
            if "impliedVolatility" in df.columns:
                df = df[df["impliedVolatility"].notna()]
            # Prices strictly positive if present
            if "lastPrice" in df.columns:
                df = df[df["lastPrice"] > 0]
            # IV strictly positive if present
            if "impliedVolatility" in df.columns:
                df = df[df["impliedVolatility"] > 0]

            df.sort_values("strike", inplace=True)
            df.reset_index(drop=True, inplace=True)
            after = len(df)
            setattr(self, name, df)
            if before != after:
                logger.debug(
                    f"[{self.expiry.date()}] Cleaned {name}: {before} -> {after} rows."
                )

    def _robust_outlier_mask(self, df: pd.DataFrame, cfg: OutlierConfig) -> np.ndarray:
        """
        Returns boolean mask of *kept* points using IV-only robust rules.
        """
        if df.empty:
            return np.array([], dtype=bool)

        k = df["k"].values
        iv = df["impliedVolatility"].values

        keep = np.isfinite(iv) & (iv >= cfg.iv_min) & (iv <= cfg.iv_max)

        # Liquidity screens if desired
        if "openInterest" in df.columns and cfg.min_open_interest > 0:
            keep &= (df["openInterest"].fillna(0).values >= cfg.min_open_interest)
        if "volume" in df.columns and cfg.min_volume > 0:
            keep &= (df["volume"].fillna(0).values >= cfg.min_volume)

        # Sort by k for rolling ops
        order = np.argsort(k)
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        k_sorted = k[order]
        iv_sorted = iv[order]

        # Rolling median + MAD in index space
        w = max(3, int(cfg.roll_window) | 1)  # force odd >=3
        half = w // 2
        med = np.empty_like(iv_sorted)
        mad = np.empty_like(iv_sorted)

        for i in range(len(iv_sorted)):
            lo = max(0, i - half)
            hi = min(len(iv_sorted), i + half + 1)
            window = iv_sorted[lo:hi]
            m = np.median(window)
            med[i] = m
            mad[i] = np.median(np.abs(window - m)) + 1e-12

        # MAD rule
        z = np.abs(iv_sorted - med) / mad
        keep_sorted = (z <= cfg.mad_thresh)

        # Local slope (finite difference) on sorted grid: |d sigma / d k|
        if len(iv_sorted) >= 3:
            dk = np.gradient(k_sorted)
            ds = np.gradient(iv_sorted)
            slope = np.abs(ds / (dk + 1e-12))
            keep_sorted &= (slope <= cfg.max_local_slope)

        # map back to original order and combine with base keep
        keep_final = np.zeros_like(keep, dtype=bool)
        keep_final[order] = keep_sorted
        keep &= keep_final

        # ensure we keep at least a minimal core near ATM (k≈0) if available
        if keep.sum() >= 3:
            pass
        else:
            # fallback: keep 3 closest to k=0 (best effort)
            idx = np.argsort(np.abs(k))
            keep = np.zeros_like(keep, dtype=bool)
            keep[idx[:min(3, len(idx))]] = True

        return keep

    # ------------------------------------------------------------------
    # IV smile construction (OTM slice -> cleaned -> convex)
    # ------------------------------------------------------------------
    @property
    def otm_iv_df(self) -> pd.DataFrame:
        """
        OTM slice (calls: K>spot, puts: K<spot), cleaned and sorted.

        Returns
        -------
        pd.DataFrame with columns ['strike', 'impliedVolatility'] sorted by strike.
        """
        calls = self.calls
        puts = self.puts
        if calls.empty and puts.empty:
            return pd.DataFrame(columns=["strike", "impliedVolatility"])

        otm_calls = calls[calls["strike"] > self.spot_price][["strike", "impliedVolatility"]]
        otm_puts = puts[puts["strike"] < self.spot_price][["strike", "impliedVolatility"]]
        data = pd.concat([otm_calls, otm_puts], ignore_index=True)

        data = data.dropna(subset=["impliedVolatility"])
        data = data[data["impliedVolatility"] > 0]
        return data.sort_values("strike").reset_index(drop=True)

    def _remove_arbitrage(self, df: pd.DataFrame, max_iv_jump: float) -> pd.DataFrame:
        """
        Simple de-arbitrage: remove extreme local jumps in IV to smooth noise.

        Parameters
        ----------
        df : DataFrame with columns ['strike','impliedVolatility']
        max_iv_jump : float
            Maximum allowed absolute gradient between adjacent IV points.

        Returns
        -------
        DataFrame filtered by gradient threshold.
        """
        if df.empty:
            return df
        out = df.copy()
        iv = out["impliedVolatility"].values
        # gradient on an irregular grid is approximate — adequate for outlier pruning
        grad = np.gradient(iv)
        mask = np.isfinite(iv) & (iv > 0) & (np.abs(grad) < max_iv_jump)
        return out.loc[mask].reset_index(drop=True)

    def _build_cleaned_smile(self, max_iv_jump: float) -> None:
        """Build and store cleaned OTM smile slice."""
        base = self.otm_iv_df
        self.cleaned_smile = self._remove_arbitrage(base, max_iv_jump=max_iv_jump)
        if self.cleaned_smile is None or self.cleaned_smile.empty:
            logger.debug(f"[{self.expiry.date()}] No valid OTM IV data after cleaning.")

    def _project_convex_smile(self, method: str = "slopes") -> None:
        """
        Project cleaned smile onto the convex cone (in strike).

        Parameters
        ----------
        method : str
            Projection method used by `project_convex` (e.g. 'slopes').
        """
        if self.cleaned_smile is None or self.cleaned_smile.empty:
            self.cleaned_smile_hat = None
            return

        K = self.cleaned_smile["strike"].values
        sigma = self.cleaned_smile["impliedVolatility"].values
        try:
            sigma_hat = project_convex(K, sigma, method=method)
            smile_hat = self.cleaned_smile.copy()
            smile_hat["impliedVolatility"] = sigma_hat
            self.cleaned_smile_hat = smile_hat
        except Exception as e:
            logger.warning(f"[{self.expiry.date()}] Convex projection failed: {e}")
            self.cleaned_smile_hat = self.cleaned_smile.copy()

    def _fit_svi(self, k: np.ndarray, iv: np.ndarray, T_years: float) -> Optional[tuple]:
        """
        Fit raw SVI to total variance w = iv^2 * T.
        Returns params (a,b,rho,m,sigv) or None on failure.
        """
        if len(k) < 5:
            return None
        w_obs = (iv**2) * T_years

        # initial guesses
        a0 = max(1e-6, np.percentile(w_obs, 10))
        b0 = max(1e-4, (np.percentile(w_obs, 90) - np.percentile(w_obs, 10)) / (np.ptp(k) + 1e-6))
        rho0 = np.clip(np.corrcoef(k, w_obs)[0,1], -0.5, 0.5) if np.isfinite(np.corrcoef(k, w_obs)[0,1]) else 0.0
        m0 = np.median(k)
        sigv0 = max(1e-3, 0.2 * (np.std(k) + 1e-6))

        x0 = np.array([a0, b0, rho0, m0, sigv0])

        def svi_w(x, kk):
            a, b, rho, m, sigv = x
            return a + b * (rho * (kk - m) + np.sqrt((kk - m)**2 + sigv**2))

        def obj(x):
            a, b, rho, m, sigv = x
            # hard constraints via penalty (in addition to bounds)
            if b <= 0 or sigv <= 0 or not (-0.999 < rho < 0.999) or a < 0:
                return 1e6
            w_model = svi_w(x, k)
            # robust (Huber-like) loss
            resid = w_model - w_obs
            c = 3.0 * np.median(np.abs(resid) + 1e-12)
            huber = np.where(np.abs(resid) <= c, 0.5 * resid**2, c*(np.abs(resid) - 0.5*c))
            return np.sum(huber)

        bounds = Bounds([0.0, 1e-8, -0.999, np.min(k)-1.0, 1e-6],
                        [10.0, 10.0, 0.999, np.max(k)+1.0, 5.0])

        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200})
        if not (res.success and np.isfinite(res.fun)):
            return None
        return tuple(res.x)

    def _build_regularized_smile(self,
                                cfg: OutlierConfig = OutlierConfig(),
                                n_grid: int = 41) -> Optional[pd.DataFrame]:
        """
        Main entry: filter outliers, then fit SVI (fallback PCHIP).
        Returns DataFrame with ['strike','k','iv'] on a dense grid.
        """
        raw = self._otm_iv_with_liquidity()
        if raw.empty:
            self.cleaned_smile = None
            self.cleaned_smile_hat = None
            return None

        mask = self._robust_outlier_mask(raw, cfg)
        filt = raw.loc[mask].sort_values("k").reset_index(drop=True)

        if len(filt) < 3:
            # Not enough points -> take what we can and linear interpolate in k
            x = filt["k"].values
            y = filt["impliedVolatility"].values
            if len(x) == 0:
                self.cleaned_smile = None
                self.cleaned_smile_hat = None
                return None
            k_grid = np.linspace(np.min(x), np.max(x), max(3, n_grid))
            iv_grid = np.interp(k_grid, x, y)
            strike = self.spot_price * np.exp(k_grid)
            out = pd.DataFrame({"strike": strike, "k": k_grid, "iv": iv_grid})
            self.cleaned_smile = filt[["strike","impliedVolatility"]].rename(columns={"impliedVolatility":"iv"})
            self.cleaned_smile_hat = out.rename(columns={"iv":"impliedVolatility"})
            return out

        # Try SVI on filtered points
        today = pd.Timestamp.today().normalize()
        T_years = max((self.expiry - today).days, 1) / 365.0

        x = filt["k"].values
        y = filt["impliedVolatility"].values
        params = self._fit_svi(x, y, T_years)

        k_grid = np.linspace(np.min(x), np.max(x), n_grid)
        if params is not None:
            a, b, rho, m, sigv = params
            w = a + b * (rho * (k_grid - m) + np.sqrt((k_grid - m)**2 + sigv**2))
            iv_grid = np.sqrt(np.maximum(w, 1e-12) / T_years)
        else:
            # Fallback: PCHIP in IV space
            spline = PchipInterpolator(x, y)
            iv_grid = spline(k_grid)

        strike = self.spot_price * np.exp(k_grid)

        # Save both the filtered raw and the smooth result
        self.cleaned_smile = filt[["strike","impliedVolatility"]].rename(columns={"impliedVolatility":"iv"})
        self.cleaned_smile_hat = pd.DataFrame({
            "strike": strike,
            "k": k_grid,
            "impliedVolatility": iv_grid
        })
        return self.cleaned_smile_hat

    def rework_smile_regularization(self,
                                    cfg: OutlierConfig = OutlierConfig(),
                                    n_grid: int = 41) -> Optional[pd.DataFrame]:
        """
        Public API: rebuild cleaned + regularized OTM smile for THIS maturity.
        """
        return self._build_regularized_smile(cfg=cfg, n_grid=n_grid)

    # ------------------------------------------------------------------
    # Convex + monotone price regularization (per instrument)
    # ------------------------------------------------------------------
    def build_convex_prices(self) -> None:
        """
        Build convex-monotone regularized call/put price curves on their own strikes.

        Produces
        --------
        self.calls_convex, self.puts_convex : DataFrames with new column 'price_convex'
        """
        def _prep(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            d = df.dropna(subset=["strike", "lastPrice"]).copy()
            d = d[d["lastPrice"] > 0].sort_values("strike").reset_index(drop=True)
            return d

        calls = _prep(self.calls)
        puts = _prep(self.puts)

        if not calls.empty:
            try:
                if len(calls) >= 3:
                    calls["price_convex"] = convex_monotone_projection(
                        calls["strike"].values,
                        calls["lastPrice"].values,
                        monotone="nonincreasing",  # C(K) non-increasing in K
                    )
                else:
                    calls["price_convex"] = calls["lastPrice"].values
            except Exception as e:
                logger.warning(f"[{self.expiry.date()}] Call convex projection failed: {e}")
                calls["price_convex"] = calls["lastPrice"].values

        if not puts.empty:
            try:
                if len(puts) >= 3:
                    puts["price_convex"] = convex_monotone_projection(
                        puts["strike"].values,
                        puts["lastPrice"].values,
                        monotone="nondecreasing",  # P(K) non-decreasing in K
                    )
                else:
                    puts["price_convex"] = puts["lastPrice"].values
            except Exception as e:
                logger.warning(f"[{self.expiry.date()}] Put convex projection failed: {e}")
                puts["price_convex"] = puts["lastPrice"].values

        self.calls_convex = calls
        self.puts_convex = puts

    # ------------------------------------------------------------------
    # Forward & discount extraction
    # ------------------------------------------------------------------
    def parity_regression_all_strikes(self) -> Tuple[float, float]:
        """
        Use convex-regularized prices across all *common* strikes to estimate (B0T, F_T)
        via least-squares fit of:  C(K) - P(K) = B0T * (F_T - K)

        Returns
        -------
        (B0T, F_T) : tuple of floats (np.nan, np.nan if insufficient/invalid)
        """
        if self.calls_convex is None or self.puts_convex is None:
            self.build_convex_prices()

        calls = self.calls_convex[["strike", "price_convex"]].rename(columns={"price_convex": "C"})
        puts = self.puts_convex[["strike", "price_convex"]].rename(columns={"price_convex": "P"})
        merged = pd.merge(calls, puts, on="strike", how="inner").sort_values("strike")

        if len(merged) < 2:
            return np.nan, np.nan

        K = merged["strike"].values
        y = (merged["C"] - merged["P"]).values

        # OLS: y = a + b K  => B0T = -b, FT = a/B0T
        A = np.vstack([np.ones_like(K), K]).T
        try:
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
            a, b = sol
            B0T = -b
            if not np.isfinite(B0T) or B0T <= 0:
                return np.nan, np.nan
            B0T = min(B0T, 1.0)  # optional cap (positive rate assumption)
            FT = a / B0T if B0T > 0 else np.nan
            return float(B0T), float(FT)
        except Exception as e:
            logger.warning(f"[{self.expiry.date()}] Parity regression failed: {e}")
            return np.nan, np.nan

    def forward_from_cp_zero_crossing(self) -> float:
        """
        Forward from the zero-crossing of C(K) - P(K) on *raw* lastPrice.
        Returns np.nan if no sign change (the slice should be dropped by caller).
        """
        calls = self.calls[["strike", "lastPrice"]].rename(columns={"lastPrice": "C"}).copy()
        puts  = self.puts[["strike", "lastPrice"]].rename(columns={"lastPrice": "P"}).copy()

        # basic guards
        calls = calls.dropna().query("strike > 0 and C > 0").sort_values("strike")
        puts  = puts.dropna().query("strike > 0 and P > 0").sort_values("strike")

        merged = pd.merge(calls, puts, on="strike", how="inner").sort_values("strike").reset_index(drop=True)
        if len(merged) < 2:
            return float("nan")

        merged["cp_diff"] = merged["C"] - merged["P"]
        K = merged["strike"].values
        D = merged["cp_diff"].values

        # candidates (K*, score) where score ~ closeness to zero around the crossing
        candidates = []

        # exact zeros
        exact_idx = np.where(np.isclose(D, 0.0))[0]
        for j in exact_idx:
            candidates.append((float(K[j]), 0.0))

        # sign changes -> linear interpolation between neighbors
        for i in range(1, len(merged)):
            d1, d2 = D[i-1], D[i]
            if d1 * d2 < 0:  # strict sign change
                k1, k2 = K[i-1], K[i]
                # linear interpolation for root
                k_star = k1 - d1 * (k2 - k1) / (d2 - d1)
                score = abs(d1) + abs(d2)  # smaller is better
                candidates.append((float(k_star), float(score)))

        if not candidates:
            return float("nan")

        # pick the most credible crossing (smallest |d1|+|d2| or exact zero)
        k_star = min(candidates, key=lambda x: x[1])[0]
        return float(k_star)

    # Optional: make estimate_forward() use only the zero-crossing
    def estimate_forward(self) -> float:
        return self.forward_from_cp_zero_crossing()

    # ------------------------------------------------------------------
    # Convenience & metrics
    # ------------------------------------------------------------------
    @property
    def mean_otm_iv(self) -> float:
        """Average OTM IV (NaN if not available)."""
        df = self.otm_iv_df
        return float(df["impliedVolatility"].mean()) if not df.empty else float("nan")

    def get_strikes(self) -> Tuple[list[float], list[float]]:
        return self.calls["strike"].tolist(), self.puts["strike"].tolist()

    def verify_convexity_iv(self, atol: float = 1e-8) -> bool:
        """
        Check convexity of the convex-projected IV smile via discrete second differences.
        """
        if self.cleaned_smile_hat is None or self.cleaned_smile_hat.empty:
            return True
        v = self.cleaned_smile_hat["impliedVolatility"].values
        if len(v) < 3:
            return True
        second = v[2:] - 2 * v[1:-1] + v[:-2]
        return bool((second >= -atol).all())

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_smile(self, option_type: str = "call", x_axis: str = "strike") -> None:
        """
        Plot raw IV smile (calls/puts/OTM).

        Parameters
        ----------
        option_type : {'call','put','OTM'}
        x_axis : {'strike','moneyness'}
        """
        if option_type == "call":
            data = self.calls
        elif option_type == "put":
            data = self.puts
        elif option_type == "OTM":
            data = self.otm_iv_df
        else:
            raise ValueError("option_type must be 'call', 'put', or 'OTM'")

        if data.empty or "impliedVolatility" not in data.columns:
            logger.info(f"[{self.expiry.date()}] No data to plot.")
            return

        strikes = data["strike"].values
        iv = data["impliedVolatility"].values

        if x_axis == "strike":
            x_values = strikes
            x_label = "Strike"
        elif x_axis == "moneyness":
            x_values = strikes / self.spot_price
            x_label = "Moneyness (K/S₀)"
        else:
            raise ValueError("x_axis must be 'strike' or 'moneyness'")

        plt.figure(figsize=(10, 6))
        plt.plot(x_values, iv, marker="o", linestyle="-", label=f"{option_type} IV")
        if x_axis == "moneyness":
            plt.axvline(1.0, color="red", linestyle="--", label="ATM (K/S₀=1)")
        else:
            plt.axvline(self.spot_price, color="red", linestyle="--", label="Spot")
        plt.title(f"{option_type} Volatility Smile - Expiry {self.expiry.date()}")
        plt.xlabel(x_label)
        plt.ylabel("Implied Volatility")
        plt.legend()
        plt.grid(True)
        show_plot()

    def plot_smile_clean(self) -> None:
        """Plot raw OTM IV, cleaned IV, and convex-projected IV."""
        if self.cleaned_smile is None or self.cleaned_smile.empty:
            logger.info(f"[{self.expiry.date()}] No valid OTM IV for plotting.")
            return

        plt.figure(figsize=(10, 6))

        raw = self.otm_iv_df
        if not raw.empty:
            plt.scatter(raw["strike"], raw["impliedVolatility"], alpha=0.6, label="Raw OTM")

        plt.plot(
            self.cleaned_smile["strike"],
            self.cleaned_smile["impliedVolatility"],
            label="Cleaned",
        )

        if self.cleaned_smile_hat is not None and not self.cleaned_smile_hat.empty:
            plt.plot(
                self.cleaned_smile_hat["strike"],
                self.cleaned_smile_hat["impliedVolatility"],
                linestyle="--",
                label="Convex",
            )

        plt.axvline(self.spot_price, color="black", linestyle="--", label="Spot")
        plt.title(f"Clean OTM Smile - Expiry {self.expiry.date()}")
        plt.xlabel("Strike")
        plt.ylabel("Implied Volatility")
        plt.legend()
        plt.grid(True)
        show_plot()

    def plot_smile_convex(self) -> None:
        """Plot convex-regularized price curves for calls and puts."""
        if self.calls_convex is None or self.puts_convex is None:
            logger.info(f"[{self.expiry.date()}] No convex prices to plot.")
            return

        plt.figure(figsize=(10, 6))

        if not self.calls_convex.empty:
            plt.plot(
                self.calls_convex["strike"],
                self.calls_convex["price_convex"],
                label="Call Price (Convex)",
            )

        if not self.puts_convex.empty:
            plt.plot(
                self.puts_convex["strike"],
                self.puts_convex["price_convex"],
                label="Put Price (Convex)",
            )

        plt.axvline(self.spot_price, color="black", linestyle="--", label="Spot")
        plt.title(f"Convex Price Curves - Expiry {self.expiry.date()}")
        plt.xlabel("Strike")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        show_plot()

