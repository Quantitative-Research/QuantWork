import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from datetime import datetime
import os


class Market:
    def __init__(self, excel_path: str, pricing_date: datetime):
        """
        Initialize market data from an Excel file.

        Excel format:
        - First column: Expiry dates (as string or datetime)
        - Second column: Forward prices
        - From third column onward:
            * First row: Absolute strikes
            * Headers: Moneyness (e.g. '100.0%')
            * Values: Implied volatilities
        """
        self._pricing_date = pricing_date
        self._xl_path = excel_path

        self._spot = None
        self._forward_curve = None
        self._forward_curve_interpolated = None
        self._vol_KT = None
        self._strikes = None
        self._maturities = None
        self._maturitiesAsDates = None

        self._load_market_data()

    def _load_market_data(self):
        df = pd.read_excel(self._xl_path, index_col=0)

        # Set pricing date as the first maturity row
        df.index = pd.to_datetime(df.index)
        df.index = df.index.where(df.index != df.index[0], self._pricing_date)

        self._maturitiesAsDates = df.index
        self._maturities = [(d - self._pricing_date).days / 365.25 for d in df.index]
        self._maturities[0] = 0

        forwards = df.iloc[:, 0].astype(float).values

        # Fill missing forward at pricing date with ATM forward (from 100% moneyness column)
        if pd.isna(forwards[0]):
            forwards[0] = df['100.0%'].iloc[0]

        self._forward_curve = forwards
        self._forward_curve_interpolated = interp1d(
            self._maturities, forwards, kind="linear", fill_value="extrapolate" #TODO check fill_value correctness
        )

        # Extract vol surface
        vol_data = df.iloc[:, 1:].copy()
        moneyness = np.array([float(col.strip('%')) / 100 for col in vol_data.columns])
        strikes_row = moneyness * forwards[0]
        vol_matrix = vol_data.iloc[1:].astype(float).values

        # Copy the first line of the vol matrix
        first_row = vol_matrix[0, :].reshape(1, -1)  # Reshape to 2D for stacking

        # Stack the first row at the beginning of the vol_matrix
        vol_matrix_with_first_row = np.vstack((first_row, vol_matrix))

        # Update the internal variables
        self._strikes = strikes_row
        self._vol_KT = vol_matrix_with_first_row / 100

        # Compute spot from strike and moneyness
        self._spot = forwards[0]  # Assuming consistent moneyness-strike structure

    # Properties
    @property
    def spot(self):
        return self._spot

    @property
    def pricing_date(self):
        return self._pricing_date

    @property
    def maturities(self):
        return self._maturities

    @property
    def maturities_as_dates(self):
        return self._maturitiesAsDates

    @property
    def forward_curve(self):
        return self._forward_curve

    @property
    def strikes(self):
        return self._strikes

    @property
    def vol_surface(self):
        return self._vol_KT

    def get_forward(self, T: float) -> float:
        """
        Get interpolated forward price for a given maturity (in years).
        """
        return float(self._forward_curve_interpolated(T))

    def get_volatility(self, strike: float, maturity: float) -> float:

        # Clamp maturity within known bounds
        if maturity < self.maturities[0]:
            maturity = self.maturities[0]
        elif maturity > self.maturities[-1]:
            maturity = self.maturities[-1]

        # Check for exact match
        try:
            strike_idx = list(self._strikes).index(strike)
            maturity_idx = self._maturities.index(maturity)
            return self._vol_KT[maturity_idx][strike_idx]
        except ValueError:
            pass

        # Exact maturity: interpolate in strike
        if maturity in self.maturities:
            return self.get_interpolated_volatility(strike, maturity)

        # Interpolate in total variance
        i = np.searchsorted(self.maturities, maturity)
        if i == 0:
            T1, T2 = self._maturities[0], self._maturities[1]
        elif i >= len(self.maturities):
            T1, T2 = self._maturities[-2], self._maturities[-1]
        else:
            T1, T2 = self._maturities[i - 1], self._maturities[i]

        fwd = self.get_forward(maturity)
        fwd_T1 = self.get_forward(T1)
        fwd_T2 = self.get_forward(T2)

        if fwd == 0 or fwd_T1 == 0 or fwd_T2 == 0:
            raise ValueError("Invalid forward spot price (zero).")

        K1 = strike * fwd_T1 / fwd
        K2 = strike * fwd_T2 / fwd

        vol_T1 = self.get_interpolated_volatility(K1, T1)
        vol_T2 = self.get_interpolated_volatility(K2, T2)

        var1 = vol_T1 ** 2 * T1
        var2 = vol_T2 ** 2 * T2

        interpolated_var = var1 + (maturity - T1) / (T2 - T1) * (var2 - var1)
        if interpolated_var < 0:
            raise ValueError("Interpolated variance is negative.")

        return np.sqrt(interpolated_var / maturity)

    def get_interpolated_volatility(self, strike: float, maturity: float) -> float:
        i = np.searchsorted(self.maturities, maturity)
        if i == len(self.maturities):
            i -= 1
        maturity_index = i

        vols = self._vol_KT[maturity_index, :]

        # Linear extrapolation below/above strike range
        if strike < self.strikes[0]:
            K0 = self.strikes[0]
            K1 = K0 * 1.01
            vol0 = vols[0]
            vol1 = self.get_volatility(K1, self.maturities[maturity_index])
            a = (vol1 - vol0) / (K1 - K0)
            b = vol0 - a * K0
            return b if strike < 0 else a * strike + b

        elif strike > self.strikes[-1]:
            K1 = self.strikes[-1]
            K0 = K1 * 0.99
            vol1 = vols[-1]
            vol0 = self.get_volatility(K0, self.maturities[maturity_index])
            a = (vol1 - vol0) / (K1 - K0)
            b = vol1 - a * K1
            return max(0.0, a * strike + b)

        # Cubic spline interpolation in strike
        spline = CubicSpline(self._strikes, vols, bc_type='natural')
        return float(spline(strike))

    def get_DiscountFactor(self, T : float) -> float:
        """
        Get Discount Factor for a given maturity (in years).
        """
        pass