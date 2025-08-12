import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d, UnivariateSpline
import numpy as np
from .convex_projection import project_convex

class OptionDataFetcher:
    def __init__(self, ticker):
        self.ticker = ticker
        self.option_maturities = {}
        self.spot_price = None
        self.forward_curve = None
    
    def build_market(self):
        """
        Fetch option data from Yahoo Finance and build the forward curve in one step.
        """
        self.fetch_options()
        self.build_forward_curve()
        self.build_vol_surface()

    def build_vol_surface(self):
        """
        Build a volatility surface matrix from convex-projected smiles.
        Rows: strikes
        Columns: maturities
        """
        all_strikes = sorted({
            strike
            for maturity in self.option_maturities.values()
            if maturity.cleaned_smile_hat is not None
            for strike in maturity.cleaned_smile_hat['strike'].values
        })

        # Create empty DataFrame
        vol_surface = pd.DataFrame(index=all_strikes)

        # Fill with convex vols for each expiry
        for expiry, maturity in self.option_maturities.items():
            if maturity.cleaned_smile_hat is None:
                continue
            smile = maturity.cleaned_smile_hat
            # Map strikes to vols, NaN for missing strikes
            vol_series = pd.Series(
                data=smile['impliedVolatility'].values,
                index=smile['strike'].values
            )
            vol_surface[pd.to_datetime(expiry)] = vol_series.reindex(all_strikes)

        self.vol_surface = vol_surface.sort_index(axis=1)
        return self.vol_surface

    def fetch_options(self):
        stock = yf.Ticker(self.ticker)
        self.spot_price = stock.history(period='1d')['Close'].iloc[-1]  # Latest close price
        expirations = stock.options

        for expiry in expirations:
            opt_chain = stock.option_chain(expiry)
            maturity = OptionMaturity(expiry, opt_chain.calls, opt_chain.puts, self.spot_price)
            self.option_maturities[expiry] = maturity

    def build_forward_curve(self):
        """
        Build and interpolate the forward curve. 
        Also sets self.forward_interpolator.

        Returns:
            set: A set of (expiry, forward) tuples
        """
        forward_set = set()
        expiries = []
        forwards = []

        for expiry, maturity in self.option_maturities.items():
            try:
                forward = maturity.estimate_forward()
                expiry_date = pd.to_datetime(expiry)
                forward_set.add((expiry_date, forward))
                expiries.append(expiry_date)
                forwards.append(forward)
            except Exception as e:
                print(f"Could not estimate forward for {expiry}: {e}")

        if expiries and forwards:
            # Sort by expiry
            expiries, forwards = zip(*sorted(zip(expiries, forwards)))

            # Convert dates to numerical values (days to expiry)
            expiry_days = [(exp - pd.Timestamp.today()).days for exp in expiries]

            # Build interpolator
            self.forward_interpolator = interp1d(expiry_days, forwards, kind='linear', fill_value="extrapolate")

    def get_forward(self, expiry):
        """
        Get interpolated forward price for a given expiry.

        Args:
            expiry (datetime.datetime, str, int, or float): 
                - if datetime or str: expiry date
                - if int/float: maturity in years

        Returns:
            float: interpolated forward price
        """
        if self.forward_interpolator is None:
            raise ValueError("Forward curve has not been built. Please call build_forward_curve() first.")

        # Handle depending on input type
        if isinstance(expiry, (pd.Timestamp, str)):
            expiry_date = pd.to_datetime(expiry)
            expiry_day = (expiry_date - pd.Timestamp.today()).days
        elif isinstance(expiry, (int, float)):
            expiry_day = int(expiry * 365.25)
  # Assuming 365.25 days in a year
        else:
            raise TypeError(f"Unsupported expiry type: {type(expiry)}. Must be datetime, str, int or float.")
        
        return float(self.forward_interpolator(expiry_day))



    def get_maturities(self):
        return list(self.option_maturities.keys())

    def get_maturity_data(self, expiry):
        return self.option_maturities.get(expiry, None)
    
## Plotting 

    def plot_forward_curve(self):
        """
        Plots the forward curve based on the interpolated data.

        Returns:
            None
        """
        if self.forward_interpolator is None:
            raise ValueError("Forward curve has not been built. Please call build_forward_curve() first.")
        
        # Get the current list of expiries and forward values
        expiries = list(self.option_maturities.keys())
        expiry_dates = [pd.to_datetime(exp) for exp in expiries]
        expiry_days = [(exp - pd.Timestamp.today()).days for exp in expiry_dates]

        # Get the interpolated forward values for plotting
        forward_values = self.forward_interpolator(expiry_days)

        # Plotting the forward curve
        plt.figure(figsize=(10, 6))
        plt.plot(expiry_days, forward_values, label="Forward Curve", color='b', marker='o')

        # Plot the actual data points
        actual_forwards = [maturity.estimate_forward() for maturity in self.option_maturities.values()]
        plt.scatter(expiry_days, actual_forwards, color='r', label="Actual Forwards", zorder=5)

        plt.xlabel('Days to Expiry')
        plt.ylabel('Forward Price')
        plt.title(f"Forward Curve for {self.ticker}")
        plt.legend()
        plt.grid(True)
        plt.show()


class OptionMaturity:
    def __init__(self, expiry, calls, puts, spot_price):
        self.expiry = pd.to_datetime(expiry)
        self.calls = calls.copy()
        self.puts = puts.copy()
        self.spot_price = spot_price
        self.cleaned_smile = None
        self.cleaned_smile_hat = None  
        self._clean_data()
        self._build_cleaned_smile()
        self._project_convex_smile(method='slopes')  # Default method for convex projection

    def _clean_data(self):
        """Remove NaN/invalid implied volatility rows."""
        for df in [self.calls, self.puts]:
            df.dropna(subset=['impliedVolatility'], inplace=True)
            df = df[df['impliedVolatility'] > 0]
            df.reset_index(drop=True, inplace=True)

    @property
    def otm_iv_df(self):
        """
        Returns a cleaned DataFrame of OTM strikes and IV, sorted by strike.
        """
        otm_calls = self.calls[self.calls['strike'] > self.spot_price]
        otm_puts = self.puts[self.puts['strike'] < self.spot_price]

        data = pd.concat([otm_calls, otm_puts])[['strike', 'impliedVolatility']]
        data = data[(data['impliedVolatility'] > 0) & (data['impliedVolatility'].notna())]
        return data.sort_values(by='strike').reset_index(drop=True)
    
    def _remove_arbitrage(self, df, max_jump=0.5):
        """
        Supprime les points créant des violations simples (sauts trop forts).
        """
        df = df.copy()
        iv = df['impliedVolatility'].values
        mask = (iv > 0) & (np.abs(np.gradient(iv)) < max_jump)
        return df[mask].reset_index(drop=True)

    def _build_cleaned_smile(self):
        """
        Construit et stocke la version nettoyée/désarbitrée du smile OTM.
        """
        self.cleaned_smile = self._remove_arbitrage(self.otm_iv_df)
    
    def _project_convex_smile(self, method):
        """
        Projette self.cleaned_smile sur l'ensemble des smiles convexes.
        """
        if self.cleaned_smile is None or self.cleaned_smile.empty:
            return
        K = self.cleaned_smile['strike'].values
        sigma = self.cleaned_smile['impliedVolatility'].values
        sigma_hat = project_convex(K, sigma, method=method)  # méthode rapide par défaut
        self.cleaned_smile_hat = self.cleaned_smile.copy()
        self.cleaned_smile_hat['impliedVolatility'] = sigma_hat


    @property
    def mean_otm_iv(self):
        """Returns the average OTM implied volatility."""
        iv_values = [iv for _, iv in self.otm_iv_set]
        return float(np.mean(iv_values)) if iv_values else np.nan

    def get_strikes(self):
        return self.calls['strike'].tolist(), self.puts['strike'].tolist()
    
    def estimate_forward(self):
        """
        Estimate forward price using put-call parity at the ATM strike.
        """
        merged = pd.merge(
            self.calls[['strike', 'lastPrice']], 
            self.puts[['strike', 'lastPrice']], 
            on='strike', 
            suffixes=('_call', '_put')
        )

        merged['abs_diff'] = (merged['strike'] - self.spot_price).abs()
        atm_row = merged.loc[merged['abs_diff'].idxmin()]

        K = atm_row['strike']
        C = atm_row['lastPrice_call']
        P = atm_row['lastPrice_put']

        return K + (C - P)
    
    def plot_smile_clean(self):
        if self.cleaned_smile is None or self.cleaned_smile.empty:
            print(f"Aucune donnée valide pour {self.expiry}")
            return
        plt.figure(figsize=(10, 6))
        plt.scatter(self.otm_iv_df['strike'], self.otm_iv_df['impliedVolatility'],
                    color='red', alpha=0.6, label='Brut')
        plt.plot(self.cleaned_smile['strike'], self.cleaned_smile['impliedVolatility'],
                 color='blue', label='Nettoyé')
        if self.cleaned_smile_hat is not None:
            plt.plot(self.cleaned_smile_hat['strike'], self.cleaned_smile_hat['impliedVolatility'],
                     color='green', linestyle='--', label='Convexe')
        plt.axvline(self.spot_price, color='black', linestyle='--', label='Spot')
        plt.title(f"Vol Smile Clean - Expiry {self.expiry.date()}")
        plt.xlabel("Strike")
        plt.ylabel("Implied Volatility")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_smile(self, option_type='OTM', x_axis='strike'):
        """
        Plot the implied volatility smile with cleaned data.
        """
        if option_type == 'call':
            data = self.calls
        elif option_type == 'put':
            data = self.puts
        elif option_type == 'OTM':
            data = self.otm_iv_df
        else:
            raise ValueError("option_type must be 'call', 'put', or 'OTM'")

        if data.empty:
            print(f"No data to plot for {self.expiry}")
            return

        strikes = data['strike']
        iv = data['impliedVolatility']

        if x_axis == 'strike':
            x_values = strikes
            x_label = 'Strike Price'
        elif x_axis == 'moneyness':
            x_values = strikes / self.spot_price
            x_label = 'Moneyness (K/S₀)'
        else:
            raise ValueError("x_axis must be 'strike' or 'moneyness'")

        plt.figure(figsize=(10, 6))
        plt.plot(x_values, iv, marker='o', linestyle='-', label=f'{option_type} IV')

        if x_axis == 'moneyness':
            plt.axvline(1.0, color='red', linestyle='--', label='ATM (K/S₀=1)')
        else:
            plt.axvline(self.spot_price, color='red', linestyle='--', label='Spot Price')

        plt.title(f'{option_type} Volatility Smile - Expiry {self.expiry.date()}')
        plt.xlabel(x_label)
        plt.ylabel('Implied Volatility')
        plt.legend()
        plt.grid(True)
        plt.show()
