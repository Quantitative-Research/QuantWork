import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from scipy.interpolate import interp1d

class TreasuryCurveFetcher:
    def __init__(self):
        self.curve = {}
        self.interpolator = None

    def __repr__(self) -> str:
        return f"TreasuryCurveFetcher(curve_points={len(self.curve)})"
    
    def fetch_curve(self):
        """
        Fetch Treasury yields from Yahoo Finance.
        """
        url = "https://finance.yahoo.com/quote/^IRX?p=^IRX"  # Using 13W T-bill rate for quick access
        
        try:
            # Fetch the page
            response = requests.get("https://finance.yahoo.com/quotes/treasury-bond-rates")
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all the rates table rows
            tables = soup.find_all('table')

            if not tables:
                raise ValueError("No table found on Yahoo Finance treasury page.")

            # Parse the first table
            treasury_table = pd.read_html(str(tables[0]))[0]
            # Example table columns: ['Name', 'Last Price', 'Change', '% Change']

            # Map Yahoo names to maturities
            maturity_map = {
                '13 Week Treasury Bill': 0.25,
                '26 Week Treasury Bill': 0.5,
                '1 Year Treasury': 1.0,
                '2 Year Treasury': 2.0,
                '5 Year Treasury': 5.0,
                '10 Year Treasury': 10.0,
                '30 Year Treasury': 30.0
            }

            for idx, row in treasury_table.iterrows():
                name = row['Name']
                last_price = row['Last Price']

                if name in maturity_map:
                    maturity = maturity_map[name]
                    rate = float(last_price.strip('%')) / 100.0  # Convert % string to decimal
                    self.curve[maturity] = rate

            self.build_interpolator()

        except Exception as e:
            print(f"Error fetching Treasury curve: {e}")

    def build_interpolator(self):
        """
        Build an interpolator from the fetched curve.
        """
        maturities = list(self.curve.keys())
        rates = list(self.curve.values())
        self.interpolator = interp1d(maturities, rates, kind='linear', fill_value='extrapolate')

    def get_rate(self, maturity_in_years):
        """
        Get interpolated risk-free rate for a given maturity.

        Args:
            maturity_in_years (float): time to maturity (e.g., 0.5 for 6 months)

        Returns:
            float: interpolated risk-free rate (annualized, in decimal)
        """
        if self.interpolator is None:
            raise ValueError("Treasury curve not built. Please call fetch_curve() first.")
        
        return float(self.interpolator(maturity_in_years))
