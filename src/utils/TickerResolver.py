"""
Module: ticker_resolver
Provides a class-based resolver mapping stock names to Yahoo Finance tickers.
Uses:
1) Internal manual map
2) Yahoo Finance autocomplete API
3) Normalized fallback
"""

import requests
import unicodedata


class TickerResolver:
    """Resolves human stock names into Yahoo Finance tickers."""

    # ---------------------------------------------------------------------
    # Manual curated stock map
    # ---------------------------------------------------------------------
    STOCK_MAP = {
        # --- France (CAC40) ---
        "lvmh": "MC.PA",
        "louis vuitton": "MC.PA",
        "air liquide": "AI.PA",
        "loreal": "OR.PA",
        "l'oreal": "OR.PA",
        "l oreal": "OR.PA",
        "hermes": "RMS.PA",
        "hermes international": "RMS.PA",
        "total": "TTE.PA",
        "total energies": "TTE.PA",
        "totalenergies": "TTE.PA",
        "danone": "BN.PA",
        "bnp": "BNP.PA",
        "bnp paribas": "BNP.PA",
        "safran": "SAF.PA",
        "thales": "HO.PA",
        "dassault systemes": "DSY.PA",
        "dassault aviation": "AM.PA",
        "dassault": "AM.PA",
        "airbus": "AIR.PA",
        "sanofi": "SAN.PA",
        "schneider electric": "SU.PA",
        "schneider": "SU.PA",
        "air france": "AF.PA",
        "renault": "RNO.PA",
        "societe generale": "GLE.PA",
        "socgen": "GLE.PA",
        "veolia": "VIE.PA",
        "michelin": "ML.PA",
        "axa": "CS.PA",
        "credit agricole": "ACA.PA",
        "crédit agricole": "ACA.PA",
        "vivendi": "VIV.PA",
        "kering": "KER.PA",
        "bouygues": "EN.PA",
        "edf": "EDF.PA",
        "electricite de france": "EDF.PA",
        "teleperformance": "TEP.PA",
        "stellantis": "STLAM.MI",
        "carrefour": "CA.PA",
        "engie": "ENGI.PA",
        "unibail rodamco": "URW.PA",
        "unibail-rodamco": "URW.PA",
        "unibail": "URW.PA",
        "westfield": "URW.PA",
        "pernod ricard": "RI.PA",
        "capgemini": "CAP.PA",
        "vinci": "DG.PA",
        "saint gobain": "SGO.PA",
        "essilorluxottica": "EL.PA",

        # --- US Large Caps ---
        "apple": "AAPL",
        "microsoft": "MSFT",
        "amazon": "AMZN",
        "tesla": "TSLA",
        "meta": "META",
        "facebook": "META",
        "google": "GOOG",
        "alphabet": "GOOG",
        "nvidia": "NVDA",
        "netflix": "NFLX",
        "amd": "AMD",
        "intel": "INTC",

        # --- Other internationals ---
        "eni": "ENI.MI",
        "eni spa": "ENI.MI",
        "shell": "SHEL",    
        "royal dutch shell": "SHEL",
        "shell plc": "SHEL",


    }

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------
    @staticmethod
    def _normalize(s: str) -> str:
        """Normalize text: lowercase, remove accents, trim."""
        s = s.strip().lower()
        s = unicodedata.normalize("NFD", s)
        return "".join(c for c in s if unicodedata.category(c) != "Mn")

    @staticmethod
    def _yahoo_autocomplete(query: str) -> str | None:
        """
        Use Yahoo Finance unofficial autocomplete API to guess a ticker.
        Returns ticker or None.
        """
        try:
            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()

            quotes = data.get("quotes", [])
            if quotes:
                return quotes[0].get("symbol")
        except Exception:
            return None
        return None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def resolve(self, user_input: str) -> str:
        """
        Resolve a stock name into a Yahoo Finance ticker.

        Resolution order:
        1) Normalize
        2) Try internal map
        3) Try Yahoo autocomplete API
        4) Fallback: return normalized uppercase input
        """
        clean = self._normalize(user_input)

        # Step 1: internal map
        if clean in self.STOCK_MAP:
            return self.STOCK_MAP[clean]

        # Step 2: Yahoo autocomplete
        ticker = self._yahoo_autocomplete(clean)
        if ticker:
            return ticker.upper()

        # Step 3: fallback — assume input is already a ticker
        return user_input.strip().upper()

def resolve_ticker(user_input: str) -> str:
    """Convenience function to resolve ticker using TickerResolver class."""
    resolver = TickerResolver()
    return resolver.resolve(user_input)