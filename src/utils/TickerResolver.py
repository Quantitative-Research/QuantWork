"""
Module: ticker_resolver
Provides a class-based resolver mapping stock names to Yahoo Finance tickers.
Uses:
1) Internal manual map
2) Yahoo Finance autocomplete API
3) Normalized fallback

Also provides reference index resolution for common market indexes.
"""

import requests
import unicodedata

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
    "vusion": "VIE.PA",
    "banco de sabadell": "SAB.MC",
    "sabadell": "SAB.MC",
    "euronext": "ENX.PA",
    "intesa sanpaolo": "ISP.MI",
    "adyen": "ADYEY",

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

# Index reference mapping for common market indexes
REFERENCE_INDEXES = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
    "CAC 40": "^FCHI",
    "DAX": "^GDAXI",
    "FTSE 100": "^FTSE",
    "EUROSTOXX 50": "^STOXX50E",
    "SX5E": "^STOXX50E",
    "Nikkei 225": "^N225",
    "Hang Seng": "^HSI",
    "Shanghai Composite": "000001.SS",
    "MSCI World": "^990100-USD-STRD",
    "MSCI World ETF": "URTH",   # iShares MSCI World ETF
}

# ---------------------------------------------------------------------
# reference safe havens

REFERENCE_PRECIOUS_METALS = {
    "gold": "GC=F",
    "silver": "SI=F",
    "platinum": "PL=F",
    "palladium": "PA=F",
    "copper": "HG=F",
}

def resolve_reference_index(reference_input: str) -> str:
    """
    Resolve user-friendly reference index name to ticker.
    If it's already a ticker or not found, return as-is.
    
    Parameters
    ----------
    reference_input : str
        Either a friendly name from REFERENCE_INDEXES or a ticker symbol
        
    Returns
    -------
    str
        The corresponding ticker symbol
    """

    if not reference_input:
        return ""
    reference_input = reference_input.strip()
    # Check if it's a friendly name
    if reference_input in REFERENCE_INDEXES:
        return REFERENCE_INDEXES[reference_input]
    elif reference_input in REFERENCE_PRECIOUS_METALS:
        return REFERENCE_PRECIOUS_METALS[reference_input]
    elif reference_input in STOCK_MAP:
        return STOCK_MAP[reference_input]
    # Otherwise return as-is (assume it's a ticker)
    return reference_input


def get_reference_indexes() -> dict:
    """
    Get the mapping of reference index friendly names to tickers.
    
    Returns
    -------
    dict
        Dictionary mapping index names to their ticker symbols
    """
    return REFERENCE_INDEXES.copy()


class TickerResolver:
    """Resolves human stock names into Yahoo Finance tickers."""

    # ---------------------------------------------------------------------
    # Manual curated stock map
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------
    def __init__(self):
        pass

    def __repr__(self) -> str:
        return "TickerResolver with manual mappings for popular stocks and indexes"
    
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
        if clean in STOCK_MAP:
            return STOCK_MAP[clean]
        elif clean in REFERENCE_PRECIOUS_METALS:
            return REFERENCE_PRECIOUS_METALS[clean]
        elif clean in REFERENCE_INDEXES:
            return REFERENCE_INDEXES[clean]
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