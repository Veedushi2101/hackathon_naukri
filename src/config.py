"""Shared configuration: universe, event taxonomy, half-lives, paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DOCS = ROOT / "docs"

UNIVERSE = {
    # ticker: (company name, sector, aliases)
    "AAPL": ("Apple", "Technology", ["apple", "iphone"]),
    "MSFT": ("Microsoft", "Technology", ["microsoft", "azure"]),
    "AMZN": ("Amazon", "Consumer", ["amazon", "aws"]),
    "GOOGL": ("Alphabet", "Technology", ["alphabet", "google"]),
    "JPM": ("JPMorgan", "Financials", ["jpmorgan", "jp morgan", "chase"]),
    "BAC": ("Bank of America", "Financials", ["bank of america", "bofa"]),
    "GS": ("Goldman Sachs", "Financials", ["goldman sachs", "goldman"]),
    "XOM": ("Exxon Mobil", "Energy", ["exxon", "exxonmobil"]),
    "CVX": ("Chevron", "Energy", ["chevron"]),
    "JNJ": ("Johnson & Johnson", "Healthcare", ["johnson & johnson", "j&j"]),
    "PFE": ("Pfizer", "Healthcare", ["pfizer"]),
    "WMT": ("Walmart", "Consumer", ["walmart"]),
    "KO": ("Coca-Cola", "Consumer", ["coca-cola", "coca cola"]),
    "BA": ("Boeing", "Industrials", ["boeing"]),
}

EVENT_TYPES = ["Credit Event", "Geopolitical", "Macroeconomic", "Merger/Acquisition",
               "Product Launch", "Earnings", "Regulatory/Legal", "Other"]

# Sentiment half-life in trading days: how long a signal stays "alive"
HALF_LIFE = {"Credit Event": 8, "Geopolitical": 10, "Macroeconomic": 6,
             "Merger/Acquisition": 5, "Product Launch": 3, "Earnings": 3,
             "Regulatory/Legal": 7, "Other": 2}

# Prior impact (1-10) used only when too few events exist to calibrate a type
PRIOR_IMPACT = {"Credit Event": 8, "Geopolitical": 7, "Macroeconomic": 6,
                "Merger/Acquisition": 6, "Product Launch": 3, "Earnings": 5,
                "Regulatory/Legal": 5, "Other": 2}
