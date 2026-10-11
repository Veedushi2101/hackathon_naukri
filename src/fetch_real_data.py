"""OPTIONAL: replace synthetic prices with real Yahoo Finance data (needs internet + `pip install yfinance`).
    python -m src.fetch_real_data --start 2025-01-01 --end 2025-12-31
Then rerun `python run_all.py`. Text feeds: drop GDELT / NewsAPI / Kaggle-tweet exports into
data/news.csv and data/tweets.csv with columns id,timestamp,source,text (see README)."""
import argparse
import pandas as pd
from src.config import DATA, UNIVERSE

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--start", default="2025-01-01"); ap.add_argument("--end", default="2025-12-31")
    a = ap.parse_args()
    import yfinance as yf
    px = yf.download(list(UNIVERSE), start=a.start, end=a.end, auto_adjust=True)["Close"].dropna()
    px.index.name = "date"; px.to_csv(DATA / "prices.csv"); print("saved", px.shape)
