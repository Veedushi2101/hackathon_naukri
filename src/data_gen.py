"""Synthetic data generator (seeded, reproducible).

Creates in /data:
  news.csv, tweets.csv      - text feeds (two independent sources)
  prices.csv                - daily closes for the universe
  portfolio.csv             - wholesale-banking book (loans, bonds, derivatives)
  replay_episodes.csv       - hand-written headlines for 'time machine' replay

IMPORTANT: everything here is SYNTHETIC. Price shocks are injected after events
so that the pipeline has a learnable signal; this is a *demo* of the machinery,
not evidence of real alpha. Use src/fetch_real_data.py for real prices.
"""
import numpy as np, pandas as pd
from src.config import DATA, UNIVERSE

rng = np.random.default_rng(42)
TICKERS = list(UNIVERSE)

T = {  # event type -> (negative templates, positive templates)
 "Credit Event": (
  ["{co} faces credit rating downgrade amid rising loan defaults",
   "{co} reports surge in bad debt and covenant breach, bond yields spike",
   "{co} warns of liquidity crunch as lenders pull credit lines"],
  ["{co} wins credit rating upgrade after strong debt reduction",
   "{co} refinances debt at lower rates, credit outlook improves"]),
 "Geopolitical": (
  ["Sanctions and war escalation threaten {co} overseas operations",
   "Trade war tariffs hit {co} as geopolitical tensions rise",
   "Supply chain disrupted for {co} after border conflict"],
  ["Ceasefire and trade deal ease geopolitical risk for {co}",
   "{co} gains from new defence and export agreement"]),
 "Macroeconomic": (
  ["Inflation surprise and rate hike pressure {co} margins",
   "Recession fears and weak GDP data weigh on {co}",
   "Central bank tightening raises {co} funding costs"],
  ["Cooling inflation and rate cut hopes lift {co}",
   "Strong GDP growth and robust consumer demand benefit {co}"]),
 "Merger/Acquisition": (
  ["{co} merger collapses as regulators block takeover bid",
   "{co} overpays for acquisition, investors worried"],
  ["{co} announces acquisition expected to boost earnings",
   "{co} agrees takeover deal at premium, shares surge"]),
 "Product Launch": (
  ["{co} product launch delayed after major defect found",
   "{co} recalls new product, customers disappointed"],
  ["{co} unveils innovative product launch to strong demand",
   "{co} launches new service, analysts upbeat"]),
 "Earnings": (
  ["{co} misses earnings estimates and cuts guidance",
   "{co} quarterly profit falls, revenue below forecast"],
  ["{co} beats earnings estimates and raises guidance",
   "{co} reports record quarterly profit, revenue growth"]),
 "Regulatory/Legal": (
  ["{co} hit with antitrust lawsuit and heavy fine",
   "{co} faces regulatory probe and compliance penalty"],
  ["{co} wins court ruling, regulatory approval granted",
   "{co} cleared of investigation, lawsuit dismissed"]),
}
FILLER = ["{co} shares trade flat as markets await data", "Analysts keep neutral view on {co}",
          "{co} to present at industry conference next week"]
SHOCK = {"Credit Event": .040, "Geopolitical": .040, "Macroeconomic": .020, "Merger/Acquisition": .035,
         "Product Launch": .012, "Earnings": .025, "Regulatory/Legal": .020}
SOC_NEG = ["omg {co} is done 😱 {w}", "selling all my {t} right now, this looks bad", "{t} crashing?? heard rumours it's worse than reported",
           "{co} fraud?? thread 🧵 {w}"]
SOC_POS = ["{t} to the moon 🚀 {w}", "loading up on {t}, great news", "{co} looking strong, bullish 📈"]


def make_all(n_days=120, start="2026-04-01"):
    DATA.mkdir(exist_ok=True)
    days = pd.bdate_range(start, periods=n_days)
    news, tweets, events = [], [], []
    for d_i, d in enumerate(days):
        if d_i < 3 or d_i > n_days - 4:
            continue
        for tk in TICKERS:
            co = UNIVERSE[tk][0]
            if rng.random() < 0.10:  # a real event hits this stock today
                et = rng.choice(list(T))
                sign = -1 if rng.random() < 0.5 else 1
                tpl = rng.choice(T[et][0 if sign < 0 else 1])
                ts = d + pd.Timedelta(hours=int(rng.integers(8, 15)))
                news.append((ts, "news", tpl.format(co=co)))
                if rng.random() < .5:  # second outlet confirms
                    news.append((ts + pd.Timedelta(hours=1), "news", tpl.format(co=co) + " - report"))
                events.append((d_i, tk, et, sign))
                for _ in range(int(rng.integers(2, 6))):  # social amplification
                    pool = SOC_NEG if sign < 0 else SOC_POS
                    tweets.append((ts + pd.Timedelta(minutes=int(rng.integers(5, 400))), "twitter",
                                   rng.choice(pool).format(co=co, t="$" + tk, w=tpl.split(" ")[-1])))
            elif rng.random() < 0.08:
                news.append((d + pd.Timedelta(hours=10), "news", rng.choice(FILLER).format(co=co)))
            # rumour episodes: social panic with NO news confirmation (divergence demo)
            if rng.random() < 0.012:
                for _ in range(int(rng.integers(4, 8))):
                    tweets.append((d + pd.Timedelta(hours=int(rng.integers(9, 18)), minutes=int(rng.integers(0, 59))),
                                   "twitter", rng.choice(SOC_NEG).format(co=co, t="$" + tk, w="")))
    news = pd.DataFrame(news, columns=["timestamp", "source", "text"]).sort_values("timestamp")
    tweets = pd.DataFrame(tweets, columns=["timestamp", "source", "text"]).sort_values("timestamp")
    news.insert(0, "id", [f"N{i}" for i in range(len(news))]); tweets.insert(0, "id", [f"T{i}" for i in range(len(tweets))])
    news.to_csv(DATA / "news.csv", index=False); tweets.to_csv(DATA / "tweets.csv", index=False)

    # prices: market factor + idiosyncratic noise + post-event shock on the next day
    mkt = rng.normal(0.0003, 0.008, n_days)
    rets = pd.DataFrame({tk: 0.9 * mkt + rng.normal(0, 0.011, n_days) for tk in TICKERS}, index=days)
    for d_i, tk, et, sign in events:
        if d_i + 1 < n_days:
            rets.iloc[d_i + 1, rets.columns.get_loc(tk)] += sign * SHOCK[et] * rng.uniform(.6, 1.4)
            if d_i + 2 < n_days:
                rets.iloc[d_i + 2, rets.columns.get_loc(tk)] += sign * SHOCK[et] * .3 * rng.uniform(.5, 1.5)
    px = 100 * (1 + rets).cumprod()
    px.index.name = "date"; px.round(4).to_csv(DATA / "prices.csv")

    make_portfolio(); make_replay()
    return news, tweets, px


def make_portfolio(n_loans=40, n_bonds=30, n_deriv=20):
    sectors = ["Energy", "Financials", "Technology", "Healthcare", "Consumer", "Industrials", "Real Estate", "Utilities"]
    countries = ["US", "UK", "DE", "IN", "BR", "TR", "ZA", "MX", "RU", "CN"]
    cps = [f"CP{i:03d}" for i in range(1, 31)]
    cp_attr = {c: (rng.choice(sectors), rng.choice(countries, p=[.3, .1, .08, .1, .08, .07, .06, .08, .05, .08])) for c in cps}
    rows = []
    for i in range(n_loans):
        c = rng.choice(cps); s, k = cp_attr[c]
        rows.append(dict(asset_id=f"L{i:03d}", asset_type="Loan", counterparty=c, sector=s, country=k, notional=float(rng.integers(5, 80) * 1e6),
                         pd_base=float(rng.choice([.005, .01, .02, .04, .08])), lgd=float(rng.choice([.35, .45, .6])), duration=np.nan, convexity=np.nan, delta=np.nan, underlying=""))
    for i in range(n_bonds):
        c = rng.choice(cps); s, k = cp_attr[c]
        d = float(rng.choice([2, 4, 7, 10]))
        rows.append(dict(asset_id=f"B{i:03d}", asset_type="Bond", counterparty=c, sector=s, country=k, notional=float(rng.integers(5, 60) * 1e6),
                         pd_base=float(rng.choice([.003, .01, .03])), lgd=.6, duration=d, convexity=d * d * 1.2, delta=np.nan, underlying=""))
    for i in range(n_deriv):
        c = rng.choice(cps); s, k = cp_attr[c]
        rows.append(dict(asset_id=f"D{i:03d}", asset_type="Derivative", counterparty=c, sector=s, country=k, notional=float(rng.integers(10, 100) * 1e6),
                         pd_base=.01, lgd=.6, duration=np.nan, convexity=np.nan, delta=float(rng.choice([-.6, -.3, .3, .6])),
                         underlying=str(rng.choice(["EQUITY", "FX", "OIL", "RATES"]))))
    pd.DataFrame(rows).to_csv(DATA / "portfolio.csv", index=False)


def make_replay():
    """Short, hand-written headlines *inspired by* well-known stress episodes (illustrative, not scraped)."""
    rows = []
    def ep(name, items):
        for i, (h, txt) in enumerate(items):
            rows.append((name, pd.Timestamp("2023-03-08") + pd.Timedelta(hours=h) if "Banking" in name else pd.Timestamp("2022-02-21") + pd.Timedelta(hours=h), "news", txt))
    ep("Banking Run (Mar-2023 style)", [
        (9, "Regional lender announces large securities loss and emergency capital raise"),
        (30, "Depositors withdraw billions as bank run fears spread, liquidity crunch at lender"),
        (36, "Regulators seize failed regional bank after deposit run, credit rating downgrade across sector"),
        (40, "JPMorgan and Goldman Sachs shares fall as contagion fears hit bank bonds and loan defaults rise"),
        (60, "Authorities announce deposit backstop, markets steady but credit spreads stay wide"),
    ])
    ep("Geopolitical Shock (Feb-2022 style)", [
        (5, "Military buildup raises war fears, sanctions threatened against energy exporters"),
        (48, "Conflict escalation: sanctions and trade war announced, oil spikes, Chevron and Exxon shares jump"),
        (52, "Supply chain disrupted, inflation surprise and rate hike fears hit global markets"),
        (72, "Boeing and Apple warn of disrupted supply chain as geopolitical tensions escalate"),
    ])
    pd.DataFrame(rows, columns=["episode", "timestamp", "source", "text"]).to_csv(DATA / "replay_episodes.csv", index=False)


if __name__ == "__main__":
    n, t, p = make_all()
    print(f"news={len(n)} tweets={len(t)} prices={p.shape}")
