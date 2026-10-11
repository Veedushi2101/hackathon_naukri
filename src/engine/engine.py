"""Ripple NLP Risk Engine: ingest -> link -> sentiment/event -> calibrated impact -> signals.jsonl

Usage:  python -m src.engine.engine [--model lexicon|transformer]
Output: data/signals.jsonl   (one JSON object per signal)
        data/impact_calibration.csv
"""
import argparse, json
import numpy as np, pandas as pd
from src.config import DATA, UNIVERSE
from src.engine.nlp import Analyzer, link_entities
from src.engine.impact import calibrate, signal_impact, decay_weight


# ---------- 1. ingestion: every source is normalised to (id, timestamp, source, text) ----------
def ingest(paths) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        frames.append(df[["id", "timestamp", "source", "text"]] if "id" in df else
                      df.assign(id=[f"{p.stem}{i}" for i in range(len(df))])[["id", "timestamp", "source", "text"]])
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


# ---------- 2-3. analysis ----------
def analyse(raw: pd.DataFrame, analyzer: Analyzer) -> pd.DataFrame:
    rows = []
    for r in raw.itertuples():
        ents = link_entities(r.text) or ["MARKET"]
        s, c, terms = analyzer.sentiment(r.text)
        et, ec = analyzer.event(r.text)
        for e in ents:
            rows.append(dict(id=r.id, timestamp=r.timestamp, source=r.source, entity=e, sentiment=s, event=et,
                             confidence=round(0.6 * c + 0.4 * ec, 3), evidence=r.text, reasoning=terms))
    return pd.DataFrame(rows)


def run(model="lexicon", prices=None, raw_paths=None, calibrate_on=None, save=True, out_name="signals.jsonl"):
    raw = ingest(raw_paths or [DATA / "news.csv", DATA / "tweets.csv"])
    sig = analyse(raw, Analyzer(model))
    prices = prices if prices is not None else pd.read_csv(DATA / "prices.csv", index_col=0, parse_dates=True)
    base = calibrate(calibrate_on if calibrate_on is not None else sig, prices, save=save)
    sig["impact"] = [signal_impact(base[e], s, c) for e, s, c in zip(sig.event, sig.sentiment, sig.confidence)]
    sig["impact_base"] = sig.event.map(base)
    if save:
        with open(DATA / out_name, "w") as f:
            for r in sig.to_dict("records"):
                r["timestamp"] = r["timestamp"].isoformat()
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return sig, base


def load_signals(name="signals.jsonl") -> pd.DataFrame:
    df = pd.read_json(DATA / name, lines=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ---------- 4. time-decayed, per-channel aggregation + news/social divergence ----------
def daily_scores(sig: pd.DataFrame, dates) -> pd.DataFrame:
    """For each trading date (end-of-day) and ticker: decayed sentiment by channel, combined score, divergence flag."""
    dates = pd.DatetimeIndex(dates)
    s = sig[sig.entity.isin(UNIVERSE)].copy()
    s["day"] = s.timestamp.dt.normalize()
    s["w"] = s.confidence
    out = []
    for d_i, d in enumerate(dates):
        past = s[s.day <= d]
        if past.empty:
            continue
        age = np.array([len(pd.bdate_range(x, d)) - 1 for x in past.day])  # trading-day age
        past = past.assign(decay=[decay_weight(a, e) for a, e in zip(age, past.event)])
        past["contrib"] = past.sentiment * past.w * past.decay
        for tk, g in past.groupby("entity"):
            n_soc_recent = ((g.source == "twitter") & (g.day >= d - pd.Timedelta(days=2))).sum()
            news = float(np.tanh(g[g.source == "news"].contrib.sum()))
            soc = float(np.tanh(0.5 * g[g.source == "twitter"].contrib.sum()))
            comb = 0.65 * news + 0.35 * soc if (news != 0 and soc != 0) else (news if news != 0 else 0.6 * soc)
            div = soc - news
            out.append(dict(date=d, ticker=tk, news=news, social=soc, score=comb, divergence=div,
                            rumor_flag=bool(soc <= -0.45 and div <= -0.35 and n_soc_recent >= 3)))
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lexicon", choices=["lexicon", "transformer"])
    a = ap.parse_args()
    sig, base = run(a.model)
    print(f"{len(sig)} signals written to data/signals.jsonl")
    print(pd.read_csv(DATA / "impact_calibration.csv").to_string(index=False))
    print(sig.tail(3).drop(columns=["reasoning"]).to_string())
