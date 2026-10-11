"""Event-study calibration of the 1-10 Impact Score + sentiment decay.

Impact(event type) = f( mean |abnormal return| over t+1..t+3 following events of that type ),
where abnormal return = stock return - equal-weight universe return. f is a saturating curve
so a ~2.5% 3-day move is ~6/10 and a 6%+ move is ~10/10. Types with <MIN_N observations fall
back to PRIOR_IMPACT (blended). The calibration table is written to data/impact_calibration.csv
so the jury can audit where each number comes from.
"""
import numpy as np, pandas as pd
from src.config import DATA, EVENT_TYPES, PRIOR_IMPACT, HALF_LIFE

MIN_N = 8


def map_to_score(x: float) -> float:
    return float(np.clip(1 + 9 * (1 - np.exp(-x / 0.03)), 1, 10))


def calibrate(sig: pd.DataFrame, prices: pd.DataFrame, horizon=3, save=True) -> dict:
    rets = prices.pct_change()
    mkt = rets.mean(axis=1)
    ab = rets.sub(mkt, axis=0)
    rows = []
    ev = sig[(sig.source == "news") & (sig.confidence >= 0.5) & (sig.event != "Other")].copy()
    ev["date"] = pd.to_datetime(ev.timestamp).dt.normalize()
    ev = ev.drop_duplicates(["entity", "date", "event"])
    idx = ab.index
    moves = {et: [] for et in EVENT_TYPES}
    for r in ev.itertuples():
        pos = idx.searchsorted(r.date, side="right")  # first trading day after the event day
        if pos + horizon <= len(idx) and r.entity in ab:
            window = ab[r.entity].iloc[pos:pos + horizon]
            moves[r.event].append(abs(window.sum()))
    cal = {}
    for et in EVENT_TYPES:
        m = moves[et]
        n = len(m)
        data_score = map_to_score(np.mean(m)) if n else np.nan
        w = min(1.0, n / (2 * MIN_N)) if n >= MIN_N else 0.0
        score = w * data_score + (1 - w) * PRIOR_IMPACT[et] if n else float(PRIOR_IMPACT[et])
        cal[et] = dict(n_events=n, mean_abs_abnormal_ret_3d=round(float(np.mean(m)), 5) if n else np.nan,
                       data_score=round(data_score, 2) if n else np.nan, prior=PRIOR_IMPACT[et],
                       data_weight=round(w, 2), impact_base=round(score, 2), half_life_days=HALF_LIFE[et])
    out = pd.DataFrame(cal).T
    out.index.name = "event_type"
    if save:
        out.to_csv(DATA / "impact_calibration.csv")
    return {k: v["impact_base"] for k, v in cal.items()}


def signal_impact(base: float, sentiment: float, confidence: float) -> int:
    """Scale event-type base impact by how emphatic (|sentiment|) and certain the reading is."""
    x = base * (0.55 + 0.45 * min(1.0, abs(sentiment) * 1.4)) * (0.8 + 0.2 * confidence)
    return int(np.clip(round(x), 1, 10))


def decay_weight(age_days: float, event: str) -> float:
    return float(0.5 ** (age_days / HALF_LIFE.get(event, 3)))
