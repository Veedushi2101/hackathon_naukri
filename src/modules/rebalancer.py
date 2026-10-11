"""Module A - Tactical sentiment-tilted index rebalancer + honest backtest.

Target weight_i  ~  (1/N) * exp(k * score_i), clipped to [w_min, w_max], renormalised.
Trading is limited by a one-way turnover cap and charged a cost in bps, so the index
cannot whipsaw. Signals known at the close of day t set weights for day t+1 (no look-ahead).
"""
import numpy as np, pandas as pd
from src.engine.engine import daily_scores

K, W_MIN, W_MAX, TURN_CAP, COST_BPS = 1.5, 0.02, 0.15, 0.10, 10


def _project(w, lo, hi):
    for _ in range(50):
        w = np.clip(w, lo, hi); w = w / w.sum()
        if w.min() >= lo - 1e-9 and w.max() <= hi + 1e-9:
            break
    return w


def target_weights(scores: pd.Series, k=K):
    n = len(scores)
    return pd.Series(_project(np.exp(k * scores.values) / n, W_MIN, W_MAX), index=scores.index)


def run_index(sig, prices, k=K, turn_cap=TURN_CAP, cost_bps=COST_BPS, score_override=None):
    """Return dict(weights, returns, turnover) over the price calendar."""
    tickers = list(prices.columns)
    scores = score_override if score_override is not None else daily_scores(sig, prices.index)
    S = scores.pivot(index="date", columns="ticker", values="score").reindex(prices.index).reindex(columns=tickers).fillna(0.0)
    rets = prices.pct_change().fillna(0.0)
    w = pd.Series(1 / len(tickers), index=tickers)
    W, R, TO = [], [], []
    for i, d in enumerate(prices.index):
        # earn today's return with yesterday-close weights
        port_ret = float((w * rets.loc[d]).sum())
        w = w * (1 + rets.loc[d]); w = w / w.sum()  # drift
        tgt = target_weights(S.loc[d], k)
        delta = tgt - w
        one_way = delta.abs().sum() / 2
        if one_way > turn_cap:
            delta *= turn_cap / one_way
            one_way = turn_cap
        w = w + delta
        R.append(port_ret - one_way * 2 * cost_bps / 1e4); TO.append(one_way); W.append(w.copy())
    return dict(weights=pd.DataFrame(W, index=prices.index), returns=pd.Series(R, index=prices.index),
                turnover=pd.Series(TO, index=prices.index), scores=S)


def benchmark(prices, kind):
    rets = prices.pct_change().fillna(0.0)
    if kind == "equal_weight_daily":
        return rets.mean(axis=1)
    w = pd.Series(1 / prices.shape[1], index=prices.columns); out = []  # buy & hold
    for d in prices.index:
        out.append(float((w * rets.loc[d]).sum())); w = w * (1 + rets.loc[d]); w /= w.sum()
    return pd.Series(out, index=prices.index)


def stats(r, turnover=None):
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    d = dict(total_return=eq.iloc[-1] - 1, ann_vol=r.std() * np.sqrt(252), sharpe=sharpe, max_drawdown=dd)
    if turnover is not None:
        d["avg_daily_turnover"] = turnover.mean()
    return d


def placebo(sig, prices, scores, n=40, seed=0):
    """Honesty check: shuffle each day's scores across tickers. If the tilt only 'works' because
    of the real signal, the true Sharpe should rank high versus these random tilts."""
    rng = np.random.default_rng(seed); shar = []
    tk = list(prices.columns)
    for _ in range(n):
        sh = scores.copy()
        perm = {t: rng.choice(tk) for t in tk}
        sh["ticker"] = sh["ticker"].map(perm)
        sh = sh.groupby(["date", "ticker"], as_index=False).mean(numeric_only=True)
        shar.append(stats(run_index(sig, prices, score_override=sh)["returns"])["sharpe"])
    return np.array(shar)


def backtest_report(sig, prices, with_placebo=True):
    res = run_index(sig, prices)
    table = pd.DataFrame({
        "Ripple sentiment-tilt": stats(res["returns"], res["turnover"]),
        "Equal-weight (daily rebal)": stats(benchmark(prices, "equal_weight_daily")),
        "Buy & hold": stats(benchmark(prices, "buy_hold")),
    }).T
    info = {}
    if with_placebo:
        ps = placebo(sig, prices, daily_scores(sig, prices.index))
        info["placebo_pctile"] = float((ps < table.loc["Ripple sentiment-tilt", "sharpe"]).mean())
    return res, table, info
