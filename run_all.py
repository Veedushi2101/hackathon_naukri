import argparse, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import DATA, ROOT, DOCS
from src import data_gen
from src.engine.engine import run, load_signals, daily_scores
from src.modules.rebalancer import backtest_report
from src.modules import stress as S

"""One-command pipeline:  python run_all.py
1. generate synthetic data (skipped if /data already populated; use --regen to force)
2. NLP engine  -> data/signals.jsonl + impact_calibration.csv
3. Module A    -> results/backtest.csv, weights.csv, docs/module_a.png
4. Module B    -> results/stress_*.csv, docs/module_b.png, replay timelines
"""

RES = ROOT / "results"


def main(regen=False, model="lexicon"):
    RES.mkdir(exist_ok=True); DOCS.mkdir(exist_ok=True)
    if regen or not (DATA / "prices.csv").exists():
        data_gen.make_all()
    px = pd.read_csv(DATA / "prices.csv", index_col=0, parse_dates=True)
    sig, base = run(model, prices=px)
    print(f"[engine] {len(sig)} signals | sources: {sig.source.value_counts().to_dict()}")

    # ---- Module A
    res, table, info = backtest_report(sig, px)
    table.round(4).to_csv(RES / "backtest.csv"); res["weights"].round(5).to_csv(RES / "weights.csv")
    res["scores"].round(4).to_csv(RES / "daily_scores.csv")
    sc = daily_scores(sig, px.index); sc.to_csv(RES / "daily_scores_long.csv", index=False)
    print("[module A] backtest\n", table.round(3).to_string(), "\n placebo percentile:", info)
    json.dump(info, open(RES / "placebo.json", "w"))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    res["weights"].plot.area(ax=ax[0], legend=False, linewidth=0, colormap="tab20"); ax[0].set_title("Index weights over time"); ax[0].set_ylim(0, 1)
    ax[0].legend(ncol=2, fontsize=6, loc="upper left")
    (1 + res["returns"]).cumprod().plot(ax=ax[1], label="SignalForge tilt")
    from src.modules.rebalancer import benchmark
    (1 + benchmark(px, "equal_weight_daily")).cumprod().plot(ax=ax[1], label="Equal weight")
    (1 + benchmark(px, "buy_hold")).cumprod().plot(ax=ax[1], label="Buy & hold"); ax[1].legend(); ax[1].set_title("Growth of 1 (net of costs) - SYNTHETIC prices")
    plt.tight_layout(); plt.savefig(DOCS / "module_a.png", dpi=150); plt.close()

    # ---- Module B
    pf = S.load_portfolio()
    scen = S.scenario_from_signals(sig)
    out = S.stress_test(pf, scen); out.to_csv(RES / "stress_assets.csv", index=False)
    tot, by_type, by_sec, by_ctry = S.summarise(out)
    by_type.to_csv(RES / "stress_by_type.csv"); by_sec.to_csv(RES / "stress_by_sector.csv"); by_ctry.to_csv(RES / "stress_by_country.csv")
    naive = S.naive_flat_shock(pf)
    print(f"[module B] scenario={scen}\n  before={tot.value_before/1e6:,.1f}m after={tot.value_after/1e6:,.1f}m pnl={tot.pnl/1e6:,.1f}m | naive flat shock pnl={naive/1e6:,.1f}m")
    raw = pd.read_csv(DATA / "replay_episodes.csv", parse_dates=["timestamp"])
    # episodes: re-run per episode so each is replayed in isolation
    tl = []
    for ep, g in raw.groupby("episode"):
        g = g.assign(id=[f"E{i}" for i in range(len(g))]); g.to_csv(RES / "_ep.csv", index=False)
        s_ep, _ = run(model, prices=px, raw_paths=[RES / "_ep.csv"], calibrate_on=sig, save=False)
        t = S.replay(s_ep, pf); t.insert(0, "episode", ep); tl.append(t)
        s_ep.assign(episode=ep).to_json(RES / f"replay_signals_{ep.split()[0].lower()}.jsonl", orient="records", lines=True, date_format="iso")
    (RES / "_ep.csv").unlink()
    tl = pd.concat(tl); tl.drop(columns=["scenario"]).to_csv(RES / "replay_timeline.csv", index=False)
    print(tl[["episode", "timestamp", "trigger", "pnl_pct"]].assign(pnl_pct=lambda d: d.pnl_pct.round(4)).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    x = np.arange(len(by_type)); ax[0].bar(x - .2, by_type.value_before / 1e6, .4, label="Before"); ax[0].bar(x + .2, by_type.value_after / 1e6, .4, label="After stress")
    ax[0].set_xticks(x); ax[0].set_xticklabels(by_type.index); ax[0].set_ylabel("USD m"); ax[0].legend(); ax[0].set_title(f"Portfolio value by asset type | total P&L {tot.pnl/1e6:,.1f}m")
    by_sec.pnl.div(1e6).plot.barh(ax=ax[1], color="firebrick"); ax[1].set_title("Stress P&L by sector (USD m)")
    plt.tight_layout(); plt.savefig(DOCS / "module_b.png", dpi=150); plt.close()
    return sig, table, tot, naive


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--regen", action="store_true"); ap.add_argument("--model", default="lexicon")
    a = ap.parse_args(); main(a.regen, a.model)
