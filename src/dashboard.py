"""Streamlit dashboard:   streamlit run src/dashboard.py
Tabs: Signals | Module A (rebalancer + replay) | Module B (stress test + contagion graph + episode replay)"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd, networkx as nx
import streamlit as st, plotly.express as px, plotly.graph_objects as go
from src.config import DATA, ROOT, EVENT_TYPES, UNIVERSE
from src.engine.engine import load_signals
from src.engine.nlp import Analyzer, link_entities
from src.engine.impact import signal_impact
from src.modules import stress as S

RES = ROOT / "results"
st.set_page_config(page_title="Ripple - NLP Risk Engine", layout="wide")
st.title("Ripple: one NLP risk engine, two time horizons")
st.caption("News + social text -> sentiment / event / calibrated impact -> (A) tactical index rebalancer, (B) strategic stress test. Prices & text are SYNTHETIC unless you loaded real data.")

if not (RES / "backtest.csv").exists():
    st.error("Run `python run_all.py` first to generate data and results."); st.stop()


@st.cache_data
def load():
    sig = load_signals()
    w = pd.read_csv(RES / "weights.csv", index_col=0, parse_dates=True)
    sc = pd.read_csv(RES / "daily_scores_long.csv", parse_dates=["date"])
    bt = pd.read_csv(RES / "backtest.csv", index_col=0)
    pxs = pd.read_csv(DATA / "prices.csv", index_col=0, parse_dates=True)
    cal = pd.read_csv(DATA / "impact_calibration.csv", index_col=0)
    return sig, w, sc, bt, pxs, cal


sig, W, SC, BT, PX, CAL = load()
tab1, tab2, tab3 = st.tabs(["1. Signals & calibration", "2. Module A - Rebalancer", "3. Module B - Stress test"])

# ------------------------------------------------------------------ tab 1
with tab1:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Try the engine live")
        txt = st.text_input("Paste a headline or tweet", "JPMorgan faces credit downgrade as loan defaults surge, lenders pull credit lines")
        an = Analyzer("lexicon"); s, c, terms = an.sentiment(txt); et, ec = an.event(txt); conf = round(.6 * c + .4 * ec, 3)
        st.json([dict(entity=e, sentiment=s, event=et, impact=signal_impact(CAL.impact_base[et], s, conf), confidence=conf,
                      evidence=txt, reasoning=terms) for e in (link_entities(txt) or ["MARKET"])])
    with c2:
        st.subheader("Impact calibration (event study)")
        st.caption("Impact = mean |abnormal return| over t+1..t+3 after events of each type, mapped to 1-10.")
        st.dataframe(CAL[["n_events", "mean_abs_abnormal_ret_3d", "impact_base", "half_life_days"]])
    st.subheader("Signal stream")
    f1, f2, f3 = st.columns(3)
    src = f1.multiselect("Source", ["news", "twitter"], ["news", "twitter"]); mi = f2.slider("Min impact", 1, 10, 1)
    ent = f3.selectbox("Entity", ["All"] + sorted(sig.entity.unique()))
    d = sig[sig.source.isin(src) & (sig.impact >= mi)]
    if ent != "All": d = d[d.entity == ent]
    st.dataframe(d.sort_values("timestamp", ascending=False)[["timestamp", "source", "entity", "sentiment", "event", "impact", "confidence", "evidence"]].head(300), use_container_width=True)
    st.subheader("News vs social divergence (rumour / early-warning flags)")
    fl = SC[SC.rumor_flag]
    st.write(f"{len(fl)} flagged ticker-days: social is sharply more negative than news.")
    st.dataframe(fl[["date", "ticker", "news", "social", "divergence"]].round(3), use_container_width=True)

# ------------------------------------------------------------------ tab 2
with tab2:
    st.subheader("Sentiment-tilted index (14 stocks, turnover-capped, 10bps costs)")
    dates = list(W.index)
    c1, c2 = st.columns([3, 1])
    idx = c1.slider("Replay date", 0, len(dates) - 1, len(dates) - 1, format="")
    speed = c2.select_slider("Replay speed", [1, 2, 5, 10], 5)
    ph = st.empty()

    def draw(i):
        w = W.iloc[: i + 1]
        fig = px.area(w, title=f"Weights up to {dates[i].date()}"); fig.update_layout(height=380, yaxis_tickformat=".0%", legend=dict(orientation="h"))
        cur = W.iloc[i].sort_values(); bar = px.bar(cur, orientation="h", title="Current weights"); bar.update_layout(height=380, showlegend=False)
        with ph.container():
            a, b = st.columns([3, 2]); a.plotly_chart(fig, use_container_width=True); b.plotly_chart(bar, use_container_width=True)

    if st.button("Play replay from start"):
        for i in range(5, len(dates), speed):
            draw(i); time.sleep(0.12)
    draw(idx)
    st.subheader("Backtest vs benchmarks (honest report)")
    st.dataframe(BT.style.format("{:.3f}"))
    try:
        import json; pl = json.load(open(RES / "placebo.json"))
        st.info(f"Placebo test: the real signal's Sharpe beats {pl['placebo_pctile']:.0%} of random-shuffled signals. "
                "CAVEAT: prices here are synthetic with shocks injected after events, so the headline return is an artefact of the generator. "
                "Run `python -m src.fetch_real_data` for real-price results.")
    except Exception:
        pass

# ------------------------------------------------------------------ tab 3
def network_fig(G):
    layer = {"event": 0, "sector": 1, "country": 1, "counterparty": 2, "asset": 3}
    for n, dd in G.nodes(data=True): dd["layer"] = layer[dd["kind"]]
    pos = nx.multipartite_layout(G, subset_key="layer")
    ex, ey = [], []
    for u, v, dd in G.edges(data=True):
        if dd["w"] > 0.05:
            ex += [pos[u][0], pos[v][0], None]; ey += [pos[u][1], pos[v][1], None]
    nodes = list(G.nodes)
    fig = go.Figure([go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=.5, color="#bbb"), hoverinfo="none"),
                     go.Scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers",
                                marker=dict(size=[16 if G.nodes[n]["kind"] in ("event", "sector", "country") else 8 for n in nodes],
                                            color=[G.nodes[n]["heat"] for n in nodes], colorscale="YlOrRd", cmin=0, cmax=1, showscale=True,
                                            colorbar=dict(title="stress heat"), line=dict(width=.5, color="#333")),
                                text=[f'{G.nodes[n]["kind"]}: {G.nodes[n]["label"]} (heat {G.nodes[n]["heat"]:.2f})' for n in nodes], hoverinfo="text")])
    fig.update_layout(height=520, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=0, r=0, t=10, b=0))
    return fig


with tab3:
    pf = S.load_portfolio()
    mode = st.radio("Scenario source", ["Auto: triggered by engine signals", "Replay historical-style episode", "Manual what-if"], horizontal=True)
    scen = {}
    if mode.startswith("Auto"):
        thr = st.slider("Trigger when impact >=", 4, 10, S.TRIGGER_IMPACT); scen = S.scenario_from_signals(sig, thr)
    elif mode.startswith("Replay"):
        tl = pd.read_csv(RES / "replay_timeline.csv", parse_dates=["timestamp"])
        ep = st.selectbox("Episode", tl.episode.unique()); t = tl[tl.episode == ep].reset_index(drop=True)
        step = st.slider("Replay step", 1, len(t), len(t)) if len(t) > 1 else 1
        row = t.iloc[step - 1]; st.write(f"**{row.timestamp}** - {row.trigger}"); st.caption(row.evidence)
        ev, imp = row.trigger.split(" (impact ")[0], int(row.trigger.split("impact ")[1].rstrip(")")); scen = {ev: imp / 10}
        st.dataframe(t[["timestamp", "trigger", "pnl_pct"]])
    else:
        c1, c2 = st.columns(2); et = c1.selectbox("Event type", EVENT_TYPES[:-1]); imp = c2.slider("Impact score", 1, 10, 9); scen = {et: imp / 10}
    st.write("Active scenario:", {k: f"severity {v:.1f}" for k, v in scen.items()} or "none (no event above threshold)")
    res = S.stress_test(pf, scen); tot, by_type, by_sec, by_cty = S.summarise(res); naive = S.naive_flat_shock(pf)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Before (USD m)", f"{tot.value_before/1e6:,.1f}"); m2.metric("After (USD m)", f"{tot.value_after/1e6:,.1f}")
    m3.metric("Contagion P&L (USD m)", f"{tot.pnl/1e6:,.1f}", f"{tot.pnl/abs(tot.value_before):.1%}"); m4.metric("Naive flat-shock P&L (USD m)", f"{naive/1e6:,.1f}")
    a, b = st.columns(2)
    bt = by_type.reset_index().melt("asset_type", ["value_before", "value_after"]); a.plotly_chart(px.bar(bt, x="asset_type", y="value", color="variable", barmode="group", title="Value before vs after"), use_container_width=True)
    b.plotly_chart(px.bar(by_sec.reset_index(), x="pnl", y="sector", orientation="h", title="P&L by sector", color_discrete_sequence=["firebrick"]), use_container_width=True)
    st.subheader("Contagion graph: event -> sector/country -> counterparty -> asset")
    st.plotly_chart(network_fig(S.build_graph(pf, scen, res)), use_container_width=True)
    st.subheader("Worst 10 assets"); st.dataframe(res.sort_values("pnl").head(10)[["asset_id", "asset_type", "counterparty", "sector", "country", "vuln", "value_before", "value_after", "pnl"]].round(2))
