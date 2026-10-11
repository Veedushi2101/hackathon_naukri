"""Builds docs/architecture.png and docs/presentation.pdf from real results. Run after run_all.py."""
import json, textwrap
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
from src.config import DATA, DOCS, ROOT

RES = ROOT / "results"
NAME, COLLEGE = "<Your Full Name>", "<Your College>"


def box(ax, x, y, w, h, text, fc="#e8f0fe", ec="#1a73e8", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5, color="#444"))


def architecture():
    fig, ax = plt.subplots(figsize=(14, 7.5)); ax.set_xlim(0, 14); ax.set_ylim(0, 7.5); ax.axis("off")
    ax.text(7, 7.15, "Ripple - architecture & data flow", ha="center", fontsize=15, weight="bold")
    for i, t in enumerate(["News feed\n(GDELT / NewsAPI /\nKaggle headlines)", "Social feed\n(X / Kaggle tweets)", "Replay episodes\n(deterministic)"]):
        box(ax, 0.3, 5.3 - i * 1.7 + 0.0, 2.4, 1.2, t, "#fff4e5", "#f29900"); arrow(ax, 2.7, 5.9 - i * 1.7, 3.5, 4.3)
    box(ax, 3.5, 3.4, 2.3, 1.8, "Ingestors\nnormalised schema\n(id, ts, source, text)\n+ entity linking", "#e6f4ea", "#188038")
    arrow(ax, 5.8, 4.3, 6.4, 4.3)
    ax.add_patch(FancyBboxPatch((6.4, 2.2), 3.3, 4.0, boxstyle="round,pad=0.03", fc="#f3e8fd", ec="#8430ce", lw=2))
    ax.text(8.05, 5.95, "NLP RISK ENGINE", ha="center", weight="bold", color="#8430ce")
    for i, t in enumerate(["Sentiment  [-1, 1]\n(FinBERT | lexicon)", "Event type\n(zero-shot | keyword rules)", "Impact 1-10\n(event-study calibrated)", "Half-life decay +\nnews/social divergence"]):
        box(ax, 6.6, 5.0 - i * 0.7, 2.9, 0.58, t, "#fff", "#8430ce", 8)
    arrow(ax, 9.7, 4.3, 10.3, 4.3)
    box(ax, 10.3, 3.6, 3.4, 1.4, "signals.jsonl  +  FastAPI\n{entity, sentiment, event, impact,\nconfidence, evidence, reasoning}", "#e8f0fe", "#1a73e8", 8.5)
    arrow(ax, 12, 3.6, 10.9, 2.3); arrow(ax, 12, 3.6, 12.9, 2.3)
    box(ax, 9.0, 0.9, 3.0, 1.4, "MODULE A  Tactical\nsoftmax tilt, turnover cap,\ncost penalty, backtest + placebo", "#e6f4ea", "#188038", 8.5)
    box(ax, 12.2, 0.9, 1.7, 1.4, "MODULE B\nStrategic\ncontagion graph\nstress test", "#fce8e6", "#d93025", 8.5)
    box(ax, 3.0, 0.4, 5.0, 1.0, "yfinance prices / synthetic prices (calibration + backtest)", "#fff4e5", "#f29900", 8.5)
    arrow(ax, 8.0, 0.9, 9.0, 1.3); arrow(ax, 5.5, 1.4, 7.0, 2.2)
    box(ax, 5.5, 6.55, 6.5, 0.5, "Streamlit dashboard: signals | weights replay | stress test + network | episode replay", "#f1f3f4", "#5f6368", 8.5)
    plt.savefig(DOCS / "architecture.png", dpi=170, bbox_inches="tight"); plt.close()


def slide(pdf, title, body=None, img=None, table=None):
    fig = plt.figure(figsize=(13.33, 7.5)); fig.patch.set_facecolor("white")
    fig.text(0.05, 0.92, title, fontsize=26, weight="bold", color="#1a237e")
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.89, 0.89], color="#1a73e8", lw=3))
    if body:
        body = "\n".join(textwrap.fill(l, 100, subsequent_indent="    ") if l.strip() else "" for l in body.split("\n"))
        fig.text(0.05, 0.85, body, fontsize=14, va="top", linespacing=1.5)
    if img:
        im = plt.imread(img)
        ax = fig.add_axes([0.03, 0.03, 0.94, 0.52] if body else [0.05, 0.04, 0.9, 0.82]); ax.imshow(im); ax.axis("off")
    pdf.savefig(fig); plt.close(fig)


def deck():
    bt = pd.read_csv(RES / "backtest.csv", index_col=0); pl = json.load(open(RES / "placebo.json"))
    cal = pd.read_csv(DATA / "impact_calibration.csv", index_col=0)
    tl = pd.read_csv(RES / "replay_timeline.csv"); res = pd.read_csv(RES / "stress_by_type.csv", index_col=0)
    import src.modules.stress as S_; naive = S_.naive_flat_shock(S_.load_portfolio())
    tot_b, tot_a = res.value_before.sum(), res.value_after.sum(); sig = pd.read_json(DATA / "signals.jsonl", lines=True)
    r = bt.loc["Ripple sentiment-tilt"]
    with PdfPages(DOCS / "presentation.pdf") as pdf:
        fig = plt.figure(figsize=(13.33, 7.5)); fig.text(.5, .62, "Ripple", ha="center", fontsize=60, weight="bold", color="#1a237e")
        fig.text(.5, .5, "A single calibrated, transparent NLP risk engine\npowering a fast index rebalancer and a deep stress tester", ha="center", fontsize=20)
        fig.text(.5, .3, f"{NAME}  |  {COLLEGE}\nS&P Global & Crisil Campus Hackathon 2026 - Phase III Case Study", ha="center", fontsize=15, color="#555")
        pdf.savefig(fig); plt.close(fig)
        slide(pdf, "Problem & approach", "The problem: news and social posts move prices faster than risk systems can digest them.\n\n"
              "Our approach: one engine converts text into structured, auditable signals used over two horizons:\n"
              "  - Module A (short-term, daily): index weights tilted by sentiment\n  - Module B (long-term, event-driven): stress test on a wholesale-banking book\n\n"
              "What sets it apart:\n  1. Impact is LEARNED from an event study rather than assumed\n  2. Sentiment fades per event type (half-lives), preventing erratic weights\n"
              "  3. A news-versus-social gap flags rumours and early warnings\n  4. Each signal includes evidence, confidence and reasoning\n  5. A contagion graph and deterministic replay mode")
        slide(pdf, "System design", img=DOCS / "architecture.png")
        slide(pdf, "Implementation highlights",
              "Engine: entity linking -> sentiment [-1,1] -> event category -> impact 1-10 -> decay and divergence\n"
              "  Back-ends: offline financial lexicon (default, deterministic) or FinBERT + BART-MNLI zero-shot (--model transformer)\n"
              "  Delivery: signals.jsonl plus a FastAPI service (/signals, /analyze, /stress)\n\n"
              "Module A: weight ~ exp(k * score) / N, bounded 2-15%, 10% turnover cap, 10 bps costs, no look-ahead\n"
              "  Compared with equal-weight and buy & hold, plus a placebo test that shuffles signals across tickers.\n\n"
              "Module B: graph from event to sector/country, counterparty and asset; each instrument revalued on its own terms\n"
              "  Loans: PD x LGD x EAD | Bonds: duration + convexity + credit loss | Derivatives: delta + CVA\n\n"
              "Reliability: unit tests, a one-command pipeline and seeded, reproducible data")
        slide(pdf, "Key results - Module A (synthetic prices)", f"Sentiment-tilt: return {r.total_return:.1%}, Sharpe {r.sharpe:.2f}, max DD {r.max_drawdown:.1%}, turnover {r.avg_daily_turnover:.1%}/day\n"
              f"Equal-weight: {bt.loc['Equal-weight (daily rebal)'].total_return:.1%}   Buy & hold: {bt.loc['Buy & hold'].total_return:.1%}   Placebo percentile: {pl['placebo_pctile']:.0%}\n"
              "CAVEAT: prices are synthetic and shocks follow events by construction - this validates the machinery, not alpha.\nRe-run with real prices (fetch_real_data.py) for a genuine test.", img=DOCS / "module_a.png")
        slide(pdf, "Key results - Module B", f"Auto-triggered scenario: portfolio {tot_b/1e6:,.0f}m -> {tot_a/1e6:,.0f}m ({(tot_a-tot_b)/tot_b:.1%}). Naive flat shock would show only {naive/1e6:,.0f}m and say nothing about WHERE the loss sits.\n"
              f"Replay: " + "; ".join(f"{e.split(' (')[0]} -> {p:.1%}" for e, p in tl.groupby('episode').pnl_pct.min().items()) + f"   |   {len(sig)} signals, {sig.source.nunique()} sources", img=DOCS / "module_b.png")
        slide(pdf, "Domain impact, limitations & next steps",
              "Why it matters: risk teams receive auditable, machine-readable early warnings and an impact rating based on observed market reaction;\n"
              "stress tests follow actual exposure routes (sector, country, counterparty) instead of flat shocks.\n\n"
              "Limitations: the demo uses synthetic data; the lexicon is coarse (FinBERT advised in production); stress parameters are illustrative;\n"
              "no intraday data; spill-overs are single-hop.\n\n"
              "Next steps: live GDELT/NewsAPI streams, a spill-over matrix calibrated on past crises, per-counterparty ratings,\n"
              "Kafka ingestion, plus model monitoring and drift alerts.")


if __name__ == "__main__":
    architecture(); deck(); print("docs built")
