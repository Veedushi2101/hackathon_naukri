"""Module B - Event-driven stress testing with a contagion graph.

Event signal (type, impact) -> sector/country vulnerability -> counterparty -> asset
and, in parallel, event -> market risk factors (rates, spreads, equity, FX, oil).
Each instrument is re-valued with its own mechanics:
  Loan        : expected loss = PD * LGD * EAD   (PD and LGD stressed by vulnerability)
  Bond        : dP/P = -D*dy + 0.5*C*dy^2  minus credit-loss on stressed PD
  Derivative  : delta * notional * underlying move, minus CVA on positive MTM
A flat 'naive' shock (equity -10%, rates +200bp, same for every asset) is reported for comparison.
"""
import numpy as np, pandas as pd, networkx as nx
from src.config import DATA

TRIGGER_IMPACT = 7

# factor shocks at severity 1.0:  rates_bp, spread_bp, equity, fx(EM depreciation), oil
FACTORS = {
    "Credit Event":       dict(rates_bp=-20, spread_bp=150, equity=-0.10, fx=-0.02, oil=0.00),
    "Geopolitical":       dict(rates_bp=-30, spread_bp=100, equity=-0.08, fx=-0.06, oil=0.25),
    "Macroeconomic":      dict(rates_bp=75,  spread_bp=60,  equity=-0.07, fx=-0.03, oil=-0.10),
    "Merger/Acquisition": dict(rates_bp=0,   spread_bp=20,  equity=-0.02, fx=0.0,   oil=0.0),
    "Product Launch":     dict(rates_bp=0,   spread_bp=5,   equity=-0.01, fx=0.0,   oil=0.0),
    "Earnings":           dict(rates_bp=0,   spread_bp=20,  equity=-0.03, fx=0.0,   oil=0.0),
    "Regulatory/Legal":   dict(rates_bp=0,   spread_bp=40,  equity=-0.04, fx=0.0,   oil=0.0),
    "Other":              dict(rates_bp=0,   spread_bp=0,   equity=0.0,   fx=0.0,   oil=0.0),
}
# event -> sector vulnerability (0-1)
SECTOR_V = {
    "Credit Event":  dict(Financials=1.0, **{"Real Estate": .7}, Industrials=.4, Consumer=.3, Energy=.3, Technology=.2),
    "Geopolitical":  dict(Energy=.9, Industrials=.7, Technology=.5, Financials=.4, Consumer=.4, Healthcare=.2),
    "Macroeconomic": dict(**{"Real Estate": .9}, Financials=.6, Consumer=.7, Industrials=.6, Technology=.5, Utilities=.4),
    "Merger/Acquisition": dict(Financials=.3, Technology=.3),
    "Regulatory/Legal": dict(Technology=.6, Financials=.6, Healthcare=.5, Energy=.4),
    "Earnings": dict(Technology=.3, Consumer=.3), "Product Launch": dict(Technology=.2), "Other": {},
}
COUNTRY_V = {
    "Geopolitical": dict(RU=1.0, TR=.7, ZA=.5, CN=.6, IN=.3, BR=.4, MX=.4, DE=.4, UK=.2, US=.1),
    "Credit Event": dict(TR=.6, BR=.5, ZA=.5, MX=.4, IN=.3, US=.3, UK=.3, DE=.2, CN=.4, RU=.5),
    "Macroeconomic": dict(TR=.7, BR=.6, ZA=.6, MX=.5, IN=.4, US=.3, UK=.3, DE=.3, CN=.3, RU=.4),
}
# one-hop sector spill-overs (source -> target)
SPILL = {("Financials", "Real Estate"): .5, ("Financials", "Industrials"): .3, ("Financials", "Consumer"): .3,
         ("Energy", "Industrials"): .4, ("Energy", "Utilities"): .3, ("Real Estate", "Financials"): .4}


def load_portfolio():
    return pd.read_csv(DATA / "portfolio.csv")


def scenario_from_signals(sig: pd.DataFrame, threshold=TRIGGER_IMPACT):
    """Adverse high-impact signals -> {event_type: severity in (0,1]} (max, no double counting)."""
    hit = sig[(sig.impact >= threshold) & (sig.sentiment < -0.2)]
    return {et: float(g.impact.max()) / 10 for et, g in hit.groupby("event")}


def vulnerabilities(pf, scenario):
    """Spill-over aware sector vulnerability + country vulnerability, combined per counterparty."""
    sec_v = {}
    for et, sev in scenario.items():
        base = {s: v * sev for s, v in SECTOR_V.get(et, {}).items()}
        spilled = dict(base)
        for (a, b), w in SPILL.items():
            if a in base:
                spilled[b] = max(spilled.get(b, 0), base[a] * w)
        for s, v in spilled.items():
            sec_v[s] = 1 - (1 - sec_v.get(s, 0)) * (1 - v)
    ctry_v = {}
    for et, sev in scenario.items():
        for c, v in COUNTRY_V.get(et, {}).items():
            ctry_v[c] = 1 - (1 - ctry_v.get(c, 0)) * (1 - v * sev)
    return sec_v, ctry_v


def combine_factors(scenario):
    f = dict(rates_bp=0.0, spread_bp=0.0, equity=0.0, fx=0.0, oil=0.0)
    for et, sev in scenario.items():
        for k, v in FACTORS[et].items():
            f[k] += v * sev
    f["equity"] = max(f["equity"], -0.45)
    return f


def stress_test(pf: pd.DataFrame, scenario: dict) -> pd.DataFrame:
    pf = pf.copy()
    sec_v, ctry_v = vulnerabilities(pf, scenario)
    f = combine_factors(scenario)
    pf["vuln"] = [1 - (1 - sec_v.get(s, 0)) * (1 - ctry_v.get(c, 0)) for s, c in zip(pf.sector, pf.country)]
    pd_s = np.minimum(1.0, pf.pd_base * (1 + 6 * pf.vuln))
    lgd_s = np.minimum(0.95, pf.lgd * (1 + 0.25 * pf.vuln))
    base_el, stress_el = pf.pd_base * pf.lgd * pf.notional, pd_s * lgd_s * pf.notional
    before, after = np.zeros(len(pf)), np.zeros(len(pf))
    for i, r in enumerate(pf.itertuples()):
        if r.asset_type == "Loan":
            before[i], after[i] = r.notional - base_el.iloc[i], r.notional - stress_el.iloc[i]
        elif r.asset_type == "Bond":
            dy = (f["rates_bp"] + f["spread_bp"] * (0.4 + r.vuln)) / 1e4
            dP = -r.duration * dy + 0.5 * r.convexity * dy ** 2
            before[i] = r.notional - base_el.iloc[i]
            after[i] = r.notional * (1 + dP) - stress_el.iloc[i]
        else:
            move = {"EQUITY": f["equity"], "FX": f["fx"] * (0.5 + r.vuln), "OIL": f["oil"],
                    "RATES": -f["rates_bp"] / 1e4 * 5}.get(r.underlying, 0.0)
            mtm0 = r.delta * r.notional * 0.05
            mtm1 = mtm0 + r.delta * r.notional * move
            cva = max(mtm1, 0) * pd_s.iloc[i] * lgd_s.iloc[i]
            before[i], after[i] = mtm0 - max(mtm0, 0) * r.pd_base * r.lgd, mtm1 - cva
    pf["value_before"], pf["value_after"] = before, after
    pf["pnl"] = pf.value_after - pf.value_before
    return pf


def naive_flat_shock(pf: pd.DataFrame) -> float:
    """Benchmark: same -10% equity / +200bp rates for everyone - ignores who the counterparty is."""
    pnl = 0.0
    for r in pf.itertuples():
        if r.asset_type == "Bond":
            pnl += r.notional * (-r.duration * 0.02 + 0.5 * r.convexity * 0.0004)
        elif r.asset_type == "Derivative" and r.underlying == "EQUITY":
            pnl += r.delta * r.notional * -0.10
    return pnl


def summarise(res: pd.DataFrame):
    by = lambda col: res.groupby(col)[["value_before", "value_after", "pnl"]].sum().assign(
        pnl_pct=lambda d: d.pnl / d.value_before.abs()).sort_values("pnl")
    tot = res[["value_before", "value_after", "pnl"]].sum()
    return tot, by("asset_type"), by("sector"), by("country")


def build_graph(pf, scenario, res=None):
    """networkx graph  event -> sector/country -> counterparty -> asset, with 'heat' per node."""
    G = nx.DiGraph()
    sec_v, ctry_v = vulnerabilities(pf, scenario)
    for et, sev in scenario.items():
        G.add_node(f"EVT:{et}", kind="event", heat=sev, label=et)
        for s, v in SECTOR_V.get(et, {}).items():
            G.add_edge(f"EVT:{et}", f"SEC:{s}", w=v * sev)
        for c, v in COUNTRY_V.get(et, {}).items():
            G.add_edge(f"EVT:{et}", f"CTY:{c}", w=v * sev)
    for s in pf.sector.unique():
        G.add_node(f"SEC:{s}", kind="sector", heat=sec_v.get(s, 0), label=s)
    for c in pf.country.unique():
        G.add_node(f"CTY:{c}", kind="country", heat=ctry_v.get(c, 0), label=c)
    agg = pf.groupby("counterparty").first()[["sector", "country"]]
    pnl_by_cp = res.groupby("counterparty").pnl.sum() if res is not None else {}
    vmax = max([abs(v) for v in pnl_by_cp.values] + [1]) if res is not None else 1
    for cp, r in agg.iterrows():
        h = 1 - (1 - sec_v.get(r.sector, 0)) * (1 - ctry_v.get(r.country, 0))
        G.add_node(f"CP:{cp}", kind="counterparty", heat=h, label=cp)
        G.add_edge(f"SEC:{r.sector}", f"CP:{cp}", w=sec_v.get(r.sector, 0)); G.add_edge(f"CTY:{r.country}", f"CP:{cp}", w=ctry_v.get(r.country, 0))
    for r in (res if res is not None else pf).itertuples():
        G.add_node(f"AST:{r.asset_id}", kind="asset", heat=0.0 if res is None else min(1, abs(r.pnl) / (abs(res.pnl).max() + 1e-9)), label=r.asset_id)
        G.add_edge(f"CP:{r.counterparty}", f"AST:{r.asset_id}", w=1.0)
    G.remove_nodes_from([n for n, d in G.nodes(data=True) if "kind" not in d])  # sectors/countries absent from the book
    return G


def replay(sig: pd.DataFrame, pf: pd.DataFrame, threshold=TRIGGER_IMPACT):
    """Step through signals in time order; after each new trigger re-run the stress test.
    Returns a DataFrame timeline of (timestamp, scenario, portfolio pnl)."""
    sig = sig.sort_values("timestamp"); rows, scen = [], {}
    for r in sig.itertuples():
        if r.impact >= threshold and r.sentiment < -0.2 and scen.get(r.event, 0) < r.impact / 10:
            scen = {**scen, r.event: r.impact / 10}
            res = stress_test(pf, scen)
            tot = res.pnl.sum()
            rows.append(dict(timestamp=r.timestamp, trigger=f"{r.event} (impact {r.impact})", evidence=r.evidence,
                             scenario=dict(scen), pnl=tot, pnl_pct=tot / res.value_before.abs().sum()))
    return pd.DataFrame(rows)
