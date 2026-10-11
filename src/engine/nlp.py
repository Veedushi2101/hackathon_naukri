"""NLP primitives: entity linking, sentiment, event classification.

Two back-ends, same interface:
  * 'lexicon' (default): offline, dependency-free, financial lexicon with negation +
    intensifiers. Fast and deterministic - used for the demo and for tests.
  * 'transformer': ProsusAI/finbert (sentiment) + facebook/bart-large-mnli (zero-shot
    event type). Enabled with  --model transformer  if `transformers` + `torch` are installed.
    Keyword rules are still used as a prior, and agreement between the two raises confidence.
"""
import math, re
from src.config import UNIVERSE, EVENT_TYPES

POS = {"beats": 1.2, "beat": 1.2, "upgrade": 1.3, "upgraded": 1.3, "record": 1.0, "surge": 0.8, "surges": 0.8, "growth": 0.8,
       "strong": 0.8, "robust": 0.8, "upbeat": 0.9, "wins": 1.0, "approval": 0.9, "approved": 0.9, "cleared": 1.0, "dismissed": 0.9,
       "raises": 0.7, "boost": 0.9, "lift": 0.7, "lifts": 0.7, "benefit": 0.8, "gains": 0.8, "innovative": 0.8, "unveils": 0.6,
       "ceasefire": 1.0, "deal": 0.5, "agreement": 0.6, "refinances": 0.6, "improves": 0.8, "cooling": 0.6, "cut hopes": 0.6,
       "bullish": 1.2, "moon": 1.0, "great": 0.8, "loading up": 0.9, "steady": 0.4, "backstop": 0.5, "premium": 0.4, "jump": 0.5,
       "profit": 0.5, "lower rates": 0.7, "looking strong": 1.0, "launches": 0.5}
NEG = {"misses": 1.2, "miss": 1.0, "downgrade": 1.3, "downgraded": 1.3, "default": 1.2, "defaults": 1.2, "bad debt": 1.2, "breach": 1.1,
       "spike": 0.7, "crunch": 1.2, "collapse": 1.4, "collapses": 1.4, "block": 0.8, "blocked": 0.8, "lawsuit": 1.0, "fine": 0.7,
       "probe": 0.8, "penalty": 0.9, "antitrust": 0.8, "recall": 1.0, "recalls": 1.0, "defect": 1.0, "delayed": 0.8, "disappointed": 0.9,
       "falls": 0.8, "fall": 0.8, "cuts": 0.8, "cut guidance": 1.0, "weak": 0.8, "recession": 1.3, "inflation": 0.5, "tightening": 0.7,
       "war": 1.3, "sanctions": 1.0, "conflict": 1.0, "tensions": 0.8, "escalation": 1.0, "escalate": 1.0, "tariffs": 0.8, "disrupted": 1.0,
       "threaten": 1.0, "fears": 0.8, "worried": 0.8, "fraud": 1.4, "crash": 1.2, "crashing": 1.2, "done": 0.6, "selling": 0.8,
       "rumours": 0.5, "bad": 0.8, "worse": 0.9, "run": 0.4, "seize": 1.0, "failed": 1.2, "withdraw": 0.7, "loss": 0.9, "contagion": 1.1,
       "wide": 0.3, "warn": 0.8, "warns": 0.8, "warned": 0.8, "hit": 0.6, "hits": 0.6, "pressure": 0.7, "weigh": 0.6, "overpays": 0.8,
       "buildup": 0.5, "worries": 0.7, "😱": 1.0, "rate hike": 0.8, "falls": 0.8, "pull": 0.4}
NEGATORS = {"no", "not", "never", "without", "fails", "cleared", "dismissed"}
BOOST = {"sharply": 1.4, "heavy": 1.3, "massive": 1.4, "huge": 1.3, "major": 1.2, "record": 1.0, "slightly": 0.6}

EVENT_KW = {
    "Credit Event": ["credit", "default", "downgrade", "upgrade", "bad debt", "liquidity", "covenant", "bond yield", "lenders", "debt", "loan", "bank run", "deposit", "spreads", "refinanc"],
    "Geopolitical": ["war", "sanction", "conflict", "geopolit", "ceasefire", "military", "border", "tariff", "trade war", "supply chain", "tension", "invasion"],
    "Macroeconomic": ["inflation", "recession", "gdp", "central bank", "rate hike", "rate cut", "interest rate", "tightening", "unemployment", "consumer demand", "oil"],
    "Merger/Acquisition": ["merger", "acquisition", "acquire", "takeover", "buyout", "bid", "deal at premium"],
    "Product Launch": ["launch", "unveil", "recall", "product", "new service", "defect"],
    "Earnings": ["earnings", "quarterly", "profit", "revenue", "guidance", "eps", "estimates"],
    "Regulatory/Legal": ["lawsuit", "antitrust", "regulat", "probe", "fine", "compliance", "court", "investigation", "ruling", "sec "],
}

_ALIAS = []
for tk, (name, _s, al) in UNIVERSE.items():
    for a in {name.lower(), *al}:
        _ALIAS.append((a, tk))
_ALIAS.sort(key=lambda x: -len(x[0]))


def link_entities(text: str):
    """Return tickers mentioned via cashtag ($JPM), ticker symbol or company alias."""
    low, found = text.lower(), []
    for m in re.findall(r"\$([A-Za-z]{1,5})", text):
        if m.upper() in UNIVERSE and m.upper() not in found:
            found.append(m.upper())
    for alias, tk in _ALIAS:
        if re.search(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])", low) and tk not in found:
            found.append(tk)
    return found


def lexicon_sentiment(text: str):
    """Return (score in [-1,1], confidence in [0,1], evidence_terms)."""
    low = text.lower()
    toks = re.findall(r"[a-z&']+|[^\sa-z0-9]", low)
    total, hits, terms = 0.0, 0, []
    joined = " " + low + " "
    for table, sign in ((POS, 1), (NEG, -1)):
        for w, wt in table.items():
            if " " in w or not w.isascii():
                if w in joined:
                    total += sign * wt; hits += 1; terms.append(w)
            else:
                for i, t in enumerate(toks):
                    if t == w:
                        mult = 1.0
                        ctx = toks[max(0, i - 3):i]
                        if any(c in NEGATORS for c in ctx if c not in ("cleared", "dismissed")):
                            mult *= -0.8
                        for c in ctx:
                            mult *= BOOST.get(c, 1.0)
                        total += sign * wt * mult; hits += 1; terms.append(w)
    score = math.tanh(0.7 * total)
    conf = min(0.95, 0.35 + 0.15 * hits) if hits else 0.25
    return round(score, 3), round(conf, 3), terms[:6]


def keyword_event(text: str):
    low = text.lower()
    scores = {et: sum(1 for k in kws if k in low) for et, kws in EVENT_KW.items()}
    best = max(scores, key=scores.get)
    tot = sum(scores.values())
    if scores[best] == 0:
        return "Other", 0.3
    return best, round(scores[best] / tot, 3)


class Analyzer:
    def __init__(self, backend: str = "lexicon"):
        self.backend = backend
        self._fb = self._zs = None
        if backend == "transformer":
            from transformers import pipeline  # optional heavy dependency
            self._fb = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)
            self._zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def sentiment(self, text):
        s, c, terms = lexicon_sentiment(text)
        if self._fb is not None:
            out = {d["label"].lower(): d["score"] for d in self._fb(text[:512])[0]}
            fs = out.get("positive", 0) - out.get("negative", 0)
            agree = (fs * s > 0) or abs(s) < .1
            return round(0.7 * fs + 0.3 * s, 3), round(min(.97, max(out.values()) * (1.0 if agree else .7)), 3), terms
        return s, c, terms

    def event(self, text):
        et, c = keyword_event(text)
        if self._zs is not None:
            r = self._zs(text[:512], EVENT_TYPES[:-1])
            if r["scores"][0] > 0.5:
                if r["labels"][0] == et:
                    return et, min(.97, (c + r["scores"][0]) / 2 + .2)
                if c < 0.6:
                    return r["labels"][0], round(r["scores"][0] * .8, 3)
        return et, c
