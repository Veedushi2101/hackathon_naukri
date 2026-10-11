"""Run: python -m pytest -q   (or: python tests/test_core.py)"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from src.engine.nlp import Analyzer, link_entities, lexicon_sentiment, keyword_event
from src.engine.impact import signal_impact, decay_weight
from src.modules.rebalancer import target_weights
from src.modules import stress as S


def test_sentiment_sign():
    assert lexicon_sentiment("Apple beats earnings estimates and raises guidance")[0] > 0.5
    assert lexicon_sentiment("Boeing recalls product after major defect, customers disappointed")[0] < -0.5

def test_entity_linking():
    assert link_entities("$JPM and Goldman Sachs fall") == ["JPM", "GS"]
    assert link_entities("nothing relevant here") == []

def test_event_types():
    assert keyword_event("sanctions and war escalation")[0] == "Geopolitical"
    assert keyword_event("rating downgrade and loan defaults")[0] == "Credit Event"

def test_impact_bounds_and_decay():
    assert 1 <= signal_impact(8.5, -0.9, 0.9) <= 10
    assert decay_weight(0, "Earnings") == 1 and decay_weight(3, "Earnings") == 0.5

def test_weights_valid():
    w = target_weights(pd.Series(np.random.default_rng(0).uniform(-1, 1, 14), index=list("abcdefghijklmn")))
    assert abs(w.sum() - 1) < 1e-9 and w.min() >= 0.02 - 1e-9 and w.max() <= 0.15 + 1e-9

def test_stress_monotonic():
    pf = S.load_portfolio()
    lo, hi = S.stress_test(pf, {"Credit Event": 0.3}).pnl.sum(), S.stress_test(pf, {"Credit Event": 0.9}).pnl.sum()
    assert hi < lo < 0  # more severe event -> bigger loss
    assert abs(S.stress_test(pf, {}).pnl.sum()) < 1e-6  # no event -> no P&L

if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"): v(); print("ok", k)
