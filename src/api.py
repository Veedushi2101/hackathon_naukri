"""FastAPI service exposing the engine's signals.   uvicorn src.api:app --reload
GET /health | /signals?entity=JPM&min_impact=7&source=news&limit=50 | /scores?ticker=JPM | /stress?min_impact=7
POST /analyze  {"text": "..."}  -> live signal for any headline/tweet"""
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.config import DATA
from src.engine.engine import load_signals
from src.engine.nlp import Analyzer, link_entities
from src.engine.impact import signal_impact
from src.modules import stress as S

app = FastAPI(title="SignalForge NLP Risk Engine", version="1.0")
_an = Analyzer("lexicon")
_cal = pd.read_csv(DATA / "impact_calibration.csv", index_col=0).impact_base.to_dict()


def _sig():
    return load_signals()


@app.get("/health")
def health():
    return {"status": "ok", "signals": len(_sig())}


@app.get("/signals")
def signals(entity: str | None = None, min_impact: int = 1, source: str | None = None, limit: int = 100):
    d = _sig(); d = d[d.impact >= min_impact]
    if entity: d = d[d.entity == entity.upper()]
    if source: d = d[d.source == source]
    d = d.sort_values("timestamp", ascending=False).head(limit).copy()
    d["timestamp"] = d.timestamp.astype(str)
    return d.to_dict("records")


class Item(BaseModel):
    text: str


@app.post("/analyze")
def analyze(item: Item):
    s, c, terms = _an.sentiment(item.text); et, ec = _an.event(item.text); conf = round(0.6 * c + 0.4 * ec, 3)
    return [dict(entity=e, sentiment=s, event=et, impact=signal_impact(_cal[et], s, conf), confidence=conf,
                 evidence=item.text, reasoning=terms) for e in (link_entities(item.text) or ["MARKET"])]


@app.get("/stress")
def stress(min_impact: int = S.TRIGGER_IMPACT):
    d = _sig(); scen = S.scenario_from_signals(d, min_impact)
    res = S.stress_test(S.load_portfolio(), scen); tot, by_type, _, _ = S.summarise(res)
    return dict(scenario=scen, value_before=tot.value_before, value_after=tot.value_after, pnl=tot.pnl, by_asset_type=by_type.reset_index().to_dict("records"))
