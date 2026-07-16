"""FastAPI serving zone + monitoring dashboard.

    POST /predict       score a transaction with the latest model
    POST /monitor       run a drift check on a batch (auto-retrains on drift)
    GET  /status        pipeline status + model versions
    GET  /metrics       Prometheus exposition
    GET  /healthz       liveness
    GET  /              live monitoring dashboard (React via CDN, no build)

The serving instance loads the latest model version and hot-reloads after an
automated retrain. FastAPI/uvicorn are optional; a baseline model is trained on
synthetic data at startup so the API is demoable instantly.
"""

from __future__ import annotations

from typing import Optional

from ..config import Settings
from ..data.mockdata import generate
from ..engine import DriftGuard
from ..models import FEATURES


def create_app(rows: int = 3000, seed: int = 7, engine: Optional[DriftGuard] = None):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
    from pydantic import BaseModel

    app = FastAPI(title="DriftGuard", version="0.1.0",
                  description="Real-time drift detection & self-retraining fraud pipeline")

    if engine is None:
        engine = DriftGuard(Settings())
        engine.fit_reference(generate(rows, seed=seed), generate(max(400, rows // 3), seed=seed + 1))
    app.state.engine = engine

    class Txn(BaseModel):
        amount: float
        hour: float
        distance_from_home: float
        txn_last_hour: float
        merchant_risk: float
        account_age_days: float
        foreign: float
        online: float

    class MonitorBody(BaseModel):
        rows: int = 1000
        drift: bool = True

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "version": "0.1.0"}

    @app.post("/predict")
    def predict(txn: Txn) -> dict:
        row = [getattr(txn, f) for f in FEATURES]
        return app.state.engine.predict(row).to_dict()

    @app.post("/monitor")
    def monitor(body: MonitorBody) -> JSONResponse:
        eng: DriftGuard = app.state.engine
        cur = generate(body.rows, seed=99, drift=body.drift)
        test = generate(max(400, body.rows // 3), seed=98, drift=body.drift)
        res = eng.run_cycle(cur, test)
        return JSONResponse({"report": res.report.to_dict(), "retrained": res.retrained,
                             "new_version": res.new_version,
                             "f1_before": round(res.prev_f1, 4), "f1_after": round(res.new_f1, 4)})

    @app.get("/status")
    def status() -> JSONResponse:
        return JSONResponse(app.state.engine.status())

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        st = app.state.engine.status()
        m = st["monitor"]
        latest = st["versions"][-1]["metrics"] if st["versions"] else {}
        lines = [
            "# TYPE driftguard_model_version gauge",
            f"driftguard_model_version {st['model_version']}",
            "# TYPE driftguard_drift_events_total counter",
            f"driftguard_drift_events_total {m['drift_events']}",
            "# TYPE driftguard_last_drift_share gauge",
            f"driftguard_last_drift_share {m['last_drift_share']}",
            "# TYPE driftguard_model_f1 gauge",
            f"driftguard_model_f1 {latest.get('f1', 0)}",
            "# TYPE driftguard_bus_lag gauge",
            f"driftguard_bus_lag {st['bus_lag']}",
        ]
        return "\n".join(lines) + "\n"

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        from .dashboard import DASHBOARD_HTML
        return DASHBOARD_HTML

    return app


def run_server(host: str = "127.0.0.1", port: int = 8000, rows: int = 3000) -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(create_app(rows=rows), host=host, port=port)
