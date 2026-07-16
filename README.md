<div align="center">

# DriftGuard

### Real-Time Streaming & Drift-Detection MLOps Pipeline

A fraud-detection system that **monitors its own health and fixes itself**. It
serves predictions, watches the live data distribution with the Population
Stability Index, and — when a new fraud pattern shifts the data — fires an alert
and **automatically retrains, registers, and hot-reloads** a new model version.

[![CI](https://github.com/avase33/driftguard/actions/workflows/ci.yml/badge.svg)](https://github.com/avase33/driftguard/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-000000.svg)](https://github.com/astral-sh/ruff)

</div>

---

## The scenario

You deploy a fraud model. Three months later a new fraud pattern emerges, changing
the statistical properties of incoming transactions — and your model silently
decays. DriftGuard catches this automatically across **four operational zones**:

```
Ingestion (producer → Kafka bus) → Serving (predict, latest model)
                                  → Monitoring (batch → PSI vs reference)
                                  → Retraining (on drift: retrain → register → reload)
```

## What's implemented (and actually runs)

Everything is **pure Python** and runs with **zero external services** offline:

- **XGBoost-style GBDT from scratch** — histogram-binned split finding, second-order
  gain `0.5·(GL²/(HL+λ)+GR²/(HR+λ)−G²/(H+λ))`, leaf values `−G/(H+λ)`,
  `min_child_weight`/`reg_lambda` regularisation. Trains a strong fraud classifier
  in under a second. (`DRIFTGUARD_MODEL=xgboost` swaps in real XGBoost.)
- **PSI drift detection from scratch** — Population Stability Index per feature,
  with the standard thresholds (<0.1 stable · 0.1–0.2 monitor · ≥0.2 retrain).
- **MLflow-style model registry** — versioned models with metrics/params; serving
  always loads the latest and hot-reloads after a retrain.
- **Kafka-style streaming** — partitioned bus + row-by-row transaction producer.
- **Automated retraining loop** — drift → alert (webhook/Slack) → retrain →
  register → reload, mirrored by a GitHub Actions `retrain.yml` workflow triggered
  by `repository_dispatch`.
- **FastAPI serving + live monitoring dashboard** (`/predict`, `/monitor`,
  `/status`, `/metrics`).

## Quickstart (no keys, no servers)

```bash
pip install -e .
driftguard demo
```

Real output:

```
[1] Baseline model v1 trained on reference data
    reference F1=0.71  AUC=0.93  precision=0.78  recall=0.65
[2] Three months later — new fraud pattern emerges
    v1 F1 on drifted data dropped to 0.48 (AUC 0.79)
[3] Monitoring zone — PSI drift check (3000 samples)
    dataset drift: True  (88% of features shifted)
      distance_from_home   PSI= 1.842  [DRIFT]
      amount               PSI= 0.531  [DRIFT]
      online               PSI= 0.402  [DRIFT]
      merchant_risk        PSI= 0.311  [DRIFT]
      ...
[4] Retraining zone — drift alert fired -> auto-retrained
    registered model v2; F1 on drifted data recovered 0.48 -> 0.74
Registry now has 2 model versions.
```

## Run the serving API + dashboard

```bash
pip install -e ".[serve]"
driftguard serve            # http://localhost:8000  (dashboard + API)
```

```
POST /predict   {amount, hour, distance_from_home, ...} -> fraud probability + model version
POST /monitor   run a PSI drift check on a batch (auto-retrains on drift)
GET  /status    model versions + monitoring state
GET  /metrics   Prometheus exposition
```

## The math: Population Stability Index

```
PSI = Σ_i (Actual_i − Expected_i) · ln(Actual_i / Expected_i)
```

`Expected_i`/`Actual_i` are the fractions of records in bin *i* for the reference
vs current samples (bins from the reference quantiles). **PSI < 0.1** stable ·
**0.1–0.2** monitor · **≥ 0.2** drift → retrain. See `driftguard/drift/psi.py`.

## Full stack (Docker Compose)

```bash
docker compose up --build     # serving + Kafka + MLflow + Prometheus
```

The **automated retraining workflow** (`.github/workflows/retrain.yml`) triggers on
a `repository_dispatch` of type `drift-detected` (the drift webhook), retrains,
registers a new version, and uploads the artifact.

## Repository layout

```
driftguard/
  data/            synthetic fraud generator (reference + drifted)
  model/           XGBoost-style GBDT (from scratch) + metrics
  drift/           PSI + drift detector
  registry.py      MLflow-style model registry (versioning + hot-reload)
  streaming/       Kafka-style bus + transaction producer
  monitoring.py    batch → drift check → alert
  retraining.py    retrain → register (auto recovery)
  engine.py        four-zone orchestration   |  serving/  FastAPI + dashboard
scripts/           train_baseline.py, producer.py, retrain.py
docker-compose.yml, Dockerfile, .github/workflows/{ci,retrain}.yml
```

## Development

```bash
pip install -e ".[serve,dev]"
pytest --cov=driftguard
ruff check driftguard scripts
python verify_driftguard.py     # offline end-to-end self-check
```

## License

MIT © Akhil Vase
