# Architecture

DriftGuard is a self-monitoring ML system: it serves a fraud model, watches its
own input distribution, and automatically retrains when the world changes. Work
flows through four operational zones.

```
   ┌──────────────┐   transactions   ┌──────────────┐
   │  Ingestion   │ ───────────────► │   Serving    │  POST /predict
   │  producer →  │                  │  latest model│  (hot-reloaded on retrain)
   │  Kafka bus   │ ───────┐         └──────────────┘
   └──────────────┘        │ batches          ▲
                           ▼                   │ new version
                  ┌──────────────┐   drift?    │
                  │  Monitoring  │ ── PSI ──►  │
                  │  vs reference│             │
                  └──────┬───────┘             │
                         │ dataset drift       │
                         ▼ (webhook)     ┌──────────────┐
                    alert / dispatch ──► │  Retraining  │
                                         │ retrain →    │
                                         │ register →   │
                                         │ redeploy     │
                                         └──────────────┘
```

## The zones

**Ingestion** — a producer streams simulated transactions (reference, then a
drifted stream with a new fraud pattern) into a Kafka-style partitioned bus,
keyed so each account keeps ordering.

**Serving** — a FastAPI service scores transactions with the **latest** model
version from the registry, and hot-reloads after an automated retrain.

**Monitoring** — batches the live stream and compares it against the training
**reference** using the Population Stability Index. A feature is "drifted" when its
PSI ≥ 0.2; the dataset has drifted when > 30% of features drift.

**Retraining** — on a drift alert, trains a fresh model on the newly-collected
data, evaluates it, registers a new version in the model registry, and hands it
back for hot-reload. In production the GitHub Actions `retrain.yml` workflow runs
these steps on a `repository_dispatch` from the drift webhook.

## The model (from scratch)

`model/gbdt.py` is a compact, faithful **XGBoost-style** gradient booster:
histogram-binned features, second-order split gain
`0.5·(GL²/(HL+λ)+GR²/(HR+λ)−G²/(H+λ))`, leaf values `−G/(H+λ)`, `min_child_weight`
and `reg_lambda` regularisation. Pure Python — trains a solid fraud classifier in
under a second. A real-XGBoost adapter slots in via `DRIFTGUARD_MODEL=xgboost`.

## The math: Population Stability Index

```
PSI = Σ_i (Actual_i − Expected_i) · ln(Actual_i / Expected_i)
```

Expected/Actual are the fractions of records per bin (bins from the reference
quantiles). Interpretation: **< 0.1** stable · **0.1–0.2** monitor · **≥ 0.2**
drift → retrain. Implemented from scratch in `drift/psi.py`.

## Offline-first

The from-scratch GBDT, local JSON registry, PSI detection and in-memory bus mean
the whole four-zone loop runs, tests and verifies with **zero external services**.
Adapters for XGBoost, MLflow, Kafka and Evidently switch on via environment
variables without changing the pipeline.
