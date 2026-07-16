import pytest

from driftguard.cli import main


def test_cli_demo(capsys):
    assert main(["--rows", "1500", "demo"]) == 0
    out = capsys.readouterr().out.lower()
    assert "baseline" in out and "drift" in out
    assert "psi=" in out


def test_cli_version():
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0


def test_api_endpoints():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from driftguard.serving.api import create_app

    client = TestClient(create_app(rows=1500))
    assert client.get("/healthz").json()["status"] == "ok"

    txn = {"amount": 1200.0, "hour": 3.0, "distance_from_home": 90.0, "txn_last_hour": 6.0,
           "merchant_risk": 0.8, "account_age_days": 20.0, "foreign": 1.0, "online": 1.0}
    pred = client.post("/predict", json=txn).json()
    assert 0.0 <= pred["fraud_probability"] <= 1.0
    assert pred["model_version"] == 1

    mon = client.post("/monitor", json={"rows": 800, "drift": True}).json()
    assert mon["report"]["dataset_drift"] is True
    assert mon["retrained"] is True and mon["new_version"] == 2

    st = client.get("/status").json()
    assert st["model_version"] == 2 and st["total_versions"] == 2

    metrics = client.get("/metrics").text
    assert "driftguard_model_version" in metrics
    assert "DriftGuard" in client.get("/").text
