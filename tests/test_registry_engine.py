import pytest

from driftguard.config import Settings
from driftguard.data.mockdata import generate
from driftguard.engine import DriftGuard
from driftguard.model.gbdt import GradientBoostedTrees
from driftguard.registry import LocalRegistry
from driftguard.streaming.bus import PartitionedBus, partition_for
from driftguard.streaming.producer import TransactionProducer


def test_registry_versions_and_latest():
    reg = LocalRegistry()
    m1 = GradientBoostedTrees().fit(generate(200, seed=1).X, generate(200, seed=1).y)
    reg.log_model(m1, {"f1": 0.5}, {"a": 1}, "initial")
    reg.log_model(m1, {"f1": 0.6}, {"a": 2}, "drift")
    assert reg.latest_version() == 2
    mv, model = reg.latest()
    assert mv.version == 2 and mv.reason == "drift"
    assert len(reg.versions()) == 2


def test_model_serialisation_roundtrip():
    train = generate(400, seed=3)
    m = GradientBoostedTrees(n_estimators=20).fit(train.X, train.y)
    state = m.to_state()
    m2 = GradientBoostedTrees.from_state(state)
    a = m.predict_proba(train.X[:20])
    b = m2.predict_proba(train.X[:20])
    assert all(abs(x - y) < 1e-9 for x, y in zip(a, b))


def test_streaming_producer_and_partitioning():
    assert partition_for("acct_1", 4) == partition_for("acct_1", 4)
    bus = PartitionedBus(4)
    n = TransactionProducer(bus).stream(generate(300, seed=2))
    assert n == 300 and bus.lag() == 300
    msgs = bus.poll()
    assert len(msgs) == 300 and "features" in msgs[0]


@pytest.fixture(scope="module")
def guarded():
    dg = DriftGuard(Settings())
    base = dg.fit_reference(generate(3000, seed=7), generate(900, seed=8))
    return dg, base


def test_baseline_trains_and_predicts(guarded):
    dg, base = guarded
    assert base["auc"] > 0.8
    pred = dg.predict(generate(1, seed=5).X[0])
    assert 0.0 <= pred.proba <= 1.0 and pred.model_version == 1


def test_drift_cycle_detects_and_retrains(guarded):
    dg, _ = guarded
    drift_test = generate(900, seed=9, drift=True)
    prev_f1 = dg.evaluate(drift_test)["f1"]
    result = dg.run_cycle(generate(3000, seed=10, drift=True), drift_test)
    assert result.report.dataset_drift
    assert result.retrained and result.new_version == 2
    # retraining on the drifted distribution should not do worse on drifted data
    assert result.new_f1 >= prev_f1
    assert dg.registry.latest_version() == 2
