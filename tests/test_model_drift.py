import random

from driftguard.config import Settings
from driftguard.data.mockdata import generate
from driftguard.model.gbdt import GradientBoostedTrees
from driftguard.model.metrics import classification_metrics, roc_auc
from driftguard.drift.psi import population_stability_index
from driftguard.drift.detector import DriftDetector


def test_gbdt_learns_fraud():
    train = generate(2000, seed=7)
    test = generate(700, seed=8)
    model = GradientBoostedTrees(n_estimators=60, max_depth=3, learning_rate=0.3)
    model.fit(train.X, train.y)
    m = classification_metrics(test.y, model.predict_proba(test.X))
    assert m["auc"] > 0.8, f"AUC too low: {m['auc']:.3f}"
    assert m["f1"] > 0.2


def test_roc_auc_bounds():
    assert roc_auc([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2]) == 1.0
    assert abs(roc_auc([1, 0], [0.5, 0.5]) - 0.5) < 1e-9


def test_psi_zero_for_same_distribution():
    r = random.Random(0)
    x = [r.gauss(0, 1) for _ in range(2000)]
    assert population_stability_index(x, x, bins=10) < 0.01


def test_psi_high_for_shift():
    r = random.Random(0)
    ref = [r.gauss(0, 1) for _ in range(2000)]
    cur = [r.gauss(2.5, 1) for _ in range(2000)]
    psi = population_stability_index(ref, cur, bins=10)
    assert psi >= 0.2, f"expected drift, PSI={psi:.3f}"


def test_detector_flags_drift_not_stable():
    ref = generate(1500, seed=7, drift=False)
    det = DriftDetector(ref, Settings())

    stable = det.detect(generate(600, seed=8, drift=False).X)
    assert not stable.dataset_drift

    drifted = det.detect(generate(600, seed=9, drift=True).X)
    assert drifted.dataset_drift
    assert drifted.drift_share > 0.3
    # amount/distance/merchant_risk should be among the drifted features
    drifted_names = {f.feature for f in drifted.features if f.drifted}
    assert "amount" in drifted_names or "distance_from_home" in drifted_names
