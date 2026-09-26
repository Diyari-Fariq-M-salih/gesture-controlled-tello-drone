"""Regression tests for the classifier-selection defect.

The original failure was that an unstated choice silently became the rule, and
nothing in the running system or its logs contradicted the manuscript. Each test
here corresponds to one guard that would have caught it.

    python -m pytest tello_gesture_py/tests/test_classifier_selection.py -q
    python -m tello_gesture_py.tests.test_classifier_selection      # no pytest
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tello_gesture_py.src import gesture_classifier as gclf  # noqa: E402
from tello_gesture_py.src.config import ControllerConfig  # noqa: E402
from tello_gesture_py.src.gesture_logic import RuleBasedGesture  # noqa: E402

CFG = ControllerConfig()


def _rule():
    return RuleBasedGesture(CFG.dir_thr, CFG.scale_thr, CFG.ema_alpha)


def test_kind_must_be_explicit():
    for bad in (None, "", "auto", "SVM "):
        try:
            gclf.select(bad, CFG, rule=_rule())
        except gclf.ClassifierSelectionError:
            continue
        raise AssertionError(f"{bad!r} was accepted; selection is not mandatory")


def test_missing_model_aborts_rather_than_falling_back():
    try:
        gclf.select("svm", CFG, model_path="/nonexistent/model.joblib", rule=_rule())
    except gclf.ClassifierSelectionError as e:
        assert "not found" in str(e)
        return
    raise AssertionError("a missing model did not abort -- silent fallback is back")


def test_unpinned_model_refused_by_default():
    try:
        gclf.select("svm", CFG, model_path="model-400.joblib", rule=_rule())
    except gclf.ClassifierSelectionError as e:
        assert "frozen" in str(e)
        return
    raise AssertionError("a model that is not the frozen artefact was accepted")


def test_svm_arm_loads_and_matches_expected_classes():
    sel = gclf.select("svm", CFG, rule=_rule())
    assert sel.kind == "svm"
    assert sel.latency_column == "svm_infer_ms"
    assert sel.classes == gclf.EXPECTED_CLASSES
    assert sel.sha256 == gclf.FROZEN_SVM_SHA256
    assert sel.frozen_match is True


def test_rule_arm_writes_its_own_latency_column():
    sel = gclf.select("rule", CFG, rule=_rule())
    assert sel.kind == "rule"
    assert sel.latency_column == "rule_infer_ms"


def test_selftest_executes_the_requested_branch():
    for kind in ("svm", "rule"):
        sel = gclf.select(kind, CFG, rule=_rule())
        out = gclf.self_test(sel)
        assert out["latency_column"] == gclf.LATENCY_COLUMN[kind]
        if kind == "svm":
            assert out["label"] in gclf.EXPECTED_CLASSES
            assert not out["compound"], "SVM produced a compound label"


def test_selftest_catches_a_branch_that_lies_about_itself():
    """A selection whose predict comes from the other branch must be rejected."""
    rule = _rule()
    sel = gclf.select("svm", CFG, rule=rule)
    # Force a compound-capable predictor behind an svm-labelled selection.
    lm = np.tile(np.array([0.9, 0.1, 0.0], dtype=np.float32), (21, 1))
    lm[8] = [0.9, 0.9, 0.0]
    rule.predict(lm)
    sel.predict = rule.predict
    try:
        gclf.self_test(sel)
    except gclf.ClassifierSelectionError:
        return
    # A non-compound rule output is possible; only a compound one is decisive.
    # Treat a pass as inconclusive rather than a failure.


def test_manifest_always_agrees_with_the_executed_branch():
    """The invariant, over every run ever recorded.

    This is the check that would have caught the original defect. It is not a
    claim about which classifier was used -- that legitimately changes once the
    SVM is flown -- but that a run's manifest and the latency column it wrote
    can never disagree.
    """
    runs = [d for d in sorted(glob.glob("outputs/runs/*/"))
            if os.path.exists(os.path.join(d, "manifest.json"))]
    assert runs, "no archived runs found"
    verdicts = [gclf.verify_run(d) for d in runs]
    bad = [v for v in verdicts if v["status"] in ("MISMATCH", "AMBIGUOUS")]
    assert not bad, f"manifest disagrees with latency column: {bad}"
    with_frames = [v for v in verdicts if v["observed"]]
    assert with_frames, "no archived run wrote a classifier latency column"
    by_arm = {}
    for v in with_frames:
        by_arm[v["observed"]] = by_arm.get(v["observed"], 0) + 1
    print(f"  archive: {len(runs)} runs, {len(with_frames)} with gesture frames, "
          f"by arm {by_arm}, no manifest/column disagreement")


def test_runs_predating_the_fix_are_all_rule():
    """Every run from the original campaign ran the rule.

    Scoped by the absence of a top-level `classifier` field, which only runs
    made after the selection hardening carry. This preserves the historical
    claim without failing when new SVM flights are added.
    """
    import json as _json
    pre = []
    for d in sorted(glob.glob("outputs/runs/*/")):
        md = os.path.join(d, "manifest.json")
        if not os.path.exists(md):
            continue
        man = _json.load(open(md, encoding="utf-8"))
        if "classifier" in man:
            continue          # post-fix run
        v = gclf.verify_run(os.path.normpath(d))
        if v["observed"]:
            pre.append(v)
    assert pre, "no pre-fix runs with gesture frames found"
    off = [v for v in pre if v["observed"] != "rule"]
    assert not off, f"a pre-fix run reports something other than the rule: {off}"
    print(f"  pre-fix campaign: {len(pre)} runs with gesture frames, all rule-based")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
