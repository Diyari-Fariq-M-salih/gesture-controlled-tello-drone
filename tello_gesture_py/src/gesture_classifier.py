"""Explicit, validated selection of the gesture classifier.

The controller previously chose between the RBF-SVM and the geometric rule by
testing whether a model path happened to be supplied, and fell back silently when
it was not. No logged launch ever supplied one, so an entire flight campaign ran
the rule while the manuscript characterised the SVM, and nothing in the running
system reported the difference.

Selection is now explicit and mandatory. `--classifier` has no default, the SVM
arm validates its model and label set before the aircraft is armed, the resolved
choice is recorded in the run manifest with the model's SHA-256, and a startup
self-test confirms that the branch which actually executes is the one that was
asked for. Each of those is a separate guard, because the original defect was
invisible to every check that existed at the time.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The seven classes the dataset and the manuscript use, in label order.
EXPECTED_CLASSES: List[str] = ["CENTER", "LEFT", "RIGHT", "UP", "DOWN",
                              "FORWARD", "BACK"]

# The SVM characterised in Section V-C and measured operationally at 0.783.
# Frozen: re-measurement compares flight against that artefact, so a different
# file is a different experiment.
FROZEN_SVM_SHA256 = "e34704a1317101572a7604ae6ec84484c35d323637faf1a40ba11fbe88a934d4"

KINDS = ("svm", "rule")
LATENCY_COLUMN = {"svm": "svm_infer_ms", "rule": "rule_infer_ms"}


class ClassifierSelectionError(RuntimeError):
    """Raised when the requested classifier cannot be honoured exactly."""


@dataclass
class Selection:
    kind: str
    predict: Callable
    latency_column: str
    model_path: Optional[str] = None
    labels_path: Optional[str] = None
    sha256: Optional[str] = None
    classes: List[str] = field(default_factory=list)
    frozen_match: Optional[bool] = None

    def manifest(self) -> Dict[str, object]:
        return {
            "classifier": self.kind,
            "latency_column": self.latency_column,
            "model_path": self.model_path,
            "labels_path": self.labels_path,
            "model_sha256": self.sha256,
            "model_matches_frozen": self.frozen_match,
            "classes": self.classes,
        }


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def select(kind: str, cfg, *, model_path: Optional[str] = None,
           labels_path: Optional[str] = None, rule=None,
           require_frozen: bool = True) -> Selection:
    """Resolve `kind` into a usable classifier, or refuse.

    Every failure here is fatal by design. A missing model used to mean "use the
    rule instead"; it now means the launch does not happen.
    """
    if kind not in KINDS:
        raise ClassifierSelectionError(
            f"--classifier must be one of {KINDS}, got {kind!r}")

    if kind == "rule":
        if rule is None:
            raise ClassifierSelectionError("rule arm selected but no rule instance given")
        return Selection(kind="rule", predict=rule.predict,
                         latency_column=LATENCY_COLUMN["rule"],
                         classes=list(EXPECTED_CLASSES))

    # ---- SVM arm: validate before the aircraft is armed --------------------
    from .model_classifier import (TrainedClassifier, _resolve_labels_path,
                                   _resolve_model_path)

    mp = _resolve_model_path(model_path or "model.joblib")
    lp = _resolve_labels_path(labels_path or "labels_example.json")

    if not os.path.exists(mp):
        raise ClassifierSelectionError(f"--classifier svm: model not found at {mp}")
    if not os.path.exists(lp):
        raise ClassifierSelectionError(f"--classifier svm: labels not found at {lp}")

    digest = _sha256(mp)
    frozen = digest == FROZEN_SVM_SHA256
    if require_frozen and not frozen:
        raise ClassifierSelectionError(
            "--classifier svm: model does not match the frozen artefact.\n"
            f"  expected {FROZEN_SVM_SHA256}\n  found    {digest}\n"
            "  The re-measurement compares flight against the model characterised\n"
            "  in Section V-C. A different file is a different experiment.\n"
            "  Pass --allow-unpinned-model only if that is deliberate.")

    try:
        clf = TrainedClassifier(mp, lp)
    except Exception as e:
        raise ClassifierSelectionError(
            f"--classifier svm: model at {mp} failed to load: {type(e).__name__}: {e}")

    names = [clf.id_to_name.get(i) for i in range(len(EXPECTED_CLASSES))]
    if names != EXPECTED_CLASSES:
        raise ClassifierSelectionError(
            "--classifier svm: label set does not match the expected seven classes.\n"
            f"  expected {EXPECTED_CLASSES}\n  found    {names}")

    n = getattr(getattr(clf, "model", None), "classes_", None)
    if n is not None and len(n) != len(EXPECTED_CLASSES):
        raise ClassifierSelectionError(
            f"--classifier svm: model exposes {len(n)} classes, expected "
            f"{len(EXPECTED_CLASSES)}")

    return Selection(kind="svm", predict=clf.predict,
                     latency_column=LATENCY_COLUMN["svm"],
                     model_path=mp, labels_path=lp, sha256=digest,
                     classes=names, frozen_match=frozen)


def self_test(sel: Selection) -> Dict[str, object]:
    """Push one synthetic landmark vector through the selected classifier.

    Confirms the branch that executes is the one requested, and that it is the
    branch whose latency column will be written. The original defect was exactly
    a disagreement between the intended and the executed branch, so this is
    checked rather than assumed.
    """
    lm = np.linspace(0.2, 0.8, 63, dtype=np.float32).reshape(21, 3)
    try:
        out = sel.predict(lm)
    except Exception as e:
        raise ClassifierSelectionError(
            f"self-test: {sel.kind} classifier raised {type(e).__name__}: {e}")

    name = getattr(out, "name", None)
    if not isinstance(name, str) or not name:
        raise ClassifierSelectionError(
            f"self-test: {sel.kind} classifier returned no label ({out!r})")

    # The rule composes compound labels by joining components; the SVM returns a
    # single name from the label file. A label shape that belongs to the other
    # branch means the wrong one ran.
    compound = "-" in name
    if sel.kind == "svm" and compound:
        raise ClassifierSelectionError(
            f"self-test: --classifier svm produced the compound label {name!r}, "
            "which only the rule can form. The wrong branch executed.")
    if sel.kind == "svm" and name not in EXPECTED_CLASSES:
        raise ClassifierSelectionError(
            f"self-test: --classifier svm produced {name!r}, not one of the "
            "seven expected classes.")

    return {"label": name, "latency_column": sel.latency_column,
            "compound": compound}


def verify_run(run_dir: str) -> Dict[str, object]:
    """Does a completed run's manifest agree with the latency column it wrote?

    Used by the regression test and by reveal_arms. Returns a verdict rather
    than raising, so a whole archive can be swept.
    """
    import csv

    md = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(md):
        return {"run": os.path.basename(run_dir), "status": "no_manifest"}
    man = json.load(open(md, encoding="utf-8"))

    declared = man.get("classifier")
    if declared is None:
        declared = (man.get("model") or {}).get("classifier")
        if declared == "rule_based":
            declared = "rule"

    observed = None
    pf = os.path.join(run_dir, "perf.csv")
    if os.path.exists(pf):
        try:
            with open(pf, encoding="utf-8") as f:
                hdr = next(csv.reader(f))
            has = {k for k, c in LATENCY_COLUMN.items() if c in hdr}
            if len(has) == 1:
                observed = has.pop()
            elif len(has) > 1:
                observed = "ambiguous"
        except Exception:
            pass

    if observed is None:
        status = "no_gesture_frames"
    elif observed == "ambiguous":
        status = "AMBIGUOUS"
    elif declared is None:
        status = "undeclared"
    elif declared == observed:
        status = "ok"
    else:
        status = "MISMATCH"

    return {"run": os.path.basename(run_dir), "declared": declared,
            "observed": observed, "status": status}
