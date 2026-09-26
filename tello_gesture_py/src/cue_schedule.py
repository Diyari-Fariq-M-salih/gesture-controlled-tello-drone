"""Cued-capture schedule for in-flight classifier measurement.

The scenario table ceilings at 149/149 and cannot distinguish a classifier that
issues the right command from one that issues a different command the operator
judges acceptable. This drives a cue instead: the operator is told a class, given
time to form it, and the frames that follow are labelled with what was asked for.

It runs inside the controller, so everything under measurement -- perception,
identity gate, arbitration, the depth-stability gate, the RC output -- is the
deployed path. The schedule only decides what the operator is asked to do and
which frames carry a label.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# What the operator physically does, per vocabulary. The two arms differ here,
# which is why the operator cannot be blinded to the arm.
CUE_TEXT: Dict[str, Dict[str, str]] = {
    "svm": {
        "CENTER": "hold the trained CENTER pose",
        "LEFT": "hold the trained LEFT pose",
        "RIGHT": "hold the trained RIGHT pose",
        "UP": "hold the trained UP pose",
        "DOWN": "hold the trained DOWN pose",
        "FORWARD": "hold the trained FORWARD pose (open palm)",
        "BACK": "hold the trained BACK pose",
    },
    # The rule reads the index-finger vector in IMAGE coordinates with no
    # mirror correction, and the camera faces the operator. Pointing to your own
    # left therefore moves the fingertip toward the right of the image and the
    # rule emits RIGHT. The cue is written in the operator's frame and names the
    # action rather than the class, so the operator never has to invert
    # anything mentally; `cue_name` still records the class being tested.
    "rule": {
        "CENTER": "finger neutral, no clear direction",
        "LEFT": "point to YOUR RIGHT",
        "RIGHT": "point to YOUR LEFT",
        "UP": "point UP",
        "DOWN": "point DOWN",
        "FORWARD": "FAST thrust toward drone, SLOW return, repeat",
        "BACK": "FAST pull away from drone, SLOW return, repeat",
    },
}

FIELDS = ["t", "seq", "frame_age_ms", "round", "hold", "cue_label", "cue_name",
          "phase", "vocabulary", "hand_detected", "hand_auth", "face_auth",
          "mode", "raw_gesture", "emitted_gesture", "command", "fb_stable_count",
          "gesture_conf", "battery", "altitude_cm", "actuation_suppressed",
          # The depth cue and the quantity it is differenced from. Without
          # these the depth channel can only be analysed by its output, not by
          # the mechanism -- and in flight the aircraft's own drift changes
          # apparent hand size, which is exactly what delta_s reads.
          "bbox_area", "delta_s"]

# The classifier's entire input, kept so a capture can be re-scored offline by
# any model. Appended after the interpreted columns so the readable fields stay
# at the front of the file.
LANDMARK_FIELDS = [f"lm{i:02d}_{a}" for i in range(21) for a in ("x", "y", "z")]


def landmark_row(lm):
    """Flatten a (21,3) landmark array into the logged columns.

    Written at %.9g, which round-trips float32 exactly. Rounding is not safe
    here: MediaPipe pins the wrist z to a near-constant value whose standard
    deviation across the training set is 3e-07, and StandardScaler divides by
    it, so a 1e-5 rounding error reaches the classifier as tens of standard
    deviations. At five decimal places that flipped 9.7% of predictions.
    """
    if lm is None:
        return {k: None for k in LANDMARK_FIELDS}
    flat = [float(v) for p in lm for v in p[:3]]
    return {k: "%.9g" % v for k, v in zip(LANDMARK_FIELDS, flat)}


@dataclass
class CueSchedule:
    """Drives the cue sequence and says which phase each frame belongs to."""

    names: Dict[int, str]
    vocabulary: str
    rounds: int = 6
    settle_s: float = 5.0
    hold_s: float = 6.0
    seed: int = 7

    schedule: List[Tuple[int, int]] = field(default_factory=list)
    idx: int = -1
    started_at: Optional[float] = None
    paused_at: Optional[float] = None
    done: bool = False

    only: Optional[List[str]] = None

    def __post_init__(self):
        rng = random.Random(self.seed)
        keep = sorted(self.names) if not self.only else [
            k for k in sorted(self.names) if self.names[k] in self.only]
        if not keep:
            raise ValueError(f"no classes left after --cue-classes {self.only}")
        sched = []
        for r in range(self.rounds):
            order = list(keep)
            rng.shuffle(order)
            sched.extend((r, lab) for lab in order)
        self.schedule = sched

    # ---------------------------------------------------------------- state
    @property
    def active(self) -> bool:
        return self.started_at is not None and not self.done

    @property
    def current(self) -> Optional[Tuple[int, int]]:
        if not self.active or self.idx < 0 or self.idx >= len(self.schedule):
            return None
        return self.schedule[self.idx]

    def cue_text(self) -> str:
        c = self.current
        return "" if c is None else CUE_TEXT[self.vocabulary][self.names[c[1]]]

    def spoken(self) -> str:
        """What the operator hears.

        For the rule arm the class name is misleading -- its LEFT is produced by
        pointing to the operator's right -- so the instruction is spoken instead
        of the label. The label is still what the frame is scored against.
        """
        c = self.current
        if c is None:
            return ""
        name = self.names[c[1]]
        if self.vocabulary == "rule" and name in ("LEFT", "RIGHT"):
            return CUE_TEXT["rule"][name]
        return name

    # ---------------------------------------------------------------- drive
    def start(self) -> None:
        """Begin, or resume from where an abort left off."""
        self.done = False
        if self.idx < 0:
            self.idx = 0
        self.started_at = time.time()

    def stop(self) -> None:
        self.started_at = None

    def skip(self) -> None:
        self._advance()

    def _advance(self) -> None:
        self.idx += 1
        if self.idx >= len(self.schedule):
            self.done = True
            self.started_at = None
        else:
            self.started_at = time.time()

    def tick(self, now: float) -> Tuple[str, float]:
        """Return (phase, seconds remaining in this hold).

        `phase` is 'settle' while the operator is forming the gesture and
        'hold' for the frames that get labelled. Settle frames are recorded with
        their phase so they can be excluded rather than silently dropped.
        """
        if not self.active or self.started_at is None:
            return "idle", 0.0
        dt = now - self.started_at
        if dt < self.settle_s:
            return "settle", self.settle_s + self.hold_s - dt
        if dt < self.settle_s + self.hold_s:
            return "hold", self.settle_s + self.hold_s - dt
        self._advance()
        return ("idle" if self.done else "settle"), 0.0

    def progress(self) -> str:
        n = len(self.schedule)
        if self.done:
            return f"complete {n}/{n}"
        return f"hold {min(self.idx + 1, n)}/{n}"

    def manifest(self) -> Dict[str, object]:
        return {
            "vocabulary": self.vocabulary,
            "rounds": self.rounds,
            "settle_s": self.settle_s,
            "hold_s": self.hold_s,
            "seed": self.seed,
            "holds_scheduled": len(self.schedule),
            "classes_cued": self.only or "all",
            "holds_completed": len(self.schedule) if self.done else max(self.idx, 0),
            "cues": CUE_TEXT[self.vocabulary],
            "blinded": False,
            "blinding_note": ("the arms require different physical gestures, so the "
                              "operator cannot be blinded to the arm"),
        }
