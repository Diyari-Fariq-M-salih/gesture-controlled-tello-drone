"""Software-in-the-loop scenario verification for the deterministic mode
arbitration + identity-gating logic (Table B in the ARO content review).

No camera, drone, or human operator is required: this replays scripted,
timestamp-controlled input sequences through the *actual* production code
(DeterministicModeManager.tick, and the same hand/face gating rule used in
controller.py) and checks the resulting mode transitions against the
expected behavior. It validates the arbitration logic itself deterministically
and reproducibly. It does NOT validate real flight dynamics, camera
detection accuracy, or physical drone behavior -- that would require
hardware that is not available for this project. Report results as
"software-in-the-loop scenario verification", not "flight trials".

The DeterministicConfig values below mirror the override values used at
runtime in Controller.__init__ (tello_gesture_py/src/controller.py), which
match the ARO manuscript's reported reproducibility parameters (hold~1.2s,
search-trigger>10s, search-duration=5s) -- the originally tested/reported
configuration.

Usage:
    py -3 -m src.gestures.simulate_scenarios
    py -3 -m src.gestures.simulate_scenarios --out_json outputs/metrics/scenario_simulation.json
"""

import argparse
import json
from pathlib import Path
from typing import Callable, List, Tuple

from .. import mode_manager as mm_module
from ..mode_manager import DeterministicConfig, DeterministicModeManager

# Mirrors Controller.__init__'s override in controller.py, which matches the
# ARO manuscript's reported reproducibility parameters (originally tested config).
RUNTIME_CFG = dict(
    battery_land_pct=15,
    nohuman_search_s=10.0,
    search_duration_s=5.0,
    search_cooldown_s=10.0,
    mode_hold_s=1.2,
    hand_release_s=0.8,
    face_release_s=0.8,
)


class FakeClock:
    def __init__(self, t0: float = 1_000_000.0):
        self.t = t0

    def time(self) -> float:
        return self.t

    def advance(self, dt: float):
        self.t += dt


class GatingSim:
    """Mirrors the AUTHORIZED-signal timer bookkeeping in controller.py's run loop:
    hand_detected = hand_raw AND face_authorized; timers reset only on authorized signals.
    """

    def __init__(self, clock: FakeClock):
        self.clock = clock
        self.last_hand_ts = clock.time()
        self.last_face_ts = clock.time()
        self.last_any_ts = clock.time()

    def state(self, hand_raw: bool, face_auth: bool, battery: float = 100.0) -> dict:
        now = self.clock.time()
        hand = bool(hand_raw and face_auth)
        face = bool(face_auth)
        if face:
            self.last_face_ts = now
            self.last_any_ts = now
        if hand:
            self.last_hand_ts = now
            self.last_any_ts = now
        return {
            "hand_detected": hand,
            "face_detected": face,
            "time_since_hand_s": now - self.last_hand_ts,
            "time_since_face_s": now - self.last_face_ts,
            "time_since_any_seen_s": now - self.last_any_ts,
            "battery": battery,
            "altitude_cm": 100.0,
            "flying": True,
        }


class Trial:
    def __init__(self, name: str):
        self.name = name
        self.log: List[str] = []
        self.passed = True

    def check(self, condition: bool, msg: str):
        tag = "OK" if condition else "FAIL"
        self.log.append(f"[{tag}] {msg}")
        if not condition:
            self.passed = False


def _new_mgr_and_clock() -> Tuple[DeterministicModeManager, FakeClock, GatingSim]:
    clock = FakeClock()
    mm_module.time.time = clock.time  # patch the module-level clock tick() reads
    mgr = DeterministicModeManager(DeterministicConfig(**RUNTIME_CFG))
    sim = GatingSim(clock)
    return mgr, clock, sim


def _run(name: str, script: Callable[[Trial, DeterministicModeManager, FakeClock, GatingSim], None]) -> Trial:
    mgr, clock, sim = _new_mgr_and_clock()
    trial = Trial(name)
    script(trial, mgr, clock, sim)
    return trial


# ---------------------------------------------------------------------------
# Scenario definitions. Each returns a list of Trial (sub-trials that stress
# a boundary condition, since the logic is deterministic and repeating an
# identical run would always give the same result).
# ---------------------------------------------------------------------------

def scenario_authorized_gesture_accepted() -> List[Trial]:
    trials = []

    def from_hover(t: Trial, mgr, clock, sim):
        mgr.tick(sim.state(hand_raw=False, face_auth=False))
        mode, reason = mgr.tick(sim.state(hand_raw=True, face_auth=True))
        t.check(mode == "gesture", f"authorized hand from hover -> gesture (got {mode}: {reason})")

    def from_no_face_yet(t: Trial, mgr, clock, sim):
        # hand physically present but face not yet authorized this tick -> must NOT enter gesture
        mode, _ = mgr.tick(sim.state(hand_raw=True, face_auth=False))
        t.check(mode != "gesture", f"hand without authorized face must not enter gesture (got {mode})")
        mode, _ = mgr.tick(sim.state(hand_raw=True, face_auth=True))
        t.check(mode == "gesture", f"once face authorized, gesture accepted same-frame (got {mode})")

    trials.append(_run("authorized_gesture_accepted#1_from_hover", from_hover))
    trials.append(_run("authorized_gesture_accepted#2_face_then_hand", from_no_face_yet))
    return trials


def scenario_unauthorized_gesture_rejected() -> List[Trial]:
    trials = []

    def basic(t: Trial, mgr, clock, sim):
        mode, reason = mgr.tick(sim.state(hand_raw=True, face_auth=False))
        t.check(mode != "gesture", f"unauthorized hand rejected (got {mode}: {reason})")

    def persists(t: Trial, mgr, clock, sim):
        # unauthorized hand held for a long time must never trigger gesture
        for _ in range(10):
            clock.advance(1.0)
            mode, _ = mgr.tick(sim.state(hand_raw=True, face_auth=False))
        t.check(mode != "gesture", f"unauthorized hand held 10s still rejected (got {mode})")

    trials.append(_run("unauthorized_gesture_rejected#1_basic", basic))
    trials.append(_run("unauthorized_gesture_rejected#2_persistent", persists))
    return trials


def scenario_face_following_activates() -> List[Trial]:
    trials = []

    def basic(t: Trial, mgr, clock, sim):
        mode, reason = mgr.tick(sim.state(hand_raw=False, face_auth=True))
        t.check(mode == "face", f"authorized face, no hand -> face (got {mode}: {reason})")

    trials.append(_run("face_following_activates#1_basic", basic))
    return trials


def scenario_gesture_preempts_face() -> List[Trial]:
    trials = []

    def basic(t: Trial, mgr, clock, sim):
        mode, _ = mgr.tick(sim.state(hand_raw=False, face_auth=True))
        t.check(mode == "face", "precondition: starts in face mode")
        mode, reason = mgr.tick(sim.state(hand_raw=True, face_auth=True))
        t.check(mode == "gesture", f"hand appears during face mode -> immediate preemption (got {mode}: {reason})")

    trials.append(_run("gesture_preempts_face#1_immediate", basic))
    return trials


def scenario_hysteresis_prevents_flicker() -> List[Trial]:
    trials = []
    hold = RUNTIME_CFG["mode_hold_s"]
    release = RUNTIME_CFG["hand_release_s"]

    def brief_dropout_survives(t: Trial, mgr, clock, sim):
        mgr.tick(sim.state(hand_raw=True, face_auth=True))
        clock.advance(hold + 0.1)
        mgr.tick(sim.state(hand_raw=True, face_auth=True))  # confirm settled in gesture past hold
        clock.advance(release * 0.5)  # brief dropout, still under release threshold
        mode, reason = mgr.tick(sim.state(hand_raw=False, face_auth=True))
        t.check(mode == "gesture", f"brief hand dropout ({release * 0.5:.2f}s < release {release}s) holds gesture (got {mode}: {reason})")

    def sustained_dropout_releases(t: Trial, mgr, clock, sim):
        mgr.tick(sim.state(hand_raw=True, face_auth=True))
        clock.advance(hold + 0.1)
        mgr.tick(sim.state(hand_raw=True, face_auth=True))
        clock.advance(release + 0.2)  # exceeds release threshold
        mode, reason = mgr.tick(sim.state(hand_raw=False, face_auth=True))
        t.check(mode == "face", f"sustained hand dropout ({release + 0.2:.2f}s > release {release}s) releases to face (got {mode}: {reason})")

    def cannot_leave_before_hold(t: Trial, mgr, clock, sim):
        mgr.tick(sim.state(hand_raw=True, face_auth=True))
        clock.advance(0.05)  # well under mode_hold_s
        mode, reason = mgr.tick(sim.state(hand_raw=False, face_auth=True))
        t.check(mode == "gesture", f"cannot leave gesture before mode_hold_s={hold}s even with hand gone (got {mode}: {reason})")

    trials.append(_run("hysteresis_prevents_flicker#1_brief_dropout", brief_dropout_survives))
    trials.append(_run("hysteresis_prevents_flicker#2_sustained_dropout", sustained_dropout_releases))
    trials.append(_run("hysteresis_prevents_flicker#3_min_hold_time", cannot_leave_before_hold))
    return trials


def scenario_search_starts_after_loss() -> List[Trial]:
    trials = []
    thr = RUNTIME_CFG["nohuman_search_s"]

    def not_yet(t: Trial, mgr, clock, sim):
        mgr.tick(sim.state(hand_raw=False, face_auth=False))
        clock.advance(thr - 0.2)
        mode, reason = mgr.tick(sim.state(hand_raw=False, face_auth=False))
        t.check(mode != "search_360", f"before threshold ({thr - 0.2:.2f}s < {thr}s) must not search yet (got {mode}: {reason})")

    def triggers(t: Trial, mgr, clock, sim):
        mgr.tick(sim.state(hand_raw=False, face_auth=False))
        clock.advance(thr + 0.2)
        mode, reason = mgr.tick(sim.state(hand_raw=False, face_auth=False))
        t.check(mode == "search_360", f"after threshold ({thr + 0.2:.2f}s >= {thr}s) search starts (got {mode}: {reason})")

    trials.append(_run("search_starts_after_loss#1_before_threshold", not_yet))
    trials.append(_run("search_starts_after_loss#2_after_threshold", triggers))
    return trials


def scenario_target_reacquired_after_search() -> List[Trial]:
    trials = []
    thr = RUNTIME_CFG["nohuman_search_s"]
    dur = RUNTIME_CFG["search_duration_s"]

    def reacquire_via_hand(t: Trial, mgr, clock, sim):
        mgr.tick(sim.state(hand_raw=False, face_auth=False))
        clock.advance(thr + 0.2)
        mode, _ = mgr.tick(sim.state(hand_raw=False, face_auth=False))
        t.check(mode == "search_360", "precondition: search started")
        clock.advance(dur * 0.3)  # well before search would time out on its own
        mode, reason = mgr.tick(sim.state(hand_raw=True, face_auth=True))
        t.check(mode == "gesture", f"authorized hand during search -> immediate exit to gesture (got {mode}: {reason})")

    def reacquire_via_face(t: Trial, mgr, clock, sim):
        mgr.tick(sim.state(hand_raw=False, face_auth=False))
        clock.advance(thr + 0.2)
        mgr.tick(sim.state(hand_raw=False, face_auth=False))
        clock.advance(dur * 0.3)
        mode, reason = mgr.tick(sim.state(hand_raw=False, face_auth=True))
        t.check(mode == "face", f"authorized face during search -> immediate exit to face (got {mode}: {reason})")

    def search_times_out_to_hover(t: Trial, mgr, clock, sim):
        mgr.tick(sim.state(hand_raw=False, face_auth=False))
        clock.advance(thr + 0.2)
        mgr.tick(sim.state(hand_raw=False, face_auth=False))
        clock.advance(dur + 0.2)
        mode, reason = mgr.tick(sim.state(hand_raw=False, face_auth=False))
        t.check(mode == "hover", f"search times out after {dur}s with nothing found -> hover (got {mode}: {reason})")

    trials.append(_run("target_reacquired_after_search#1_via_hand", reacquire_via_hand))
    trials.append(_run("target_reacquired_after_search#2_via_face", reacquire_via_face))
    trials.append(_run("target_reacquired_after_search#3_timeout_to_hover", search_times_out_to_hover))
    return trials


def scenario_battery_failsafe_landing() -> List[Trial]:
    trials = []
    thr = RUNTIME_CFG["battery_land_pct"]

    def at_threshold(t: Trial, mgr, clock, sim):
        mode, reason = mgr.tick(sim.state(hand_raw=False, face_auth=False, battery=float(thr)))
        t.check(mode == "land", f"battery == {thr}% triggers land (got {mode}: {reason})")

    def above_threshold_no_land(t: Trial, mgr, clock, sim):
        mode, reason = mgr.tick(sim.state(hand_raw=False, face_auth=False, battery=float(thr) + 1.0))
        t.check(mode != "land", f"battery == {thr + 1}% must not trigger land (got {mode}: {reason})")

    def overrides_active_gesture(t: Trial, mgr, clock, sim):
        mode, _ = mgr.tick(sim.state(hand_raw=True, face_auth=True, battery=100.0))
        t.check(mode == "gesture", "precondition: actively in gesture mode")
        mode, reason = mgr.tick(sim.state(hand_raw=True, face_auth=True, battery=float(thr)))
        t.check(mode == "land", f"battery failsafe overrides an active authorized gesture (got {mode}: {reason})")

    trials.append(_run("battery_failsafe_landing#1_at_threshold", at_threshold))
    trials.append(_run("battery_failsafe_landing#2_above_threshold", above_threshold_no_land))
    trials.append(_run("battery_failsafe_landing#3_overrides_gesture", overrides_active_gesture))
    return trials


SCENARIOS = {
    "authorized_gesture_accepted": scenario_authorized_gesture_accepted,
    "unauthorized_gesture_rejected": scenario_unauthorized_gesture_rejected,
    "face_following_activates": scenario_face_following_activates,
    "gesture_preempts_face": scenario_gesture_preempts_face,
    "hysteresis_prevents_flicker": scenario_hysteresis_prevents_flicker,
    "search_starts_after_loss": scenario_search_starts_after_loss,
    "target_reacquired_after_search": scenario_target_reacquired_after_search,
    "battery_failsafe_landing": scenario_battery_failsafe_landing,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_json", default="outputs/metrics/scenario_simulation.json")
    args = ap.parse_args()

    orig_time = mm_module.time.time
    results = {}
    all_trials: List[Trial] = []

    try:
        for name, fn in SCENARIOS.items():
            trials = fn()
            results[name] = trials
            all_trials.extend(trials)
    finally:
        mm_module.time.time = orig_time  # restore real clock

    print("\n--- Table B: Scenario-Level System Tests (software-in-the-loop verification) ---")
    print("| Test case | Trials | Successes | Success rate |")
    print("|---|---:|---:|---:|")
    summary = {}
    for name, trials in results.items():
        n = len(trials)
        ok = sum(1 for t in trials if t.passed)
        rate = f"{100.0 * ok / n:.0f}%" if n else "n/a"
        print(f"| {name} | {n} | {ok} | {rate} |")
        summary[name] = {"trials": n, "successes": ok}

    print("\nDetailed sub-trial log:")
    for t in all_trials:
        status = "PASS" if t.passed else "FAIL"
        print(f"\n[{status}] {t.name}")
        for line in t.log:
            print(f"    {line}")

    n_total = len(all_trials)
    n_pass = sum(1 for t in all_trials if t.passed)
    print(f"\nTOTAL: {n_pass}/{n_total} sub-trials passed")

    out = {
        "config": RUNTIME_CFG,
        "summary": summary,
        "sub_trials": [
            {"name": t.name, "passed": t.passed, "log": t.log} for t in all_trials
        ],
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    if n_pass != n_total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
