"""Cued gesture capture through the aircraft's own video pipeline, on the ground.

Gesture accuracy in the manuscript is offline on both protocols: the 0.997
within-session and 0.872 held-out figures score the classifier on webcam frames.
This harness scores it on frames the Tello encoded, transmitted and the host
decoded, against a cue the script issues, so operational accuracy can be
reported on the same footing as the identity gate.

WHY BOTH CLASSIFIERS ARE RUN
----------------------------
Every logged run in this repository carries `classifier: rule_based` and no
`--model` was ever passed, so the RBF-SVM that Section III-B describes and
Section V-C evaluates has never been in the flight loop. The compound labels in
the decision logs (DOWN-FORWARD, UP-FORWARD) are RuleBasedGesture outputs; the
SVM emits single class names. The two therefore have to be measured together on
identical frames: the SVM because it is what the paper characterises, the rule
path because it is what produced every flight result reported. Both are
deployed classes selected between in Controller.__init__; neither is
reimplemented here, and nothing in either is modified.

The depth cue is read out of the deployed rule object rather than recomputed.
RuleBasedGesture.predict() stores the smoothed bounding-box scale in
`_last_scale` and forms its cue as (s_t - s_{t-1}) / s_{t-1}; sampling that
attribute either side of the call recovers the same delta_s the classifier
acted on, without touching the classifier.

Nothing here commands the aircraft: the SDK socket carries `command`,
`streamon` and `streamoff` only. Run it with the propellers removed. What the
measurement needs from the drone is its codec, its optics and its link.

Usage:
    python -m tello_gesture_py.scripts.operational_gesture_eval \\
        --model model.joblib --labels labels_example.json --run-id capture_gesture-svm
    python -m tello_gesture_py.scripts.operational_gesture_eval --resume outputs/runs/<dir>

Keys: SPACE pause/resume   s skip the current hold   q abort (progress is kept)
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import cv2

from tello_gesture_py.src.config import ControllerConfig
from tello_gesture_py.src.gesture_logic import DepthStabilityGate, RuleBasedGesture
from tello_gesture_py.src.hand_gesture import HandGesture
from tello_gesture_py.src.latest_frame import LatestFrame
from tello_gesture_py.src.run_context import RunContext
from tello_gesture_py.src.state_listener import StateListener
from tello_gesture_py.src.tello_udp import TelloUDP
from tello_gesture_py.src.video_stream import VideoStream

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIELDS = ["t", "seq", "frame_age_ms", "round", "hold", "cued_label", "cued_name",
          "phase", "hand_detected", "bbox_area", "delta_s",
          "svm_label", "svm_name", "svm_prob", "svm_margin",
          "rule_raw", "rule_name", "rule_conf", "fb_stable_count", "battery"]


# How to perform each class, per vocabulary. "pose" is the dataset vocabulary the
# SVM was trained on; "pointing" is what the rule classifier decodes and what the
# operator actually performed in flight.
CUE = {
    "pose": {c: "hold the trained pose" for c in
             ("CENTER", "LEFT", "RIGHT", "UP", "DOWN", "FORWARD", "BACK")},
    "pointing": {
        "CENTER":  "index finger neutral, no clear direction",
        "LEFT":    "point the index finger LEFT",
        "RIGHT":   "point the index finger RIGHT",
        "UP":      "point the index finger UP",
        "DOWN":    "point the index finger DOWN",
        # Fast one way, slow back. A symmetric motion fires the opposite class
        # on the return stroke; a slow one fires nothing.
        "FORWARD": "FAST thrust toward camera, SLOW return. Repeat ~3x",
        "BACK":    "FAST pull away from camera, SLOW return. Repeat ~3x",
    },
}


def _hud(frame, lines, colour):
    for i, line in enumerate(lines):
        o = (12, 34 + i * 30)
        cv2.putText(frame, line, o, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, o, cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 1, cv2.LINE_AA)


def build_schedule(names, rounds, seed):
    """Randomised class order within each round, so drift cannot align with a class."""
    rng = random.Random(seed)
    sched = []
    for r in range(rounds):
        order = sorted(names)
        rng.shuffle(order)
        sched.extend((r, lab) for lab in order)
    return sched


def connect(cfg):
    tello = TelloUDP(cfg.tello_ip, cfg.cmd_port, local_port=cfg.local_cmd_port)
    ok, resp = False, ""
    for i in range(1, 4):
        ok, resp = tello.send_cmd("command", timeout_ms=4000)
        if ok and resp.lower() == "ok":
            break
        print(f"SDK handshake attempt {i}/3: {ok} {resp}")
        time.sleep(0.5)
    if not ok or resp.lower() != "ok":
        raise SystemExit("Failed to enter SDK mode. Is this PC on the TELLO-XXXXXX network?")
    tello.send_cmd("streamoff", timeout_ms=2000)
    tello.send_cmd("streamon", timeout_ms=6000)
    return tello


def done_holds(path):
    """(round, cued_label) pairs already captured, so a resumed run skips them."""
    if not os.path.exists(path):
        return set()
    seen = set()
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("phase") == "hold":
                seen.add((int(row["round"]), int(row["cued_label"])))
    return seen


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="joblib SVM; omit to record the rule path only")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--run-id", default="capture_gesture")
    ap.add_argument("--note", default="")
    ap.add_argument("--settle-s", type=float, default=5.0,
                    help="Seconds to change gesture before recording resumes.")
    ap.add_argument("--hold-s", type=float, default=6.0,
                    help="Seconds recorded per hold. Frames within one hold are "
                         "near-duplicates, so more holds beats longer holds for "
                         "independent information; the analysis reports a "
                         "hold-level rate alongside the frame-level one.")
    ap.add_argument("--vocabulary", choices=("pose", "pointing"), default="pose",
                    help="Which gesture vocabulary to cue. 'pose' is the dataset "
                         "vocabulary the SVM was trained on; 'pointing' is what "
                         "the rule classifier decodes and what was flown. "
                         "FORWARD and BACK are movements under 'pointing', "
                         "because the rule derives them from frame-to-frame area "
                         "change and a static hold cannot produce them.")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resume", default=None, metavar="RUNDIR",
                    help="Continue a run, skipping holds already recorded.")
    args = ap.parse_args()

    # Resolve the labels path the way the deployed classifier resolves it, so a
    # bare filename means the same thing here as it does on the command line
    # that flies the drone.
    from tello_gesture_py.src.model_classifier import _resolve_labels_path
    lp = _resolve_labels_path(args.labels or "labels_example.json")
    if not os.path.exists(lp):
        raise SystemExit(f"Labels file not found: {lp}\n"
                         "Pass --labels with a path, or put the file in data/labels/.")
    names = {int(k): v for k, v in json.load(open(lp, encoding="utf-8")).items()}

    cfg = ControllerConfig()
    cfg.allow_takeoff = False

    if args.resume:
        run = RunContext.__new__(RunContext)
        run.dir = Path(args.resume)
        run.name = run.dir.name
        run.run_id = args.run_id
        run.note = args.note
        run.started_at = time.time()
        run._manifest = json.load(open(run.dir / "manifest.json", encoding="utf-8"))
        print(f"[resume] {run.dir}")
    else:
        run = RunContext(run_id=args.run_id, note=args.note)

    csv_path = run.path("op_gesture.csv")
    already = done_holds(csv_path)
    new_file = not os.path.exists(csv_path)

    # The SVM is optional only so the harness still runs if no model is present;
    # without it the capture cannot answer the question the paper asks.
    clf = None
    if args.model:
        from tello_gesture_py.src.model_classifier import TrainedClassifier
        clf = TrainedClassifier(args.model, lp)
    else:
        print("WARNING: no --model. The SVM columns will be empty, and this capture")
        print("         will not measure the classifier Section V-C characterises.")

    hand = HandGesture(max_num_hands=1)
    # Same constants the controller constructs the rule classifier with.
    rule = RuleBasedGesture(dir_thr=cfg.dir_thr, scale_thr=cfg.scale_thr,
                            ema_alpha=cfg.ema_alpha)
    # The same gate the controller applies between the classifier and the
    # command. Without it the capture measures the classifier, not the system.
    fb_gate = DepthStabilityGate(cfg.gesture_fb_streak_on)

    run.record("config", cfg)
    run.record("harness", "operational_gesture_eval")
    run.record("model", {
        "gesture_model": args.model, "gesture_labels": lp,
        "classifier": "svm+rule_based" if clf is not None else "rule_based",
        "fb_streak_on": cfg.gesture_fb_streak_on,
        "note": "both classifiers scored on identical frames; neither commands the aircraft",
    })
    run.record("protocol", {"settle_s": args.settle_s, "hold_s": args.hold_s,
                            "rounds": args.rounds, "seed": args.seed,
                            "classes": names, "vocabulary": args.vocabulary,
                            "cues": CUE[args.vocabulary],
                            "condition": "ground, propellers off"})

    print("=" * 68)
    print("  Ground capture through the drone stream. PROPELLERS OFF.")
    print("  This harness never sends takeoff or rc -- it only reads video.")
    print("=" * 68)

    tello = connect(cfg)
    state = StateListener(cfg.state_port)
    state.start()
    latest = LatestFrame()
    video = VideoStream(latest, f"udp://0.0.0.0:{cfg.video_port}")
    if not video.start():
        raise SystemExit("Video stream not opened. Check firewall UDP 11111.")

    cv2.namedWindow("OPERATIONAL GESTURE", cv2.WINDOW_AUTOSIZE)
    t0 = time.time()
    while latest.get()[1] is None:
        if time.time() - t0 > 12:
            raise SystemExit("No frames decoded after 12 s.")
        time.sleep(0.1)

    fh = open(csv_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        w.writeheader()

    full = build_schedule(names, args.rounds, args.seed)
    # A hold is identified by where it sits in the *full* schedule, not in the
    # outstanding remainder, so a resumed session does not renumber holds it
    # shares with the session it continues.
    hold_id = {(r, lab): i for i, (r, lab) in enumerate(full)}
    schedule = [(r, lab) for r, lab in full if (r, lab) not in already]
    print(f"\nvocabulary: {args.vocabulary}")
    for c, how in CUE[args.vocabulary].items():
        print(f"   {c:<8} {how}")
    print(f"\n{len(schedule)} holds to capture"
          + (f" ({len(already)} already done)" if already else ""))

    n_rows, last_seq, aborted, idx = 0, -1, False, 0
    try:
        for idx, (rnd, lab) in enumerate(schedule):
            # A fresh hold is a fresh gesture: clear the smoothing and scale
            # history exactly as the controller does when the hand is lost, or
            # delta_s on the first frame is a difference against the last class.
            rule.reset()
            fb_gate.reset()
            t_start = time.time()
            phase_end = t_start + args.settle_s
            hold_end = phase_end + args.hold_s
            skip = False

            while time.time() < hold_end:
                ok, frame, seq, ts = latest.get(copy=True)
                now = time.time()
                if not (ok and frame is not None and seq != last_seq):
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        aborted = True
                        break
                    continue
                last_seq = seq
                phase = "settle" if now < phase_end else "hold"

                # Throttled as the controller throttles it, so the frame rate
                # this measures is the frame rate the deployed loop sees.
                if seq % max(cfg.hand_every_n, 1) != 0:
                    continue

                det = hand.detect(frame)
                detected = bool(det.has_hand and det.landmarks is not None)

                bbox_area = delta_s = None
                svm_label = svm_name = svm_prob = svm_margin = None
                rule_raw = rule_name = rule_conf = None

                if detected:
                    prev_scale = rule._last_scale
                    gr = rule.predict(det.landmarks)      # deployed rule path
                    rule_raw, rule_conf = gr.name, gr.confidence
                    rule_name = fb_gate.apply(gr.name)    # what the controller commands
                    bbox_area = rule._last_scale
                    if prev_scale is not None:
                        delta_s = (bbox_area - prev_scale) / max(prev_scale, 1e-6)

                    if clf is not None:
                        sr = clf.predict(det.landmarks)   # deployed SVM path
                        svm_name = sr.name
                        svm_prob = sr.confidence
                        svm_label = next((k for k, v in names.items() if v == sr.name), None)
                        try:
                            p = sorted(clf.model.predict_proba(
                                det.landmarks[:, :3].reshape(1, -1))[0])
                            svm_margin = float(p[-1] - p[-2])
                        except Exception:
                            svm_margin = None

                if phase == "hold" and not skip:
                    st = state.snapshot()
                    w.writerow({
                        "t": now, "seq": seq, "frame_age_ms": round((now - ts) * 1000, 2),
                        "round": rnd, "hold": hold_id[(rnd, lab)],
                        "cued_label": lab, "cued_name": names[lab],
                        "phase": phase, "hand_detected": int(detected),
                        "bbox_area": None if bbox_area is None else round(bbox_area, 8),
                        "delta_s": None if delta_s is None else round(delta_s, 6),
                        "svm_label": svm_label, "svm_name": svm_name,
                        "svm_prob": None if svm_prob is None else round(svm_prob, 4),
                        "svm_margin": None if svm_margin is None else round(svm_margin, 4),
                        "rule_raw": rule_raw, "rule_name": rule_name,
                        "rule_conf": None if rule_conf is None else round(rule_conf, 4),
                        "fb_stable_count": fb_gate.stable_count,
                        "battery": st.get("bat"),
                    })
                    n_rows += 1
                    if n_rows % 40 == 0:
                        fh.flush()

                remain = hold_end - now
                _hud(frame, [
                    f"{'SETTLE' if phase == 'settle' else 'HOLD  '}   {names[lab]}"
                    f"   [{args.vocabulary}]",
                    CUE[args.vocabulary][names[lab]],
                    f"hold {idx + 1}/{len(schedule)}   round {rnd + 1}/{args.rounds}"
                    f"   {remain:4.1f}s",
                    f"hand={int(detected)}  rule={rule_name or '--'}  svm={svm_name or '--'}",
                    f"rows={n_rows}    SPACE pause   s skip   q abort",
                ], (0, 255, 0) if phase == "hold" else (0, 200, 255))
                cv2.imshow("OPERATIONAL GESTURE", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    aborted = True
                    break
                if key == ord("s"):
                    skip = True
                    break
                if key == ord(" "):
                    # Pause stops the clock as well, otherwise the hold window
                    # elapses while the operator is resting their arm.
                    paused_at = time.time()
                    while True:
                        _, f2, _, _ = latest.get(copy=True)
                        if f2 is not None:
                            _hud(f2, ["PAUSED", "SPACE resume   q abort"], (0, 200, 255))
                            cv2.imshow("OPERATIONAL GESTURE", f2)
                        k2 = cv2.waitKey(30) & 0xFF
                        if k2 == ord(" "):
                            break
                        if k2 == ord("q"):
                            aborted = True
                            break
                    d = time.time() - paused_at
                    phase_end += d
                    hold_end += d
                    if aborted:
                        break
            if aborted:
                break
    finally:
        fh.flush()
        fh.close()
        video.stop()
        try:
            state.stop()
        except Exception:
            pass
        tello.send_cmd("streamoff", timeout_ms=2000)
        tello.close()
        cv2.destroyAllWindows()
        run.record("capture", {"rows": n_rows, "holds_attempted": idx + (0 if aborted else 1),
                               "holds_scheduled": len(schedule), "aborted": aborted,
                               "holds_previously_done": len(already)})
        run.record("stream", {"reopen_count": video.reopen_count,
                              "read_failures": video.read_failures})
        run.write()
        print(f"\n{n_rows} hold-window frames -> {csv_path}")
        if aborted:
            print(f"Aborted. Resume with:\n  --resume {run.dir}")
        print(f"[run] {run.dir}")


if __name__ == "__main__":
    main()
