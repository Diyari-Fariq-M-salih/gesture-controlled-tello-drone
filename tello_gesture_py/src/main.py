import argparse

from .config import ControllerConfig
from .controller import Controller
from .run_context import RunContext


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m tello_gesture_py.src.main",
        description="Tello identity-gated gesture controller.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    clf = ap.add_argument_group("gesture classifier (required)")
    g = clf.add_mutually_exclusive_group(required=True)
    g.add_argument("--classifier", choices=("svm", "rule"),
                   help="Which gesture classifier to fly. No default: an "
                        "unstated choice is what allowed a whole campaign to "
                        "run the rule while the SVM was the documented path.")
    g.add_argument("--arm-randomise", action="store_true",
                   help="Pick the arm by seeded RNG and withhold it from the "
                        "operator for the duration of the run.")
    clf.add_argument("--arm-seed", type=int, default=None,
                     help="Seed for --arm-randomise. Recorded in the manifest.")
    clf.add_argument("--model", default=None, help="joblib model for the svm arm.")
    clf.add_argument("--labels", default=None, help="labels.json for the svm arm.")
    cue = ap.add_argument_group("cued in-flight capture")
    cue.add_argument("--cue-rounds", type=int, default=0,
                     help="Rounds of seven cued classes. 0 disables. Press 'c' "
                          "in flight to start, 'k' to skip a hold.")
    cue.add_argument("--cue-settle-s", type=float, default=5.0)
    cue.add_argument("--cue-hold-s", type=float, default=6.0)
    cue.add_argument("--cue-seed", type=int, default=7)
    cue.add_argument("--cue-classes", default="",
                     help="Comma-separated subset to cue, e.g. LEFT,RIGHT. "
                          "Default cues all seven.")
    cue.add_argument("--freeflight-log", action="store_true",
                     help="Dense per-frame log for scoring against an external "
                          "video recording. Press 'm' and clap to stamp a sync "
                          "marker. No cue schedule.")
    cue.add_argument("--no-actuate", action="store_true",
                     help="Hover lock: compute and log every command as normal "
                          "but transmit rc 0 0 0 0, so the aircraft holds "
                          "station. Required for cued holds indoors -- a 6 s "
                          "directional hold travels about 3.9 m.")
    cue.add_argument("--no-landmarks", action="store_true",
                     help="Omit the 63 landmark columns. They are what makes a "
                          "capture re-scorable by another model offline, so "
                          "only drop them if disk is genuinely short.")
    cue.add_argument("--no-cue-voice", action="store_true",
                     help="Silence the spoken cue. Only do this if you have a "
                          "second person calling the classes.")
    clf.add_argument("--allow-unpinned-model", action="store_true",
                     help="Permit an SVM whose hash differs from the frozen "
                          "artefact characterised in Section V-C.")

    run = ap.add_argument_group("run identity")
    run.add_argument("--run-id", default="drone",
                     help="<type>_<what> tag, e.g. drone_n8-hover-cued-svm; names the output directory.")
    run.add_argument("--note", default="",
                     help="Free-text note stored in the run manifest.")

    abl = ap.add_argument_group("ablations (one design element off per flight)")
    abl.add_argument("--no-identity-gate", action="store_true",
                     help="Accept any detected hand/face, ignoring enrollment.")
    abl.add_argument("--no-hysteresis", action="store_true",
                     help="Zero mode hold and release times.")
    abl.add_argument("--no-ema", action="store_true",
                     help="Disable landmark smoothing.")
    abl.add_argument("--no-fb-streak", action="store_true",
                     help="Disable the FORWARD/BACK stability gate.")
    abl.add_argument("--no-auth-hysteresis", action="store_true",
                     help="Bare per-frame identity threshold (pre-fix behaviour).")

    assoc = ap.add_argument_group("hand-face association (off unless --associate-hands)")
    assoc.add_argument("--associate-hands", action="store_true",
                       help="Accept only a hand on the verified person's arm. Changes which "
                            "hand may command; the paper's flights ran without it.")
    assoc.add_argument("--target-side", choices=("right", "left"), default=None,
                       help="Anatomical arm that commands (default: right).")
    assoc.add_argument("--no-arm-check", action="store_true",
                       help="Accept a hand even when its arm is out of frame (the "
                            "original behaviour; Pose then guesses the arm).")
    assoc.add_argument("--live-pose", action="store_true",
                       help="Run pose on every frame (debugging; costs frame rate). Toggle with 'b'.")
    assoc.add_argument("--debug-overlay", action="store_true",
                       help="Start with the landmark overlay on. Toggle with 'v'.")
    assoc.add_argument("--mirrored", action="store_true",
                       help="The frame is horizontally flipped. Never for the Tello feed.")

    safety = ap.add_argument_group("safety")
    safety.add_argument("--no-fly", action="store_true",
                        help="Block takeoff entirely. Use for bench checks.")

    tune = ap.add_argument_group("overrides")
    tune.add_argument("--rc-speed", type=int, default=None, help="RC magnitude for gesture commands.")
    tune.add_argument("--battery-land-pct", type=int, default=None, help="Failsafe landing threshold.")
    tune.add_argument("--failsafe-step", type=int, default=None,
                      help="After each failsafe, lower the threshold by this many points "
                           "so you can take off and trigger another. 0 disables.")
    tune.add_argument("--log-hz", type=float, default=None, help="Telemetry sample rate (0 = every loop).")
    tune.add_argument("--search-yaw", type=int, default=None,
                      help="Yaw RC magnitude during search. ~0.43 deg/s per unit.")
    tune.add_argument("--search-duration", type=float, default=None,
                      help="Seconds per search sweep. yaw*0.43*duration ~= degrees covered.")
    tune.add_argument("--nohuman-search", type=float, default=None,
                      help="Seconds without an authorized human before searching.")
    tune.add_argument("--no-llm", action="store_true", help="Disable the reason-only LLM.")
    rec = ap.add_argument_group("recording (into the run directory; git-ignored)")
    rec.add_argument("--record", action="store_true",
                     help="Save what the flight window shows (HUD included) as session.mp4.")
    rec.add_argument("--record-raw", action="store_true",
                     help="Also save the clean camera frames as raw.mp4, for re-analysis.")
    rec.add_argument("--record-fps", type=float, default=None,
                     help="Video frame rate (default 20). Playback always runs in real time.")
    tune.add_argument("--hide-llm-text", action="store_true",
                      help="Start with the LLM's text hidden in the HUD (toggle with 'i'); it still runs and logs.")

    return ap


def main():
    args = build_parser().parse_args()

    cfg = ControllerConfig()

    cfg.no_identity_gate = args.no_identity_gate
    cfg.no_hysteresis = args.no_hysteresis
    cfg.no_ema = args.no_ema
    cfg.no_fb_streak = args.no_fb_streak
    cfg.no_auth_hysteresis = args.no_auth_hysteresis
    cfg.allow_takeoff = not args.no_fly

    if args.rc_speed is not None:
        cfg.rc_speed = args.rc_speed
    if args.battery_land_pct is not None:
        cfg.battery_land_pct = args.battery_land_pct
    if args.failsafe_step is not None:
        cfg.failsafe_step_pct = args.failsafe_step
    if args.log_hz is not None:
        cfg.log_hz = args.log_hz
    if args.search_yaw is not None:
        cfg.search_yaw_cmd = args.search_yaw
    if args.search_duration is not None:
        cfg.search_duration_s = args.search_duration
    if args.nohuman_search is not None:
        cfg.nohuman_search_s = args.nohuman_search
    if args.no_llm:
        cfg.llm_enabled = False
    cfg.hand_association = args.associate_hands
    cfg.assoc_mirrored = args.mirrored
    cfg.assoc_require_arm = not args.no_arm_check
    cfg.assoc_live_pose = args.live_pose
    cfg.debug_overlay = args.debug_overlay
    cfg.hud_show_llm = not args.hide_llm_text
    cfg.record_video = args.record or args.record_raw
    cfg.record_raw = args.record_raw
    if args.record_fps is not None:
        cfg.record_fps = args.record_fps
    if args.target_side is not None:
        cfg.assoc_target_side = args.target_side

    run = RunContext(run_id=args.run_id, note=args.note)

    # Arm assignment. Under randomisation the operator is not told which arm is
    # flying: stream rate varies by venue and run, so arms have to interleave,
    # and an operator who knows the arm can steer the result.
    blind = False
    if args.arm_randomise:
        import random
        seed = args.arm_seed if args.arm_seed is not None else random.SystemRandom().randrange(1 << 30)
        classifier = random.Random(seed).choice(("svm", "rule"))
        run.record("arm_assignment", {"randomised": True, "seed": seed,
                                      "blinded_during_run": True})
        blind = True
        print("[arm] randomised; assignment withheld until reveal_arms.py")
    else:
        classifier = args.classifier
        run.record("arm_assignment", {"randomised": False, "seed": None,
                                      "blinded_during_run": False})

    from .gesture_classifier import ClassifierSelectionError
    try:
        ctrl = Controller(cfg, model_path=args.model, labels_path=args.labels,
                          run=run, classifier=classifier,
                          require_frozen=not args.allow_unpinned_model,
                          blind=blind, cue_rounds=args.cue_rounds,
                          cue_settle_s=args.cue_settle_s,
                          cue_hold_s=args.cue_hold_s, cue_seed=args.cue_seed, cue_classes=args.cue_classes,
                          cue_voice=not args.no_cue_voice,
                          freeflight=args.freeflight_log,
                          log_landmarks=not args.no_landmarks,
                          no_actuate=args.no_actuate)
    except ClassifierSelectionError as e:
        run.record("aborted", {"reason": str(e)})
        run.write()
        raise SystemExit(f"\nclassifier selection failed:\n{e}\n")
    raise SystemExit(ctrl.run_loop())


if __name__ == "__main__":
    main()
