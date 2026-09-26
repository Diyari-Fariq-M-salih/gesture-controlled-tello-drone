from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ControllerConfig:
    # timing
    ui_dt: float = 1.0 / 30.0      # ~30 Hz UI loop
    rc_dt: float = 1.0 / 10.0      # ~10 Hz RC send loop

    # RC control
    rc_speed: int = 30
    rc_deadband: int = 6
    rc_limit: int = 100

    # gesture thresholds
    dir_thr: float = 0.10          # index vector threshold (normalized)
    scale_thr: float = 0.18        # forward/back by hand scale change
    ema_alpha: float = 0.35        # landmark smoothing

    # perception throttling
    hand_every_n: int = 2
    face_every_n: int = 4

    # streak gating
    hand_streak_on: int = 2
    face_streak_on: int = 1
    gesture_fb_streak_on: int = 2

    # arbitration (mirrors DeterministicConfig; recorded in the run manifest)
    battery_land_pct: int = 15
    # After a battery failsafe fires, drop the threshold by this many points so
    # the operator can take off again and trigger a fresh one. Lets a single
    # charge yield several failsafe trials instead of one. 0 disables stepping.
    failsafe_step_pct: int = 0
    # Never step below this: under ~15% the radio link itself becomes unreliable,
    # so a trial there measures the link rather than the failsafe.
    failsafe_floor_pct: int = 15
    nohuman_search_s: float = 10.0
    # 28 s at search_yaw_cmd=30 (~13 deg/s measured) completes one full
    # rotation. Yaw is kept low deliberately: the sweep exists to detect a
    # face, and a faster spin smears the frame enough that the detector
    # misses the operator it is rotating past.
    search_duration_s: float = 28.0
    search_cooldown_s: float = 10.0
    mode_hold_s: float = 1.2
    hand_release_s: float = 0.8
    face_release_s: float = 0.8
    search_yaw_cmd: int = 30

    # face identity
    faceid_cosine_thr: float = 0.55
    faceid_enroll_samples: int = 20
    faceid_hysteresis: bool = True      # Schmitt trigger on the authorization decision
    faceid_release_thr: float = 0.45    # de-authorize only below this
    faceid_release_frames: int = 3      # ...and only after this many consecutive frames

    # hand-face association (hand_association.py). Off by default: the paper's
    # flights ran without it, and turning it on changes which hand may command.
    hand_association: bool = False
    assoc_max_hands: int = 4            # candidates: the operator's two hands + a bystander's
    assoc_target_side: str = "right"    # anatomical arm that commands
    assoc_mirrored: bool = False        # the Tello feed is not flipped; a selfie webcam is
    assoc_num_poses: int = 2            # >1, or a bystander's skeleton can crowd out the operator's
    assoc_every_n: int = 20             # fresh hand detections between pose re-anchors
    # Shoulder-widths, hand to target arm. 0.7 admits the operator's own
    # gesturing hand in 91% of calibration frames (0.5: 85%; FORWARD alone
    # has a median of 0.50). scripts/association_calibration.py regenerates
    # the evidence. A bystander's hand measured ~4.2 on one photo; confirm that
    # margin with two people before relying on it.
    assoc_dist_thresh: float = 0.7
    assoc_side_margin: float = 0.85     # target arm must beat the other arm by this factor
    assoc_face_frac: float = 0.6        # share of pose face points inside the verified face box
    assoc_min_iou: float = 0.15         # below this the IoU track counts as lost
    assoc_handedness_conf: float = 0.9  # a handedness label vetoes only above this
    # Require the target arm's elbow and wrist in frame with Pose visibility >=
    # assoc_arm_vis. Pose invents out-of-frame limbs, so without this a hand with
    # no arm in shot matches its own guessed arm. Keeps 76% of the operator's
    # close-range calibration gestures (the rest have the elbow below the frame).
    assoc_require_arm: bool = True
    assoc_arm_vis: float = 0.5
    # Pose on every fresh hand detection instead of every assoc_every_n. Debug
    # aid: tracks the body live at a cost in frame rate. Toggled with 'b'.
    assoc_live_pose: bool = False

    # landmark overlay in the flight window (display only; toggled with 'v')
    debug_overlay: bool = False
    # show the LLM's reasoning text in the HUD (display only; toggled with 'i').
    # The model still runs and logs either way; --no-llm turns it off.
    hud_show_llm: bool = True

    # session recording into the run directory (session_recorder.py). The
    # videos show faces and are git-ignored; raw.mp4 doubles the encoding cost.
    record_video: bool = False
    record_raw: bool = False
    record_fps: float = 20.0

    # LLM reasoner
    llm_enabled: bool = True
    llm_model: str = "qwen2.5:0.5b-instruct"
    llm_decision_hz: float = 1.0
    llm_timeout_s: float = 4.0

    # ---- ablations (each disables one design element for A/B flights) ----
    no_identity_gate: bool = False   # accept any hand/face, ignoring enrollment
    no_hysteresis: bool = False      # zero hold/release times
    no_ema: bool = False             # raw landmarks, no smoothing
    no_fb_streak: bool = False       # no stability gate on FORWARD/BACK
    no_auth_hysteresis: bool = False # bare per-frame threshold on identity

    # ---- safety ----
    allow_takeoff: bool = True       # False blocks 't' entirely (bench checks)

    # telemetry logging
    log_hz: float = 1.0

    # tello
    tello_ip: str = "192.168.10.1"
    cmd_port: int = 8889
    state_port: int = 8890
    video_port: int = 11111
    local_cmd_port: int = 9013

    def apply_ablations(self) -> None:
        """Fold ablation switches into the numeric parameters they suppress."""
        if self.no_hysteresis:
            self.mode_hold_s = 0.0
            self.hand_release_s = 0.0
            self.face_release_s = 0.0
        if self.no_ema:
            self.ema_alpha = 1.0
        if self.no_fb_streak:
            self.gesture_fb_streak_on = 1
        if self.no_auth_hysteresis:
            self.faceid_hysteresis = False

    def active_ablations(self) -> list:
        return [
            name for name, on in (
                ("no_identity_gate", self.no_identity_gate),
                ("no_hysteresis", self.no_hysteresis),
                ("no_ema", self.no_ema),
                ("no_fb_streak", self.no_fb_streak),
                ("no_auth_hysteresis", self.no_auth_hysteresis),
            ) if on
        ]
