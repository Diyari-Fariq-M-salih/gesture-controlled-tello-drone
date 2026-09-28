"""Every tunable setting of the flight stack, in one place.

One dataclass per component, then ControllerConfig, which the command line
fills and the controller passes down. Each module imports its own class from
here (`from .config import FaceIDConfig`), so the old import paths still work.
Values set on the command line land in ControllerConfig and are recorded in
every run's manifest.

    FaceIDConfig          identity gate: embedding model, thresholds, hysteresis
    FaceFollowConfig      face following: target, gains, forward/back law
    BackFollowConfig      follow behind: re-ID model, enrolment, distance/height hold
    DeterministicConfig   mode arbitration timings
    LLMReasonConfig       the reason-only language model
    ControllerConfig      the flight loop, gestures, association, recording, ablations
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REID_MODEL = PROJECT_ROOT / "models" / "third_party" / "osnet_x0_25_msmt17.onnx"
DEFAULT_POSE_MODEL = PROJECT_ROOT / "models" / "mediapipe" / "pose_landmarker_full.task"


# ------------------------------------------------- identity gate (face_id.py)
@dataclass
class FaceIDConfig:
    model_path: str = str(PROJECT_ROOT / "models" / "third_party" / "arcface.onnx")
    input_size: int = 112

    # Many TF-exported ArcFace models expect RGB. If your scores look wrong, flip this.
    rgb: bool = True

    # Common ArcFace normalization: (x - 127.5) / 128
    mean: float = 127.5
    std: float = 128.0

    enroll_samples: int = 20
    enroll_min_face_px: int = 60

    cosine_thr: float = 0.55

    # --- authorization hysteresis (Schmitt trigger) ---
    #
    # A bare per-frame `score >= cosine_thr` chatters whenever the operator sits
    # near the threshold: consecutive frames of the same person score e.g. 0.56,
    # 0.49, 0.54, 0.67, and authorization flickers on and off. Everything
    # downstream -- the hand gate, and through it the whole mode manager -- is
    # built to arbitrate over a stable signal, so the flicker is amplified into
    # spurious mode changes and refused commands.
    #
    # With hysteresis the gate opens at `cosine_thr` but only closes below
    # `release_thr`, so marginal frames hold their previous decision instead of
    # oscillating. Set hysteresis=False to recover the original bare-threshold
    # behaviour for comparison.
    hysteresis: bool = True
    release_thr: float = 0.45

    # Frames below release_thr must accumulate before authorization drops, so a
    # single bad crop (blink, motion blur, half-turned head) cannot revoke it.
    release_frames: int = 3


# -------------------------------------------- face following (face_follow.py)
@dataclass
class FaceFollowConfig:
    target_area_frac: float = 0.075

    kp_yaw: float = 0.2
    kp_ud: float = 0.12
    kp_fb: float = 0.30

    max_yaw: int = 20
    max_ud: int = 40
    max_fb: int = 40

    deadband_px: int = 18
    deadband_area: float = 0.008

    # Forward/back law. "distance" (default since 2026-09-26) steers on the
    # standoff estimated from the face box, d = k_dist_m / sqrt(area), so a metre
    # too far and a metre too close command the same speed. "area" is the law
    # every run up to then flew (arXiv v1): it steers on area, which falls with
    # the square of distance, so it approached slowly (never above ~22) and
    # backed off hard (up to max_fb). The target distance is the same in both:
    # k_dist_m / sqrt(target_area_frac) = 0.94 m.
    fb_law: str = "distance"
    k_dist_m: float = 0.258          # sqrt(area) calibration, audit claim V-H
    kp_fb_per_m: float = 40.0        # rc units per metre of standoff error
    deadband_dist_m: float = 0.06    # about the area law's deadband at the target

    lost_timeout_s: float = 0.7

    # Cheaper detection -> smoother stream
    detect_w: int = 960
    detect_h: int = 720
    detect_every_n: int = 3
    control_hz: float = 15.0

    area_ema_alpha: float = 0.25

    # Face-ID crop padding (relative)
    crop_pad: float = 0.15

    # Identity crops must come from a FRESH detection.
    #
    # `lost_timeout_s` deliberately holds face_detected() true across gaps so
    # mode arbitration does not flicker on a low-frame-rate stream. That hold is
    # right for arbitration and wrong for identity: re-embedding a stale box
    # feeds ArcFace whatever is now inside a box the face has already left,
    # which scores like a stranger and revokes authorization from the real
    # operator. Identity therefore gets its own, much tighter freshness window.
    crop_max_age_s: float = 0.35
    crop_min_px: int = 60


# --------------------------------------------- follow behind (back_follow.py)
@dataclass
class BackFollowConfig:
    reid_model: str = str(DEFAULT_REID_MODEL)
    pose_model: str = str(DEFAULT_POSE_MODEL)
    num_poses: int = 2
    pose_every_n: int = 2            # pose on every n-th frame it is needed

    enroll_delay_s: float = 6.0      # time to turn round after face enrolment
    enroll_samples: int = 20
    enroll_every_s: float = 0.12     # spacing between enrolment crops
    enroll_timeout_s: float = 20.0
    enroll_min_samples: int = 8      # accept a short enrolment after the timeout

    match_on: float = 0.65           # cosine to lock onto the operator
    match_off: float = 0.50          # cosine below which a locked track is dropped
    hold_s: float = 1.5              # ride through short pose dropouts before "lost"
    vis_thresh: float = 0.5          # shoulders and the box
    hip_vis_thresh: float = 0.3      # hips are often dim or at the frame edge
    smooth: float = 0.5              # EMA weight of a new box and distance (1 = none)
    min_torso_px: float = 40.0

    follow_distance_m: float = 1.5
    follow_height_m: float = 2.0
    torso_m: float = 0.50            # shoulder-mid to hip-mid, a typical adult
    focal_px_960: float = 680.0      # Tello, 82.6 deg diagonal FOV, 960 px wide

    kp_yaw: float = 0.12             # rc per px of horizontal error
    max_yaw: int = 30
    deadband_px: int = 30
    kp_fb_per_m: float = 40.0         # same linear law and gain as face follow
    max_fb: int = 40
    deadband_m: float = 0.15          # the torso-based estimate is noisier than the face one
    kp_ud_per_m: float = 60.0
    max_ud: int = 30
    deadband_h_m: float = 0.10


# ------------------ mode arbitration and the language model (mode_manager.py)
@dataclass
class DeterministicConfig:
    """Arbitration parameters.

    The controller always constructs this from `ControllerConfig`, so the
    defaults below are reachable only by a direct instantiation in a test. The
    three search values in particular are *not* the deployed ones: the deployed
    sweep is 28 s after the coverage measurement, against the 5 s originally
    specified. Read `ControllerConfig` for what flies.
    """

    battery_land_pct: int = 15

    # Not deployed values -- see the note above; ControllerConfig has 10/28/10.
    nohuman_search_s: float = 5.0
    search_duration_s: float = 12.0
    search_cooldown_s: float = 5.0

    mode_hold_s: float = 1.2
    hand_release_s: float = 0.8
    face_release_s: float = 0.8
    back_release_s: float = 1.0      # follow-behind (opt-in; inert when no back is reported)


@dataclass
class LLMReasonConfig:
    enabled: bool = True
    # 1.5B since 2026-09-27; every run before flew qwen2.5:0.5b-instruct (arXiv v1).
    # Chosen with scripts/llm_bench.py on 40 logged moments: the 0.5B invented
    # battery warnings in 23-42% of replies, the 1.5B in 7%, at 0.9 s a reply.
    model: str = "qwen2.5:1.5b-instruct"
    url: str = "http://127.0.0.1:11434/api/chat"
    decision_hz: float = 1.0
    timeout_s: float = 4.0
    # Ollama unloads an idle model after 5 min and reloading it outlasts the
    # timeout, which is where most "LLM error (ReadTimeout)" rows came from.
    keep_alive: str = "30m"
    max_tokens: int = 60             # one sentence; also bounds the latency
    temperature: float = 0.2


# --------------------------- the flight loop; the command line fills this one
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
    # follow-behind (opt-in): face enrolment, then the operator's back; with no
    # face in view the aircraft trails the enrolled back (back_follow.py)
    follow_behind: bool = False
    follow_distance_m: float = BackFollowConfig.follow_distance_m
    follow_height_m: float = BackFollowConfig.follow_height_m
    back_enroll_delay_s: float = BackFollowConfig.enroll_delay_s

    # face-follow forward/back law: "distance" (default) or "area" (as flown in v1)
    follow_law: str = "distance"

    record_video: bool = False
    record_raw: bool = False
    record_fps: float = 20.0

    # LLM reasoner
    llm_enabled: bool = True
    llm_model: str = LLMReasonConfig.model
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
