import time
import cv2
from typing import Optional
import textwrap

from .config import ControllerConfig
from .run_context import RunContext
from .tello_udp import TelloUDP
from .state_listener import StateListener
from .latest_frame import LatestFrame
from .video_stream import VideoStream
from .hand_gesture import HandGesture
from .hand_association import AssociationTracker, HandAssociator
from .association_overlay import draw_debug
from .session_recorder import SessionRecorder
from .gesture_logic import DepthStabilityGate, RuleBasedGesture, rc_from_gesture_name
from .keyboard import rc_from_key, RC
from .telemetry_logger import TelemetryLogger, DecisionLogger, ScenarioLogger
from .perf_logger import PerfLogger, StageTimer
from .face_follow import FaceFollower

from .face_id import FaceID, FaceIDConfig
from . import gesture_classifier as gclf
from .cue_schedule import (CueSchedule, FIELDS as CUE_FIELDS,
                           LANDMARK_FIELDS, landmark_row)
from .cue_voice import Voice

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

from .mode_manager import (
    DeterministicModeManager,
    DeterministicConfig,
    LLMReasoner,
    LLMReasonConfig,
)

SCENARIO_KEYS = {
    ord("1"): "authorized_gesture_accepted",
    ord("2"): "unauthorized_gesture_rejected",
    ord("3"): "face_following_activates",
    ord("4"): "gesture_preempts_face",
    ord("5"): "hysteresis_prevents_flicker",
    ord("6"): "search_starts_after_loss",
    ord("7"): "target_reacquired_after_search",
    ord("8"): "battery_failsafe_landing",
}


def _put_text_box(
    img,
    text: str,
    org,
    *,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    scale=0.55,
    thickness=1,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0),
    alpha=0.55,
    pad=4,
):
    """Draw text with a translucent background box for readability."""
    x, y = org
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    x1, y1 = x - pad, y - th - pad
    x2, y2 = x + tw + pad, y + baseline + pad

    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.putText(img, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)


def _fit(text: str, max_px: int, scale: float, thickness: int = 1) -> str:
    """Trim `text` so it renders no wider than max_px (fields never overlap)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    if cv2.getTextSize(text, font, scale, thickness)[0][0] <= max_px:
        return text
    while text and cv2.getTextSize(text + "..", font, scale, thickness)[0][0] > max_px:
        text = text[:-1]
    return text + ".."


_MODE_COLOR = {"GESTURE": (0, 230, 0), "FACE": (255, 220, 0), "SEARCH_360": (0, 220, 255),
               "HOVER": (200, 200, 200), "LAND": (0, 0, 255)}


def draw_hud(
    frame,
    *,
    mode: str,
    flying: bool,
    hand_detected_raw: bool,
    hand_detected: bool,
    raw_face: bool,
    face_detected: bool,
    gesture_name: str,
    face_id: FaceID,
    run_name: str = "",
    ablations: Optional[list] = None,
    trial: Optional[dict] = None,
    tally: Optional[dict] = None,
    fps: Optional[float] = None,
    battery=None,
    failsafe_thr=None,
    llm_reason: Optional[str] = None,
    assoc: Optional[str] = None,
    cue: Optional[dict] = None,
    show_llm: bool = True,
    show_help: bool = False,
    recording: bool = False,
):
    """Status as a fixed-size band across the top of the frame.

    Three rows on a fixed column grid. The band's size never depends on its
    text, every field starts at a fixed x and is trimmed to its column, and
    numbers use fixed-width formats, so a changing value neither resizes the
    background nor shifts its neighbours: nothing flickers. Text scales with
    the frame width. Keys are behind 'h' rather than on screen permanently.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(max(0.42 * w / 960.0, 0.34), 0.6)
    line_h = int(22 * scale / 0.42)
    pad = max(4, line_h // 4)
    white, dim = (255, 255, 255), (170, 170, 170)

    def band(y0, rows, alpha=0.55, color=(0, 0, 0)):
        over = frame.copy()
        cv2.rectangle(over, (0, y0), (w, y0 + rows * line_h + pad), color, -1)
        cv2.addWeighted(over, alpha, frame, 1 - alpha, 0, frame)

    def put(col_from, col_to, row, text, color=white, y0=0, bold=False):
        x = int(col_from * w) + 8
        max_px = int((col_to - col_from) * w) - 14
        y = y0 + (row + 1) * line_h - pad // 2
        th = 2 if bold else 1
        cv2.putText(frame, _fit(text, max_px, scale, th), (x, y), font, scale, color, th, cv2.LINE_AA)

    band(0, 3)

    # Row 1 -- aircraft and run
    m = mode.upper()
    put(0.00, 0.14, 0, f"{m}", _MODE_COLOR.get(m, white), bold=True)
    put(0.14, 0.23, 0, f"FLY {'Y' if flying else 'N'}", (0, 230, 0) if flying else dim)
    bat = "BAT  --" if battery is None else f"BAT {battery:3.0f}%"
    if failsafe_thr is not None:
        bat += f" (<={failsafe_thr:.0f})"
    low = battery is not None and failsafe_thr is not None and battery <= failsafe_thr + 5
    put(0.23, 0.40, 0, bat, (0, 0, 255) if low else white)
    put(0.40, 0.50, 0, "FPS  --" if not fps else f"FPS {fps:4.1f}")
    put(0.50, 0.72, 0, f"GESTURE {gesture_name}")
    abl = f"  abl:{','.join(ablations)}" if ablations else ""
    put(0.72, 0.93 if recording else 1.00, 0, f"{run_name}{abl}", dim)
    if recording:
        put(0.93, 1.00, 0, "REC", (0, 0, 255), bold=True)

    # Row 2 -- perception and identity
    n, N = face_id.enroll_progress()
    put(0.00, 0.14, 1, f"FACE {int(raw_face)}/{int(face_detected)}",
        (0, 230, 0) if face_detected else white)
    enr = f"ENROLLING {n:2d}/{N}" if face_id.enrolling else ("ID enrolled" if face_id.enrolled else "ID none (p)")
    put(0.14, 0.40, 1, f"{enr}  s={face_id.last_score:.3f}/{face_id.cfg.cosine_thr:.2f}",
        (0, 220, 255) if face_id.enrolling else white)
    put(0.40, 0.50, 1, f"HAND {int(hand_detected_raw)}/{int(hand_detected)}",
        (0, 230, 0) if hand_detected else white)
    if assoc:
        put(0.50, 1.00, 1, assoc)

    # Row 3 -- trial and language model
    if trial is not None:
        elapsed = time.time() - float(trial["t_start"])
        over = frame.copy()
        cv2.rectangle(over, (0, 2 * line_h + pad // 2), (int(0.40 * w), 3 * line_h + pad),
                      (0, 90, 160), -1)
        cv2.addWeighted(over, 0.75, frame, 0.25, 0, frame)
        put(0.00, 0.40, 2, f"TRIAL {trial['scenario']} #{trial['trial']} {elapsed:5.1f}s  y/n/x")
    elif tally:
        put(0.00, 0.40, 2, "trials " + " ".join(f"{k[:10]}:{v}" for k, v in sorted(tally.items())), dim)
    else:
        put(0.00, 0.40, 2, "no trial open (1-8)", dim)
    if not show_llm:
        llm = "LLM text hidden (i)"
    else:
        llm = "LLM: " + (" ".join(llm_reason.split()) if llm_reason else "-")
    put(0.40, 0.93, 2, llm, white if show_llm else dim)
    put(0.93, 1.00, 2, "h help", dim)

    # Cued capture: the one thing the operator reads, so it gets its own band.
    if cue:
        hold = cue["phase"] == "hold"
        y0 = 3 * line_h + pad + 4
        band(y0, 2, alpha=0.65)
        put(0.0, 1.0, 0, f"{'HOLD  ' if hold else 'SETTLE'}  {cue['name']}   {cue['progress']}",
            (0, 255, 0) if hold else (0, 200, 255), y0=y0, bold=True)
        put(0.0, 1.0, 1, cue["text"], y0=y0)

    if show_help:
        keys = ["t takeoff   l land   e EMERGENCY   q quit",
                "p enroll   o clear   1-8 trial   y/n/x pass/fail/discard",
                "v landmarks   b live pose   i LLM text   h help",
                "c cue start/pause   k cue skip   m sync marker"]
        y0 = h - len(keys) * line_h - pad - 6
        over = frame.copy()
        cv2.rectangle(over, (0, y0), (int(0.62 * w), h), (0, 0, 0), -1)
        cv2.addWeighted(over, 0.7, frame, 0.3, 0, frame)
        for i, line in enumerate(keys):
            cv2.putText(frame, line, (8, y0 + (i + 1) * line_h - pad // 2), font, scale,
                        white, 1, cv2.LINE_AA)


class Controller:
    def __init__(
        self,
        cfg: ControllerConfig,
        model_path: Optional[str] = None,
        labels_path: Optional[str] = None,
        classifier: str = None,
        require_frozen: bool = True,
        blind: bool = False,
        cue_rounds: int = 0,
        cue_settle_s: float = 5.0,
        cue_hold_s: float = 6.0,
        cue_seed: int = 7,
        cue_classes: str = "",
        cue_voice: bool = True,
        log_landmarks: bool = True,
        no_actuate: bool = False,
        freeflight: bool = False,
        run: Optional[RunContext] = None,
    ):
        cfg.apply_ablations()
        self.cfg = cfg
        self.run_ctx = run or RunContext("run")

        self.tello = TelloUDP(cfg.tello_ip, cfg.cmd_port, cfg.local_cmd_port)
        self.state = StateListener(cfg.state_port)
        self.latest = LatestFrame()

        self.video = VideoStream(
            self.latest,
            # timeout is microseconds: give up on a stalled stream after ~2s and
            # reopen, rather than blocking on ffmpeg's 30s default. On a congested
            # 2.4GHz band the Tello stream stalls in bursts; the cost of a stall is
            # dominated by how long we wait before reconnecting, not the stall itself.
            f"udp://0.0.0.0:{cfg.video_port}"
            f"?fifo_size=5000000&overrun_nonfatal=1&timeout=2000000&buffer_size=1000000"
        )

        # One hand, as flown in the paper, unless hand-face association is on:
        # then every hand is a candidate and the associator picks the operator's.
        self.hand = HandGesture(max_num_hands=cfg.assoc_max_hands if cfg.hand_association else 1)
        self.assoc = None
        if cfg.hand_association:
            self.assoc = AssociationTracker(HandAssociator(
                target_side=cfg.assoc_target_side, mirrored=cfg.assoc_mirrored,
                dist_thresh=cfg.assoc_dist_thresh, side_margin=cfg.assoc_side_margin,
                face_frac_thresh=cfg.assoc_face_frac, min_iou=cfg.assoc_min_iou,
                handedness_conf=cfg.assoc_handedness_conf, num_poses=cfg.assoc_num_poses,
                require_arm_visible=cfg.assoc_require_arm, arm_vis_thresh=cfg.assoc_arm_vis),
                every_n=0 if cfg.assoc_live_pose else cfg.assoc_every_n)
        self._debug_overlay = bool(cfg.debug_overlay)
        self._show_llm = bool(cfg.hud_show_llm)
        self._show_help = False
        self._recorder = (SessionRecorder(self.run_ctx.dir, fps=cfg.record_fps,
                                          record_raw=cfg.record_raw)
                          if cfg.record_video else None)
        self._assoc_det = None
        self.rule = RuleBasedGesture(cfg.dir_thr, cfg.scale_thr, cfg.ema_alpha)

        self.face = FaceFollower()
        # Hold face_detected() across stream gaps so arbitration does not
        # flicker; identity crops are gated separately and much more tightly
        # (FaceFollowConfig.crop_max_age_s).
        self.face.cfg.lost_timeout_s = 2.0
        # The controller already throttles observe(); leaving FaceFollower's own
        # detect_every_n above 1 would compound the two into a much rarer real
        # detection than either setting implies.
        self.face.cfg.detect_every_n = 1

        # FaceID (identity lock)
        self.face_id = FaceID(FaceIDConfig(
            model_path=str(PROJECT_ROOT / "models" / "third_party" / "arcface.onnx"),
            cosine_thr=cfg.faceid_cosine_thr,
            enroll_samples=cfg.faceid_enroll_samples,
            hysteresis=cfg.faceid_hysteresis,
            release_thr=cfg.faceid_release_thr,
            release_frames=cfg.faceid_release_frames,
            rgb=True,
        ))

        self.mode_mgr = DeterministicModeManager(DeterministicConfig(
            battery_land_pct=cfg.battery_land_pct,
            nohuman_search_s=cfg.nohuman_search_s,
            search_duration_s=cfg.search_duration_s,
            search_cooldown_s=cfg.search_cooldown_s,
            mode_hold_s=cfg.mode_hold_s,
            hand_release_s=cfg.hand_release_s,
            face_release_s=cfg.face_release_s,
        ))

        self._search_yaw_cmd = cfg.search_yaw_cmd

        self.reasoner = LLMReasoner(LLMReasonConfig(
            enabled=cfg.llm_enabled,
            model=cfg.llm_model,
            decision_hz=cfg.llm_decision_hz,
            timeout_s=cfg.llm_timeout_s,
        ))

        self.flying = False

        # ---- run-scoped outputs: one directory per launch, never overwritten ----
        self.logger = TelemetryLogger(
            fields=["bat", "h", "yaw", "vgx", "vgy", "vgz", "tof"],
            path=self.run_ctx.path("telemetry.csv"),
            hz=cfg.log_hz,
        )
        self.decisions = DecisionLogger(path=self.run_ctx.path("decisions.csv"))
        self.perf = PerfLogger(path=self.run_ctx.path("perf.csv"))
        self.scenarios = ScenarioLogger(path=self.run_ctx.path("scenarios.csv"), run_id=self.run_ctx.name)
        self._scenario_keys = SCENARIO_KEYS

        # Timers (AUTHORIZED signals only, unless the identity gate is ablated)
        self._last_hand_ts = time.time()
        self._last_face_ts = time.time()
        self._last_any_seen_ts = time.time()

        # Throttling
        self._hand_frame_i = 0
        self._last_hand_det = None
        self._face_frame_i = 0

        # Streak gating
        self._hand_streak = 0
        self._face_streak = 0

        # Gesture stability gating
        self._gesture_stable_name = "CENTER"
        self._gesture_stable_count = 0

        # CSV logging control
        self._prev_mode = ""
        self._prev_cmd = ""
        self._prev_llm_reason = ""
        self._last_log_ts = 0.0
        self._log_every_s = 1.0

        self._warned_deadband = False
        self._window_sized = False

        # frame bookkeeping
        self._last_seq = -1
        self._display = None
        self._fps_ema = None

        self._fb_gate = DepthStabilityGate(cfg.gesture_fb_streak_on)
        # Explicit selection. `classifier` is required upstream; there is no
        # path here that falls back silently, which is what let an entire
        # campaign run the rule while the paper characterised the SVM.
        self.selection = gclf.select(
            classifier, cfg, model_path=model_path, labels_path=labels_path,
            rule=self.rule, require_frozen=require_frozen)
        self._selftest = gclf.self_test(self.selection)
        self._trained = None if self.selection.kind == "rule" else self.selection
        self._blind = bool(blind)
        # Hover lock. Applied at the transmit boundary only; nothing upstream
        # branches on it, so the logged command stays the command that would
        # have been sent.
        self._no_actuate = bool(no_actuate)
        self._suppressed_nonzero = 0

        # Cued capture. Inactive until the operator presses 'c'; when active it
        # labels frames with the class the operator was asked for.
        self.cue = None
        self._cue_writer = None
        self._cue_rows = 0
        self._freeflight = bool(freeflight)
        if cue_rounds or self._freeflight:
            import csv as _csv
            self.cue = None if self._freeflight else CueSchedule(
                names={i: n for i, n in enumerate(
                    ["CENTER", "LEFT", "RIGHT", "UP", "DOWN", "FORWARD", "BACK"])},
                vocabulary=self.selection.kind, rounds=int(cue_rounds),
                settle_s=float(cue_settle_s), hold_s=float(cue_hold_s),
                seed=int(cue_seed),
                only=[x.strip().upper() for x in cue_classes.split(",")] if cue_classes else None)
            self._cue_fh = open(self.run_ctx.path("cued.csv"), "w", newline="",
                                encoding="utf-8")
            self._log_landmarks = bool(log_landmarks)
            fields = CUE_FIELDS + (LANDMARK_FIELDS if self._log_landmarks else [])
            self._cue_writer = _csv.DictWriter(self._cue_fh, fieldnames=fields)
            self._cue_writer.writeheader()
            # Spoken cue: the operator watches the aircraft, not the laptop.
            self._voice = Voice(enabled=cue_voice)
            self._cue_spoken = None
            self._cue_phase = None

        # provenance recorded up front, so an aborted run still explains itself
        self.run_ctx.record("config", cfg)
        self.run_ctx.record("ablations", cfg.active_ablations())
        if self.assoc is not None:
            import hashlib
            mp_path = self.assoc.associator.model_path
            self.run_ctx.record("hand_association", {
                "pose_model": mp_path,
                "pose_model_sha256": hashlib.sha256(open(mp_path, "rb").read()).hexdigest(),
                "num_poses": self.assoc.associator.num_poses,
                "target_side": cfg.assoc_target_side,
                "mirrored": cfg.assoc_mirrored,
            })
        # Top-level field: the one place a reader should have to look.
        self.run_ctx.record("classifier", self.selection.kind)
        self.run_ctx.record("classifier_detail", self.selection.manifest())
        self.run_ctx.record("classifier_selftest", self._selftest)
        self.run_ctx.record("blinded_during_run", self._blind)
        self.run_ctx.record("actuation_suppressed", self._no_actuate)
        if self.cue is not None:
            self.run_ctx.record("cued_protocol", self.cue.manifest())
        self.run_ctx.record("model", {
            "gesture_model": self.selection.model_path,
            "gesture_labels": self.selection.labels_path,
            "classifier": self.selection.kind,
            "faceid_model": self.face_id.cfg.model_path,
            "faceid_enabled": bool(self.face_id.enabled),
        })

    def _route_ok(self) -> bool:
        """True if this host currently has an address on the drone's subnet.

        Checked before the handshake because the common failure is not a lost
        packet but Windows silently leaving the drone's access point for one
        with internet -- which otherwise surfaces only as an opaque timeout.
        """
        import socket
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.settimeout(0.5)
            probe.connect((self.cfg.tello_ip, self.cfg.cmd_port))
            addr = probe.getsockname()[0]
            probe.close()
            return addr.rsplit(".", 1)[0] == self.cfg.tello_ip.rsplit(".", 1)[0]
        except Exception:
            return False

    def _sdk_init(self, attempts: int = 3) -> bool:
        if not self._route_ok():
            print(f"No route to {self.cfg.tello_ip} -- this PC is not on the drone's network.")
            print("  Rejoin the TELLO-XXXXXX Wi-Fi. Windows leaves access points")
            print("  that have no internet, so pin it:")
            print('    netsh wlan set profileparameter name="<other SSID>" connectionmode=manual')
            return False

        # The drone drops out of SDK mode after ~15 s without a command, and the
        # first packet after that is often lost. Retry before giving up rather
        # than making the operator relaunch.
        ok, resp = False, ""
        for i in range(1, attempts + 1):
            ok, resp = self.tello.send_cmd("command", timeout_ms=4000)
            if ok and resp.lower() == "ok":
                break
            print(f"SDK handshake attempt {i}/{attempts}: {ok} {resp}")
            time.sleep(0.5)
        if not ok or resp.lower() != "ok":
            print("Failed SDK mode:", ok, resp)
            print("  If tello_check just ran, wait ~5 s: the drone replies to the")
            print("  endpoint it latched onto and needs a moment to re-latch.")
            return False
        self.tello.send_cmd("streamoff", timeout_ms=2000)
        self.tello.send_cmd("streamon", timeout_ms=6000)
        return True

    def _rc_to_command_str(self, rc: RC) -> str:
        if not self.flying:
            return "ground"
        return f"rc lr={rc.lr} fb={rc.fb} ud={rc.ud} yaw={rc.yaw}"

    def _send_rc_locked(self, cfg) -> None:
        """Transmit a hold-station command under hover lock, and prove it.

        The values are constructed here rather than passed in, and asserted
        before transmission, so a future edit upstream cannot leak a live
        command into a suppressed run without failing loudly.
        """
        lr = fb = ud = yaw = 0
        if (lr, fb, ud, yaw) != (0, 0, 0, 0):
            raise RuntimeError("hover lock violated: non-zero rc on the wire")
        self.tello.send_rc(lr, fb, ud, yaw,
                           limit=cfg.rc_limit, deadband=cfg.rc_deadband)

    def _handle_sync_key(self, key: int):
        """'m' stamps a sync marker. Clap at the same instant.

        The phone recording and the flight log have independent clocks. One
        shared, sharply-defined event is what lets an annotation made from the
        video be mapped onto logged frames; without it the two can only be
        aligned by guesswork.
        """
        if key == ord("m"):
            self._sync_n = getattr(self, "_sync_n", 0) + 1
            t = time.time()
            self.perf.log_event(f"sync_marker_{self._sync_n}")
            self._sync_markers = getattr(self, "_sync_markers", [])
            self._sync_markers.append(t)
            print(f"[sync] marker {self._sync_n} at {t:.3f}  <- CLAP NOW")
            try:
                import winsound
                winsound.Beep(1600, 150)
            except Exception:
                pass

    def _handle_cue_keys(self, key: int):
        if self.cue is None:
            return
        if key == ord("c"):
            if self.cue.active:
                self.cue.stop()
                print(f"[cue] paused at {self.cue.progress()}")
            else:
                self.cue.start()
                print(f"[cue] running, {self.cue.progress()}")
        if key == ord("k") and self.cue.active:
            self.cue.skip()
            print(f"[cue] skipped -> {self.cue.progress()}")

    def _handle_debug_keys(self, key: int):
        # v = landmark overlay; b = pose on every frame (body tracked live)
        if key == ord("v"):
            self._debug_overlay = not self._debug_overlay
            self.perf.log_event(f"debug_overlay_{'on' if self._debug_overlay else 'off'}")
            print(f"[debug] landmark overlay {'ON' if self._debug_overlay else 'off'}")
        if key == ord("i"):
            self._show_llm = not self._show_llm
            print(f"[hud] LLM text {'shown' if self._show_llm else 'hidden'}")
        if key == ord("h"):
            self._show_help = not self._show_help
        if key == ord("b") and self.assoc is not None:
            live = self.assoc.every_n != 0
            self.assoc.every_n = 0 if live else self.cfg.assoc_every_n
            self.assoc.reset()
            self.perf.log_event(f"assoc_live_pose_{'on' if live else 'off'}")
            print(f"[debug] pose every frame {'ON' if live else 'off'}")

    def _handle_faceid_keys(self, key: int):
        # p = start/stop enrollment
        if key == ord("p"):
            if not self.face_id.enrolling:
                self.face_id.start_enroll()
                print(f"[FaceID] Enrollment started (need {self.cfg.faceid_enroll_samples} crops).")
            else:
                self.face_id.cancel_enroll()
                print("[FaceID] Enrollment cancelled.")

        # o = clear
        if key == ord("o"):
            self.face_id.clear()
            print("[FaceID] Cleared enrolled template.")

    def run_loop(self) -> int:
        cfg = self.cfg
        print(f"\n[run] {self.run_ctx.name}   ablations: {cfg.active_ablations() or 'none'}")
        print(f"[run] outputs -> {self.run_ctx.dir}")
        if self._blind:
            print("[run] arm BLINDED. Do not open the manifest until flying is "
                  "complete; use scripts/reveal_arms.py.")
        else:
            print(f"[run] classifier: {self.selection.kind} "
                  f"(writes {self.selection.latency_column})")
        if self._no_actuate:
            print("[run] HOVER LOCK: commands are computed and logged but not "
                  "transmitted. The aircraft will hold station.")
        if not cfg.allow_takeoff:
            print("[run] SAFETY: takeoff disabled (--no-fly). 't' will be ignored.")
        if not self.face_id.enabled:
            print("[run] WARNING: FaceID model unavailable; no face can be authorized.")

        if not self._sdk_init():
            return 1

        self.state.start()
        self.video.on_reopen = lambda: self.perf.log_event("video_reopen")
        if not self.video.start():
            print("Video stream not opened. Check firewall UDP 11111.")

        cv2.namedWindow("TELLO", cv2.WINDOW_AUTOSIZE)

        rc = RC(active=False)
        last_rc_send = time.time()
        gesture_name = "NOHAND"
        mode, det_reason = "hover", ""
        llm_reason = ""
        raw_face = face_detected = hand_detected_raw = hand_detected = False
        authorized_face = False

        try:
            while True:
                ok, frame, seq, frame_ts = self.latest.get(copy=True)
                now = time.time()

                # Only run perception and log a row when the decoder has actually
                # delivered a new frame. The control loop spins faster than the
                # stream, so counting repeats inflates the measured frame rate.
                is_new = bool(ok and frame is not None and seq != self._last_seq)

                if is_new:
                    self._last_seq = seq
                    raw_for_rec = (frame.copy() if self._recorder is not None
                                   and self.cfg.record_raw else None)
                    frame_age_ms = (now - frame_ts) * 1000.0
                    timer = StageTimer()

                    if not self._window_sized:
                        self._window_sized = True

                    # --- Hand detect (throttled) ---
                    self._hand_frame_i += 1
                    run_hand = (self._hand_frame_i % max(cfg.hand_every_n, 1) == 0)
                    if run_hand:
                        det = self.hand.detect(frame)
                        self._last_hand_det = det
                    else:
                        det = self._last_hand_det
                    timer.mark("hand_detect_ms", ran=run_hand)

                    raw_hand = bool(det is not None and det.has_hand and det.landmarks is not None)
                    if not raw_hand:
                        self._last_lm = None
                    self._hand_streak = min(self._hand_streak + 1, 10) if raw_hand else 0
                    hand_detected_raw = self._hand_streak >= cfg.hand_streak_on

                    # --- Face observe (throttled) ---
                    self._face_frame_i += 1
                    run_face = (self._face_frame_i % max(cfg.face_every_n, 1) == 0)
                    if run_face:
                        self.face.observe(frame)
                    timer.mark("face_detect_ms", ran=run_face)

                    raw_face = bool(self.face.face_detected())

                    # --- FaceID crop + enroll + authorize ---
                    face_crop = self.face.crop_face(frame) if raw_face else None

                    if self.face_id.enrolling and face_crop is not None:
                        self.face_id.add_sample(face_crop)

                    # is_authorized() returns early without embedding when nothing
                    # is enrolled, so `ran` must reflect real inference or the
                    # reported latency is diluted by no-ops.
                    ran_faceid = bool(
                        face_crop is not None
                        and self.face_id.enabled
                        and (self.face_id.enrolled or self.face_id.enrolling)
                    )
                    authorized_face = bool(face_crop is not None and self.face_id.is_authorized(face_crop))
                    timer.mark("face_id_ms", ran=ran_faceid)

                    # --- Identity gate (ablatable) ---
                    gate_face = raw_face if cfg.no_identity_gate else authorized_face
                    self._face_streak = min(self._face_streak + 1, 10) if gate_face else 0
                    face_detected = self._face_streak >= cfg.face_streak_on

                    if cfg.no_identity_gate:
                        hand_detected = hand_detected_raw
                    else:
                        hand_detected = bool(hand_detected_raw and face_detected)

                    # --- Hand-face association (opt-in) ---
                    # Only a hand at the end of the verified person's target arm
                    # counts. Applied before mode selection, so a bystander's
                    # raised hand cannot pull the aircraft into gesture mode.
                    if self.assoc is not None:
                        t_assoc = time.perf_counter()
                        auth_bbox = self.face.get_last_bbox() if face_detected else None
                        assoc_det = self.assoc.step(frame, det, auth_bbox, fresh=run_hand)
                        self._assoc_det = assoc_det
                        if self.assoc.last.mode in ("pose", "reacquire"):
                            timer.record("assoc_pose_ms", (time.perf_counter() - t_assoc) * 1000.0)
                        hand_detected = bool(hand_detected and assoc_det is not None)
                        if assoc_det is not None:
                            det = assoc_det
                        timer.mark("assoc_ms", ran=self.assoc.last.mode in ("pose", "iou", "reacquire"))

                    # Update timers
                    if face_detected:
                        self._last_face_ts = now
                        self._last_any_seen_ts = now
                    if hand_detected:
                        self._last_hand_ts = now
                        self._last_any_seen_ts = now

                    # --- Telemetry snapshot ---
                    st = self.state.snapshot()
                    bat = st.get("bat", None)
                    alt = st.get("h", None)

                    time_since_hand = now - self._last_hand_ts
                    time_since_face = now - self._last_face_ts
                    time_since_any = now - self._last_any_seen_ts

                    # Reset temporal gesture state when the gating hand is gone
                    if (now - self._last_hand_ts) > float(self.mode_mgr.cfg.hand_release_s):
                        try:
                            self.rule.reset()
                        except Exception:
                            pass

                    state_for_mode = {
                        "hand_detected": bool(hand_detected),
                        "face_detected": bool(face_detected),
                        "time_since_hand_s": float(time_since_hand),
                        "time_since_face_s": float(time_since_face),
                        "time_since_any_seen_s": float(time_since_any),
                        "battery": bat if bat is None else float(bat),
                        "altitude_cm": alt if alt is None else float(alt),
                        "flying": bool(self.flying),
                    }

                    mode, det_reason = self.mode_mgr.tick(state_for_mode)
                    timer.mark("mode_manager_ms")

                    gesture_conf = None

                    # --- Execute deterministic mode ---
                    if mode == "gesture":
                        if hand_detected and det is not None and det.landmarks is not None:
                            t_clf = time.perf_counter()
                            prev_scale = self.rule._last_scale
                            if self.selection.kind != "rule":
                                # Keep the rule's scale history advancing so the
                                # depth cue is observable in the SVM arm too,
                                # without its output being used.
                                self.rule.predict(det.landmarks)
                                prev_scale = getattr(self, "_prev_scale", None)
                                self._prev_scale = self.rule._last_scale
                            gr = self.selection.predict(det.landmarks)
                            timer.record(self.selection.latency_column,
                                         (time.perf_counter() - t_clf) * 1000.0)

                            gesture_conf = float(gr.confidence)

                            self._raw_gesture = gr.name
                            self._last_lm = det.landmarks
                            # Sampled from the deployed rule object either side
                            # of its own predict(), so this is the value the
                            # classifier acted on rather than a recomputation.
                            cur_scale = self.rule._last_scale
                            self._bbox_area = cur_scale
                            self._delta_s = (
                                None if (prev_scale is None or cur_scale is None)
                                else (cur_scale - prev_scale) / max(prev_scale, 1e-6))
                            gesture_name = self._fb_gate.apply(gr.name)
                            self._gesture_stable_name = self._fb_gate._name
                            self._gesture_stable_count = self._fb_gate.stable_count

                            rc = rc_from_gesture_name(gesture_name, cfg.rc_speed) if self.flying else RC(active=False)
                        else:
                            gesture_name = "NOHAND"
                            self._fb_gate.reset()
                            self._gesture_stable_name = "CENTER"
                            self._gesture_stable_count = 0
                            rc = RC(active=True) if self.flying else RC(active=False)

                    elif mode == "face":
                        if face_detected:
                            cmd, frame = self.face.update(frame)
                            rc = cmd if self.flying else RC(active=False)
                            gesture_name = "FACE"
                        else:
                            rc = RC(active=True) if self.flying else RC(active=False)
                            gesture_name = "HOVER"

                    elif mode == "search_360":
                        rc = RC(lr=0, fb=0, ud=0, yaw=self._search_yaw_cmd, active=True) if self.flying else RC(active=False)
                        gesture_name = "SEARCH"

                    elif mode == "hover":
                        rc = RC(active=True) if self.flying else RC(active=False)
                        gesture_name = "HOVER"

                    elif mode == "land":
                        if self.flying:
                            self.tello.send_cmd("land", timeout_ms=8000)
                            self.perf.log_event("battery_failsafe_land")
                            # Step the threshold down so the operator can take off
                            # again and trigger a fresh failsafe. Without this the
                            # battery stays below the line and every takeoff lands
                            # instantly, giving one trial per charge.
                            step = int(cfg.failsafe_step_pct)
                            if step > 0:
                                nxt = self.mode_mgr.cfg.battery_land_pct - step
                                if nxt >= int(cfg.failsafe_floor_pct):
                                    self.mode_mgr.cfg.battery_land_pct = nxt
                                    cfg.battery_land_pct = nxt
                                    print(f"[failsafe] fired; next threshold -> {nxt}%  (press t to fly again)")
                                else:
                                    print(f"[failsafe] fired; floor {cfg.failsafe_floor_pct}% reached, not stepping further")
                        self.flying = False
                        rc = RC(active=False)
                        gesture_name = "LAND"

                    else:
                        rc = RC(active=True) if self.flying else RC(active=False)
                        gesture_name = "OTHER"

                    timer.mark("command_exec_ms")
                    command_str = self._rc_to_command_str(rc)
                    camera_to_command_ms = timer.elapsed_ms()

                    # --- LLM reason (reason-only) ---
                    llm_payload = {
                        "mode": mode,
                        "command": command_str,
                        "deterministic_reason": det_reason,
                        "hand_detected": bool(hand_detected),
                        "face_detected": bool(face_detected),
                        "battery": bat,
                        "altitude_cm": alt,
                        "time_since_any_seen_s": float(time_since_any),
                        "flying": bool(self.flying),
                    }
                    self.reasoner.tick(llm_payload)
                    llm_reason = self.reasoner.get_reason()

                    # --- Log decisions ---
                    should_log = False
                    if (now - self._last_log_ts) >= self._log_every_s:
                        should_log = True
                    if mode != self._prev_mode or command_str != self._prev_cmd:
                        should_log = True
                    if llm_reason and llm_reason != self._prev_llm_reason:
                        should_log = True

                    if should_log:
                        n, N = self.face_id.enroll_progress()
                        llm_latency_ms = self.reasoner.get_latency_ms()
                        self.decisions.add({
                            "mode": mode,
                            "command": command_str,
                            "llm_reason": llm_reason,
                            "llm_latency_ms": llm_latency_ms,
                            "det_reason": det_reason,
                            "battery": bat,
                            "altitude_cm": alt,
                            "gesture": gesture_name,
                            "gesture_conf": gesture_conf,
                            "face_raw": bool(raw_face),
                            "face_auth": bool(authorized_face),
                            "hand_raw": bool(hand_detected_raw),
                            "hand_auth": bool(hand_detected),
                            "faceid_enrolled": bool(self.face_id.enrolled),
                            "faceid_enrolling": bool(self.face_id.enrolling),
                            "faceid_progress": f"{n}/{N}",
                            "faceid_score": float(self.face_id.last_score),
                            "face_bbox_w": (self.face.get_last_bbox() or (0, 0, None, None))[2],
                            "face_bbox_h": (self.face.get_last_bbox() or (0, 0, None, None))[3],
                            "crop_ok": bool(face_crop is not None),
                            "t_any": float(time_since_any),
                            "flying": bool(self.flying),
                            "seq": seq,
                            **self._assoc_fields(),
                        })
                        if llm_reason and llm_reason != self._prev_llm_reason:
                            self.perf.log_event("llm_explanation", latency_ms=llm_latency_ms)
                        self._prev_mode = mode
                        self._prev_cmd = command_str
                        self._prev_llm_reason = llm_reason
                        self._last_log_ts = now

                    # Cue track. Written after the command is resolved so the
                    # emitted component is the one the aircraft actually received.
                    if self._freeflight and self._cue_writer is not None:
                        self._cue_writer.writerow({
                            "t": now, "seq": seq,
                            "frame_age_ms": round(frame_age_ms, 2),
                            "round": None, "hold": None,
                            "cue_label": None, "cue_name": None,
                            "phase": "freeflight",
                            "vocabulary": self.selection.kind,
                            "hand_detected": int(hand_detected_raw),
                            "hand_auth": int(hand_detected),
                            "face_auth": int(authorized_face),
                            "mode": mode,
                            "raw_gesture": getattr(self, "_raw_gesture", None),
                            "emitted_gesture": gesture_name,
                            "command": command_str,
                            "fb_stable_count": self._fb_gate.stable_count,
                            "gesture_conf": gesture_conf,
                            "battery": bat, "altitude_cm": alt,
                            "actuation_suppressed": int(self._no_actuate),
                            "bbox_area": getattr(self, "_bbox_area", None),
                            "delta_s": getattr(self, "_delta_s", None),
                            **(landmark_row(getattr(self, "_last_lm", None))
                               if self._log_landmarks else {}),
                        })
                        self._cue_rows += 1
                        if self._cue_rows % 40 == 0:
                            self._cue_fh.flush()

                    if self.cue is not None and self.cue.active:
                        phase, _rem = self.cue.tick(now)
                        cur = self.cue.current
                        # Speak the class when a new hold opens, and mark the
                        # settle/hold boundary with a tone: that instant decides
                        # which frames are labelled.
                        if cur is not None:
                            if self.cue.idx != self._cue_spoken:
                                self._voice.cue(self.cue.spoken())
                                self._cue_spoken = self.cue.idx
                                self._cue_phase = "settle"
                            if phase == "hold" and self._cue_phase != "hold":
                                self._voice.go()
                                self._cue_phase = "hold"
                        if cur is not None and phase in ("settle", "hold"):
                            rnd, lab = cur
                            self._cue_writer.writerow({
                                "t": now, "seq": seq,
                                "frame_age_ms": round(frame_age_ms, 2),
                                "round": rnd, "hold": self.cue.idx,
                                "cue_label": lab,
                                "cue_name": self.cue.names[lab],
                                "phase": phase,
                                "vocabulary": self.cue.vocabulary,
                                "hand_detected": int(hand_detected_raw),
                                "hand_auth": int(hand_detected),
                                "face_auth": int(authorized_face),
                                "mode": mode,
                                "raw_gesture": getattr(self, "_raw_gesture", None),
                                "emitted_gesture": gesture_name,
                                "command": command_str,
                                "fb_stable_count": self._fb_gate.stable_count,
                                "gesture_conf": gesture_conf,
                                "battery": bat, "altitude_cm": alt,
                                "actuation_suppressed": int(self._no_actuate),
                                "bbox_area": getattr(self, "_bbox_area", None),
                                "delta_s": getattr(self, "_delta_s", None),
                                **(landmark_row(getattr(self, "_last_lm", None))
                                   if self._log_landmarks else {}),
                            })
                            self._cue_rows += 1
                            if self._cue_rows % 40 == 0:
                                self._cue_fh.flush()

                    logged = self.perf.log_frame(
                        timer,
                        camera_to_command_ms,
                        extra={"mode": mode, "flying": int(self.flying), **self._assoc_fields()},
                        seq=seq,
                        frame_age_ms=frame_age_ms,
                    )
                    if logged:
                        dt = self.perf.rows[-1].get("dt_s")
                        if dt:
                            inst = 1.0 / float(dt)
                            self._fps_ema = inst if self._fps_ema is None else 0.9 * self._fps_ema + 0.1 * inst

                    # --- Landmark overlay (display only, after latency is logged) ---
                    if self._debug_overlay:
                        draw_debug(frame, self._last_hand_det, self._assoc_det,
                                   None if self.assoc is None else self.assoc.associator,
                                   self.face.get_last_bbox(), face_detected)

                    # --- Overlay (clean HUD) ---
                    draw_hud(
                        frame,
                        mode=mode,
                        flying=self.flying,
                        hand_detected_raw=hand_detected_raw,
                        hand_detected=hand_detected,
                        raw_face=raw_face,
                        face_detected=face_detected,
                        gesture_name=gesture_name,
                        face_id=self.face_id,
                        run_name=self.run_ctx.name,
                        ablations=cfg.active_ablations(),
                        trial=self.scenarios.open_trial,
                        tally=self.scenarios.tally(),
                        fps=self._fps_ema,
                        battery=bat,
                        failsafe_thr=self.mode_mgr.cfg.battery_land_pct,
                        llm_reason=llm_reason,
                        assoc=self._assoc_hud(),
                        show_llm=self._show_llm,
                        show_help=self._show_help,
                        recording=self._recorder is not None,
                        cue=(None if self.cue is None or not self.cue.active else
                             {"phase": self.cue.tick(now)[0],
                              "name": self.cue.names[self.cue.current[1]]
                              if self.cue.current else "",
                              "text": self.cue.cue_text(),
                              "progress": self.cue.progress()}),
                    )
                    self._display = frame
                    if self._recorder is not None:
                        self._recorder.submit(frame, seq=seq, raw=raw_for_rec)

                if self._display is not None:
                    cv2.imshow("TELLO", self._display)
                else:
                    # Waiting screen (avoid using `frame` here)
                    h, w = 240, 320
                    blank = 255 * (cv2.UMat(h, w, cv2.CV_8UC3).get())
                    cv2.putText(
                        blank,
                        "Waiting for video...",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("TELLO", blank)

                # --- Keyboard ---
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                # FaceID keys
                self._handle_faceid_keys(key)
                self._handle_debug_keys(key)
                self._handle_cue_keys(key)
                self._handle_sync_key(key)

                # Scenario-trial keys
                if key in self._scenario_keys:
                    bat_now = self.state.snapshot().get("bat", None)
                    self.scenarios.start(self._scenario_keys[key], mode_at_start=mode, battery=bat_now)
                elif key == ord("y"):
                    self.scenarios.resolve(True, mode_at_end=mode)
                elif key == ord("n"):
                    self.scenarios.resolve(False, mode_at_end=mode)
                elif key == ord("x"):
                    self.scenarios.discard()

                if key == ord("t"):
                    if not cfg.allow_takeoff:
                        print("[safety] takeoff blocked by --no-fly")
                    else:
                        ok2, resp2 = self.tello.send_cmd("takeoff", timeout_ms=8000)
                        print("takeoff:", ok2, resp2)
                        self.flying = ok2 and (resp2 or "").lower() == "ok"

                if key == ord("l"):
                    ok2, resp2 = self.tello.send_cmd("land", timeout_ms=8000)
                    print("land:", ok2, resp2)
                    self.flying = False

                if key == ord("e"):
                    ok2, resp2 = self.tello.send_cmd("emergency", timeout_ms=3000)
                    print("emergency:", ok2, resp2)
                    self.flying = False
                    self.perf.log_event("emergency_stop")

                # Manual RC override
                if self.flying and (not self._warned_deadband):
                    try:
                        if int(cfg.rc_deadband) >= int(cfg.rc_speed):
                            print(
                                f"[WARN] rc_deadband ({cfg.rc_deadband}) >= rc_speed ({cfg.rc_speed}). "
                                "Gesture/face RC may be clamped to 0. Lower deadband or raise speed."
                            )
                        self._warned_deadband = True
                    except Exception:
                        self._warned_deadband = True

                if self.flying:
                    krc = rc_from_key(key, cfg.rc_speed)
                    if krc.active:
                        rc = krc

                # --- Send RC at fixed rate ---
                now2 = time.time()
                if now2 - last_rc_send >= cfg.rc_dt:
                    if self.flying:
                        if self._no_actuate:
                            # Hold station. The computed rc is already logged;
                            # what goes on the wire is zero.
                            if (rc.lr or rc.fb or rc.ud or rc.yaw):
                                self._suppressed_nonzero += 1
                            self._send_rc_locked(cfg)
                            self.perf.log_event("rc_send_suppressed")
                        else:
                            self.tello.send_rc(
                                rc.lr, rc.fb, rc.ud, rc.yaw,
                                limit=cfg.rc_limit,
                                deadband=cfg.rc_deadband,
                            )
                            self.perf.log_event("rc_send")
                    last_rc_send = now2

                self.logger.add(self.state.snapshot())

        finally:
            try:
                self.reasoner.stop()
            except Exception:
                pass
            try:
                if self.flying:
                    self.tello.send_cmd("land", timeout_ms=8000)
            except Exception:
                pass
            try:
                self.tello.send_cmd("streamoff", timeout_ms=2000)
            except Exception:
                pass

            self.video.stop()
            self.state.stop()
            self.tello.close()
            cv2.destroyAllWindows()

            for name, logger in (
                ("telemetry", self.logger),
                ("decisions", self.decisions),
                ("perf", self.perf),
                ("scenarios", self.scenarios),
            ):
                try:
                    logger.export()
                except Exception as e:
                    print(f"{name} export failed:", e)

            if self._recorder is not None:
                try:
                    info = self._recorder.close()
                    self.run_ctx.record("recording", info)
                    for name, meta in info["files"].items():
                        print(f"[rec] {name}: {meta['path']}  {meta['seconds']} s")
                except Exception as e:
                    print("recording close failed:", e)

            try:
                self.run_ctx.record("summary", {
                    "frames_logged": len(self.perf.rows),
                    "duplicate_frames_skipped": self.perf.duplicate_frames,
                    "telemetry_samples": len(self.logger.rows),
                    "decisions": len(self.decisions.rows),
                    "scenario_trials": len(self.scenarios.rows),
                    "scenario_tally": self.scenarios.tally(),
                    "video_reopens": getattr(self.video, "reopen_count", None),
                    "video_read_failures": getattr(self.video, "read_failures", None),
                })
                self.run_ctx.write()
            except Exception as e:
                print("manifest write failed:", e)

            print(f"\n[run] {self.run_ctx.name} complete")
            print(f"[run] {len(self.perf.rows)} frames "
                  f"({self.perf.duplicate_frames} duplicate observations skipped), "
                  f"{len(self.logger.rows)} telemetry samples, "
                  f"{len(self.scenarios.rows)} scenario trials")
            tally = self.scenarios.tally()
            for k, v in sorted(tally.items()):
                print(f"        {k:38s} {v}")
            if not self._blind:
                print(f"[run] classifier was {self.selection.kind}")
            if getattr(self, "_sync_markers", None):
                self.run_ctx.record("sync_markers", self._sync_markers)
            if self._cue_writer is not None:
                if self.cue is not None:
                    self._voice.stop()
                self._cue_fh.flush()
                self._cue_fh.close()
                if self.cue is not None:
                    self.run_ctx.record("cued_protocol", self.cue.manifest())
                    print(f"[cue] {self._cue_rows} cued frames, {self.cue.progress()}")
                else:
                    self.run_ctx.record("freeflight_log",
                                        {"rows": self._cue_rows,
                                         "landmarks_logged": self._log_landmarks})
                    print(f"[log] {self._cue_rows} dense frames for video scoring")
            if self._no_actuate:
                self.run_ctx.record("actuation_summary", {
                    "suppressed": True,
                    "nonzero_commands_withheld": self._suppressed_nonzero})
                print(f"[run] hover lock: {self._suppressed_nonzero} non-zero "
                      "commands computed and withheld")
            print(f"[run] saved to {self.run_ctx.dir}")

        return 0

    # Backwards-compatible alias for existing callers.
    def _assoc_fields(self) -> dict:
        """Per-frame association outcome for the logs; empty when it is off, so
        the paper's configuration writes exactly the columns it always did."""
        if self.assoc is None:
            return {}
        a = self.assoc.last
        return {
            "assoc_mode": a.mode,
            "assoc_ok": int(a.ok),
            "assoc_hands": a.n_hands,
            "assoc_skeleton_hits": a.skeleton_hits,
            "assoc_d": None if a.d_target is None else round(float(a.d_target), 3),
            "assoc_reason": a.reason,
        }

    def _assoc_hud(self):
        if self.assoc is None:
            return None
        a = self.assoc.last
        d = " d=----" if a.d_target is None else f" d={a.d_target:4.2f}"
        live = " LIVE" if self.assoc.every_n == 0 else "     "
        return (f"ASSOC {'OK' if a.ok else '--'} {(a.mode or '-'):<9}{d} hands={a.n_hands}"
                f"{live}  {'' if a.ok else a.reason}")

    def run(self) -> int:  # type: ignore[override]
        return self.run_loop()
