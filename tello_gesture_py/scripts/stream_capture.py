"""Face verification through the aircraft's own video pipeline, on the ground.

Figure 2's authorization envelope rests on bins holding two and three frames and
carries a concrete design recommendation, and the on-ground row of the identity
table rests on 122 frames while reporting a worse rate than flight -- an
inversion the manuscript does not explain. Both are capturable on the ground at
far higher n, which is what this harness produces.

The operator stands at marked distances and declares each one. Face detection,
cropping, embedding and the Schmitt-trigger authorization decision run exactly
as deployed -- same template, same tau_on/tau_off/k, same ONNX model -- and
apparent face size is logged beside every decision. A live per-bin counter shows
which bounding-box bins are still short of target, against the same bin edges
the analysis uses, so a sweep can be steered while it happens rather than
discovered to be short afterwards.

Nothing here commands the aircraft: the SDK socket carries `command`, `streamon`
and `streamoff` only. Run it with the propellers removed.

Gesture capture lives in scripts/operational_gesture_eval.py, which cues classes
on a timed protocol; this file no longer carries a gesture mode, so there is one
gesture harness rather than two that could drift apart.

Usage:
    python -m tello_gesture_py.scripts.stream_capture --run-id capture_envelope

Pressing a distance key opens a segment but does not start recording: the
operator needs to walk to the mark first, and frames captured on the way would
be binned at a distance they were not taken from. Recording begins --walkup-s
seconds later (default 3), and the overlay counts it down.

Keys: p enrol (hold still)   number keys select a mark from --distances
      SPACE stop recording   q quit
"""

import argparse
import csv
import time
from pathlib import Path

import cv2

from tello_gesture_py.src.config import ControllerConfig
from tello_gesture_py.src.face_follow import FaceFollower
from tello_gesture_py.src.face_id import FaceID, FaceIDConfig
from tello_gesture_py.src.latest_frame import LatestFrame
from tello_gesture_py.src.run_context import RunContext
from tello_gesture_py.src.state_listener import StateListener
from tello_gesture_py.src.tello_udp import TelloUDP
from tello_gesture_py.src.video_stream import VideoStream

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The bin edges Figure 2 is drawn on. Filling these is the point of the
# identity capture, so the harness counts against the same edges the analysis
# will use rather than against a convenient round number.
BBOX_BINS = [0, 150, 180, 220, 280, 350, 480, 620]
BIN_TARGET = 200

# Marks chosen so each lands in a different bounding-box bin, given that
# apparent face height goes as about 432/d px on this camera. 0.5 m sits above
# the top bin on purpose: that is the near-field regime where the padded crop
# clips the frame edge, which is the candidate explanation for the on-ground
# false rejection rate exceeding the in-flight one.
DEFAULT_DISTANCES = [0.5, 0.8, 1.0, 1.5, 1.75, 2.0, 2.5, 3.0]


def _bin_of(h):
    if h is None:
        return None
    for lo, hi in zip(BBOX_BINS[:-1], BBOX_BINS[1:]):
        if lo <= h < hi:
            return f"{lo}-{hi}"
    return None


def _hud(frame, lines, colour=(0, 255, 0)):
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, colour, 1, cv2.LINE_AA)


class Writer:
    """Append-as-you-go CSV, so a crash or a yanked battery keeps the frames."""

    def __init__(self, path, fields):
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=fields)
        self.w.writeheader()
        self.n = 0

    def add(self, row):
        self.w.writerow(row)
        self.n += 1
        if self.n % 50 == 0:
            self.f.flush()

    def close(self):
        self.f.flush()
        self.f.close()


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


def run_identity(args, cfg, run, latest, state):
    """Face verification through the drone stream at declared standoff distances."""
    face = FaceFollower()
    fid = FaceID(FaceIDConfig(
        cosine_thr=cfg.faceid_cosine_thr,
        enroll_samples=cfg.faceid_enroll_samples,
        hysteresis=cfg.faceid_hysteresis,
        release_thr=cfg.faceid_release_thr,
        release_frames=cfg.faceid_release_frames,
    ))
    if not fid.enabled:
        raise SystemExit("FaceID model unavailable; nothing to measure.")

    w = Writer(run.path("stream_identity.csv"),
               ["t", "seq", "frame_age_ms", "distance_m", "face_raw", "crop_ok",
                "face_bbox_w", "face_bbox_h", "faceid_score", "face_auth",
                "embed_ms", "battery", "segment"])

    dist, segment, last_seq = None, 0, -1
    record_from = 0.0          # wall clock at which the current segment starts
    fills = {f"{lo}-{hi}": 0 for lo, hi in zip(BBOX_BINS[:-1], BBOX_BINS[1:])}
    marks = args.distances
    print("\np to enrol, then a number key to declare which mark you are on:")
    for i, m in enumerate(marks, 1):
        print(f"   {i} = {m:.2f} m")
    print(f"Each segment waits {args.walkup_s:.0f} s before recording, so you can "
          "reach the mark.")
    print(f"Target {BIN_TARGET} scored frames per bounding-box bin: "
          + ", ".join(fills))

    while True:
        ok, frame, seq, ts = latest.get(copy=True)
        now = time.time()
        if ok and frame is not None and seq != last_seq:
            last_seq = seq
            age_ms = (now - ts) * 1000.0

            if seq % max(cfg.face_every_n, 1) == 0:
                face.observe(frame)
            raw = bool(face.face_detected())
            crop = face.crop_face(frame) if raw else None

            if fid.enrolling and crop is not None:
                fid.add_sample(crop)

            score = auth = None
            embed_ms = None
            if crop is not None and fid.enrolled:
                t0 = time.perf_counter()
                auth = bool(fid.is_authorized(crop))
                embed_ms = (time.perf_counter() - t0) * 1000.0
                score = fid.last_score

            bbox = face.get_last_bbox() or (0, 0, None, None)
            walking = dist is not None and now < record_from
            if dist is not None and fid.enrolled and not walking:
                st = state.snapshot()
                w.add({"t": now, "seq": seq, "frame_age_ms": round(age_ms, 2),
                       "distance_m": dist, "face_raw": int(raw),
                       "crop_ok": int(crop is not None),
                       "face_bbox_w": bbox[2], "face_bbox_h": bbox[3],
                       "faceid_score": None if score is None else round(score, 4),
                       "face_auth": None if auth is None else int(auth),
                       "embed_ms": None if embed_ms is None else round(embed_ms, 3),
                       "battery": st.get("bat"), "segment": segment})
                # Only a scored frame counts toward a bin: a frame with no
                # detection carries no authorization decision, so counting it
                # would report a bin as full of data the analysis discards.
                if score is not None:
                    bk = _bin_of(bbox[3])
                    if bk:
                        fills[bk] += 1

            if bbox[2]:
                x, y, bw, bh = bbox
                cv2.rectangle(frame, (int(x), int(y)), (int(x + bw), int(y + bh)),
                              (0, 255, 0) if auth else (0, 165, 255), 2)
            n, N = fid.enroll_progress()
            cur = _bin_of(bbox[3])
            short = [f"{k}:{v}" for k, v in fills.items() if v < BIN_TARGET]
            if dist is None:
                status = "recording=N"
            elif walking:
                status = "WALK TO %.1f m ... %.1f s" % (dist, record_from - now)
            else:
                status = "RECORDING %.1f m" % dist
            _hud(frame, [
                f"MODE=identity   enrolled={'Y' if fid.enrolled else 'N'}"
                + (f"  enrolling {n}/{N}" if fid.enrolling else ""),
                f"{status}   frames={w.n}",
                f"bbox_h={bbox[3]}   bin={cur or '--'}"
                f" {fills.get(cur, 0) if cur else 0}/{BIN_TARGET}"
                f"   score={'--' if score is None else '%.3f' % score}"
                f"   auth={'--' if auth is None else int(auth)}",
                "short: " + (", ".join(short) if short else "none -- all bins full"),
                "p enrol   1-%d mark   SPACE stop   q quit" % len(marks),
            ], (0, 200, 255) if (dist is None or walking) else (0, 255, 0))
            cv2.imshow("STREAM CAPTURE", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            fid.start_enroll()
            print("[FaceID] enrolling; hold still and face the drone.")
        if key == ord(" "):
            dist = None
        if ord("1") <= key <= ord("9"):
            i = key - ord("1")
            if i >= len(marks):
                print(f"[rec] no mark {i + 1}; {len(marks)} marks configured")
                continue
            dist = marks[i]
            segment += 1
            record_from = time.time() + float(args.walkup_s)
            print(f"[rec] segment {segment}: {dist:.1f} m "
                  f"(recording starts in {args.walkup_s:.0f} s)")

    w.close()
    run.record("capture", {"mode": "identity", "frames": w.n,
                           "threshold": fid.cfg.cosine_thr,
                           "hysteresis": fid.cfg.hysteresis,
                           "walkup_s": args.walkup_s,
                           "distances_m": marks,
                           "bin_fills": fills, "bin_target": BIN_TARGET,
                           "bins_short": {k: v for k, v in fills.items() if v < BIN_TARGET}})
    print(f"\n{w.n} labelled frames -> {run.path('stream_identity.csv')}")
    print(f"{'bbox bin':<12}{'scored':>8}{'target':>8}")
    for k, v in fills.items():
        print(f"{k:<12}{v:>8}{BIN_TARGET:>8}" + ("" if v >= BIN_TARGET else "   SHORT"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--distances", default=",".join(str(x) for x in DEFAULT_DISTANCES),
                    type=lambda s: [float(x) for x in s.split(",") if x.strip()],
                    help="Comma-separated distances in metres, in the order the "
                         "number keys select them. The default set places one "
                         "mark inside each bounding-box bin for a room of about "
                         "3 m; pass your own if your space differs.")
    ap.add_argument("--walkup-s", type=float, default=3.0,
                    help="Seconds between pressing a distance key and recording, "
                         "so the operator can reach the mark. Frames during the "
                         "walk are discarded, not binned.")
    args = ap.parse_args()

    cfg = ControllerConfig()
    cfg.allow_takeoff = False
    run = RunContext(run_id=args.run_id or "capture_envelope", note=args.note)
    run.record("config", cfg)
    run.record("harness", "stream_capture")
    run.record("mode", "identity")

    print("=" * 66)
    print("  Ground capture through the drone stream. Propellers OFF.")
    print("  This harness never sends takeoff or rc -- it only reads video.")
    print("=" * 66)

    tello = connect(cfg)
    state = StateListener(cfg.state_port)
    state.start()
    latest = LatestFrame()
    video = VideoStream(latest, f"udp://0.0.0.0:{cfg.video_port}")
    if not video.start():
        raise SystemExit("Video stream not opened. Check firewall UDP 11111.")

    cv2.namedWindow("STREAM CAPTURE", cv2.WINDOW_AUTOSIZE)
    t0 = time.time()
    while latest.get()[1] is None:
        if time.time() - t0 > 12:
            raise SystemExit("No frames decoded after 12 s.")
        time.sleep(0.1)

    try:
        run_identity(args, cfg, run, latest, state)
    finally:
        video.stop()
        try:
            state.stop()
        except Exception:
            pass
        tello.send_cmd("streamoff", timeout_ms=2000)
        tello.close()
        cv2.destroyAllWindows()
        run.record("stream", {"reopen_count": video.reopen_count,
                              "read_failures": video.read_failures})
        run.write()
        print(f"[run] {run.dir}")


if __name__ == "__main__":
    main()
