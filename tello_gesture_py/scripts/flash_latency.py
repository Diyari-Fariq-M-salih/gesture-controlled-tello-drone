"""Camera-to-host latency of the Tello video path, measured with screen flashes.

The per-frame logs time the host only: `frame_age_ms` from the decoder handing a
frame over to the control loop taking it, `camera_to_command_ms` from there to
the command. What happens before the frame reaches the host (exposure, H.264
encode on the aircraft, Wi-Fi, FFmpeg buffering and decode) is not in them, and
cannot be recovered from them: the stream carries no capture time, and the
aircraft's clock is not the host's. This script puts a shared clock into the
picture instead.

The aircraft sits on the ground, propellers off, its camera facing this screen.
The screen switches between black and white at random intervals; the switch
time is taken on the host clock, and so is the arrival of the first frame that
shows it. The difference is the camera-to-host delay:

    delay_arrival_ms   switch -> decoded frame in the single-slot buffer
    delay_pickup_ms    switch -> the loop takes the frame (adds frame age)

Adding the logged `camera_to_command_ms` to delay_pickup_ms gives camera to
command. The figure includes the screen's own response (about one refresh),
so it is an upper bound on the camera path.

The deployed perception load (hands every 2nd frame, face detection and a face
embedding every 4th) runs on every frame by default, so decoding competes for
the CPU as it does in flight. `--no-load` turns it off, for comparison.

    python -m tello_gesture_py.scripts.flash_latency                    # the drone
    python -m tello_gesture_py.scripts.flash_latency --selftest 120     # no drone:
        # a synthetic camera that sees the screen 120 ms late; checks the method

Shares UDP port 9013 with the controller and tello_check: run it alone.
Writes a new outputs/runs/<stamp>_flash_latency/ (flash_latency.csv, manifest).
"""

import argparse
import csv
import os
import random
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tello_gesture_py.src.config import ControllerConfig  # noqa: E402
from tello_gesture_py.src.latest_frame import LatestFrame  # noqa: E402
from tello_gesture_py.src.run_context import PROJECT_ROOT, RunContext  # noqa: E402

GRID = (8, 6)            # brightness is read per cell, so the screen need not fill the view
MIN_CONTRAST = 25.0      # grey levels between black and white for a cell to count as screen
TIMEOUT_S = 1.5          # a switch not seen by then is a miss


def pct(x, q):
    return round(float(np.percentile(x, q)), 1) if len(x) else None


class Probe:
    """Consumes frames exactly as the control loop does and watches for the switch."""

    def __init__(self, latest: LatestFrame, load: bool, cfg: ControllerConfig):
        self.latest = latest
        self.load = load
        self.cfg = cfg
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.recent = []            # (t_pick, cells) for calibration
        self.dark = self.bright = self.mask = None
        self.pending = None         # dict set by the display side
        self.results = []
        self.frames = 0
        self.load_ms = []
        self.last_v = None
        if load:
            from tello_gesture_py.src.face_follow import FaceFollower
            from tello_gesture_py.src.face_id import FaceID, FaceIDConfig
            from tello_gesture_py.src.hand_gesture import HandGesture
            self.hand = HandGesture(max_num_hands=1)
            self.face = FaceFollower()
            self.face.cfg.detect_every_n = 1
            self.face_id = FaceID(FaceIDConfig(
                model_path=str(PROJECT_ROOT / "models" / "third_party" / "arcface.onnx"), rgb=True))
            if not self.face_id.enabled:
                print("[load] face embedding model missing; the load omits it")

    def value(self, cells):
        """0 at the calibrated black, 1 at white, averaged over the screen's cells."""
        span = (self.bright - self.dark)[self.mask]
        return float(np.mean((cells[self.mask] - self.dark[self.mask]) / span))

    def run(self):
        last_seq = -1
        i = 0
        while not self.stop.is_set():
            ok, frame, seq, ts = self.latest.get(copy=True)
            if not ok or seq == last_seq:
                time.sleep(0.001)
                continue
            last_seq = seq
            t_pick = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cells = cv2.resize(gray, GRID, interpolation=cv2.INTER_AREA).astype(np.float32)
            self.frames += 1
            with self.lock:
                self.recent.append((t_pick, cells))
                del self.recent[:-60]
                p = self.pending
                if p is not None and self.mask is not None:
                    v = self.value(cells)
                    if ts > p["t_switch"]:
                        crossed = v > 0.5 if p["target"] == 1 else v < 0.5
                        if crossed:
                            p.update(seq=seq, t_arrival=ts, t_pick=t_pick, v_at=round(v, 3),
                                     v_before=self.last_v, detected=True)
                            self.results.append(p)
                            self.pending = None
                    self.last_v = round(v, 3)
                elif self.mask is not None:
                    self.last_v = round(self.value(cells), 3)
            if self.load:
                t0 = time.perf_counter()
                i += 1
                if i % self.cfg.hand_every_n == 0:
                    self.hand.detect(frame)
                if i % self.cfg.face_every_n == 0:
                    self.face.observe(frame)
                    crop = self.face.crop_face(frame) if self.face.face_detected() else None
                    if crop is None:
                        h, w = frame.shape[:2]
                        crop = frame[h // 2 - 56:h // 2 + 56, w // 2 - 56:w // 2 + 56]
                    self.face_id.embed(crop)
                self.load_ms.append((time.perf_counter() - t0) * 1000.0)

    def settle_cells(self, since):
        with self.lock:
            c = [cells for t, cells in self.recent if t >= since]
        return np.median(np.stack(c), axis=0) if c else None


class Screen:
    """Full-screen black/white window, or a virtual one for the self-test."""

    def __init__(self, window: bool):
        self.window = window
        self.level = 0
        self.history = [(0.0, 0)]    # (host time, level), for the self-test camera
        self.imgs = {0: np.zeros((720, 1280, 3), np.uint8),
                     1: np.full((720, 1280, 3), 255, np.uint8)}
        if window:
            cv2.namedWindow("FLASH", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("FLASH", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("FLASH", self.imgs[0])
            cv2.waitKey(50)

    def show(self, level):
        """Switch and return the host time at which the paint was handed to the OS."""
        if self.window:
            cv2.imshow("FLASH", self.imgs[level])
            cv2.waitKey(1)
        self.level = level
        t = time.time()
        self.history.append((t, level))
        return t

    def wait(self, seconds):
        """Keep the window alive; False if the operator pressed q or Esc."""
        end = time.time() + seconds
        while time.time() < end:
            if self.window:
                if cv2.waitKey(5) & 0xFF in (ord("q"), 27):
                    return False
            else:
                time.sleep(0.005)
        return True


def synthetic_camera(latest, screen, delay_s, stop):
    """A camera that sees the screen exactly delay_s late, at a jittery ~30 fps.

    Only the frame timing quantises, so the measured delay should be delay_s
    plus on average half a frame interval (about 17 ms).
    """
    rng = np.random.default_rng(1)
    frames = {}                                 # made in advance: no time between "capture" and put
    for lv in (0, 1):
        frames[lv] = []
        for _ in range(4):
            f = np.full((720, 960, 3), 30 + 180 * lv, np.uint8)
            f[:, :200] = 90                     # part of the view is not the screen
            frames[lv].append(cv2.add(f, rng.integers(0, 8, f.shape, dtype=np.uint8)))
    i = 0
    while not stop.is_set():
        now = time.time()
        lv = [lv for t, lv in list(screen.history) if t <= now - delay_s][-1]
        i += 1
        latest.put(frames[lv][i % 4])
        time.sleep(max(0.0, rng.normal(1 / 30, 0.004)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=60, help="Screen switches (half each way).")
    ap.add_argument("--hold", type=float, nargs=2, default=(1.0, 2.0), metavar=("MIN", "MAX"),
                    help="Random time between switches, seconds.")
    ap.add_argument("--no-load", action="store_true", help="Do not run the perception load.")
    ap.add_argument("--selftest", type=float, default=None, metavar="MS",
                    help="No drone: a synthetic camera that sees the screen MS late.")
    ap.add_argument("--window", action="store_true",
                    help="With --selftest, still show the full-screen flashes.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--run-id", default="latency_flash")
    args = ap.parse_args()

    cfg = ControllerConfig()
    selftest = args.selftest is not None
    run = RunContext(args.run_id + ("-selftest" if selftest else ""))
    seed = args.seed if args.seed is not None else int(time.time())
    rnd = random.Random(seed)

    latest = LatestFrame()
    probe = Probe(latest, load=not args.no_load, cfg=cfg)
    screen = Screen(window=(not selftest) or args.window)
    stop = threading.Event()
    tello = video = None
    battery = {"start": None, "end": None}

    try:
        if selftest:
            threading.Thread(target=synthetic_camera, daemon=True,
                             args=(latest, screen, args.selftest / 1000.0, stop)).start()
        else:
            from tello_gesture_py.src.tello_udp import TelloUDP
            from tello_gesture_py.src.video_stream import VideoStream
            tello = TelloUDP(cfg.tello_ip, cfg.cmd_port, cfg.local_cmd_port)
            ok, resp = False, ""
            for _ in range(5):
                ok, resp = tello.send_cmd("command", timeout_ms=4000)
                if ok and resp.lower() == "ok":
                    break
                time.sleep(0.5)
            if not ok or resp.lower() != "ok":
                sys.exit(f"No SDK handshake ({ok} {resp}). On the TELLO Wi-Fi? tello_check not running?")
            ok, b = tello.send_cmd("battery?", timeout_ms=2000)
            battery["start"] = b if ok else None
            print(f"[tello] connected, battery {battery['start']}%")
            tello.send_cmd("streamoff", timeout_ms=2000)
            tello.send_cmd("streamon", timeout_ms=6000)
            video = VideoStream(latest, f"udp://0.0.0.0:{cfg.video_port}"
                                f"?fifo_size=5000000&overrun_nonfatal=1&timeout=2000000&buffer_size=1000000")
            if not video.start():
                sys.exit("Could not open the video stream.")

            def keepalive():
                # The Tello leaves SDK mode after ~15 s without a command.
                while not stop.wait(5.0):
                    ok, b = tello.send_cmd("battery?", timeout_ms=1500)
                    if ok and b.strip().isdigit():
                        battery["end"] = b.strip()
            threading.Thread(target=keepalive, daemon=True).start()

        threading.Thread(target=probe.run, daemon=True).start()

        t_wait = time.time() + 20
        while probe.frames < 30:
            if time.time() > t_wait:
                sys.exit("No video after 20 s.")
            if not screen.wait(0.1):
                return

        # Calibrate: what black and white look like to the camera, per cell.
        print("[cal] black ...")
        screen.show(0)
        screen.wait(2.0)
        dark = probe.settle_cells(time.time() - 1.0)
        print("[cal] white ...")
        screen.show(1)
        screen.wait(2.0)
        bright = probe.settle_cells(time.time() - 1.0)
        mask = (bright - dark) > MIN_CONTRAST
        print(f"[cal] {int(mask.sum())}/{mask.size} cells see the screen, "
              f"contrast median {np.median((bright - dark)[mask]) if mask.any() else 0:.0f} grey levels")
        if mask.sum() < 3:
            sys.exit("The camera cannot see the screen: move the drone closer, face it at the "
                     "screen, dim the room.")
        with probe.lock:
            probe.dark, probe.bright, probe.mask = dark, bright, mask
        screen.show(0)
        screen.wait(1.5)

        print(f"[run] {args.n} switches, load {'on' if probe.load else 'off'}; q to stop")
        for k in range(args.n):
            target = 1 - screen.level
            with probe.lock:
                v_before = probe.last_v
            t_switch = screen.show(target)
            p = {"k": k, "edge": "rise" if target == 1 else "fall", "target": target,
                 "t_switch": t_switch, "v_before_switch": v_before, "detected": False}
            with probe.lock:
                probe.pending = p
            hold = rnd.uniform(*args.hold)
            if not screen.wait(max(hold, TIMEOUT_S)):
                break
            with probe.lock:
                if probe.pending is p:          # never seen
                    probe.results.append(p)
                    probe.pending = None
            r = probe.results[-1]
            if r["detected"]:
                print(f"  {k + 1:3d}/{args.n} {r['edge']:4s} "
                      f"{(r['t_arrival'] - t_switch) * 1000:6.1f} ms")
            else:
                print(f"  {k + 1:3d}/{args.n} {r['edge']:4s}  missed")
    finally:
        stop.set()
        probe.stop.set()
        if video is not None:
            video.stop()
        if tello is not None:
            tello.send_cmd("streamoff", timeout_ms=2000)
            tello.close()
        if screen.window:
            cv2.destroyAllWindows()

    # ---- write and summarise ----
    rows = []
    for r in probe.results:
        row = {"k": r["k"], "edge": r["edge"], "detected": int(r["detected"]),
               "t_switch": round(r["t_switch"], 6), "v_before_switch": r.get("v_before_switch")}
        if r["detected"]:
            row.update(seq=r["seq"], t_arrival=round(r["t_arrival"], 6), t_pick=round(r["t_pick"], 6),
                       delay_arrival_ms=round((r["t_arrival"] - r["t_switch"]) * 1000, 2),
                       delay_pickup_ms=round((r["t_pick"] - r["t_switch"]) * 1000, 2),
                       frame_age_ms=round((r["t_pick"] - r["t_arrival"]) * 1000, 2),
                       v_at=r["v_at"])
        rows.append(row)
    cols = ["k", "edge", "detected", "t_switch", "seq", "t_arrival", "t_pick",
            "delay_arrival_ms", "delay_pickup_ms", "frame_age_ms", "v_before_switch", "v_at"]
    with open(run.path("flash_latency.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    ok_rows = [r for r in rows if r["detected"]]
    arr = [r["delay_arrival_ms"] for r in ok_rows]
    pick = [r["delay_pickup_ms"] for r in ok_rows]
    summary = {
        "source": f"selftest {args.selftest} ms" if selftest else "tello",
        "perception_load": probe.load,
        "switches": len(rows), "detected": len(ok_rows),
        "delay_arrival_ms": {"median": pct(arr, 50), "p95": pct(arr, 95),
                             "min": pct(arr, 0), "max": pct(arr, 100)},
        "delay_pickup_ms": {"median": pct(pick, 50), "p95": pct(pick, 95)},
        "by_edge_arrival_median_ms": {e: pct([r["delay_arrival_ms"] for r in ok_rows
                                              if r["edge"] == e], 50) for e in ("rise", "fall")},
        # Most frames skip the throttled stages, so a median is ~0; the mean is the load.
        "load_ms_per_frame_mean": round(float(np.mean(probe.load_ms)), 1) if probe.load_ms else None,
        "frames": probe.frames,
        "video_reopens": getattr(video, "reopen_count", None),
        "battery_pct": battery,
        "seed": seed,
        "note": "includes the screen's own response time: an upper bound on the camera path",
    }
    run.record("flash_latency", summary)
    run.write()

    print(f"\n{len(ok_rows)}/{len(rows)} switches seen")
    print(f"camera -> host buffer : median {summary['delay_arrival_ms']['median']} ms, "
          f"p95 {summary['delay_arrival_ms']['p95']} ms")
    print(f"camera -> loop pickup : median {summary['delay_pickup_ms']['median']} ms, "
          f"p95 {summary['delay_pickup_ms']['p95']} ms")
    print(f"rise / fall medians   : {summary['by_edge_arrival_median_ms']}")
    print(f"wrote {run.dir}")


if __name__ == "__main__":
    main()
