"""Command-to-motion latency, measured with small yaw nudges while hovering.

flash_latency.py times the camera path (scene -> frame on the host). What it
cannot see is the other direction: how long after the host sends an RC command
the aircraft actually moves. This script closes that loop. While the Tello
hovers, it sends a short yaw pulse, alternating left and right, and times the
first sign of the turn two ways, both on the host clock:

    video_ms      command sent -> first frame whose image has started to slide.
                  The whole loop: command out, the aircraft reacts, the video of
                  the reaction comes back. Minus the camera path (flash test)
                  it leaves command-to-motion.
    telemetry_ms  command sent -> first state packet whose heading has changed.
                  Command-to-motion plus the state link, with no video in it;
                  coarse, since heading comes in whole degrees at ~10 Hz.

Image motion is the global horizontal shift between consecutive frames (phase
correlation), so the camera should face a textured scene, not a blank wall.
`video_extrap_ms` extrapolates the start of the slide back to zero from the
first frames that show it, since a threshold only fires once the motion has
built up.

Flying: the operator takes off with `t` and starts the test with `g`; nothing
flies on its own. Pulses are yaw only (default 40 of 100 for 0.4 s, about 15
degrees), alternating direction so the heading stays put. `e` is the emergency
stop, `l` lands, `q` lands and quits. It lands by itself when the test ends.

    python -m tello_gesture_py.scripts.yaw_latency
    python -m tello_gesture_py.scripts.yaw_latency --selftest 150 600
        # no drone: a simulated aircraft that turns 150 ms after a command, seen
        # through a 600 ms video path; checks the method

Shares UDP ports 8889/8890/9013 with the controller: run it alone.
Writes outputs/runs/<stamp>_latency_yaw/ (yaw_latency.csv, frames.csv, manifest).
"""

import argparse
import csv
import os
import random
import socket
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tello_gesture_py.src.config import ControllerConfig  # noqa: E402
from tello_gesture_py.src.latest_frame import LatestFrame  # noqa: E402
from tello_gesture_py.src.run_context import RunContext  # noqa: E402

SMALL = (160, 120)
MIN_BATTERY = 30


def pct(x, q):
    return round(float(np.percentile(x, q)), 1) if len(x) else None


class Link:
    """RC out at 20 Hz and state in, both time-stamped on the host clock."""

    def __init__(self, cfg, sim=None):
        self.sim = sim
        self.rc = (0, 0, 0, 0)
        self.rc_log = []                 # (t, yaw) at every change
        self.state = []                  # (t_recv, yaw_deg, bat)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.tello = None
        if sim is None:
            from tello_gesture_py.src.tello_udp import TelloUDP
            self.tello = TelloUDP(cfg.tello_ip, cfg.cmd_port, cfg.local_cmd_port)
            self.state_port = cfg.state_port

    def set_yaw(self, yaw):
        with self.lock:
            self.rc = (0, 0, 0, int(yaw))
        t = self._send()                 # send at once; do not wait for the tick
        self.rc_log.append((t, int(yaw)))
        return t

    def _send(self):
        with self.lock:
            lr, fb, ud, yw = self.rc
        if self.sim is not None:
            self.sim.command(yw)
        else:
            self.tello.send_cmd_noack(f"rc {lr} {fb} {ud} {yw}")
        return time.time()

    def rc_loop(self):
        while not self.stop.wait(0.05):
            self._send()

    def state_loop(self):
        if self.sim is not None:
            while not self.stop.wait(0.1):
                y = self.sim.heading_seen(time.time())
                self.state.append((time.time(), float(round(y)), 90.0))
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("", self.state_port))
        sock.settimeout(1.0)
        while not self.stop.is_set():
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            t = time.time()
            kv = dict(p.split(":", 1) for p in data.decode(errors="ignore").split(";") if ":" in p)
            try:
                self.state.append((t, float(kv.get("yaw", "nan")), float(kv.get("bat", "nan"))))
            except ValueError:
                pass
        sock.close()

    def cmd(self, msg, timeout_ms=8000):
        if self.sim is not None:
            return True, "ok"
        return self.tello.send_cmd(msg, timeout_ms=timeout_ms)


class SimAircraft:
    """Self-test: turns cmd_s after a yaw command; a camera sees it vid_s late."""

    RATE = 1.0                           # deg/s per rc unit

    def __init__(self, cmd_s, vid_s, latest, stop):
        self.cmd_s, self.vid_s = cmd_s, vid_s
        self.latest, self.stop = latest, stop
        self.cmds = [(0.0, 0)]
        self.lock = threading.Lock()
        rng = np.random.default_rng(3)
        tex = cv2.GaussianBlur(rng.integers(0, 255, (240, 6000)).astype(np.uint8), (0, 0), 3)
        self.tex = cv2.cvtColor(cv2.normalize(tex, None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)

    def command(self, yaw):
        with self.lock:
            if self.cmds[-1][1] != yaw:
                self.cmds.append((time.time(), yaw))

    def heading(self, t):
        """Integrated heading at time t, the rate stepping cmd_s after each command."""
        with self.lock:
            cmds = list(self.cmds)
        h = 0.0
        for (t0, y), nxt in zip(cmds, cmds[1:] + [(float("inf"), None)]):
            a, b = t0 + self.cmd_s, min(nxt[0] + self.cmd_s, t)
            if b > a:
                h += y * self.RATE * (b - a)
        return h

    def heading_seen(self, t):
        return self.heading(t - 0.03)    # state link a little late

    def camera(self):
        while not self.stop.is_set():
            h = self.heading(time.time() - self.vid_s)
            x = int(2500 + h * 12) % 5000
            self.latest.put(np.ascontiguousarray(self.tex[:, x:x + 320]))
            time.sleep(1 / 30)


class Flow:
    """Global horizontal image shift between consecutive new frames."""

    def __init__(self, latest):
        self.latest = latest
        self.rows = []                   # (seq, t_arrival, t_pick, dx)
        self.stop = threading.Event()
        self.last = None

    def run(self):
        last_seq, prev = -1, None
        win = cv2.createHanningWindow(SMALL, cv2.CV_32F)
        while not self.stop.is_set():
            ok, frame, seq, ts = self.latest.get(copy=True)
            if not ok or seq == last_seq:
                time.sleep(0.001)
                continue
            last_seq = seq
            t_pick = time.time()
            g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), SMALL,
                           interpolation=cv2.INTER_AREA).astype(np.float32)
            if prev is not None:
                (dx, _dy), _ = cv2.phaseCorrelate(prev, g, win)
                self.rows.append((seq, ts, t_pick, float(dx)))
            prev = g
            self.last = frame


def analyse(pulses, flow_rows, state, rc_log):
    """Onsets per pulse, from the recorded series (after the flight)."""
    fr = np.array([(r[1], r[3]) for r in flow_rows]) if flow_rows else np.zeros((0, 2))
    st = np.array([(s[0], s[1]) for s in state if s[1] == s[1]]) if state else np.zeros((0, 2))
    out = []
    for p in pulses:
        t0, sgn = p["t_cmd"], p["dir"]
        row = dict(p, detected=0)
        base = fr[(fr[:, 0] > t0 - 1.2) & (fr[:, 0] <= t0)] if len(fr) else fr
        after = fr[(fr[:, 0] > t0) & (fr[:, 0] < t0 + 2.5)] if len(fr) else fr
        if len(base) >= 5 and len(after) >= 3:
            noise = float(np.std(base[:, 1]))
            thr = max(4 * noise, 0.6)
            mag = np.abs(after[:, 1])
            hit = [i for i in range(len(after) - 1)
                   if mag[i] > thr and mag[i + 1] > thr
                   and np.sign(after[i, 1]) == np.sign(after[i + 1, 1])]
            if hit:
                i = hit[0]
                row.update(detected=1, noise_px=round(noise, 3), thr_px=round(thr, 2),
                           image_sign=int(np.sign(after[i, 1])),
                           video_ms=round((after[i, 0] - t0) * 1000, 1))
                # back-extrapolate the rising edge to zero
                j = i
                while j + 1 < len(after) and mag[j + 1] > mag[j] and j - i < 3:
                    j += 1
                seg = after[max(i - 1, 0):j + 1]
                if len(seg) >= 2 and np.ptp(np.abs(seg[:, 1])) > 0:
                    a, b = np.polyfit(seg[:, 0], np.abs(seg[:, 1]), 1)
                    if a > 0:
                        t_zero = max(-b / a, t0)
                        row["video_extrap_ms"] = round((t_zero - t0) * 1000, 1)
        if len(st):
            before = st[st[:, 0] <= t0]
            later = st[(st[:, 0] > t0) & (st[:, 0] < t0 + 2.5)]
            if len(before) and len(later):
                y0 = before[-1, 1]
                moved = later[np.abs(((later[:, 1] - y0) + 180) % 360 - 180) >= 1]
                if len(moved):
                    row["telemetry_ms"] = round((moved[0, 0] - t0) * 1000, 1)
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=16, help="Yaw pulses (alternating direction).")
    ap.add_argument("--amp", type=int, default=40, help="Yaw RC value of a pulse, 1-100.")
    ap.add_argument("--pulse", type=float, default=0.4, help="Pulse length, seconds.")
    ap.add_argument("--gap", type=float, nargs=2, default=(2.5, 3.5), metavar=("MIN", "MAX"),
                    help="Random quiet time between pulses, seconds.")
    ap.add_argument("--selftest", type=float, nargs=2, default=None, metavar=("CMD_MS", "VIDEO_MS"),
                    help="No drone: simulated aircraft and camera delays.")
    ap.add_argument("--video-delay-ms", type=float, default=None,
                    help="Camera-to-host delay from flash_latency, to split the loop.")
    ap.add_argument("--run-id", default="latency_yaw")
    args = ap.parse_args()
    if not 1 <= args.amp <= 60:
        sys.exit("--amp must be 1-60")

    cfg = ControllerConfig()
    selftest = args.selftest is not None
    run = RunContext(args.run_id + ("-selftest" if selftest else ""))
    latest = LatestFrame()
    stop = threading.Event()
    sim = None
    if selftest:
        sim = SimAircraft(args.selftest[0] / 1000, args.selftest[1] / 1000, latest, stop)
    link = Link(cfg, sim)
    flow = Flow(latest)
    video = None
    flying = False
    pulses = []
    rnd = random.Random(int(time.time()))
    status = "t: take off   g: start test   l: land   e: EMERGENCY   q: quit"

    try:
        threading.Thread(target=link.state_loop, daemon=True).start()
        if selftest:
            threading.Thread(target=sim.camera, daemon=True).start()
        else:
            from tello_gesture_py.src.video_stream import VideoStream
            ok, resp = False, ""
            for _ in range(5):
                ok, resp = link.cmd("command", 4000)
                if ok and resp.lower() == "ok":
                    break
                time.sleep(0.5)
            if not ok or resp.lower() != "ok":
                sys.exit(f"No SDK handshake ({ok} {resp}).")
            link.cmd("streamoff", 2000)
            link.cmd("streamon", 6000)
            video = VideoStream(latest, f"udp://0.0.0.0:{cfg.video_port}"
                                f"?fifo_size=5000000&overrun_nonfatal=1&timeout=2000000&buffer_size=1000000")
            if not video.start():
                sys.exit("Could not open the video stream.")
        threading.Thread(target=flow.run, daemon=True).start()
        threading.Thread(target=link.rc_loop, daemon=True).start()

        phase, t_next, k = "idle", 0.0, 0
        if selftest:
            flying, phase, t_next = True, "baseline", time.time() + 2.0
        while True:
            frame = flow.last
            if frame is not None:
                v = frame.copy()
                bat = link.state[-1][2] if link.state else float("nan")
                cv2.putText(v, f"{phase}  pulse {k}/{args.n}  battery {bat:.0f}%", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(v, status, (10, v.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1)
                cv2.imshow("YAW LATENCY", v)
            key = cv2.waitKey(5) & 0xFF
            now = time.time()
            if key == ord("e"):
                link.cmd("emergency", 3000)
                flying = False
                print("[EMERGENCY] motors stopped")
                break
            if key in (ord("q"), ord("l")):
                break
            if key == ord("t") and not flying and not selftest:
                bat = link.state[-1][2] if link.state else 0
                if bat < MIN_BATTERY:
                    print(f"[refused] battery {bat:.0f}% < {MIN_BATTERY}%")
                else:
                    ok, resp = link.cmd("takeoff", 10000)
                    flying = ok and resp.lower() == "ok"
                    print(f"[takeoff] {ok} {resp}")
            if key == ord("g") and flying and phase == "idle":
                phase, t_next = "baseline", now + 2.0
                print("[test] started; hold still, keep the area clear")

            if phase == "baseline" and now >= t_next:
                phase, t_next = "pulse", now
            if phase == "pulse" and now >= t_next:
                d = 1 if k % 2 == 0 else -1
                t_cmd = link.set_yaw(d * args.amp)
                pulses.append({"k": k, "dir": d, "amp": args.amp, "t_cmd": t_cmd})
                phase, t_next = "stop", t_cmd + args.pulse
            if phase == "stop" and now >= t_next:
                link.set_yaw(0)
                k += 1
                if k >= args.n:
                    phase, t_next = "settle", now + 2.5
                else:
                    phase, t_next = "pulse", now + rnd.uniform(*args.gap)
            if phase == "settle" and now >= t_next:
                print("[test] done")
                break
    finally:
        try:
            link.set_yaw(0)
        except Exception:
            pass
        if flying and not selftest:
            print("[land]", link.cmd("land", 10000))
        stop.set()
        link.stop.set()
        flow.stop.set()
        time.sleep(0.2)
        if video is not None:
            video.stop()
        if link.tello is not None:
            link.cmd("streamoff", 2000)
            link.tello.close()
        cv2.destroyAllWindows()

    rows = analyse(pulses, flow.rows, link.state, link.rc_log)
    cols = ["k", "dir", "amp", "t_cmd", "detected", "video_ms", "video_extrap_ms",
            "telemetry_ms", "image_sign", "noise_px", "thr_px"]
    with open(run.path("yaw_latency.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(run.path("frames.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seq", "t_arrival", "t_pick", "dx_px"])
        w.writerows(flow.rows)

    def col(c):
        return [r[c] for r in rows if r.get(c) is not None]
    vid, ext, tel = col("video_ms"), col("video_extrap_ms"), col("telemetry_ms")
    signs = [(r["dir"], r["image_sign"]) for r in rows if r.get("detected")]
    consistent = len({d * s for d, s in signs}) == 1 if signs else None
    summary = {
        "source": f"selftest cmd {args.selftest[0]} ms, video {args.selftest[1]} ms" if selftest else "tello",
        "pulses": len(rows), "detected": len(vid),
        "video_ms": {"median": pct(vid, 50), "p95": pct(vid, 95), "min": pct(vid, 0), "max": pct(vid, 100)},
        "video_extrap_ms": {"median": pct(ext, 50), "p95": pct(ext, 95)},
        "telemetry_ms": {"median": pct(tel, 50), "p95": pct(tel, 95), "n": len(tel)},
        "image_direction_consistent": consistent,
        "amp": args.amp, "pulse_s": args.pulse,
        "note": "video_ms is the whole loop: command out, the aircraft turns, the video returns",
    }
    if args.video_delay_ms is not None and vid:
        summary["command_to_motion_ms_est"] = round(pct(vid, 50) - args.video_delay_ms, 1)
    run.record("yaw_latency", summary)
    run.write()

    print(f"\n{len(vid)}/{len(rows)} pulses seen in the video; direction consistent: {consistent}")
    print(f"loop (command -> turn seen in video): median {summary['video_ms']['median']} ms, "
          f"p95 {summary['video_ms']['p95']} ms; extrapolated start {summary['video_extrap_ms']['median']} ms")
    print(f"command -> heading change in telemetry: median {summary['telemetry_ms']['median']} ms "
          f"({len(tel)} pulses)")
    if "command_to_motion_ms_est" in summary:
        print(f"command -> motion (loop - {args.video_delay_ms:.0f} ms camera path): "
              f"{summary['command_to_motion_ms_est']} ms")
    print(f"wrote {run.dir}")


if __name__ == "__main__":
    main()
