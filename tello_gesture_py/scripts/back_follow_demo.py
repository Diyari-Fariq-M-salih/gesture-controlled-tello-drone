"""Try follow-behind on a webcam: back enrolment, matching, and the commands it would send.

No drone, and it works alone: the screen is recorded to session.mp4 in the run
directory (as --record does in flight), and beeps tell you what is happening
while your back is turned:
    1 short beep   back enrolment has started: stand still, back to the camera
    2 beeps        back enrolled: walk about
    1 long beep    enrolment failed (try again with p)
Press p, turn round within the countdown, and stand 2-3 m away so your whole
back is in view while it enrols; then walk about, away and back, left and right,
and turn to face the camera now and then. Afterwards, watch session.mp4: the box
shows the match score and the estimated distance, and the bottom line the RC
command the aircraft would have been sent. Facing the camera stops following,
as it does in flight.

    python -m tello_gesture_py.scripts.back_follow_demo
    python -m tello_gesture_py.scripts.back_follow_demo --camera 1 --delay 6

Keys: p enrol the back (after the countdown), o clear, [ ] distance, - = height,
q quit. --no-record turns the recording off. Scores are logged to outputs/runs/<stamp>_webcam_back-follow/ so the
match thresholds can be calibrated from real sessions.

A laptop webcam has a different focal length from the Tello, so the distance
reading is only indicative here; --focal-px sets it.
"""
import argparse
import csv
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tello_gesture_py.src.back_follow import BackFollowConfig, BackFollower  # noqa: E402
from tello_gesture_py.src.face_follow import FaceFollower  # noqa: E402
from tello_gesture_py.src.run_context import RunContext  # noqa: E402
from tello_gesture_py.src.session_recorder import SessionRecorder  # noqa: E402


def beep(n=1, ms=180, hz=1500):
    try:
        import winsound
        for i in range(n):
            winsound.Beep(hz, ms)
    except Exception:
        print("", end="", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--delay", type=float, default=6.0, help="Seconds to turn round.")
    ap.add_argument("--focal-px", type=float, default=None,
                    help="Focal length in px at 960 wide (Tello default 680).")
    ap.add_argument("--run-id", default="webcam_back-follow")
    ap.add_argument("--no-record", action="store_true", help="Do not save session.mp4.")
    a = ap.parse_args()

    cfg = BackFollowConfig(enroll_delay_s=a.delay, pose_every_n=1)
    if a.focal_px:
        cfg.focal_px_960 = a.focal_px
    back = BackFollower(cfg)
    face = FaceFollower()
    face.cfg.detect_every_n = 1
    face.cfg.lost_timeout_s = 0.3
    run = RunContext(a.run_id)
    run.record("back_follow", {k: v for k, v in vars(cfg).items()})
    rec = None if a.no_record else SessionRecorder(run.dir, fps=20.0)
    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        sys.exit(f"camera {a.camera} did not open")
    f = open(run.path("back_follow.csv"), "w", newline="", encoding="utf-8")
    log = None
    prev_phase, prev_enrolled = back.phase, back.enrolled
    print("p: enrol back   o: clear   [ ]: distance   - =: height   q: quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.time()
            face.observe(frame)
            face_visible = bool(face.face_detected())
            t0 = time.perf_counter()
            obs = back.step(frame, now, face_visible)
            ms = (time.perf_counter() - t0) * 1000.0
            rc = back.command(obs, frame.shape[1], altitude_cm=None)
            row = {"t": round(now, 3), "face_visible": int(face_visible), "step_ms": round(ms, 1),
                   **back.fields(), "rc_fb": rc.fb, "rc_yaw": rc.yaw}
            if log is None:
                log = csv.DictWriter(f, fieldnames=list(row))
                log.writeheader()
            log.writerow(row)

            # audible state changes, for an operator with their back turned
            if back.phase == "capturing" and prev_phase != "capturing":
                beep(1)
            if back.enrolled and not prev_enrolled:
                beep(2)
            if prev_phase == "capturing" and back.phase == "idle" and not back.enrolled:
                beep(1, ms=700, hz=600)
            prev_phase, prev_enrolled = back.phase, back.enrolled

            back.draw(frame, now)
            state = obs.state + ("  (face seen)" if face_visible else "")
            txt = (f"{state}   best {obs.best_score if obs.best_score is not None else 0:.2f}"
                   f"   people {obs.n_people}   would send fb {rc.fb:+d} yaw {rc.yaw:+d}"
                   f"   {ms:.0f} ms")
            cv2.putText(frame, txt, (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, txt, (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("BACK FOLLOW (webcam)", frame)
            if rec is not None:
                rec.submit(frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("p"):
                back.start_enroll(now)
                print(f"[back] turn around: enrolment in {cfg.enroll_delay_s:.0f} s")
            elif k == ord("o"):
                back.clear()
                print("[back] cleared")
            else:
                msg = back.adjust(k)
                if msg:
                    print(f"[back] {msg}")
    finally:
        if rec is not None:
            info = rec.close()
            run.record("recording", info)
            for name, meta in info["files"].items():
                print(f"[rec] {name}: {run.dir / meta['path']}  {meta['seconds']} s")
        f.close()
        cap.release()
        cv2.destroyAllWindows()
        run.write()
        print(f"wrote {run.dir}")


if __name__ == "__main__":
    main()
