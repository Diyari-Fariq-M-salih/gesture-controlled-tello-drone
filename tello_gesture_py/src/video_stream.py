import threading
import time
import cv2

from .latest_frame import LatestFrame


class VideoStream:
    """
    Robust UDP video reader for Tello.
    - Reads frames in a background thread
    - Writes latest frame into LatestFrame
    - If OpenCV/FFmpeg decoder crashes (common on UDP loss), we reopen the capture and continue.
    """

    def __init__(self, latest: LatestFrame, url: str):
        self._latest = latest
        self._url = url
        self._cap = None
        self._running = False
        self._th = None
        self.reopen_count = 0
        self.read_failures = 0
        self._last_reopen = 0.0
        self.on_reopen = None   # optional callback, used to log a perf event

    def start(self) -> bool:
        if self._running:
            return True

        # Force FFmpeg backend on Windows for UDP H.264
        self._cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            return False

        self._running = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return True

    def stop(self):
        self._running = False
        if self._th is not None:
            self._th.join(timeout=1.0)
        self._th = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None

    def _reopen(self):
        self.reopen_count += 1
        self._last_reopen = time.time()
        if self.on_reopen is not None:
            try:
                self.on_reopen()
            except Exception:
                pass
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass

        time.sleep(0.3)
        self._cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)

    def _loop(self):
        backoff = 0.02
        consecutive = 0
        while self._running:
            try:
                if self._cap is None or not self._cap.isOpened():
                    self._reopen()
                    time.sleep(0.1)
                    continue

                ok, frame = self._cap.read()
                if not ok or frame is None:
                    # Packet loss or decoder stall. A handful of misses is normal
                    # on UDP; a sustained run means the stream is gone and only a
                    # reopen recovers it.
                    self.read_failures += 1
                    consecutive += 1
                    # Reopening is not free: the decoder must wait for the next
                    # SPS/PPS keyframe before it can produce a frame again, so a
                    # trigger-happy reopen turns a brief RF dropout into a longer
                    # blackout and can thrash. Require a sustained stall, and a
                    # cooldown so repeated reopens cannot pile up.
                    if consecutive >= 90 and (time.time() - self._last_reopen) > 5.0:
                        print(f"Video stalled ({consecutive} failed reads). Reopening.")
                        self._reopen()
                        consecutive = 0
                    time.sleep(backoff)
                    continue
                consecutive = 0

                # publish
                self._latest.put(frame)

            except cv2.error as e:
                print("Video decode error (OpenCV). Reopening stream:", e)
                self._reopen()
                time.sleep(0.1)

            except Exception as e:
                print("Video thread error. Reopening stream:", e)
                self._reopen()
                time.sleep(0.1)
