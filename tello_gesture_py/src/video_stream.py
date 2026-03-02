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
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass

        time.sleep(0.3)
        self._cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)

    def _loop(self):
        backoff = 0.02
        while self._running:
            try:
                if self._cap is None or not self._cap.isOpened():
                    self._reopen()
                    time.sleep(0.1)
                    continue

                ok, frame = self._cap.read()
                if not ok or frame is None:
                    # Packet loss / decoder stall. Don't die, just retry.
                    time.sleep(backoff)
                    continue

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
