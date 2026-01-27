import threading
from typing import Optional
import cv2
from .latest_frame import LatestFrame

class VideoStream:
    """OpenCV/FFmpeg video reader pushing latest frames into LatestFrame."""

    def __init__(self, latest: LatestFrame, url: str):
        self.latest = latest
        self.url = url
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap = None

    def start(self) -> bool:
        if self._running:
            return True
        self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            return False
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()
            self._cap = None

    def _loop(self):
        while self._running and self._cap:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                continue
            self.latest.set(frame)
