import threading
import time
from typing import Optional, Tuple
import numpy as np

class LatestFrame:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._ts = 0.0

    def set(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
            self._seq += 1
            self._ts = time.time()

    def get(self, copy: bool = True) -> Tuple[bool, Optional[np.ndarray], int, float]:
        with self._lock:
            if self._frame is None:
                return False, None, self._seq, self._ts
            f = self._frame.copy() if copy else self._frame
            return True, f, self._seq, self._ts

    def age_ms(self) -> float:
        with self._lock:
            if self._frame is None:
                return 1e9
            return (time.time() - self._ts) * 1000.0
        
    def put(self, frame):
        """
        Compatibility alias for VideoStream.
        Calls the existing method used in this project to store the latest frame.
        """
        if hasattr(self, "set"):
            return self.set(frame)
        if hasattr(self, "update"):
            return self.update(frame)
        if hasattr(self, "write"):
            return self.write(frame)
        # Fallback: store directly if this class uses attributes
        self._frame = frame
        self._ts = time.time()
        self._seq = getattr(self, "_seq", 0) + 1

