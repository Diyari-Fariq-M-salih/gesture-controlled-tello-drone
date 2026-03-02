import socket
import threading
from typing import Dict, Optional

class StateListener:
    """Receives telemetry on UDP 8890 and parses key:value; into dict."""

    def __init__(self, port: int = 8890):
        self.port = port
        self._lock = threading.Lock()
        self._state: Dict[str, float] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        if self._running:
            return True
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._state)

    def get(self, key: str, default=None):
        with self._lock:
            return self._state.get(key, default)

    def _loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("", self.port))
        sock.settimeout(1.0)

        while self._running:
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break

            s = data.decode("utf-8", errors="ignore")
            parsed = {}
            for kv in s.split(";"):
                if ":" not in kv:
                    continue
                k, v = kv.split(":", 1)
                try:
                    parsed[k] = float(v)
                except ValueError:
                    pass

            if parsed:
                with self._lock:
                    self._state.update(parsed)

        sock.close()
