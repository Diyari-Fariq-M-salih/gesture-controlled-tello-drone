import socket
import select
from typing import Tuple

class TelloUDP:
    """UDP command helper for Tello SDK (8889)."""

    def __init__(self, ip="192.168.10.1", cmd_port=8889, local_port=9012):
        self.ip = ip
        self.cmd_port = cmd_port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("", local_port))
        self._sock.setblocking(False)
        self._closed = False

    def close(self):
        if not self._closed:
            self._sock.close()
            self._closed = True

    def send_cmd(self, msg: str, timeout_ms: int = 6000) -> Tuple[bool, str]:
        if self._closed:
            return False, "socket closed"
        self._sock.sendto(msg.encode("utf-8"), (self.ip, self.cmd_port))
        r, _, _ = select.select([self._sock], [], [], timeout_ms / 1000.0)
        if not r:
            return False, "timeout"
        try:
            resp, _ = self._sock.recvfrom(2048)
            return True, resp.decode("utf-8", errors="ignore").strip()
        except Exception as e:
            return False, str(e)

    def send_cmd_noack(self, msg: str) -> None:
        if self._closed:
            return
        self._sock.sendto(msg.encode("utf-8"), (self.ip, self.cmd_port))

    @staticmethod
    def _clamp_rc(v: int, limit: int, deadband: int) -> int:
        v = max(-limit, min(limit, v))
        if -deadband < v < deadband:
            v = 0
        return int(v)

    def send_rc(self, lr: int, fb: int, ud: int, yaw: int, limit=100, deadband=6):
        lr = self._clamp_rc(lr, limit, deadband)
        fb = self._clamp_rc(fb, limit, deadband)
        ud = self._clamp_rc(ud, limit, deadband)
        yaw = self._clamp_rc(yaw, limit, deadband)
        self.send_cmd_noack(f"rc {lr} {fb} {ud} {yaw}")
