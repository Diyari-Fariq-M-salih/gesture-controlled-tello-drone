import time

def now_s() -> float:
    return time.time()

def clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v

def apply_deadband(v: int, deadband: int) -> int:
    return 0 if -deadband < v < deadband else v

class Rate:
    def __init__(self, dt: float):
        self.dt = dt
        self._next = time.time()

    def sleep(self):
        t = time.time()
        if t < self._next:
            time.sleep(self._next - t)
        self._next = max(self._next + self.dt, time.time())
