"""Speak the cue instead of displaying it.

The cued protocol asks the operator to read a class name off the laptop while
flying a drone in front of them. That is not a protocol an operator can follow:
attention has to leave the aircraft to read the screen, which is both unsafe and
a source of exactly the timing error the cue exists to eliminate.

Speech moves the cue to a channel that does not compete with watching the
aircraft. The operator hears the class, performs it, and keeps their eyes on the
drone throughout.

Windows SAPI is driven through PowerShell so nothing has to be installed. A
`winsound` beep marks the settle/hold boundary, which is the moment that decides
whether a frame is labelled, and is worth signalling even if speech fails.
"""

from __future__ import annotations

import platform
import queue
import shutil
import subprocess
import threading
from typing import Optional

_IS_WINDOWS = platform.system() == "Windows"


class Voice:
    """Non-blocking speech and tones for the cue track.

    Every call returns immediately. Audio runs on a worker thread, because a
    blocking speak() inside the control loop would stall frame handling and
    corrupt the latency measurements this study reports.
    """

    def __init__(self, enabled: bool = True, rate: int = 2):
        self.enabled = bool(enabled) and _IS_WINDOWS
        self.rate = int(rate)
        self.available = False
        self._q: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._th: Optional[threading.Thread] = None

        if self.enabled:
            self.available = shutil.which("powershell") is not None
            if self.available:
                self._th = threading.Thread(target=self._loop, daemon=True)
                self._th.start()

    # ------------------------------------------------------------------ api
    def say(self, text: str) -> None:
        if self.available:
            self._q.put(("say", text))

    def beep(self, freq: int = 880, ms: int = 120) -> None:
        if self.enabled:
            self._q.put(("beep", (freq, ms)))

    def cue(self, name: str) -> None:
        """Announce the class at the start of its settle window."""
        self.say(name.replace("-", " "))

    def go(self) -> None:
        """Mark the settle/hold boundary: recording starts now."""
        self.beep(1200, 90)

    def done(self) -> None:
        """Mark the end of a hold."""
        self.beep(520, 70)

    def stop(self) -> None:
        if self._th is not None:
            self._q.put(None)
            self._th.join(timeout=1.0)
            self._th = None

    # --------------------------------------------------------------- worker
    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            kind, payload = item
            try:
                if kind == "say":
                    self._speak(payload)
                elif kind == "beep":
                    import winsound
                    winsound.Beep(int(payload[0]), int(payload[1]))
            except Exception:
                # Audio is a convenience. Losing it must never take the flight
                # down, so failures are swallowed rather than raised.
                pass

    def _speak(self, text: str) -> None:
        safe = "".join(ch for ch in text if ch.isalnum() or ch in " -")
        subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Add-Type -AssemblyName System.Speech; "
             "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
             f"$s.Rate = {self.rate}; $s.Speak('{safe}')"],
            capture_output=True, timeout=6)


def selftest() -> bool:
    """Speak a short sequence so the operator can set volume before flying."""
    v = Voice()
    if not v.available:
        print("[voice] unavailable -- the cue will be on screen only")
        return False
    print("[voice] speaking a test sequence; set your volume now")
    for name in ("left", "up", "forward"):
        v.cue(name)
        v.go()
    v.say("voice check complete")
    import time
    time.sleep(4.0)
    v.stop()
    return True


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
