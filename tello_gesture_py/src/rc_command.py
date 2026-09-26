from dataclasses import dataclass

@dataclass
class RCCommand:
    lr: int = 0   # left/right
    fb: int = 0   # forward/back
    ud: int = 0   # up/down
    yaw: int = 0  # yaw
    active: bool = True  # if False, treated as "no command"
