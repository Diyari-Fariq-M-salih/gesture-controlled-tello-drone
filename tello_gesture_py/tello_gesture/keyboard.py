from dataclasses import dataclass

@dataclass
class RC:
    lr: int = 0
    fb: int = 0
    ud: int = 0
    yaw: int = 0

def rc_from_key(key: int, speed: int) -> RC:
    if key == ord("w"):
        return RC(fb=speed)
    if key == ord("s"):
        return RC(fb=-speed)
    if key == ord("a"):
        return RC(lr=-speed)
    if key == ord("d"):
        return RC(lr=speed)
    if key == ord("r"):
        return RC(ud=speed)
    if key == ord("f"):
        return RC(ud=-speed)
    if key == ord("j"):
        return RC(yaw=-speed)
    if key == ord("k"):
        return RC(yaw=speed)
    return RC()
