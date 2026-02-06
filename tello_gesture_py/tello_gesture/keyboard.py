from .rc_command import RCCommand as RC

def rc_from_key(key: int, speed: int) -> RC:
    if key == ord("w"):
        return RC(fb=speed, active=True)
    if key == ord("s"):
        return RC(fb=-speed, active=True)
    if key == ord("a"):
        return RC(lr=-speed, active=True)
    if key == ord("d"):
        return RC(lr=speed, active=True)
    if key == ord("r"):
        return RC(ud=speed, active=True)
    if key == ord("f"):
        return RC(ud=-speed, active=True)
    if key == ord("j"):
        return RC(yaw=-speed, active=True)
    if key == ord("k"):
        return RC(yaw=speed, active=True)
    return RC(active=False)
