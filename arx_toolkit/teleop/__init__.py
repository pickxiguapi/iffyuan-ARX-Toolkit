from arx_toolkit.teleop.leader_follower import LeaderFollowerTeleop

__all__ = ["LeaderFollowerTeleop", "VRTeleop"]


def __getattr__(name):
    if name == "VRTeleop":
        from arx_toolkit.teleop.vr_teleop import VRTeleop
        return VRTeleop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
