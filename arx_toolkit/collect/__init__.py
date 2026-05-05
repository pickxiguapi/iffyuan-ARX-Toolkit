"""Data collection module for ARX LIFT2."""

from arx_toolkit.collect.collector import Collector

__all__ = ["Collector", "collect_vr_episode"]


def __getattr__(name):
    if name == "collect_vr_episode":
        from arx_toolkit.collect.vr_collector import collect_vr_episode

        return collect_vr_episode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
