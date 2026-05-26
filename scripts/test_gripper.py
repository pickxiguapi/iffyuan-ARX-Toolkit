#!/usr/bin/env python3
"""Check VR gripper signals before recording.

This script is intentionally read-only: it subscribes to arm status and VR
command topics, prints raw gripper values, and reports whether the selected
side changes while the operator opens/closes the VR gripper.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

GRIPPER_OPEN_RAW = -3.4
GRIPPER_CLOSE_RAW = 0.0


def gripper_normalize(raw: float) -> float:
    normalized = (
        (float(raw) - GRIPPER_OPEN_RAW)
        / (GRIPPER_CLOSE_RAW - GRIPPER_OPEN_RAW)
    )
    return max(0.0, min(1.0, normalized))


@dataclass
class GripperSnapshot:
    status_raw: float | None = None
    status_norm: float | None = None
    vr_raw: float | None = None
    vr_norm: float | None = None
    status_stamp: float | None = None
    vr_stamp: float | None = None


def _fmt(value: float | None) -> str:
    return "None" if value is None else f"{value: .4f}"


def _joint_gripper(msg: Any) -> float | None:
    joint_pos = getattr(msg, "joint_pos", None)
    if joint_pos is None or len(joint_pos) <= 6:
        return None
    return float(joint_pos[6])


class GripperMonitor:
    def __init__(
        self,
        *,
        left_status_topic: str,
        right_status_topic: str,
        left_vr_topic: str,
        right_vr_topic: str,
    ) -> None:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from arx5_arm_msg.msg._robot_status import RobotStatus
        from arm_control.msg._pos_cmd import PosCmd

        if not rclpy.ok():
            rclpy.init()

        self._lock = threading.Lock()
        self.snapshots = {
            "left": GripperSnapshot(),
            "right": GripperSnapshot(),
        }

        class MonitorNode(Node):
            pass

        self.node = MonitorNode("arx_gripper_precheck")
        self.node.create_subscription(
            RobotStatus,
            left_status_topic,
            lambda msg: self._on_status("left", msg),
            10,
        )
        self.node.create_subscription(
            RobotStatus,
            right_status_topic,
            lambda msg: self._on_status("right", msg),
            10,
        )
        self.node.create_subscription(
            PosCmd,
            left_vr_topic,
            lambda msg: self._on_vr("left", msg),
            10,
        )
        self.node.create_subscription(
            PosCmd,
            right_vr_topic,
            lambda msg: self._on_vr("right", msg),
            10,
        )

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def _on_status(self, side: str, msg: Any) -> None:
        raw = _joint_gripper(msg)
        with self._lock:
            snap = self.snapshots[side]
            snap.status_raw = raw
            snap.status_norm = None if raw is None else gripper_normalize(raw)
            snap.status_stamp = time.time()

    def _on_vr(self, side: str, msg: Any) -> None:
        raw = float(getattr(msg, "gripper", math.nan))
        if math.isnan(raw):
            raw = None
        with self._lock:
            snap = self.snapshots[side]
            snap.vr_raw = raw
            snap.vr_norm = None if raw is None else gripper_normalize(raw)
            snap.vr_stamp = time.time()

    def snapshot(self) -> dict[str, GripperSnapshot]:
        with self._lock:
            return {
                side: GripperSnapshot(**vars(snap))
                for side, snap in self.snapshots.items()
            }

    def close(self) -> None:
        import rclpy

        self.executor.shutdown()
        self.node.destroy_node()
        self.thread.join(timeout=2.0)
        if rclpy.ok():
            rclpy.shutdown()


def _span(values: list[float | None]) -> float:
    nums = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not nums:
        return 0.0
    return max(nums) - min(nums)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight check for ARX VR gripper values.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--min-span", type=float, default=0.05)
    parser.add_argument("--left-status-topic", default="arm_status_l")
    parser.add_argument("--right-status-topic", default="arm_status_r")
    parser.add_argument("--left-vr-topic", default="/ARX_VR_L")
    parser.add_argument("--right-vr-topic", default="/ARX_VR_R")
    args = parser.parse_args()

    sides = ("left", "right") if args.side == "both" else (args.side,)
    monitor = GripperMonitor(
        left_status_topic=args.left_status_topic,
        right_status_topic=args.right_status_topic,
        left_vr_topic=args.left_vr_topic,
        right_vr_topic=args.right_vr_topic,
    )
    vr_history: dict[str, list[float | None]] = {side: [] for side in sides}
    status_history: dict[str, list[float | None]] = {side: [] for side in sides}

    print("Start gripper precheck. Open/close the VR gripper now.")
    print("Values are shown as raw hardware value and normalized [0, 1].")
    period = 1.0 / max(float(args.hz), 1e-6)
    end_time = time.time() + max(float(args.duration), 0.0)

    try:
        while time.time() < end_time:
            snapshots = monitor.snapshot()
            for side in sides:
                snap = snapshots[side]
                vr_history[side].append(snap.vr_norm)
                status_history[side].append(snap.status_norm)
                print(
                    f"{side:5s} "
                    f"status raw={_fmt(snap.status_raw)} norm={_fmt(snap.status_norm)} | "
                    f"vr raw={_fmt(snap.vr_raw)} norm={_fmt(snap.vr_norm)}"
                )
            time.sleep(period)
    finally:
        monitor.close()

    print("\nSummary:")
    failed = False
    for side in sides:
        vr_span = _span(vr_history[side])
        status_span = _span(status_history[side])
        ok = vr_span >= float(args.min_span)
        failed = failed or not ok
        mark = "OK" if ok else "FAIL"
        print(
            f"{mark} {side}: vr_span={vr_span:.4f}, "
            f"status_span={status_span:.4f}, min_span={args.min_span:.4f}"
        )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
