#!/usr/bin/env python3
"""Check a collected ARX Zarr dataset for stale streams and gripper values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def _load_config(meta) -> dict:
    raw = meta.attrs.get("config")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _episode_ranges(data, meta) -> list[tuple[int, int]]:
    if "episode_ends" in meta:
        ends = [int(x) for x in meta["episode_ends"][:]]
    elif "episode" in data:
        episode = data["episode"][:]
        ends = []
        total = 0
        for ep in np.unique(episode):
            total += int(np.sum(episode == ep))
            ends.append(total)
    else:
        total = int(data["timestamp"].shape[0])
        ends = [total] if total > 0 else []
    starts = [0] + ends[:-1]
    return list(zip(starts, ends))


def _camera_names(data) -> list[str]:
    names = []
    for key in sorted(data.keys()):
        if key.startswith("rgb_") and getattr(data[key], "ndim", 0) == 4:
            names.append(key.removeprefix("rgb_"))
    return names


def _key_span(data, key: str, start: int, end: int, dim: int | None = None) -> float | None:
    if key not in data:
        return None
    arr = np.asarray(data[key][start:end], dtype=np.float32)
    if arr.size == 0:
        return 0.0
    if dim is not None:
        if arr.ndim < 2 or arr.shape[1] <= dim:
            return None
        arr = arr[:, dim]
    return float(np.nanmax(arr) - np.nanmin(arr))


def _image_delta(data, cam: str, start: int, end: int) -> float | None:
    key = f"rgb_{cam}"
    if key not in data or end - start < 2:
        return None
    first = np.asarray(data[key][start], dtype=np.float32)
    last = np.asarray(data[key][end - 1], dtype=np.float32)
    return float(np.mean(np.abs(last - first)))


def _check_action_side(
    data,
    *,
    side: str,
    start: int,
    end: int,
    min_gripper_span: float,
) -> list[str]:
    issues = []
    key = f"action_{side}"
    span = _key_span(data, key, start, end, dim=6)
    if span is None:
        issues.append(f"{key} missing or not 7D")
    elif span < min_gripper_span:
        issues.append(f"{key} gripper stale span={span:.4f}")
    return issues


def _selected_sides(config: dict, action_side: str) -> tuple[str, ...]:
    if action_side != "auto":
        return ("left", "right") if action_side == "both" else (action_side,)
    if config.get("collection_kind") == "vr_single":
        leader_side = str(config.get("leader_side") or "left")
        return (leader_side,) if leader_side in {"left", "right"} else ("left",)
    return ("left", "right")


def _print_episode_report(
    *,
    ep_idx: int,
    start: int,
    end: int,
    issues: list[str],
    info: Iterable[str],
) -> None:
    prefix = f"episode {ep_idx} [{start}:{end}]"
    if issues:
        print(f"FAIL {prefix}")
        for item in issues:
            print(f"  - {item}")
    else:
        print(f"OK   {prefix}")
    for item in info:
        print(f"  {item}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-collection Zarr health check for ARX datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", "-d", required=True, help="Zarr dataset path")
    parser.add_argument(
        "--action-side",
        choices=["auto", "left", "right", "both"],
        default="auto",
        help="Which action gripper should be checked for movement.",
    )
    parser.add_argument("--min-gripper-span", type=float, default=0.05)
    parser.add_argument("--min-state-span", type=float, default=1e-4)
    parser.add_argument("--min-image-delta", type=float, default=0.1)
    parser.add_argument("--episodes", type=int, default=0, help="Check first N episodes; 0 = all")
    args = parser.parse_args()

    global np
    import numpy as np
    import zarr

    root = Path(args.dataset)
    store = zarr.open(str(root), "r")
    data = store["data"]
    meta = store["meta"]
    config = _load_config(meta)
    ranges = _episode_ranges(data, meta)
    if args.episodes > 0:
        ranges = ranges[: int(args.episodes)]
    if not ranges:
        print(f"FAIL no episodes found in {root}")
        raise SystemExit(1)

    action_sides = _selected_sides(config, args.action_side)
    cameras = _camera_names(data)
    print(f"Dataset: {root}")
    print(f"Action sides checked: {', '.join(action_sides)}")
    print(f"Cameras checked: {', '.join(cameras) if cameras else 'none'}")

    failed = False
    for ep_idx, (start, end) in enumerate(ranges):
        issues: list[str] = []
        info: list[str] = []
        length = end - start
        if length <= 0:
            issues.append("empty episode")

        for key in ("left_joint_pos", "right_joint_pos", "left_eef_pos", "right_eef_pos"):
            span = _key_span(data, key, start, end)
            if span is None:
                issues.append(f"{key} missing")
            elif span < float(args.min_state_span):
                issues.append(f"{key} stale span={span:.6f}")

        for side in action_sides:
            issues.extend(
                _check_action_side(
                    data,
                    side=side,
                    start=start,
                    end=end,
                    min_gripper_span=float(args.min_gripper_span),
                )
            )
            span = _key_span(data, f"action_{side}", start, end, dim=6)
            info.append(f"action_{side}[gripper] span={0.0 if span is None else span:.4f}")

        for cam in cameras:
            delta = _image_delta(data, cam, start, end)
            if delta is None:
                issues.append(f"rgb_{cam} missing or too short")
            elif delta < float(args.min_image_delta):
                issues.append(f"rgb_{cam} stale delta={delta:.4f}")
            info.append(f"rgb_{cam} first-last mean delta={0.0 if delta is None else delta:.4f}")

        failed = failed or bool(issues)
        _print_episode_report(
            ep_idx=ep_idx,
            start=start,
            end=end,
            issues=issues,
            info=info,
        )

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
