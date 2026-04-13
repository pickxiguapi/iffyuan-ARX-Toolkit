#!/usr/bin/env python3
"""Zarr → LeRobot v3 数据集转换脚本 (ARX LIFT2).

将 arx_toolkit Collector 采集的 Zarr 数据集转换为 LeRobot v3.0 格式。

特征映射::

    observation.state  = 14D  — concat [left_joint_pos(7) + right_joint_pos(7)]
    observation.eef_pos = 14D — concat [left_eef_pos(7) + right_eef_pos(7)]  (可选)
    observation.base_state = 1D — base_height
    observation.images.camera_l/h/r — RGB 图像
    action = 18D — concat [action_left(7) + action_right(7) + action_base(3) + action_lift(1)]

用法::

    python scripts/convert_to_lerobot_v3.py \
        --zarr datasets/pick_cup.zarr \
        --output lerobot_datasets/pick_cup \
        --repo-id iffyuan/arx_pick_cup \
        --task "pick up the cup" \
        --fps 50 \
        --use-videos

    # dry-run (只检查映射逻辑，不写入)
    python scripts/convert_to_lerobot_v3.py \
        --zarr datasets/pick_cup.zarr \
        --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def _check_deps():
    missing = []
    for pkg in ("zarr", "lerobot", "PIL"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append("Pillow" if pkg == "PIL" else pkg)
    if missing:
        print(f"[ERROR] 缺少依赖: {', '.join(missing)}")
        print("  pip install -e '.[lerobot]'")
        sys.exit(1)


def _get_episode_ranges(
    data_group,
    meta_group,
    max_episodes: int | None = None,
) -> list[tuple[int, int]]:
    """从 meta/episode_ends 或 data/episode 提取各 episode 的 [start, end) 范围."""
    if "episode_ends" in meta_group:
        episode_ends = meta_group["episode_ends"][:]
    else:
        print("[WARN] meta/episode_ends 缺失，从 data/episode 自动重建...")
        all_ep = data_group["episode"][:]
        unique_eps = np.unique(all_ep)
        ends = []
        running = 0
        for ep in unique_eps:
            running += int(np.sum(all_ep == ep))
            ends.append(running)
        episode_ends = np.array(ends, dtype=np.uint32)
        print(f"[INFO] 重建完成: {len(episode_ends)} 个 episode, ends={ends}")

    if max_episodes is not None and max_episodes < len(episode_ends):
        print(f"[INFO] 只使用前 {max_episodes} / {len(episode_ends)} 个 episode")
        episode_ends = episode_ends[:max_episodes]

    ranges = []
    prev = 0
    for end in episode_ends:
        ranges.append((int(prev), int(end)))
        prev = end
    return ranges


def dry_run(zarr_path: str, max_episodes: int | None = None):
    """只读取 Zarr 并打印映射信息，不做任何转换."""
    import zarr

    store = zarr.open(str(zarr_path), "r")
    data = store["data"]
    meta = store["meta"]

    print(f"\n{'=' * 60}")
    print(f"  Zarr 数据集: {zarr_path}")
    print(f"{'=' * 60}")

    # List all arrays
    print("\n  [data/ 数组]")
    for key in sorted(data.keys()):
        arr = data[key]
        print(f"    {key:25s} shape={arr.shape} dtype={arr.dtype}")

    # Episode info
    ep_ranges = _get_episode_ranges(data, meta, max_episodes)
    n_eps = len(ep_ranges)
    n_frames = sum(end - start for start, end in ep_ranges)
    print(f"\n  Episodes: {n_eps}, 总帧数: {n_frames}")

    # Check required keys
    required = [
        "left_joint_pos", "right_joint_pos",
        "action_left", "action_right", "action_base", "action_lift",
    ]
    camera_keys = [f"rgb_{cam}" for cam in ("camera_l", "camera_h", "camera_r")]

    print("\n  [特征映射]")
    print(f"    observation.state   = left_joint_pos(7) + right_joint_pos(7) = 14D")
    print(f"    action              = action_left(7) + action_right(7) + action_base(3) + action_lift(1) = 18D")
    print(f"    observation.images  = camera_l, camera_h, camera_r")

    missing = [k for k in required + camera_keys if k not in data]
    if missing:
        print(f"\n  ⚠ 缺少数组: {missing}")
    else:
        print(f"\n  ✓ 所有必需数组已就绪")

    # Optional keys
    optional = {
        "left_eef_pos": "observation.eef_pos[:7]",
        "right_eef_pos": "observation.eef_pos[7:14]",
        "base_height": "observation.base_state",
    }
    for key, target in optional.items():
        status = "✓" if key in data else "✗ (缺失)"
        print(f"    {key:25s} → {target:30s} {status}")

    # Sample first frame
    if n_frames > 0:
        print(f"\n  [首帧采样]")
        for key in required:
            if key in data:
                print(f"    {key}: {data[key][0]}")

    print(f"\n{'=' * 60}\n")


def convert(
    zarr_path: str,
    output_dir: str,
    repo_id: str,
    fps: int = 30,
    robot_type: str = "arx_lift2",
    task_name: str | None = None,
    max_episodes: int | None = None,
    use_videos: bool = False,
    include_eef: bool = True,
    include_base_state: bool = True,
):
    """执行 Zarr → LeRobot v3 转换."""
    import zarr
    from PIL import Image

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if task_name is None:
        task_name = Path(zarr_path).stem

    # --- Read Zarr ---
    store = zarr.open(str(zarr_path), "r")
    data = store["data"]
    meta = store["meta"]

    # Detect image shape from first camera
    _, C, H, W = data["rgb_camera_l"].shape
    image_shape = (H, W, C)
    print(f"[INFO] 图像尺寸: {W}×{H}")

    ep_ranges = _get_episode_ranges(data, meta, max_episodes)
    n_eps = len(ep_ranges)
    n_frames = sum(end - start for start, end in ep_ranges)
    print(f"[INFO] Episodes: {n_eps}, 总帧数: {n_frames}")

    # --- Prepare output ---
    output_path = Path(output_dir).resolve()
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Build features ---
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (18,),
            "names": ["actions"],
        },
    }

    # 3 camera images
    for cam in ("camera_l", "camera_h", "camera_r"):
        features[f"observation.images.{cam}"] = {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        }

    # Optional: eef pos
    if include_eef and "left_eef_pos" in data:
        features["observation.eef_pos"] = {
            "dtype": "float32",
            "shape": (14,),
            "names": ["eef_pos"],
        }

    # Optional: base state
    if include_base_state and "base_height" in data:
        features["observation.base_state"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": ["base_state"],
        }

    print(f"[INFO] 创建 LeRobot 数据集: repo_id={repo_id}, fps={fps}")
    print(f"[INFO] 输出路径: {output_path}")
    print(f"[INFO] use_videos={use_videos}")
    print(f"[INFO] Features: {list(features.keys())}")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        root=output_path,
        use_videos=use_videos,
        image_writer_threads=4,
    )

    # --- Convert episode by episode ---
    t0 = time.time()

    for ep_idx, (start, end) in enumerate(ep_ranges):
        ep_len = end - start
        print(f"  Episode {ep_idx}: frames [{start}, {end}) = {ep_len} steps")

        # Batch read
        left_jp = data["left_joint_pos"][start:end]    # (L, 7)
        right_jp = data["right_joint_pos"][start:end]   # (L, 7)
        act_l = data["action_left"][start:end]           # (L, 7)
        act_r = data["action_right"][start:end]          # (L, 7)
        act_b = data["action_base"][start:end]           # (L, 3)
        act_lift = data["action_lift"][start:end]        # (L, 1)

        # Camera batches
        rgb_l = data["rgb_camera_l"][start:end]  # (L, 3, H, W)
        rgb_h = data["rgb_camera_h"][start:end]
        rgb_r = data["rgb_camera_r"][start:end]

        # Optional batches
        has_eef = include_eef and "left_eef_pos" in data
        if has_eef:
            left_eef = data["left_eef_pos"][start:end]   # (L, 7)
            right_eef = data["right_eef_pos"][start:end]  # (L, 7)

        has_base = include_base_state and "base_height" in data
        if has_base:
            base_h = data["base_height"][start:end]  # (L, 1)

        for i in range(ep_len):
            # observation.state = 14D
            state = np.concatenate([
                left_jp[i].astype(np.float32),
                right_jp[i].astype(np.float32),
            ])  # (14,)

            # action = 18D
            action_vec = np.concatenate([
                act_l[i].astype(np.float32),
                act_r[i].astype(np.float32),
                act_b[i].astype(np.float32),
                act_lift[i].astype(np.float32).reshape(-1),
            ])  # (18,)

            # Images: (C, H, W) → (H, W, C) → PIL
            img_l = Image.fromarray(rgb_l[i].transpose(1, 2, 0))
            img_h = Image.fromarray(rgb_h[i].transpose(1, 2, 0))
            img_r = Image.fromarray(rgb_r[i].transpose(1, 2, 0))

            frame = {
                "observation.state": state,
                "action": action_vec,
                "observation.images.camera_l": img_l,
                "observation.images.camera_h": img_h,
                "observation.images.camera_r": img_r,
                "task": task_name,
            }

            # Optional features
            if has_eef:
                eef_pos = np.concatenate([
                    left_eef[i].astype(np.float32),
                    right_eef[i].astype(np.float32),
                ])  # (12,)
                frame["observation.eef_pos"] = eef_pos

            if has_base:
                frame["observation.base_state"] = base_h[i].astype(np.float32).reshape(-1)

            dataset.add_frame(frame)

        dataset.save_episode()

    # --- Done ---
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"[DONE] 转换完成!")
    print(f"  输出: {output_path}")
    print(f"  Episodes: {n_eps}, Frames: {n_frames}")
    print(f"  Features: state(14D) + action(18D) + 3 cameras")
    print(f"  耗时: {elapsed:.1f}s ({n_frames / max(elapsed, 0.1):.0f} frames/s)")
    print(f"{'=' * 60}")

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Zarr → LeRobot v3 数据集转换 (ARX LIFT2)",
    )
    parser.add_argument("--zarr", "-i", required=True, help="输入 Zarr 数据集路径")
    parser.add_argument("--output", "-o", default=None, help="LeRobot 输出目录")
    parser.add_argument("--repo-id", default=None, help="数据集 repo ID")
    parser.add_argument("--fps", type=int, default=30, help="帧率 (默认 30)")
    parser.add_argument("--robot-type", default="arx_lift2", help="机器人类型")
    parser.add_argument("--task", default=None, help="任务描述")
    parser.add_argument("--episodes", type=int, default=None,
                        help="只转换前 N 个 episode")
    parser.add_argument("--use-videos", action="store_true",
                        help="使用视频格式存储图像（推荐大数据集）")
    parser.add_argument("--no-eef", action="store_true",
                        help="不包含 eef_pos 特征")
    parser.add_argument("--no-base-state", action="store_true",
                        help="不包含 base_state 特征")
    parser.add_argument("--dry-run", action="store_true",
                        help="只检查映射逻辑，不执行转换")

    args = parser.parse_args()

    zarr_path = Path(args.zarr)
    if not zarr_path.exists():
        print(f"[ERROR] 输入路径不存在: {args.zarr}")
        sys.exit(1)

    if args.dry_run:
        import zarr  # only need zarr for dry-run
        dry_run(str(zarr_path), max_episodes=args.episodes)
        return

    # Need full deps for actual conversion
    _check_deps()

    if args.output is None:
        print("[ERROR] 请指定 --output 输出路径")
        sys.exit(1)
    if args.repo_id is None:
        print("[ERROR] 请指定 --repo-id")
        sys.exit(1)

    convert(
        zarr_path=str(zarr_path),
        output_dir=args.output,
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type=args.robot_type,
        task_name=args.task,
        max_episodes=args.episodes,
        use_videos=args.use_videos,
        include_eef=not args.no_eef,
        include_base_state=not args.no_base_state,
    )


if __name__ == "__main__":
    main()
