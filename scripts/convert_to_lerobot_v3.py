#!/usr/bin/env python3
"""Zarr → LeRobot v3 数据集转换脚本 (ARX LIFT2).

支持交互式选择 observation.state / action / cameras 的字段组合，
也支持通过 CLI 参数直传（跳过交互）。

用法::

    # 交互模式 — 运行后按提示选择字段
    python scripts/convert_to_lerobot_v3.py \
        --zarr datasets/pick_cup.zarr \
        --output lerobot_datasets/pick_cup \
        --repo-id iffyuan/arx_pick_cup \
        --task "pick up the cup" --fps 30

    # 非交互模式 — CLI 直传
    python scripts/convert_to_lerobot_v3.py \
        --zarr datasets/pick_cup.zarr \
        --output lerobot_datasets/pick_cup \
        --repo-id iffyuan/arx_pick_cup \
        --task "pick up the cup" --fps 30 \
        --state left_joint_pos,right_joint_pos \
        --action action_left,action_right \
        --cameras camera_l,camera_h,camera_r

    # dry-run — 只列出 Zarr 中的数组
    python scripts/convert_to_lerobot_v3.py \
        --zarr datasets/pick_cup.zarr --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 排除列表：这些 key 不参与 state / action 选择
_EXCLUDE_KEYS = {"timestamp", "episode"}
_IMAGE_PREFIXES = ("rgb_", "depth_")


def _is_numeric_array(key: str) -> bool:
    """判断 key 是否为可选数值数组（排除图像和 metadata）."""
    if key in _EXCLUDE_KEYS:
        return False
    for prefix in _IMAGE_PREFIXES:
        if key.startswith(prefix):
            return False
    return True


# ---------------------------------------------------------------------------
# 依赖检查
# ---------------------------------------------------------------------------

def _check_deps():
    missing = []
    for pkg in ("zarr", "lerobot", "PIL"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append("Pillow" if pkg == "PIL" else pkg)
    if missing:
        print(f"[ERROR] 缺少依赖: {', '.join(missing)}")
        print("  uv pip install -e .")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Episode 工具
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Zarr 探测
# ---------------------------------------------------------------------------

def _discover_arrays(data_group) -> tuple[list[tuple[str, tuple, str]], list[str]]:
    """扫描 Zarr data/ 组，返回 (数值数组列表, 相机名列表).

    Returns
    -------
    numeric_arrays : list of (key, shape, dtype_str)
    camera_names   : list of str  (不含 rgb_ 前缀，e.g. "camera_l")
    """
    numeric_arrays: list[tuple[str, tuple, str]] = []
    camera_set: set[str] = set()

    for key in sorted(data_group.keys()):
        arr = data_group[key]
        if _is_numeric_array(key):
            numeric_arrays.append((key, arr.shape, str(arr.dtype)))
        elif key.startswith("rgb_"):
            cam_name = key[4:]  # strip "rgb_"
            camera_set.add(cam_name)

    camera_names = sorted(camera_set)
    return numeric_arrays, camera_names


# ---------------------------------------------------------------------------
# 交互式选择
# ---------------------------------------------------------------------------

def _print_header(zarr_path: str, n_eps: int, n_frames: int):
    print(f"\n=== Zarr 数据集: {zarr_path} ===")
    print(f"Episodes: {n_eps}, 总帧数: {n_frames}\n")


def _print_numeric_arrays(arrays: list[tuple[str, tuple, str]]):
    print("  可用数组:")
    for idx, (key, shape, dtype) in enumerate(arrays):
        print(f"    [{idx}] {key:25s} {str(shape):20s} {dtype}")


def _print_cameras(cameras: list[str], data_group):
    print("\n  相机:")
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, cam in enumerate(cameras):
        rgb_key = f"rgb_{cam}"
        shape = data_group[rgb_key].shape if rgb_key in data_group else "?"
        label = labels[i] if i < len(labels) else str(i)
        print(f"    [{label}] {cam:25s} {str(shape)}")


def _parse_numeric_selection(
    raw: str,
    arrays: list[tuple[str, tuple, str]],
    label: str,
) -> list[str]:
    """解析用户输入的编号（逗号分隔），返回选中的 key 列表."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    selected: list[str] = []
    for p in parts:
        try:
            idx = int(p)
        except ValueError:
            print(f"  [ERROR] 无效编号 '{p}'，请重新输入")
            return []
        if idx < 0 or idx >= len(arrays):
            print(f"  [ERROR] 编号 {idx} 超出范围 [0, {len(arrays) - 1}]")
            return []
        selected.append(arrays[idx][0])
    if not selected:
        print(f"  [ERROR] {label} 至少选择一个数组")
    return selected


def _parse_camera_selection(
    raw: str,
    cameras: list[str],
) -> list[str]:
    """解析相机选择（字母或编号，直接回车=全选）."""
    if not raw.strip():
        return list(cameras)  # 全选

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    parts = [p.strip().upper() for p in raw.split(",") if p.strip()]
    selected: list[str] = []
    for p in parts:
        # 尝试字母标签
        if len(p) == 1 and p in labels:
            idx = labels.index(p)
        else:
            try:
                idx = int(p)
            except ValueError:
                print(f"  [ERROR] 无效输入 '{p}'")
                return []
        if idx < 0 or idx >= len(cameras):
            print(f"  [ERROR] 编号 {idx} 超出范围")
            return []
        selected.append(cameras[idx])
    return selected


def _calc_dim(keys: list[str], data_group) -> int:
    """计算选中 key 拼接后的总维度."""
    total = 0
    for key in keys:
        shape = data_group[key].shape
        dim = shape[1] if len(shape) > 1 else 1
        total += dim
    return total


def _format_selection(keys: list[str], data_group) -> str:
    """格式化选中内容，例如 'left_joint_pos(7) + right_joint_pos(7)'."""
    parts = []
    for key in keys:
        shape = data_group[key].shape
        dim = shape[1] if len(shape) > 1 else 1
        parts.append(f"{key}({dim})")
    return " + ".join(parts)


def interactive_select(
    data_group,
    numeric_arrays: list[tuple[str, tuple, str]],
    camera_names: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """交互式选择 state / action / cameras，返回三个 key 列表."""

    _print_numeric_arrays(numeric_arrays)
    _print_cameras(camera_names, data_group)

    # --- Step 1: state ---
    print("\nStep 1: 选择 observation.state 的组成")
    while True:
        raw = input("  输入编号（逗号分隔）: ").strip()
        state_keys = _parse_numeric_selection(raw, numeric_arrays, "state")
        if state_keys:
            dim = _calc_dim(state_keys, data_group)
            desc = _format_selection(state_keys, data_group)
            print(f"  → state = {desc} = {dim}D\n")
            break

    # --- Step 2: action ---
    print("Step 2: 选择 action 的组成")
    while True:
        raw = input("  输入编号（逗号分隔）: ").strip()
        action_keys = _parse_numeric_selection(raw, numeric_arrays, "action")
        if action_keys:
            dim = _calc_dim(action_keys, data_group)
            desc = _format_selection(action_keys, data_group)
            print(f"  → action = {desc} = {dim}D\n")
            break

    # --- Step 3: cameras ---
    print("Step 3: 选择相机（直接回车=全选）")
    while True:
        raw = input("  输入编号（逗号分隔）: ").strip()
        cam_keys = _parse_camera_selection(raw, camera_names)
        if cam_keys:
            print(f"  → cameras = {', '.join(cam_keys)}\n")
            break

    # --- 确认 ---
    state_dim = _calc_dim(state_keys, data_group)
    action_dim = _calc_dim(action_keys, data_group)
    state_names = " + ".join(state_keys)
    action_names = " + ".join(action_keys)

    print("确认:")
    print(f"  observation.state = {state_dim}D ({state_names})")
    print(f"  action            = {action_dim}D ({action_names})")
    print(f"  images            = {', '.join(cam_keys)}")

    confirm = input("  继续? [Y/n] ").strip().lower()
    if confirm and confirm != "y":
        print("[ABORT] 已取消")
        sys.exit(0)

    return state_keys, action_keys, cam_keys


# ---------------------------------------------------------------------------
# dry-run
# ---------------------------------------------------------------------------

def dry_run(zarr_path: str, max_episodes: int | None = None):
    """只读取 Zarr 并打印可用数组信息."""
    import zarr

    store = zarr.open(str(zarr_path), "r")
    data = store["data"]
    meta = store["meta"]

    ep_ranges = _get_episode_ranges(data, meta, max_episodes)
    n_eps = len(ep_ranges)
    n_frames = sum(end - start for start, end in ep_ranges)

    numeric_arrays, camera_names = _discover_arrays(data)

    _print_header(zarr_path, n_eps, n_frames)
    _print_numeric_arrays(numeric_arrays)
    _print_cameras(camera_names, data)

    print(f"\n  [data/ 全部数组]")
    for key in sorted(data.keys()):
        arr = data[key]
        print(f"    {key:25s} shape={str(arr.shape):20s} dtype={arr.dtype}")

    print()


# ---------------------------------------------------------------------------
# 转换核心
# ---------------------------------------------------------------------------

def convert(
    zarr_path: str,
    output_dir: str,
    repo_id: str,
    state_keys: list[str],
    action_keys: list[str],
    camera_names: list[str],
    fps: int = 30,
    robot_type: str = "arx_lift2",
    task_name: str | None = None,
    max_episodes: int | None = None,
    use_videos: bool = False,
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

    ep_ranges = _get_episode_ranges(data, meta, max_episodes)
    n_eps = len(ep_ranges)
    n_frames = sum(end - start for start, end in ep_ranges)

    # --- 计算维度 ---
    state_dim = _calc_dim(state_keys, data)
    action_dim = _calc_dim(action_keys, data)

    # --- 图像尺寸 (从第一个相机获取) ---
    first_cam_key = f"rgb_{camera_names[0]}"
    _, C, H, W = data[first_cam_key].shape
    image_shape = (H, W, C)
    print(f"[INFO] 图像尺寸: {W}×{H}")
    print(f"[INFO] Episodes: {n_eps}, 总帧数: {n_frames}")
    print(f"[INFO] state = {state_dim}D ({' + '.join(state_keys)})")
    print(f"[INFO] action = {action_dim}D ({' + '.join(action_keys)})")
    print(f"[INFO] cameras = {', '.join(camera_names)}")

    # --- Prepare output ---
    output_path = Path(output_dir).resolve()
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Build features ---
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["actions"],
        },
    }

    image_feature_dtype = "video" if use_videos else "image"
    for cam in camera_names:
        features[f"observation.images.{cam}"] = {
            "dtype": image_feature_dtype,
            "shape": image_shape,
            "names": ["height", "width", "channel"],
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

    # --- 预读辅助函数 ---
    def _read_and_concat(keys: list[str], start: int, end: int) -> np.ndarray:
        """批量读取 keys 并按列拼接，返回 (L, D) float32."""
        arrays = []
        for key in keys:
            arr = data[key][start:end].astype(np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            arrays.append(arr)
        return np.concatenate(arrays, axis=1)

    # --- Convert episode by episode ---
    t0 = time.time()

    for ep_idx, (start, end) in enumerate(ep_ranges):
        ep_len = end - start
        print(f"  Episode {ep_idx}: {ep_len} steps")

        # Batch read state & action
        state_batch = _read_and_concat(state_keys, start, end)   # (L, state_dim)
        action_batch = _read_and_concat(action_keys, start, end)  # (L, action_dim)

        # Batch read cameras
        cam_batches = {}
        for cam in camera_names:
            cam_batches[cam] = data[f"rgb_{cam}"][start:end]  # (L, C, H, W)

        for i in range(ep_len):
            frame = {
                "observation.state": state_batch[i],
                "action": action_batch[i],
                "task": task_name,
            }

            # Images: (C, H, W) → (H, W, C)
            for cam in camera_names:
                img_hwc = cam_batches[cam][i].transpose(1, 2, 0)
                if use_videos:
                    # LeRobot v3 视频编码路径使用 ndarray(H, W, C)
                    frame[f"observation.images.{cam}"] = img_hwc
                else:
                    frame[f"observation.images.{cam}"] = Image.fromarray(img_hwc)

            dataset.add_frame(frame)

        dataset.save_episode()

    # --- Done ---
    elapsed = time.time() - t0
    if use_videos:
        # 清理中间图像目录，避免与 videos 重复占用空间。
        image_dir = output_path / "images"
        if image_dir.exists():
            shutil.rmtree(image_dir, ignore_errors=True)

    print(f"\n{'=' * 60}")
    print(f"[DONE] 转换完成!")
    print(f"  输出: {output_path}")
    print(f"  Episodes: {n_eps}, Frames: {n_frames}")
    print(f"  state({state_dim}D) + action({action_dim}D) + {len(camera_names)} cameras")
    print(f"  耗时: {elapsed:.1f}s ({n_frames / max(elapsed, 0.1):.0f} frames/s)")
    print(f"{'=' * 60}")

    return dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Zarr → LeRobot v3 数据集转换 (ARX LIFT2) — 交互式选择 state/action",
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
    parser.add_argument("--dry-run", action="store_true",
                        help="只列出 Zarr 中的数组，不进入选择/转换")

    # 非交互模式参数
    parser.add_argument("--state", default=None,
                        help="非交互: state 字段名（逗号分隔），如 left_joint_pos,right_joint_pos")
    parser.add_argument("--action", default=None,
                        help="非交互: action 字段名（逗号分隔），如 action_left,action_right")
    parser.add_argument("--cameras", default=None,
                        help="非交互: 相机名（逗号分隔），如 camera_l,camera_h,camera_r")

    args = parser.parse_args()

    zarr_path = Path(args.zarr)
    if not zarr_path.exists():
        print(f"[ERROR] 输入路径不存在: {args.zarr}")
        sys.exit(1)

    # --- dry-run ---
    if args.dry_run:
        import zarr  # noqa: F811
        dry_run(str(zarr_path), max_episodes=args.episodes)
        return

    # --- 正式转换需要完整依赖 ---
    _check_deps()
    import zarr

    if args.output is None:
        print("[ERROR] 请指定 --output 输出路径")
        sys.exit(1)
    if args.repo_id is None:
        print("[ERROR] 请指定 --repo-id")
        sys.exit(1)

    # 打开 Zarr 进行探测
    store = zarr.open(str(zarr_path), "r")
    data = store["data"]
    meta = store["meta"]

    ep_ranges = _get_episode_ranges(data, meta, args.episodes)
    n_eps = len(ep_ranges)
    n_frames = sum(end - start for start, end in ep_ranges)

    numeric_arrays, camera_names = _discover_arrays(data)

    # --- 非交互 vs 交互 ---
    if args.state is not None and args.action is not None:
        # 非交互模式
        state_keys = [k.strip() for k in args.state.split(",") if k.strip()]
        action_keys = [k.strip() for k in args.action.split(",") if k.strip()]

        if args.cameras is not None:
            cam_keys = [k.strip() for k in args.cameras.split(",") if k.strip()]
        else:
            cam_keys = camera_names  # 默认全选

        # 校验 key 存在
        all_numeric_names = {a[0] for a in numeric_arrays}
        for key in state_keys + action_keys:
            if key not in all_numeric_names:
                print(f"[ERROR] 数组 '{key}' 不存在于 Zarr data/ 中")
                print(f"  可用: {sorted(all_numeric_names)}")
                sys.exit(1)
        for cam in cam_keys:
            if cam not in camera_names:
                print(f"[ERROR] 相机 '{cam}' 不存在")
                print(f"  可用: {camera_names}")
                sys.exit(1)

        state_dim = _calc_dim(state_keys, data)
        action_dim = _calc_dim(action_keys, data)
        print(f"[INFO] 非交互模式")
        print(f"  state  = {_format_selection(state_keys, data)} = {state_dim}D")
        print(f"  action = {_format_selection(action_keys, data)} = {action_dim}D")
        print(f"  cameras = {', '.join(cam_keys)}")
    else:
        # 交互模式
        _print_header(str(zarr_path), n_eps, n_frames)
        state_keys, action_keys, cam_keys = interactive_select(
            data, numeric_arrays, camera_names,
        )

    print(f"\n转换中...")
    convert(
        zarr_path=str(zarr_path),
        output_dir=args.output,
        repo_id=args.repo_id,
        state_keys=state_keys,
        action_keys=action_keys,
        camera_names=cam_keys,
        fps=args.fps,
        robot_type=args.robot_type,
        task_name=args.task,
        max_episodes=args.episodes,
        use_videos=args.use_videos,
    )


if __name__ == "__main__":
    main()
