"""Data collector: env + action_source → Zarr dataset.

Manages the full collect loop for ARX LIFT2 dual-arm robot:
  1. Init env (with cameras)
  2. Per-episode: wait for start signal → record steps at fixed Hz
     → save buffer → repeat
  3. Compute episode_ends metadata

The Collector is **agnostic** to the action source — it accepts any callable
that returns the current action dict.  For leader-follower teleop the teleop
thread runs independently; Collector just samples ``action_source()`` each tick.

Features:
  - 定频采集 (``--hz``, 默认 30 Hz)
  - 断点续采 (``--episodes N`` 表示目标总数)
    - 键盘控制 (raw terminal: Space 开始 / Enter 结束并确认 / Ctrl+C 退出)
  - Episode 崩溃恢复（单个 episode 报错不杀采集）
  - 启动横幅 + Episode 摘要 + 最终总结
  - 可选保存回放视频 (``--save-video``)
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
import zarr

from arx_toolkit.utils.logger import get_logger

logger = get_logger("arx_toolkit.collect")

# Camera name → Zarr key mapping
_CAM_NAMES = ("camera_l", "camera_h", "camera_r")


# ---------------------------------------------------------------------------
# EpisodeStats
# ---------------------------------------------------------------------------

@dataclass
class EpisodeStats:
    """Statistics for a single completed episode."""
    steps: int
    duration: float
    fps: float


# ---------------------------------------------------------------------------
# Keyboard listener (non-blocking, raw terminal)
# ---------------------------------------------------------------------------

class _KeyboardListener:
    """Non-blocking keyboard reader in raw terminal mode."""

    def __init__(self):
        import termios
        self._termios = termios
        self._old_settings = termios.tcgetattr(sys.stdin)

    def start(self):
        import tty
        tty.setraw(sys.stdin.fileno())

    def stop(self):
        self._termios.tcsetattr(sys.stdin, self._termios.TCSADRAIN, self._old_settings)

    def get_key(self) -> str | None:
        import select
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1).lower()
        return None


# ---------------------------------------------------------------------------
# Zarr dataset helper
# ---------------------------------------------------------------------------

def _open_or_create_zarr(
    path: str,
    image_shape: tuple[int, ...],
    cam_mode: str = "rgbd",
    config: dict | None = None,
) -> tuple[zarr.Group, zarr.Group, int]:
    """Open existing or create new Zarr dataset.

    Returns (data_group, meta_group, start_episode).
    """
    dataset_path = str(path)
    exists = os.path.exists(dataset_path)

    if exists:
        logger.info("Opening existing dataset: %s", dataset_path)
        ds = zarr.open(dataset_path, mode="a")
        data = ds["data"]
        meta = ds["meta"]
        if config:
            meta.attrs["config"] = json.dumps(config, ensure_ascii=False, default=str)
        if "episode" in data:
            start_ep = len(np.unique(data["episode"][:]))
            logger.info("Found %d existing episodes", start_ep)
        else:
            start_ep = 0
        return data, meta, start_ep

    logger.info("Creating new dataset: %s (cam_mode=%s)", dataset_path, cam_mode)
    ds = zarr.open(dataset_path, mode="w")
    data = ds.create_group("data")
    meta = ds.create_group("meta")

    # Image compressor: Blosc zstd
    try:
        from numcodecs import Blosc
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    except ImportError:
        compressor = None

    H, W = image_shape[1], image_shape[2]  # image_shape = (3, H, W)

    # --- RGB for 3 cameras ---
    for cam in _CAM_NAMES:
        data.require_dataset(
            f"rgb_{cam}", shape=(0, *image_shape), dtype=np.uint8,
            chunks=(1, *image_shape), compressor=compressor,
        )

    # --- Depth (rgbd mode) ---
    if cam_mode == "rgbd":
        depth_shape = (1, H, W)
        for cam in _CAM_NAMES:
            data.require_dataset(
                f"depth_{cam}", shape=(0, *depth_shape), dtype=np.uint16,
                chunks=(1, *depth_shape), compressor=compressor,
            )

    # --- Robot state ---
    data.require_dataset("left_eef_pos", shape=(0, 7), dtype=np.float32)
    data.require_dataset("left_joint_pos", shape=(0, 7), dtype=np.float32)
    data.require_dataset("right_eef_pos", shape=(0, 7), dtype=np.float32)
    data.require_dataset("right_joint_pos", shape=(0, 7), dtype=np.float32)
    data.require_dataset("base_height", shape=(0, 1), dtype=np.float32)

    # --- Actions ---
    data.require_dataset("action_left", shape=(0, 7), dtype=np.float32)
    data.require_dataset("action_right", shape=(0, 7), dtype=np.float32)
    data.require_dataset("action_base", shape=(0, 3), dtype=np.float32)
    data.require_dataset("action_lift", shape=(0, 1), dtype=np.float32)

    # --- Timestamps & episode ---
    data.require_dataset("timestamp", shape=(0,), dtype=np.float64)
    data.require_dataset("episode", shape=(0,), dtype=np.uint16)

    # --- Save config as meta attrs ---
    if config:
        meta.attrs["config"] = json.dumps(config, ensure_ascii=False, default=str)

    return data, meta, 0


def _compute_episode_ends(data: zarr.Group, meta: zarr.Group):
    """Recompute ``meta/episode_ends`` from ``data/episode``."""
    all_ep = data["episode"][:]
    unique_eps = np.unique(all_ep)
    ends = []
    running = 0
    for ep in unique_eps:
        running += int(np.sum(all_ep == ep))
        ends.append(running)
    if "episode_ends" in meta:
        del meta["episode_ends"]
    meta.require_dataset("episode_ends", shape=(len(ends),), dtype=np.uint32)
    meta["episode_ends"][:] = np.array(ends, dtype=np.uint32)
    logger.info("episode_ends: %s (total %d episodes)", ends, len(ends))


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class Collector:
    """Orchestrates data collection: env + action_source → Zarr.

    The Collector does NOT care where the actions come from.  ``action_source``
    is a callable that returns the current action dict each time it is called.
    For leader-follower teleop, the teleop thread runs independently and
    ``action_source`` simply reads ``teleop.last_command``.

    Parameters
    ----------
    env : ARXEnv
        Fully initialised environment with ``action_mode``, ``camera_type``
        and ``camera_view`` set as needed.
    action_source : callable
        A function ``() -> dict`` that returns the current action.
        Expected format::

            {
                "left":  np.ndarray(7,) | None,   # 6 joint + gripper
                "right": np.ndarray(7,) | None,
                "base":  np.ndarray(3,) | None,   # [vx, vy, vz]
                "lift":  float | None,             # height [0, 20]
            }

    dataset_path : str
        Path to Zarr dataset (created if not exists).
    num_episodes : int
        **Target total** number of episodes.  If the dataset already has some
        episodes the collector only records the remainder.
    hz : float
        Target collection frequency (default 30).
    cam_mode : str
        ``"rgb"`` or ``"rgbd"`` (default ``"rgbd"``).
    image_size : tuple[int, int]
        Target (W, H) for saved images.
    task : str
        Task description string saved to metadata.
    action_mode : str
        Action semantics for ``action_left`` / ``action_right``.
    save_video : bool
        If True, save per-episode MP4 videos alongside the dataset.
    video_fps : float
        FPS for saved videos (default uses ``hz``).
    """

    def __init__(
        self,
        env,
        action_source: Callable[[], Dict[str, Any]],
        dataset_path: str = "datasets/demo.zarr",
        num_episodes: int = 3,
        hz: float = 30.0,
        cam_mode: str = "rgbd",
        image_size: tuple[int, int] = (640, 480),
        task: str = "",
        action_mode: str = "absolute_joint",
        save_video: bool = False,
        video_fps: float | None = None,
    ):
        self.env = env
        self.action_source = action_source

        self.dataset_path = dataset_path
        self.num_episodes = num_episodes
        self.hz = hz
        self.cam_mode = cam_mode
        self.image_w, self.image_h = image_size
        self.task = task
        self.action_mode = action_mode
        self.save_video = save_video
        self.video_fps = video_fps or hz

        self._save_depth = cam_mode == "rgbd"

    # ------------------------------------------------------------------
    # Helpers: disk usage, banner, summaries
    # ------------------------------------------------------------------

    @staticmethod
    def _get_disk_usage(path: str) -> str:
        total = 0
        for dirpath, _dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
        if total < 1024:
            return f"{total} B"
        elif total < 1024 ** 2:
            return f"{total / 1024:.1f} KB"
        elif total < 1024 ** 3:
            return f"{total / 1024 ** 2:.1f} MB"
        else:
            return f"{total / 1024 ** 3:.2f} GB"

    def _print_banner(self, start_ep: int, total_steps: int, target_new: int):
        disk = self._get_disk_usage(self.dataset_path) if os.path.exists(self.dataset_path) else "0 B"
        lines = [
            "",
            "┌──────────────────────────────────────────────────────┐",
            "│          ARX Toolkit — Data Collector                │",
            "├──────────────────────────────────────────────────────┤",
            f"│  Dataset  : {self.dataset_path}",
            f"│  Task     : {self.task or '(未指定)'}",
            f"│  Action   : {self.action_mode}",
            f"│  Cam mode : {self.cam_mode}    Image: {self.image_w}×{self.image_h}",
            f"│  Hz       : {self.hz}    Video: {'ON' if self.save_video else 'OFF'}",
            "├──────────────────────────────────────────────────────┤",
        ]
        if start_ep > 0:
            lines.append(f"│  续采模式 : 已有 {start_ep} episodes, {total_steps:,} steps")
            lines.append(f"│  磁盘占用 : {disk}")
        else:
            lines.append("│  新建数据集")
        lines.append(f"│  本次采集 : {target_new} episodes (目标总数 {self.num_episodes})")
        lines.append("├──────────────────────────────────────────────────────┤")
        lines.append("│  Space: 开始录制 | Enter: 结束并确认 | Ctrl+C: 退出 │")
        lines.append("└──────────────────────────────────────────────────────┘")
        lines.append("")
        print("\r\n".join(lines) + "\r\n")

    def _print_episode_summary(
        self,
        current_ep: int,
        episodes_saved: int,
        data: zarr.Group,
        stats: EpisodeStats,
        target_new: int,
    ):
        total_episodes = current_ep + 1
        total_steps = len(data["episode"]) if "episode" in data else 0
        disk = self._get_disk_usage(self.dataset_path)
        remain = target_new - episodes_saved

        lines = [
            "",
            f"  ┌─ Episode {current_ep} ────────────────────────",
            f"  │ Steps   : {stats.steps} ({stats.duration:.1f}s, avg {stats.fps:.1f} FPS)",
            f"  │ Dataset : {total_episodes} episodes, {total_steps:,} total steps",
            f"  │ Disk    : {disk}",
            f"  │ Remain  : {remain} episodes",
            f"  └─────────────────────────────────────",
            "",
        ]
        print("\r\n".join(lines) + "\r\n")

    def _print_final_summary(self, episodes_saved: int, data: zarr.Group, start_ep: int):
        total_episodes = start_ep + episodes_saved
        total_steps = len(data["episode"]) if "episode" in data else 0
        disk = self._get_disk_usage(self.dataset_path) if os.path.exists(self.dataset_path) else "0 B"

        lines = [
            "",
            "╔══════════════════════════════════════════════════════╗",
            "║               采集结束 — 最终总结                    ║",
            "╠══════════════════════════════════════════════════════╣",
            f"║  本次新增 : {episodes_saved} episodes",
            f"║  数据总量 : {total_episodes} episodes, {total_steps:,} steps",
            f"║  磁盘占用 : {disk}",
            f"║  路径     : {self.dataset_path}",
            "╚══════════════════════════════════════════════════════╝",
            "",
        ]
        print("\r\n".join(lines) + "\r\n")

    # ------------------------------------------------------------------
    # Main collection loop
    # ------------------------------------------------------------------

    def run(self):
        """Run the collection session (blocking)."""
        image_shape = (3, self.image_h, self.image_w)

        config_snapshot = {
            "task": self.task,
            "hz": self.hz,
            "cam_mode": self.cam_mode,
            "action_mode": self.action_mode,
            "image_size": [self.image_w, self.image_h],
            "num_episodes": self.num_episodes,
        }

        data, meta, start_ep = _open_or_create_zarr(
            self.dataset_path,
            image_shape=image_shape,
            cam_mode=self.cam_mode,
            config=config_snapshot,
        )

        # How many new episodes to record
        target_new = max(0, self.num_episodes - start_ep)
        if target_new == 0:
            logger.info("Dataset already has %d episodes (target %d). Nothing to do.",
                        start_ep, self.num_episodes)
            return

        total_steps = len(data["episode"]) if "episode" in data and len(data["episode"]) > 0 else 0
        self._print_banner(start_ep, total_steps, target_new)

        kb = _KeyboardListener()
        episodes_saved = 0

        try:
            kb.start()

            while episodes_saved < target_new:
                current_ep = start_ep + episodes_saved

                try:
                    stats = self._run_episode(
                        current_ep=current_ep,
                        saved_idx=episodes_saved,
                        target_new=target_new,
                        data=data,
                        kb=kb,
                    )

                    if stats is not None:
                        episodes_saved += 1
                        _compute_episode_ends(data, meta)
                        self._print_episode_summary(
                            current_ep, episodes_saved, data, stats, target_new,
                        )

                except KeyboardInterrupt:
                    print("\r\n\r\n  ⚠ Ctrl+C 检测到，正在退出...\r\n")
                    break

                except Exception:
                    err_msg = traceback.format_exc()
                    print(
                        f"\r\n  ✗ Episode {current_ep} 出错，数据已丢弃:\r\n"
                        f"  {err_msg}\r\n"
                    )
                    print("  按 Space 继续下一个 episode | Ctrl+C 退出\r\n")
                    while True:
                        key = kb.get_key()
                        if key == " ":
                            break
                        elif key == "\x03":
                            print("\r\n  ⚠ Ctrl+C 检测到，正在退出...\r\n")
                            raise KeyboardInterrupt
                        time.sleep(0.05)
                    continue

        except KeyboardInterrupt:
            pass

        finally:
            kb.stop()
            self._print_final_summary(episodes_saved, data, start_ep)

    # ------------------------------------------------------------------
    # Single episode recording
    # ------------------------------------------------------------------

    def _run_episode(
        self,
        current_ep: int,
        saved_idx: int,
        target_new: int,
        data: zarr.Group,
        kb: _KeyboardListener,
    ) -> EpisodeStats | None:
        """Record one episode. Returns EpisodeStats or None if 0 steps."""
        print(
            f"\r\n=== Episode {current_ep} ({saved_idx + 1}/{target_new}) ===\r\n"
            f"  Space: start recording | Enter: end & confirm | Ctrl+C: abort\r\n"
        )

        # --- Wait for start (space key) ---
        while True:
            key = kb.get_key()
            if key == " ":
                break
            elif key == "\x03":
                raise KeyboardInterrupt
            time.sleep(0.05)

        print(f"\r\n  ● 录制 episode {current_ep}... (Enter 结束)\r\n")

        dt = 1.0 / self.hz

        # --- Video frame buffer (optional) ---
        video_frames: dict[str, list[np.ndarray]] = {}
        video_dir = ""
        if self.save_video:
            base = os.path.splitext(self.dataset_path)[0]
            video_dir = f"{base}_videos"
            os.makedirs(video_dir, exist_ok=True)
            for cam in _CAM_NAMES:
                video_frames[cam] = []
            logger.info("Video buffering enabled: will save after confirmation.")

        # --- Buffer ---
        buffer: dict[str, list] = {
            "left_eef_pos": [], "left_joint_pos": [],
            "right_eef_pos": [], "right_joint_pos": [],
            "base_height": [],
            "action_left": [], "action_right": [],
            "action_base": [], "action_lift": [],
            "timestamp": [], "episode": [],
        }
        for cam in _CAM_NAMES:
            buffer[f"rgb_{cam}"] = []
        if self._save_depth:
            for cam in _CAM_NAMES:
                buffer[f"depth_{cam}"] = []

        steps = 0
        t_start = time.time()

        while True:
            t_loop = time.monotonic()

            key = kb.get_key()
            if key == "\r":
                print(f"\r\n  Episode {current_ep} ended.\r\n")
                break
            elif key == "\x03":
                raise KeyboardInterrupt

            # 1. Observation
            obs = self.env.get_observation()

            # 2. Action from source
            action = self.action_source()

            # 3. Extract images (with resize)
            for cam in _CAM_NAMES:
                color_key = f"{cam}_color"
                rgb = obs.get(color_key)
                if rgb is not None:
                    rgb = cv2.resize(rgb, (self.image_w, self.image_h))
                else:
                    rgb = np.zeros((self.image_h, self.image_w, 3), dtype=np.uint8)

                # Video frame: buffer in memory, write on confirm-save.
                if cam in video_frames:
                    video_frames[cam].append(rgb.copy())

                # Store as (3, H, W) uint8
                buffer[f"rgb_{cam}"].append(rgb.transpose(2, 0, 1)[None])

                # Depth
                if self._save_depth:
                    depth_key = f"{cam}_aligned_depth_to_color"
                    depth = obs.get(depth_key)
                    if depth is not None:
                        depth = cv2.resize(
                            depth, (self.image_w, self.image_h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    else:
                        depth = np.zeros((self.image_h, self.image_w), dtype=np.uint16)
                    buffer[f"depth_{cam}"].append(depth[None, None])  # (1, 1, H, W)

            # 4. Robot state
            buffer["left_eef_pos"].append(
                obs.get("left_eef_pos", np.zeros(7, dtype=np.float32)).astype(np.float32)[None]
            )
            buffer["left_joint_pos"].append(
                obs.get("left_joint_pos", np.zeros(7, dtype=np.float32)).astype(np.float32)[None]
            )
            buffer["right_eef_pos"].append(
                obs.get("right_eef_pos", np.zeros(7, dtype=np.float32)).astype(np.float32)[None]
            )
            buffer["right_joint_pos"].append(
                obs.get("right_joint_pos", np.zeros(7, dtype=np.float32)).astype(np.float32)[None]
            )
            buffer["base_height"].append(
                obs.get("base_height", np.zeros(1, dtype=np.float32)).astype(np.float32).reshape(1, 1)
            )

            # 5. Actions (extract from action dict, default to zeros)
            action_left = action.get("left") if action else None
            action_right = action.get("right") if action else None
            action_base = action.get("base") if action else None
            action_lift = action.get("lift") if action else None

            buffer["action_left"].append(
                np.asarray(action_left if action_left is not None else np.zeros(7),
                           dtype=np.float32).reshape(1, 7)
            )
            buffer["action_right"].append(
                np.asarray(action_right if action_right is not None else np.zeros(7),
                           dtype=np.float32).reshape(1, 7)
            )
            buffer["action_base"].append(
                np.asarray(action_base if action_base is not None else np.zeros(3),
                           dtype=np.float32).reshape(1, 3)
            )
            buffer["action_lift"].append(
                np.array([[float(action_lift) if action_lift is not None else 0.0]],
                         dtype=np.float32)
            )

            # 6. Timestamp & episode
            buffer["timestamp"].append(np.array([time.time()], dtype=np.float64))
            buffer["episode"].append(np.array([current_ep], dtype=np.uint16))

            steps += 1

            # Print stats every 100 steps
            if steps % 100 == 0:
                elapsed = time.time() - t_start
                fps = steps / elapsed if elapsed > 0 else 0
                print(
                    f"\r\n  Step {steps:5d} | FPS {fps:.1f} | "
                    f"L_grip {obs.get('left_joint_pos', np.zeros(7))[6]:.2f} | "
                    f"R_grip {obs.get('right_joint_pos', np.zeros(7))[6]:.2f}\r\n"
                )

            # Hz limiting
            elapsed_loop = time.monotonic() - t_loop
            sleep_time = dt - elapsed_loop
            if sleep_time > 0:
                time.sleep(sleep_time)

        # --- Save buffer ---
        if steps == 0:
            logger.warning("Episode %d: 0 steps, skipping save.", current_ep)
            print(f"\r\n  Episode {current_ep}: 跳过 (0 steps)\r\n")
            return None

        duration = time.time() - t_start
        fps = steps / duration if duration > 0 else 0

        print(
            f"\r\n  Episode {current_ep} 录制完成: {steps} steps "
            f"({duration:.1f}s, avg {fps:.1f} FPS)\r\n"
            "  [S] 保存并落盘 | [D] 丢弃本段 | [Ctrl+C] 退出\r\n"
        )

        while True:
            key = kb.get_key()
            if key == "s":
                break
            if key == "d":
                print(f"\r\n  Episode {current_ep} 已丢弃，不落盘。\r\n")
                return None
            if key == "\x03":
                raise KeyboardInterrupt
            time.sleep(0.05)

        logger.info("Episode %d: saving %d steps...", current_ep, steps)
        for key, val in buffer.items():
            data[key].append(np.concatenate(val, axis=0))

        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            size = (self.image_w, self.image_h)
            for cam, frames in video_frames.items():
                video_path = f"{video_dir}/ep{current_ep}_{cam}.mp4"
                vw = cv2.VideoWriter(video_path, fourcc, self.video_fps, size)
                try:
                    for frame_rgb in frames:
                        vw.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                finally:
                    vw.release()
            logger.info("Episode %d: video saved to %s/ep%d_*.mp4", current_ep, video_dir, current_ep)

        logger.info("Episode %d saved.", current_ep)
        return EpisodeStats(steps=steps, duration=duration, fps=fps)
