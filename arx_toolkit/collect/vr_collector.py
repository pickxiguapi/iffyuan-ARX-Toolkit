"""VR Zarr collector.

The VR control node drives the robot outside this collector.  This module only
samples ARXEnv public observations and saves the result into the same Zarr
schema used by leader-follower collection.
"""

from __future__ import annotations

import select
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional

import cv2
import numpy as np

from arm_control.msg._pos_cmd import PosCmd

from arx_toolkit.collect.collector import (
    _CAM_NAMES,
    EpisodeStats,
    _compute_episode_ends,
    _open_or_create_zarr,
)
from arx_toolkit.env import ARXEnv


def _poll_stdin_line() -> Optional[str]:
    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not ready:
        return None
    return sys.stdin.readline().strip().lower()


def _default_camera_map(camera_names: Iterable[str]) -> Dict[str, str]:
    return {str(name): str(name) for name in camera_names}


def _vr_base_from_msg(msg) -> np.ndarray:
    return np.array(
        [
            float(getattr(msg, "chx", 0.0)),
            float(getattr(msg, "chy", 0.0)),
            float(getattr(msg, "chz", 0.0)),
            float(getattr(msg, "height", 0.0)),
        ],
        dtype=np.float32,
    )


def _extract_camera_frames(
    obs: Dict[str, np.ndarray],
    camera_map: Dict[str, str],
    include_depth: bool,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    colors: Dict[str, np.ndarray] = {}
    depths: Dict[str, np.ndarray] = {}
    for physical_name, logical_name in camera_map.items():
        color_key = f"{physical_name}_color"
        depth_key = f"{physical_name}_aligned_depth_to_color"
        if color_key in obs:
            colors[logical_name] = np.asarray(obs[color_key])
        if include_depth and depth_key in obs:
            depths[logical_name] = np.asarray(obs[depth_key])
    return colors, depths


def _camera_ready(
    obs: Dict[str, np.ndarray],
    camera_names: Iterable[str],
    camera_type: str,
) -> list[str]:
    missing = []
    for camera_name in camera_names:
        color_key = f"{camera_name}_color"
        depth_key = f"{camera_name}_aligned_depth_to_color"
        if color_key not in obs:
            missing.append(color_key)
        if camera_type == "rgbd" and depth_key not in obs:
            missing.append(depth_key)
    return missing


class VRCommandMirror:
    """Mirror the VR ROS2 PosCmd topics for base command recording."""

    def __init__(
        self,
        left_topic: str = "/ARX_VR_L",
        right_topic: str = "/ARX_VR_R",
    ):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node

        if not rclpy.ok():
            raise RuntimeError("rclpy must be initialized before VRCommandMirror")

        self._lock = threading.Lock()
        self._latest_left = None
        self._latest_right = None
        self._latest_left_stamp = None
        self._latest_right_stamp = None

        class MirrorNode(Node):
            pass

        self.node = MirrorNode("collect_vr_mirror")
        self.node.create_subscription(PosCmd, left_topic, self._on_left, 10)
        self.node.create_subscription(PosCmd, right_topic, self._on_right, 10)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def _on_left(self, msg) -> None:
        with self._lock:
            self._latest_left = msg
            self._latest_left_stamp = time.time()

    def _on_right(self, msg) -> None:
        with self._lock:
            self._latest_right = msg
            self._latest_right_stamp = time.time()

    def snapshot(self) -> tuple[Any, Any, Optional[float], Optional[float]]:
        with self._lock:
            return (
                self._latest_left,
                self._latest_right,
                self._latest_left_stamp,
                self._latest_right_stamp,
            )

    def close(self) -> None:
        self.executor.shutdown()
        self.node.destroy_node()
        self.thread.join(timeout=2.0)


class DualVRZarrCollector:
    """Capture VR-controlled samples from ARXEnv public observations."""

    def __init__(
        self,
        env: ARXEnv,
        camera_names: Iterable[str] = ("camera_h",),
        include_camera: bool = True,
        include_base: bool = False,
        use_depth: bool = False,
        action_mode: Literal["absolute_joint", "absolute_eef"] = "absolute_joint",
        left_vr_topic: str = "/ARX_VR_L",
        right_vr_topic: str = "/ARX_VR_R",
        img_size: tuple[int, int] = (640, 480),
        require_vr: bool = False,
    ):
        self.env = env
        self.include_camera = bool(include_camera)
        self.include_base = bool(include_base)
        self.require_vr = bool(require_vr or include_base)
        self.action_mode = _normalize_vr_action_mode(action_mode)
        self.camera_names = (
            [str(name) for name in camera_names] if self.include_camera else []
        )
        self.camera_type = "rgbd" if use_depth else "rgb"
        self.img_size = tuple(img_size)
        if self.include_camera:
            env_cameras = set(str(name) for name in getattr(self.env, "camera_view", ()))
            missing_cameras = [
                name for name in self.camera_names if name not in env_cameras
            ]
            if missing_cameras:
                raise ValueError(
                    "env.camera_view does not include requested cameras: "
                    f"{missing_cameras}"
                )
            if use_depth and getattr(self.env, "camera_type", "rgb") != "rgbd":
                raise ValueError(
                    "collect requested depth frames, but env.camera_type is not rgbd"
                )
        self.vr_mirror = (
            VRCommandMirror(left_topic=left_vr_topic, right_topic=right_vr_topic)
            if self.require_vr
            else None
        )

    def readiness(self) -> tuple[bool, list[str]]:
        missing = []
        try:
            obs = self.env.get_observation(
                include_arm=True,
                include_camera=self.include_camera,
                include_base=self.include_base,
            )
        except Exception as exc:
            return False, [f"observation: {exc}"]

        if "left_joint_pos" not in obs:
            missing.append("left_arm_status")
        if "right_joint_pos" not in obs:
            missing.append("right_arm_status")
        if self.include_base and "base_height" not in obs:
            missing.append("base_status")
        if self.vr_mirror is not None:
            left_vr, right_vr, _, _ = self.vr_mirror.snapshot()
            if left_vr is None:
                missing.append("vr_left")
            if right_vr is None:
                missing.append("vr_right")
        if self.include_camera:
            missing.extend(_camera_ready(obs, self.camera_names, self.camera_type))
        return len(missing) == 0, missing

    def wait_until_ready(self) -> None:
        last_report = 0.0
        while True:
            ready, missing = self.readiness()
            if ready:
                return
            now = time.time()
            if now - last_report > 1.0:
                print(f"Waiting for streams: {', '.join(missing)}")
                last_report = now
            time.sleep(0.2)

    def capture_sample(
        self,
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        try:
            obs = self.env.get_observation(
                include_arm=True,
                include_camera=self.include_camera,
                include_base=self.include_base,
            )
        except Exception as exc:
            return None, f"observation not ready: {exc}"

        if "left_joint_pos" not in obs or "right_joint_pos" not in obs:
            return None, "arm status not ready"

        color_frames, depth_frames = _extract_camera_frames(
            obs,
            camera_map=_default_camera_map(self.camera_names),
            include_depth=self.camera_type == "rgbd",
        )
        expected_cameras = set(_default_camera_map(self.camera_names).values())
        if self.include_camera and set(color_frames.keys()) != expected_cameras:
            return None, "camera color frames not ready"
        if (
            self.include_camera
            and self.camera_type == "rgbd"
            and set(depth_frames.keys()) != expected_cameras
        ):
            return None, "camera depth frames not ready"

        left_vr = right_vr = None
        if self.vr_mirror is not None:
            left_vr, right_vr, _left_stamp, _right_stamp = self.vr_mirror.snapshot()
            if left_vr is None or right_vr is None:
                return None, "vr topics not ready"

        left_qpos = np.asarray(obs["left_joint_pos"], dtype=np.float32).reshape(-1)[:7]
        right_qpos = np.asarray(obs["right_joint_pos"], dtype=np.float32).reshape(-1)[:7]
        left_qvel = np.asarray(
            obs.get("left_joint_qvel", np.zeros(7, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)[:7]
        right_qvel = np.asarray(
            obs.get("right_joint_qvel", np.zeros(7, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)[:7]
        left_effort = np.asarray(
            obs.get("left_joint_effort", np.zeros(7, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)[:7]
        right_effort = np.asarray(
            obs.get("right_joint_effort", np.zeros(7, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)[:7]
        left_eef = np.asarray(obs["left_eef_pos"], dtype=np.float32).reshape(-1)[:7]
        right_eef = np.asarray(obs["right_eef_pos"], dtype=np.float32).reshape(-1)[:7]
        action_base = _vr_base_from_msg(left_vr) if self.include_base else None

        action_left = (
            left_qpos.copy()
            if self.action_mode == "absolute_joint"
            else left_eef.copy()
        )
        action_right = (
            right_qpos.copy()
            if self.action_mode == "absolute_joint"
            else right_eef.copy()
        )
        action_lift = 0.0
        if self.include_base:
            action_lift = float(action_base[3]) if action_base is not None else 0.0

        return {
            "timestamp": float(time.time()),
            "images": color_frames if self.include_camera else {},
            "images_depth": depth_frames if self.include_camera else {},
            "left_eef_pos": left_eef,
            "left_joint_pos": left_qpos,
            "left_joint_qvel": left_qvel,
            "left_joint_effort": left_effort,
            "right_eef_pos": right_eef,
            "right_joint_pos": right_qpos,
            "right_joint_qvel": right_qvel,
            "right_joint_effort": right_effort,
            "base_height": np.asarray(
                obs.get("base_height", np.zeros(1, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)[:1],
            "action_left": action_left,
            "action_right": action_right,
            "action_base": (
                np.asarray(action_base[:3], dtype=np.float32)
                if action_base is not None
                else np.zeros(3, dtype=np.float32)
            ),
            "action_lift": float(action_lift),
        }, None

    def close(self) -> None:
        if self.vr_mirror is not None:
            self.vr_mirror.close()


def _normalize_vr_action_mode(
    action_mode: str,
) -> Literal["absolute_joint", "absolute_eef"]:
    mode = str(action_mode).strip().lower()
    if mode in {"absolute_joint", "absolute_eef"}:
        return mode  # type: ignore[return-value]
    raise ValueError(
        "VR collect only supports action_mode='absolute_joint' or 'absolute_eef'"
    )


def _normalize_camera_names(camera_names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    names = tuple(str(name) for name in camera_names)
    if not names:
        raise ValueError("camera_names cannot be empty")
    return names


def _countdown(seconds: int = 5, use_tts: bool = True) -> None:
    spd_say = shutil.which("spd-say") if use_tts else None
    for i in range(int(seconds), 0, -1):
        if spd_say:
            subprocess.Popen([spd_say, str(i)])
        print(f"{i}...")
        time.sleep(1.0)
    if spd_say:
        subprocess.Popen([spd_say, "go"])
    print("Go!")


def _new_zarr_episode_buffer(save_depth: bool) -> dict[str, list]:
    buffer: dict[str, list] = {
        "left_eef_pos": [], "left_joint_pos": [],
        "left_joint_qvel": [], "left_joint_effort": [],
        "right_eef_pos": [], "right_joint_pos": [],
        "right_joint_qvel": [], "right_joint_effort": [],
        "base_height": [],
        "action_left": [], "action_right": [],
        "action_base": [], "action_lift": [],
        "timestamp": [], "episode": [],
    }
    for cam in _CAM_NAMES:
        buffer[f"rgb_{cam}"] = []
    if save_depth:
        for cam in _CAM_NAMES:
            buffer[f"depth_{cam}"] = []
    return buffer


def _append_vr_sample(
    buffer: dict[str, list],
    sample: dict[str, Any],
    episode_idx: int,
    image_size: tuple[int, int],
    save_depth: bool,
    action_side: str | None = None,
) -> None:
    image_w, image_h = image_size
    images = sample["images"]
    images_depth = sample["images_depth"]

    for cam in _CAM_NAMES:
        rgb = images.get(cam)
        if rgb is not None:
            rgb = cv2.resize(rgb, (image_w, image_h))
        else:
            rgb = np.zeros((image_h, image_w, 3), dtype=np.uint8)
        buffer[f"rgb_{cam}"].append(rgb.transpose(2, 0, 1)[None])

        if save_depth:
            depth = images_depth.get(cam)
            if depth is not None:
                depth = cv2.resize(
                    depth,
                    (image_w, image_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                depth = np.zeros((image_h, image_w), dtype=np.uint16)
            buffer[f"depth_{cam}"].append(depth[None, None])

    for key in (
        "left_eef_pos", "left_joint_pos", "left_joint_qvel", "left_joint_effort",
        "right_eef_pos", "right_joint_pos", "right_joint_qvel", "right_joint_effort",
    ):
        buffer[key].append(
            np.asarray(sample[key], dtype=np.float32).reshape(1, 7)
        )
    buffer["base_height"].append(
        np.asarray(sample["base_height"], dtype=np.float32).reshape(1, 1)
    )

    action_left = np.asarray(sample["action_left"], dtype=np.float32).reshape(7)
    action_right = np.asarray(sample["action_right"], dtype=np.float32).reshape(7)
    if action_side == "left":
        action_right = np.zeros(7, dtype=np.float32)
    elif action_side == "right":
        action_left = np.zeros(7, dtype=np.float32)

    buffer["action_left"].append(action_left.reshape(1, 7))
    buffer["action_right"].append(action_right.reshape(1, 7))
    buffer["action_base"].append(
        np.asarray(sample["action_base"], dtype=np.float32).reshape(1, 3)
    )
    buffer["action_lift"].append(
        np.array([[float(sample["action_lift"])]], dtype=np.float32)
    )
    buffer["timestamp"].append(np.array([sample["timestamp"]], dtype=np.float64))
    buffer["episode"].append(np.array([episode_idx], dtype=np.uint16))


def _write_zarr_episode(data, buffer: dict[str, list]) -> None:
    for key, values in buffer.items():
        data[key].append(np.concatenate(values, axis=0))


def record_episode_interactive(
    collector: DualVRZarrCollector,
    data,
    episode_idx: int,
    frame_rate: float,
    image_size: tuple[int, int],
    save_depth: bool,
    max_frames: int = 0,
    action_side: str | None = None,
) -> tuple[EpisodeStats | None, bool]:
    period = 1.0 / max(float(frame_rate), 1e-6)
    quit_requested = False
    last_error_report = 0.0
    print("Recording started. Press Enter to stop.")

    buffer = _new_zarr_episode_buffer(save_depth=save_depth)
    steps = 0
    t_start = time.time()
    while True:
        if max_frames > 0 and steps >= int(max_frames):
            print("Reached max frame count.")
            break
        loop_start = time.perf_counter()
        command = _poll_stdin_line()
        if command == "":
            break

        sample, error = collector.capture_sample()
        if sample is None:
            now = time.time()
            if error and now - last_error_report > 1.0:
                print(f"Skipping frame: {error}")
                last_error_report = now
        else:
            _append_vr_sample(
                buffer,
                sample,
                episode_idx=episode_idx,
                image_size=image_size,
                save_depth=save_depth,
                action_side=action_side,
            )
            steps += 1
            if steps % 20 == 0:
                print(f"Captured {steps} frames")

        sleep_need = period - (time.perf_counter() - loop_start)
        if sleep_need > 0.0:
            time.sleep(sleep_need)

    if steps == 0:
        return None, quit_requested

    duration = time.time() - t_start
    fps = steps / duration if duration > 0 else 0.0
    print(
        f"Episode {episode_idx} recorded: {steps} steps "
        f"({duration:.1f}s, avg {fps:.1f} FPS)"
    )

    while True:
        save_choice = input("Save episode? [y] save / [n] discard / [q] quit: ").strip().lower()
        if save_choice in {"", "y"}:
            _write_zarr_episode(data, buffer)
            return EpisodeStats(steps=steps, duration=duration, fps=fps), quit_requested
        if save_choice == "n":
            print("Episode discarded.")
            return None, quit_requested
        if save_choice == "q":
            print("Episode discarded. Quit requested.")
            return None, True
        print("Invalid choice. Please enter y, n, or q.")


def collect_vr_episode(
    env: ARXEnv,
    arm_mode: str = "dual",
    dataset_path: Path | str = "datasets/vr.zarr",
    frame_rate: float = 20.0,
    max_frames: int = 0,
    max_episodes: int = 0,
    action_mode: str = "absolute_joint",
    camera_names: tuple[str, ...] = ("camera_h",),
    with_depth: bool = False,
    img_size: tuple[int, int] = (640, 480),
    task: str = "",
    leader_side: str = "left",
    include_base: bool = False,
    countdown_sec: int = 5,
    use_tts: bool = True,
) -> Path | None:
    """Interactively record VR-controlled episodes into a Zarr dataset."""
    arm_mode = str(arm_mode).strip().lower()
    action_mode = _normalize_vr_action_mode(action_mode)
    camera_names = _normalize_camera_names(camera_names)
    dataset_path = Path(dataset_path)
    saved_episodes = 0

    config_snapshot = {
        "task": task,
        "hz": frame_rate,
        "cam_mode": "rgbd" if with_depth else "rgb",
        "collection_kind": f"vr_{arm_mode}",
        "action_mode": action_mode,
        "leader_side": leader_side if arm_mode == "single" else None,
        "include_base": include_base,
        "camera_names": list(camera_names),
        "image_size": list(img_size),
        "max_episodes": max_episodes,
    }
    data, meta, start_ep = _open_or_create_zarr(
        str(dataset_path),
        image_shape=(3, int(img_size[1]), int(img_size[0])),
        cam_mode="rgbd" if with_depth else "rgb",
        config=config_snapshot,
    )

    collector = DualVRZarrCollector(
        env=env,
        camera_names=camera_names,
        include_camera=True,
        include_base=(arm_mode == "dual" and include_base),
        use_depth=with_depth,
        action_mode=action_mode,
        left_vr_topic="/ARX_VR_L",
        right_vr_topic="/ARX_VR_R",
        img_size=img_size,
        require_vr=(arm_mode == "dual" and include_base),
    )

    if arm_mode == "single":
        start_prompt = "Press Enter to start VR single-arm recording, or type 'q' to quit: "
    elif arm_mode == "dual":
        start_prompt = "Press Enter to start VR dual-arm recording, or type 'q' to quit: "
    else:
        raise ValueError("arm_mode must be 'single' or 'dual'")

    try:
        collector.wait_until_ready()
        while True:
            if max_episodes > 0 and saved_episodes >= int(max_episodes):
                print(f"Reached max saved episodes: {saved_episodes}")
                return dataset_path

            command = input(start_prompt).strip().lower()
            if command == "q":
                return dataset_path
            if command != "":
                print("Invalid choice. Press Enter to start or 'q' to quit.")
                continue

            print("Get ready...")
            if countdown_sec > 0:
                _countdown(countdown_sec, use_tts=use_tts)

            current_ep = start_ep + saved_episodes
            stats, quit_requested = record_episode_interactive(
                collector=collector,
                data=data,
                episode_idx=current_ep,
                frame_rate=frame_rate,
                image_size=img_size,
                save_depth=with_depth,
                max_frames=max_frames,
                action_side=leader_side if arm_mode == "single" else None,
            )

            if stats is None:
                print("Episode not saved.")
                if quit_requested:
                    return dataset_path
                continue

            saved_episodes += 1
            _compute_episode_ends(data, meta)
            print(
                f"Saved episode {current_ep} to {dataset_path} "
                f"({stats.steps} steps, avg {stats.fps:.1f} FPS)"
            )

            if quit_requested:
                print("Quit requested.")
                return dataset_path
    finally:
        collector.close()
