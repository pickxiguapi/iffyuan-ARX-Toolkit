"""ROS2 communication layer for ARX LIFT2.

Handles all ROS2 pub/sub, camera synchronization and optional video saving.
This is the low-level IO that ARXEnv delegates to — user code should
NOT import this module directly.

Ported and simplified from the reference arx_ros2_env_utils.py.
"""

from __future__ import annotations

import os
import time
import threading
import queue
from typing import Dict, Iterable, Literal, Optional

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image

from arm_control.msg._pos_cmd import PosCmd
from arx5_arm_msg.msg._robot_cmd import RobotCmd
from arx5_arm_msg.msg._robot_status import RobotStatus

import cv2
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer


# ---------------------------------------------------------------------------
# RobotIO Node
# ---------------------------------------------------------------------------


class RobotIO(Node):
    """Single ROS2 node that handles all arm / base / camera communication.

    Video saving is supported but **off by default**. Pass ``save_video=True``
    together with ``save_dir`` to enable continuous background recording.
    """

    def __init__(
        self,
        camera_type: Literal["rgb", "rgbd"] = "rgbd",
        camera_view: Iterable[str] = ("camera_l", "camera_h"),
        target_size: Optional[tuple[int, int]] = None,
        save_video: bool = False,
        video_fps: float = 20.0,
        save_dir: Optional[str] = None,
        video_name: Optional[str] = None,
    ):
        super().__init__("robot_io")
        self.bridge = CvBridge()
        self.default_target_size = tuple(target_size) if target_size else None

        # ---- Publishers ----
        self.cmd_pub_l = self.create_publisher(RobotCmd, "arm_cmd_l", 5)
        self.cmd_pub_r = self.create_publisher(RobotCmd, "arm_cmd_r", 5)
        self.cmd_pub_base = self.create_publisher(PosCmd, "/ARX_VR_L", 5)

        # ---- State storage ----
        self.latest_status: Dict[str, Optional[RobotStatus]] = {
            "left": None, "right": None,
        }
        self.latest_base: Optional[PosCmd] = None
        self.status_snapshot: Optional[Dict] = None
        self.status_lock = threading.Lock()

        # ---- Subscribers: arm status ----
        self.create_subscription(
            RobotStatus, "arm_status_l",
            lambda msg: self._on_status("left", msg), 5,
        )
        self.create_subscription(
            RobotStatus, "arm_status_r",
            lambda msg: self._on_status("right", msg), 5,
        )
        self.create_subscription(PosCmd, "body_information", self._on_base_status, 1)

        # ---- Video saving (off by default) ----
        self.save_video = bool(save_video)
        self.video_fps = float(video_fps)
        self.save_dir = os.fspath(save_dir) if save_dir else None
        self.video_name = str(video_name).strip() if video_name else None
        self.continuous_video = bool(self.save_video and self.save_dir)
        self.video_writers: Dict[tuple[str, str], cv2.VideoWriter] = {}
        self.video_shapes: Dict[tuple[str, str], tuple[int, int]] = {}

        self.save_queue: queue.Queue = queue.Queue()
        self.saver_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.saver_thread.start()

        # ---- Camera subscriptions ----
        self.camera_view = list(camera_view) if camera_view else []
        self.cam_lock = threading.Lock()
        self.latest_images: Dict[str, Image] = {}

        subs: list[Subscriber] = []
        labels: list[str] = []
        types = (
            ["color", "aligned_depth_to_color"]
            if camera_type == "rgbd"
            else ["color"]
        )
        for cam in self.camera_view:
            for typ in types:
                if "aligned" in typ:
                    topic = f"/{cam}_namespace/{cam}/aligned_depth_to_color/image_raw"
                else:
                    topic = f"/{cam}_namespace/{cam}/{typ}/image_rect_raw"
                subs.append(Subscriber(self, Image, topic, qos_profile=5))
                labels.append(f"{cam}_{typ}")

        self.labels = labels
        if subs:
            self.sync = ApproximateTimeSynchronizer(subs, queue_size=5, slop=0.02)
            self.sync.registerCallback(self._on_images)
            self.get_logger().info(
                f"Camera topics: {[s.topic for s in subs]}"
            )
        else:
            self.get_logger().warn("No camera subscriptions configured.")

    # ---- Callbacks ----

    def _on_status(self, side: str, msg: RobotStatus):
        with self.status_lock:
            self.latest_status[side] = msg

    def _on_base_status(self, msg: PosCmd):
        with self.status_lock:
            self.latest_base = msg

    def _on_images(self, *msgs):
        with self.status_lock:
            snap = dict(self.latest_status)
            snap["base"] = self.latest_base
            self.status_snapshot = snap

        with self.cam_lock:
            for label, msg in zip(self.labels, msgs):
                self.latest_images[label] = msg

        # Continuous video recording (only when enabled)
        if self.continuous_video:
            for label, msg in zip(self.labels, msgs):
                img = self._decode(label, msg, self.default_target_size)
                if img is None:
                    continue
                ts = self._stamp(msg)
                self.save_queue.put((self.save_dir, label, ts, img, True))

    # ---- Public send ----

    def send_control_msg(self, side: str, cmd: RobotCmd) -> bool:
        pub = self.cmd_pub_l if side == "left" else self.cmd_pub_r
        if not rclpy.ok():
            return False
        if pub.get_subscription_count() == 0:
            self.get_logger().warn(f"{side} no subscribers")
            return False
        pub.publish(cmd)
        return True

    def send_base_msg(self, cmd: PosCmd) -> bool:
        if not rclpy.ok():
            return False
        if self.cmd_pub_base.get_subscription_count() == 0:
            self.get_logger().warn("No base subscribers")
            return False
        self.cmd_pub_base.publish(cmd)
        return True

    # ---- Public read ----

    def get_robot_status(self) -> Dict:
        with self.status_lock:
            status = dict(self.latest_status)
            status["base"] = self.latest_base
            return status

    def get_camera(
        self,
        target_size: Optional[tuple[int, int]] = None,
        return_status: bool = False,
    ):
        """Return latest synced camera frames as numpy arrays.

        Returns:
            frames (dict[str, np.ndarray]): e.g. {"camera_h_color": (H,W,3)}
            If return_status=True: (frames, status_snapshot)
        """
        size = target_size or self.default_target_size
        frames: Dict[str, np.ndarray] = {}

        with self.cam_lock:
            items = list(self.latest_images.items())

        for key, msg in items:
            img = self._decode(key, msg, size)
            if img is not None:
                frames[key] = img

        if return_status:
            with self.status_lock:
                snap = (
                    dict(self.status_snapshot)
                    if self.status_snapshot is not None
                    else dict(self.latest_status)
                )
                if "base" not in snap:
                    snap["base"] = self.latest_base
            return frames, snap
        return frames

    # ---- Image helpers ----

    def _decode(self, key: str, msg: Image, target_size) -> Optional[np.ndarray]:
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if "depth" not in key:
                img = img[:, :, ::-1]  # BGR -> RGB
            if target_size:
                img = cv2.resize(img, target_size)
            return img
        except Exception as e:
            self.get_logger().warn(f"decode {key} failed: {e}")
            return None

    @staticmethod
    def _stamp(msg) -> float:
        header = getattr(msg, "header", None)
        if header:
            return header.stamp.sec + header.stamp.nanosec * 1e-9
        return time.time()

    # ---- Save worker (background thread) ----

    def _save_worker(self):
        while True:
            task = self.save_queue.get()
            if task is None:
                self.save_queue.task_done()
                break
            save_dir, key, ts, img, as_video = task
            try:
                os.makedirs(save_dir, exist_ok=True)
                if as_video:
                    self._save_video_frame(save_dir, key, img)
                else:
                    base_path = os.path.join(save_dir, f"{key}_{ts}")
                    if "depth" in key:
                        np.save(base_path + ".npy", img)
                    else:
                        cv2.imwrite(
                            base_path + ".png", img,
                            [cv2.IMWRITE_PNG_COMPRESSION, 0],
                        )
            except Exception as e:
                self.get_logger().warn(f"save {key} failed: {e}")
            finally:
                self.save_queue.task_done()
        self._release_video_writers()

    def _save_video_frame(self, save_dir: str, key: str, img: np.ndarray):
        # Prepare frame: depth -> grayscale vis, ensure uint8 BGR
        frame = img
        if "depth" in key:
            depth_f = img.astype(np.float32)
            finite = depth_f[np.isfinite(depth_f)]
            vmax = float(np.percentile(finite, 99)) if finite.size > 0 else 1.0
            vmax = max(vmax, 1e-6)
            frame = cv2.convertScaleAbs(depth_f, alpha=255.0 / vmax)
        if frame.dtype != np.uint8:
            frame = cv2.convertScaleAbs(frame)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)

        h, w = frame.shape[:2]
        writer_key = (save_dir, key)
        writer = self.video_writers.get(writer_key)
        prev_shape = self.video_shapes.get(writer_key)
        # Rebuild writer if resolution changed
        if writer is not None and prev_shape != (w, h):
            writer.release()
            writer = None
        if writer is None:
            fname = f"{self.video_name}_{key}.mp4" if self.video_name else f"{key}.mp4"
            path = os.path.join(save_dir, fname)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(path, fourcc, self.video_fps, (w, h), True)
            self.video_writers[writer_key] = writer
            self.video_shapes[writer_key] = (w, h)
        writer.write(frame)

    def _release_video_writers(self):
        for w in self.video_writers.values():
            try:
                w.release()
            except Exception:
                pass
        self.video_writers.clear()
        self.video_shapes.clear()

    def stop_saver(self):
        """Signal the save worker to finish and wait. Safe to call multiple times."""
        if not getattr(self, "_saver_stopped", False):
            self._saver_stopped = True
            self.save_queue.put(None)
            if getattr(self, "saver_thread", None) is not None:
                self.saver_thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def start_robot_io(
    camera_type="rgbd",
    camera_view=("camera_l", "camera_h"),
    target_size=None,
    save_video=False,
    video_fps=20.0,
    save_dir=None,
    video_name=None,
):
    """Create RobotIO node + executor, spin in background thread."""
    node = RobotIO(
        camera_type=camera_type,
        camera_view=camera_view,
        target_size=target_size,
        save_video=save_video,
        video_fps=video_fps,
        save_dir=save_dir,
        video_name=video_name,
    )
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    t = threading.Thread(target=executor.spin, daemon=True)
    t.start()
    return node, executor, t


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------


def build_observation(
    camera_all: Dict[str, np.ndarray],
    status_all: Optional[Dict],
    include_arm: bool = True,
    include_camera: bool = True,
    include_base: bool = True,
) -> Dict[str, np.ndarray]:
    """Pack status + camera frames into a flat observation dict."""
    obs: Dict[str, np.ndarray] = {}

    if include_arm and isinstance(status_all, dict):
        for side in ("left", "right"):
            s = status_all.get(side)
            if s is None:
                continue
            obs[f"{side}_eef_pos"] = np.concatenate([
                np.array(s.end_pos, dtype=np.float32),
                np.array([s.joint_pos[6] if len(s.joint_pos) > 6 else 0.0], dtype=np.float32),
            ])  # (7,): [x, y, z, r, p, y, gripper_raw]
            obs[f"{side}_joint_pos"] = np.array(s.joint_pos, dtype=np.float32)

    if include_base and isinstance(status_all, dict):
        base = status_all.get("base")
        if base is not None:
            obs["base_height"] = np.array([base.height], dtype=np.float32)

    if include_camera:
        for key, img in (camera_all or {}).items():
            if img is None:
                continue
            if isinstance(img, np.ndarray):
                obs[key] = img
            else:
                obs[key] = np.asarray(img)

    return obs
