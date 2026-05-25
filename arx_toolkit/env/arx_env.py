"""Unified ARX LIFT2 robot environment.

Design philosophy: one ``step(action, action_mode=...)`` controls the entire robot.

Robot
-----
ARX LIFT2 dual-arm mobile manipulator:
- 2 × 6-DOF arms (left / right), each with 1 gripper
- 3-wheel omnidirectional base chassis
- Lift (vertical linear stage)
- Up to 3 RealSense D405 cameras

Action — ``step(action, action_mode=...) -> obs``
-------------------------------------------------
``action`` is a dict with **4 required keys** (set value to ``None`` to skip):

.. code-block:: python

    action = {
        "left":  np.ndarray(7,) | None,
        "right": np.ndarray(7,) | None,
        "base":  np.ndarray(3,) | None,
        "lift":  float | None,
    }

**Arm action (left / right)** — 7D. ``step(..., action_mode=...)`` selects the
control path explicitly:

+-----------------+-------+--------------------------------------------------+
| action_mode     | dim   | meaning                                          |
+=================+=======+==================================================+
| absolute_joint  | 7     | [j0, j1, j2, j3, j4, j5, gripper]              |
+-----------------+-------+--------------------------------------------------+
| absolute_eef    | 7     | [x, y, z, roll, pitch, yaw, gripper]            |
+-----------------+-------+--------------------------------------------------+
| smooth_eef      | 7     | EEF target, sent through smooth sequence         |
+-----------------+-------+--------------------------------------------------+
| delta_eef       | 7     | [dx, dy, dz, droll, dpitch, dyaw, dg]           |
+-----------------+-------+--------------------------------------------------+

- Position xyz: meters, in base frame.
- Orientation rpy: radians.
- Gripper uses normalized [0, 1] at the ARXEnv API.
  0 = fully open, 1 = fully closed. ARXEnv converts to hardware raw values
  before publishing commands.

**Base action** — 3D velocity command:

+-------+---------+------------------------------------+
| index | name    | range                              |
+=======+=========+====================================+
| 0     | vx      | [-1.5, 1.5]  forward / backward    |
+-------+---------+------------------------------------+
| 1     | vy      | [-1.5, 1.5]  left / right          |
+-------+---------+------------------------------------+
| 2     | vz      | [-2.0, 2.0]  rotation              |
+-------+---------+------------------------------------+

**Lift action** — scalar float:

- height ∈ [0, 20], where 0 = lowest, 20 = highest.

Observation — ``obs = env.reset()`` / ``env.step(action, action_mode=...)``
---------------------------------------------------------------------------
``obs`` is a flat dict. Available keys depend on ``camera_type`` and
``camera_view``:

**Arm state** (per side, always present):

+---------------------+----------+----------------------------------------------+
| key                 | shape    | description                                  |
+=====================+==========+==============================================+
| {side}_eef_pos      | (7,)     | [x, y, z, roll, pitch, yaw, gripper] end-eff  |
+---------------------+----------+----------------------------------------------+
| {side}_joint_pos    | (7,)     | 6 joint angles + gripper [0,1] normalized     |
+---------------------+----------+----------------------------------------------+
| {side}_joint_qvel   | (7,)     | 6 joint velocities + gripper velocity         |
+---------------------+----------+----------------------------------------------+
| {side}_joint_effort | (7,)     | 6 joint currents/efforts + gripper effort     |
+---------------------+----------+----------------------------------------------+

where ``{side}`` is ``left`` or ``right``.

**Base state** (always present):

+---------------------+----------+----------------------------------------------+
| key                 | shape    | description                                  |
+=====================+==========+==============================================+
| base_height         | (1,)     | current lift height [0, 20]                  |
+---------------------+----------+----------------------------------------------+

**Camera images** (depends on ``camera_type`` and ``camera_view``):

+------------------------------------+---------------+---------------------------+
| key pattern                        | shape         | description               |
+====================================+===============+===========================+
| {cam}_color                        | (H, W, 3)    | RGB uint8                 |
+------------------------------------+---------------+---------------------------+
| {cam}_aligned_depth_to_color       | (H, W)       | depth uint16 (mm)         |
+------------------------------------+---------------+---------------------------+

where ``{cam}`` ∈ camera_view, e.g. ``camera_l``, ``camera_h``, ``camera_r``.
Depth images only present when ``camera_type="rgbd"``.

Lifecycle
---------
.. code-block:: python

    env = ARXEnv(camera_type="rgbd")
    obs = env.reset()       # home arms, lift=0, base stop
    obs = env.step(action, action_mode="absolute_eef")
    env.close()             # safe shutdown (also registered via atexit)

``step_arm()`` controls the upper body, and ``step_base_lift()`` controls the
lower body. The primary interface remains ``step(action, action_mode=...)``.
"""

from __future__ import annotations

import atexit
import threading
import time
from typing import Dict, Iterable, Literal, Optional, Tuple

import numpy as np

from arx_toolkit.utils.logger import get_logger
from arx_toolkit.utils.smooth import plan_smooth_eef_sequences
from arx_toolkit.utils.transforms import (
    quat_from_rpy,
    quat_multiply,
    rpy_from_quat,
)

logger = get_logger("arx_toolkit.env")

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ActionMode = Literal["absolute_joint", "absolute_eef", "smooth_eef", "delta_eef"]
Side = Literal["left", "right", "both"]

_VALID_ACTION_MODES: set[str] = {
    "absolute_joint",
    "absolute_eef",
    "smooth_eef",
    "delta_eef",
}

# RobotCmd mode constants (from ARX firmware)
_MODE_EEF = 4
_MODE_JOINT = 5

# Gripper: public API is normalized [0, 1], firmware uses [-3.4, 0.0].
#   public 0.0 = fully open,  1.0 = fully closed
#   raw   -3.4 = fully open,  0.0 = fully closed
GRIPPER_OPEN_RAW = -3.4
GRIPPER_CLOSE_RAW = 0.0


def gripper_normalize(raw: float) -> float:
    """Hardware value [-3.4, 0.0] -> normalized [0, 1] (0=open, 1=closed)."""
    return float(np.clip(
        (raw - GRIPPER_OPEN_RAW) / (GRIPPER_CLOSE_RAW - GRIPPER_OPEN_RAW),
        0.0, 1.0,
    ))


def gripper_denormalize(normalized: float) -> float:
    """Normalized [0, 1] (0=open, 1=closed) -> hardware value [-3.4, 0.0]."""
    normalized = float(np.clip(normalized, 0.0, 1.0))
    return GRIPPER_OPEN_RAW + normalized * (GRIPPER_CLOSE_RAW - GRIPPER_OPEN_RAW)

# ---------------------------------------------------------------------------
# ARXEnv
# ---------------------------------------------------------------------------
class ARXEnv:
    """Unified environment for the ARX LIFT2 dual-arm robot.

    Parameters
    ----------
    camera_type : ``"rgb"`` or ``"rgbd"``
        ``"rgb"`` subscribes color only; ``"rgbd"`` subscribes color + depth.
    camera_view : Iterable[str]
        Camera names to subscribe, e.g. ``("camera_l", "camera_h", "camera_r")``.
    img_size : tuple[int, int] | None
        Resize images to this (W, H). None = no resize.
    """

    def __init__(
        self,
        camera_type: Literal["rgb", "rgbd"] = "rgbd",
        camera_view: Iterable[str] = ("camera_l", "camera_h", "camera_r"),
        img_size: Optional[Tuple[int, int]] = (640, 480),
    ):
        if camera_type not in ("rgb", "rgbd"):
            raise ValueError(
                f"Invalid camera_type={camera_type!r}. Choose 'rgb' or 'rgbd'"
            )

        self.camera_type = camera_type
        self.camera_view = list(camera_view)
        self.img_size = img_size
        self._closed = False

        # ---- Connect via ROS2 ----
        self._init_ros2()
        self._init_base_lift_state()

        atexit.register(self.close)

    # ------------------------------------------------------------------
    # ROS2 init / teardown
    # ------------------------------------------------------------------

    def _init_ros2(self):
        """Start ROS2 node, publishers, subscribers."""
        import rclpy
        from arx_toolkit.env._ros2_io import start_robot_io

        rclpy.init()
        self.node, self.executor, self._executor_thread = start_robot_io(
            camera_type=self.camera_type,
            camera_view=self.camera_view,
            target_size=self.img_size,
        )
        if not self.node or not self.executor:
            raise RuntimeError("Failed to start ROS2 node")
        logger.info("ROS2 node started.")

    def _shutdown_ros2(self):
        """Stop ROS2 node and executor, release all resources."""
        import rclpy

        # 1. Stop video saver first (flushes pending frames)
        if getattr(self, "node", None) is not None:
            try:
                self.node.stop_saver()
            except Exception:
                pass

        # 2. Shutdown executor (stops spinning) before destroying node
        if getattr(self, "executor", None) is not None:
            self.executor.shutdown()
            self.executor = None

        # 3. Wait for executor thread to finish
        if getattr(self, "_executor_thread", None) is not None:
            self._executor_thread.join(timeout=3.0)
            self._executor_thread = None

        # 4. Now safe to destroy node
        if getattr(self, "node", None) is not None:
            self.node.destroy_node()
            self.node = None

        # 5. Finally shutdown rclpy context
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass  # already shutdown or never inited

        logger.info("ROS2 shutdown.")

    # ------------------------------------------------------------------
    # Internal: base/lift command state
    # ------------------------------------------------------------------

    def _init_base_lift_state(self) -> None:
        """Initialize non-blocking base/lift command state."""
        self._base_lift_lock = threading.Lock()
        self._base_cmd = (0.0, 0.0, 0.0)
        initial_height = self._read_base_height(default=0.0)
        self._lift_current = initial_height
        self._lift_target = initial_height
        self._lift_stop_event = threading.Event()
        self._lift_thread: threading.Thread | None = None
        # Match the legacy ARX_Realenv lift ramp: 0.03 height units per 0.01s.
        self._lift_rate_hz = 100.0
        self._lift_speed_per_s = 3.0
        self._lift_epsilon = 1e-3

    def _read_base_height(self, default: float = 0.0) -> float:
        try:
            status = self.node.get_robot_status()
            base = status.get("base") if isinstance(status, dict) else None
            return float(base.height) if base is not None else float(default)
        except Exception:
            return float(default)

    def _publish_base_lift_once(
        self,
        vx: float,
        vy: float,
        vz: float,
        height: float,
    ) -> bool:
        """Publish one ROS2 base/lift command."""
        from arm_control.msg._pos_cmd import PosCmd

        msg = PosCmd()
        msg.chx = float(vx)
        msg.chy = float(vy)
        msg.chz = float(vz)
        msg.mode1 = 1
        msg.height = float(np.clip(height, 0.0, 20.0))

        ok = self.node.send_base_msg(msg)
        if not ok:
            logger.warning("base_lift command not sent")
        return ok

    def _ensure_base_lift_smoother(self) -> None:
        if self._lift_thread is not None and self._lift_thread.is_alive():
            return
        self._lift_stop_event.clear()
        self._lift_thread = threading.Thread(
            target=self._base_lift_smoother_loop,
            name="arx_lift_smoother",
            daemon=True,
        )
        self._lift_thread.start()

    def _base_lift_smoother_loop(self) -> None:
        period = 1.0 / max(self._lift_rate_hz, 1.0)
        max_step = self._lift_speed_per_s * period
        while not self._lift_stop_event.is_set():
            should_publish = False
            with self._base_lift_lock:
                target = float(self._lift_target)
                current = float(self._lift_current)
                delta = target - current
                if abs(delta) > self._lift_epsilon:
                    if abs(delta) <= max_step:
                        current = target
                    else:
                        current += max_step if delta > 0.0 else -max_step
                    self._lift_current = current
                    vx, vy, vz = self._base_cmd
                    should_publish = True
                else:
                    vx, vy, vz = self._base_cmd

            if should_publish:
                self._publish_base_lift_once(vx, vy, vz, current)
            time.sleep(period)

    def _wait_lift_target(self, timeout: float | None = None) -> bool:
        """Block until the non-blocking lift smoother reaches its target."""
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        period = 1.0 / max(self._lift_rate_hz, 1.0)
        while True:
            with self._base_lift_lock:
                reached = abs(float(self._lift_current) - float(self._lift_target)) <= self._lift_epsilon
            if reached:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(min(period, 0.05))

    def _stop_base_lift_smoother(self) -> None:
        event = getattr(self, "_lift_stop_event", None)
        if event is not None:
            event.set()
        thread = getattr(self, "_lift_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._lift_thread = None

    # ------------------------------------------------------------------
    # Internal: send arm command
    # ------------------------------------------------------------------

    def _send_arm_cmd(self, side: str, mode: int,
                      end_pos: list[float] | None = None,
                      joint_pos: list[float] | None = None,
                      gripper: float = 0.0):
        """Build and publish one RobotCmd."""
        from arx5_arm_msg.msg._robot_cmd import RobotCmd

        msg = RobotCmd()
        msg.mode = mode
        if end_pos is not None:
            msg.end_pos = end_pos
        if joint_pos is not None:
            msg.joint_pos = joint_pos
        msg.gripper = gripper

        ok = self.node.send_control_msg(side, msg)
        if not ok:
            logger.warning("arm command not sent for %s", side)
        return ok

    def _apply_absolute_eef(self, action: Dict[str, np.ndarray]):
        """Send EEF targets (mode=4). Gripper is normalized [0,1]."""
        for side, target in action.items():
            self._send_arm_cmd(
                side, _MODE_EEF,
                end_pos=[float(x) for x in target[:6]],
                gripper=gripper_denormalize(target[6]),
            )

    def _apply_absolute_joint(self, action: Dict[str, np.ndarray]):
        """Send joint targets (mode=5). Gripper is normalized [0,1]."""
        for side, target in action.items():
            self._send_arm_cmd(
                side, _MODE_JOINT,
                joint_pos=[float(x) for x in target[:6]],
                gripper=gripper_denormalize(target[6]),
            )

    @staticmethod
    def _raw_arm_state(status_all: dict, side: str) -> tuple[np.ndarray, np.ndarray]:
        """Return raw [x,y,z,r,p,y,gripper] EEF and joint state for one arm."""
        status = status_all.get(side) if isinstance(status_all, dict) else None
        if status is None:
            raise RuntimeError(f"{side}: current status unavailable")
        end_pos = np.asarray(status.end_pos, dtype=np.float32).reshape(-1)
        joint_pos = np.asarray(status.joint_pos, dtype=np.float32).reshape(-1)
        if end_pos.shape[0] < 6 or joint_pos.shape[0] < 7:
            raise RuntimeError(f"{side}: malformed current status")
        eef = np.concatenate([end_pos[:6], joint_pos[6:7]]).astype(np.float32)
        return eef, joint_pos[:7].astype(np.float32)

    def _apply_smooth_eef(self, action: Dict[str, np.ndarray]):
        """Send EEF targets through a smoothed command sequence."""
        duration_per_step = 1.0 / 20.0
        status_all = self.node.get_robot_status()
        start_eef = {
            side: self._raw_arm_state(status_all, side)[0]
            for side in action
        }
        sequences = plan_smooth_eef_sequences(
            action=action,
            start_eef=start_eef,
            normalize_gripper=gripper_normalize,
        )
        max_len = max((len(seq) for seq in sequences.values()), default=0)
        for idx in range(max_len):
            t0 = time.time()
            for side, seq in sequences.items():
                if idx < len(seq):
                    target = seq[idx]
                    self._send_arm_cmd(
                        side, _MODE_EEF,
                        end_pos=[float(x) for x in target[:6]],
                        gripper=gripper_denormalize(target[6]),
                    )
            sleep_need = duration_per_step - (time.time() - t0)
            if sleep_need > 0.0:
                time.sleep(sleep_need)

    def _apply_delta_eef(self, action: Dict[str, np.ndarray]):
        """Compute absolute EEF targets from deltas, then send them.

        Pose deltas are in hardware EEF units. Gripper delta is in normalized
        [0,1] space, positive means more closed.
        """
        status_all = self.node.get_robot_status()
        target_action: Dict[str, np.ndarray] = {}
        for side, delta in action.items():
            curr_eef, _curr_joint = self._raw_arm_state(status_all, side)
            target_xyz = curr_eef[:3] + delta[:3]
            q_curr = quat_from_rpy(curr_eef[3:6])
            q_delta = quat_from_rpy(delta[3:6])
            q_target = quat_multiply(q_delta, q_curr)
            target_rpy = rpy_from_quat(q_target)
            target_gripper = float(np.clip(
                gripper_normalize(float(curr_eef[6])) + float(delta[6]),
                0.0,
                1.0,
            ))
            target_action[side] = np.concatenate([
                target_xyz, target_rpy, [target_gripper],
            ]).astype(np.float32)
        self._apply_absolute_eef(target_action)

    # ------------------------------------------------------------------
    # Internal: validate action dict
    # ------------------------------------------------------------------

    _REQUIRED_KEYS = {"left", "right", "base", "lift"}

    @staticmethod
    def _validate_action(action: dict) -> dict:
        """Validate the unified action dict.

        Required format::

            {
                "left":  np.ndarray(7,) | None,   # arm action (action_mode dependent)
                "right": np.ndarray(7,) | None,   # arm action (action_mode dependent)
                "base":  np.ndarray(3,) | None,   # [vx, vy, vz]
                "lift":  float | None,             # height 0~20
            }

        ``None`` means "don't move this part".
        """
        if not isinstance(action, dict):
            raise TypeError("action must be a dict with keys: left, right, base, lift")

        missing = ARXEnv._REQUIRED_KEYS - action.keys()
        if missing:
            raise ValueError(f"action dict missing required keys: {missing}")

        result: dict = {}

        # -- Arms --
        for side in ("left", "right"):
            val = action[side]
            if val is None:
                result[side] = None
                continue
            arr = np.asarray(val, dtype=np.float32).reshape(-1)
            if arr.shape[0] != 7:
                raise ValueError(
                    f"{side} action must have shape (7,), got {arr.shape}"
                )
            result[side] = arr

        # -- Base --
        base_val = action["base"]
        if base_val is None:
            result["base"] = None
        else:
            base_arr = np.asarray(base_val, dtype=np.float32).reshape(-1)
            if base_arr.shape[0] != 3:
                raise ValueError(
                    f"base action must have shape (3,), got {base_arr.shape}"
                )
            result["base"] = base_arr

        # -- Lift --
        lift_val = action["lift"]
        if lift_val is None:
            result["lift"] = None
        else:
            result["lift"] = float(lift_val)

        return result

    @staticmethod
    def _validate_arm_action(
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
    ) -> Dict[str, np.ndarray | None]:
        result: Dict[str, np.ndarray | None] = {}
        for side, val in (("left", left), ("right", right)):
            if val is None:
                result[side] = None
                continue
            arr = np.asarray(val, dtype=np.float32).reshape(-1)
            if arr.shape[0] != 7:
                raise ValueError(
                    f"{side} action must have shape (7,), got {arr.shape}"
                )
            result[side] = arr
        return result

    @staticmethod
    def _normalize_action_mode(action_mode: str) -> ActionMode:
        """Validate the public action mode."""
        normalized = str(action_mode).strip().lower()
        if normalized not in _VALID_ACTION_MODES:
            raise ValueError(
                "Invalid action_mode="
                f"{action_mode!r}. Choose from {_VALID_ACTION_MODES}"
            )
        return normalized  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public API — Observation
    # ------------------------------------------------------------------

    def get_observation(
        self,
        include_arm: bool = True,
        include_camera: bool = True,
        include_base: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Fetch latest observation.

        Gripper values in ``{side}_joint_pos[6]`` are normalized to [0, 1]
        (0 = fully open, 1 = fully closed).

        Args:
            include_arm: Include arm state in obs.
            include_camera: Include camera images in obs.
            include_base: Include base/lift state in obs.

        Returns:
            Flat dict. Full example with ``camera_type="rgbd"``,
            ``camera_view=("camera_l", "camera_h", "camera_r")``,
            ``img_size=(640, 480)``::

                {
                    # ---- Left arm ----
                    "left_eef_pos":    np.float32(7,),   # [x, y, z, roll, pitch, yaw, gripper]
                                                         #  gripper normalized [0,1] (0=open, 1=closed)
                    "left_joint_pos":  np.float32(7,),   # [j0, j1, j2, j3, j4, j5, gripper]
                                                         #  gripper normalized [0,1] (0=open, 1=closed)
                    "left_joint_qvel": np.float32(7,),   # joint velocity
                    "left_joint_effort": np.float32(7,), # joint current / effort

                    # ---- Right arm ----
                    "right_eef_pos":   np.float32(7,),
                    "right_joint_pos": np.float32(7,),
                    "right_joint_qvel": np.float32(7,),
                    "right_joint_effort": np.float32(7,),

                    # ---- Base / Lift ----
                    "base_height":     np.float32(1,),   # lift height [0, 20]

                    # ---- Cameras (rgb mode: only *_color; rgbd mode: *_color + *_aligned_depth_to_color) ----
                    "camera_l_color":                        np.uint8(480, 640, 3),   # RGB
                    "camera_l_aligned_depth_to_color":       np.uint16(480, 640),     # depth in mm
                    "camera_h_color":                        np.uint8(480, 640, 3),
                    "camera_h_aligned_depth_to_color":       np.uint16(480, 640),
                    "camera_r_color":                        np.uint8(480, 640, 3),
                    "camera_r_aligned_depth_to_color":       np.uint16(480, 640),
                }

            If ``include_arm=False``, arm keys are omitted.
            If ``include_camera=False``, camera keys are omitted.
            If ``include_base=False``, base keys are omitted.
            If ``camera_type="rgb"``, ``*_aligned_depth_to_color`` keys are absent.
        """
        from arx_toolkit.env._ros2_io import build_observation

        if include_camera:
            camera_all, status_all = self.node.get_camera(
                target_size=self.img_size,
                return_status=True,
            )
        else:
            camera_all = {}
            status_all = self.node.get_robot_status()

        obs = build_observation(
            camera_all, status_all,
            include_arm=include_arm,
            include_camera=include_camera,
            include_base=include_base,
        )

        # Normalize gripper in joint_pos[6] and eef_pos[6]: raw [-3.4, 0] -> [0, 1]
        if include_arm:
            for side in ("left", "right"):
                jp_key = f"{side}_joint_pos"
                if jp_key in obs and obs[jp_key].shape[0] >= 7:
                    obs[jp_key][6] = gripper_normalize(obs[jp_key][6])
                eef_key = f"{side}_eef_pos"
                if eef_key in obs and obs[eef_key].shape[0] >= 7:
                    obs[eef_key][6] = gripper_normalize(obs[eef_key][6])

        if not obs:
            raise RuntimeError("Empty observation — is the robot connected?")
        return obs

    # ------------------------------------------------------------------
    # Public API — Step
    # ------------------------------------------------------------------

    def step(
        self,
        action: dict,
        action_mode: ActionMode | str,
        return_observation: bool = True,
    ) -> Dict[str, np.ndarray] | None:
        """Execute one control step for the whole robot.

        Args:
            action: Unified action dict with 4 required keys::

                {
                    "left":  np.ndarray(7,) | None,
                    "right": np.ndarray(7,) | None,
                    "base":  np.ndarray(3,) | None,  # [vx, vy, vz]
                    "lift":  float | None,            # height 0~20
                }

                Arm action semantics are selected by ``action_mode``:
                - ``"absolute_joint"``: joint target [j0, j1, j2, j3, j4, j5, gripper]
                - ``"absolute_eef"``: EEF target [x, y, z, r, p, y, gripper]
                - ``"smooth_eef"``: EEF target, sent through a smoothed sequence
                - ``"delta_eef"``: EEF delta [dx, dy, dz, dr, dp, dy, gripper_delta]

                Gripper is always normalized [0,1] at this API. ARXEnv
                converts it to firmware raw [-3.4,0] before publishing.

                ``None`` = don't move that part.
            action_mode: Per-call arm control mode. Accepts
                ``"absolute_joint"``, ``"absolute_eef"``, ``"smooth_eef"``,
                or ``"delta_eef"``.
            return_observation: If True, fetch and return a fresh observation
                after publishing commands. Set False for high-rate teleop loops
                that only need to send commands.

        Returns:
            Observation dict after commands are sent, or ``None`` when
            ``return_observation=False``.
        """
        action = self._validate_action(action)
        action_mode = self._normalize_action_mode(action_mode)

        if action["left"] is not None or action["right"] is not None:
            self.step_arm(
                left=action["left"],
                right=action["right"],
                action_mode=action_mode,
                return_observation=False,
            )

        # -- Base & Lift --
        base_val = action["base"]
        lift_val = action["lift"]

        if base_val is not None or lift_val is not None:
            vx, vy, vz = (
                (float(base_val[0]), float(base_val[1]), float(base_val[2]))
                if base_val is not None
                else (None, None, None)
            )
            self.step_base_lift(vx=vx, vy=vy, vz=vz, height=lift_val)

        if not return_observation:
            return None
        return self.get_observation()

    def step_arm(
        self,
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
        action_mode: ActionMode | str = "absolute_eef",
        return_observation: bool = True,
    ) -> Dict[str, np.ndarray] | None:
        """Send one arm command for either or both arms."""
        action = self._validate_arm_action(left=left, right=right)
        action_mode = self._normalize_action_mode(action_mode)

        arm_action = {
            side: target
            for side, target in action.items()
            if target is not None
        }
        if arm_action:
            if action_mode == "absolute_joint":
                self._apply_absolute_joint(arm_action)
            elif action_mode == "absolute_eef":
                self._apply_absolute_eef(arm_action)
            elif action_mode == "smooth_eef":
                self._apply_smooth_eef(arm_action)
            elif action_mode == "delta_eef":
                self._apply_delta_eef(arm_action)

        if not return_observation:
            return None
        return self.get_observation()

    # ------------------------------------------------------------------
    # Public API — Base & Lift
    # ------------------------------------------------------------------

    def step_base_lift(
        self,
        vx: float | None = None,
        vy: float | None = None,
        vz: float | None = None,
        height: Optional[float] = None,
    ) -> None:
        """Send one combined base chassis + lift command.

        This is the preferred method for base/lift control.

        Args:
            vx: Forward/backward speed, range [-1.5, 1.5]. None = keep current.
            vy: Left/right speed, range [-1.5, 1.5]. None = keep current.
            vz: Rotation speed, range [-2.0, 2.0]. None = keep current.
            height: Lift target height, range [0, 20]. None = keep current.
        """
        base_changed = vx is not None or vy is not None or vz is not None
        with self._base_lift_lock:
            cur_vx, cur_vy, cur_vz = self._base_cmd
            next_vx = cur_vx if vx is None else float(vx)
            next_vy = cur_vy if vy is None else float(vy)
            next_vz = cur_vz if vz is None else float(vz)
            self._base_cmd = (next_vx, next_vy, next_vz)

            if height is not None:
                if self._lift_current is None:
                    self._lift_current = self._read_base_height(default=0.0)
                self._lift_target = float(np.clip(height, 0.0, 20.0))

            current_height = float(self._lift_current)

        if base_changed:
            self._publish_base_lift_once(next_vx, next_vy, next_vz, current_height)
        if height is not None:
            self._ensure_base_lift_smoother()

    # ------------------------------------------------------------------
    # Public API — Mode switch
    # ------------------------------------------------------------------

    def set_mode(self, mode: int, side: Side = "both") -> None:
        """Set special mode for one or both arms.

        Args:
            mode: 0=soft, 1=home, 2=protect, 3=gravity.
            side: ``"left"``, ``"right"``, or ``"both"``.
        """
        _MODE_NAMES = {0: "soft", 1: "home", 2: "protect", 3: "gravity"}

        if side not in {"left", "right", "both"}:
            raise ValueError(f"Invalid side={side!r}")

        targets = ("left", "right") if side == "both" else (side,)
        status = self.node.get_robot_status()

        for target in targets:
            cmd = self._build_mode_cmd(mode, status.get(target))
            ok = self.node.send_control_msg(target, cmd)
            if not ok:
                logger.warning("set_mode(%d) failed for %s", mode, target)

        logger.info("set_mode %s for %s", _MODE_NAMES.get(mode, "?"), side)

    @staticmethod
    def _build_mode_cmd(mode: int, status):
        """Build a mode-switch RobotCmd preserving current targets."""
        from arx5_arm_msg.msg._robot_cmd import RobotCmd

        cmd = RobotCmd()
        cmd.mode = int(mode)
        if status is None:
            return cmd
        try:
            end_pos = np.asarray(status.end_pos, dtype=np.float32).reshape(-1)
            if end_pos.shape[0] >= 6:
                cmd.end_pos = [float(x) for x in end_pos[:6]]
        except Exception:
            pass
        try:
            joint_pos = np.asarray(status.joint_pos, dtype=np.float32).reshape(-1)
            if joint_pos.shape[0] >= 6:
                cmd.joint_pos = [float(x) for x in joint_pos[:6]]
            if joint_pos.shape[0] >= 7:
                cmd.gripper = float(joint_pos[6])
        except Exception:
            pass
        return cmd

    def _go_home(self, side: Side = "both"):
        """Send mode=1 (home) to the firmware — arms return to factory initial pose.

        Args:
            side: Which arm(s) to home.
        """
        targets = ("left", "right") if side == "both" else (side,)
        status = self.node.get_robot_status()

        for target in targets:
            cmd = self._build_mode_cmd(1, status.get(target))
            ok = self.node.send_control_msg(target, cmd)
            if not ok:
                logger.warning("go_home (mode=1) failed for %s", target)

        logger.info("%s arm(s) homed (mode=1)", side)

    def _safe_stop_robot(self) -> None:
        """Stop lower body, home arms, and wait for lift to return to zero."""
        self.step_base_lift(vx=0.0, vy=0.0, vz=0.0)
        time.sleep(1.0)
        self._go_home(side="both")
        self.step_base_lift(height=0.0)
        self._wait_lift_target(timeout=15.0)

    # ------------------------------------------------------------------
    # Public API — Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset robot: home both arms, lift to 0, stop base.

        Returns:
            Observation dict after reset.
        """
        logger.info("Resetting ...")
        time.sleep(1.5)
        self._safe_stop_robot()

        obs = self.get_observation()
        logger.info("Reset done.")
        return obs

    def close(self) -> None:
        """Safe shutdown: stop base, home arms, lift to 0, teardown ROS2."""
        if self._closed:
            return
        self._closed = True

        logger.info("Closing ...")
        try:
            self._safe_stop_robot()
        except Exception as e:
            logger.warning("Error during close cleanup: %s", e)

        self._stop_base_lift_smoother()
        self._shutdown_ros2()
        logger.info("Closed.")


if __name__ == "__main__":
    import time as _time

    env = ARXEnv(
        camera_type="rgbd",
        camera_view=("camera_h",),
        img_size=(640, 480),
    )

    # ========== 1. reset ==========
    obs = env.reset()
    print("[reset] obs keys:", sorted(obs.keys()))
    print("  left_eef_pos:", obs["left_eef_pos"])
    print("  left_joint_pos:", obs["left_joint_pos"])
    print("  left gripper (normalized):", obs["left_joint_pos"][6])
    print("  base_height:", obs["base_height"])
    print("  camera_h_color shape:", obs["camera_h_color"].shape)
    _time.sleep(3.0)

    # ========== 2. step — 双臂 eef ==========
    obs = env.step({
        "left":  np.array([0.02, 0, 0.03, 0, 0, 0, 0.0], dtype=np.float32),
        "right": np.array([0.02, 0, 0.03, 0, 0, 0, 0.0], dtype=np.float32),
        "base": None,
        "lift": None,
    }, action_mode="absolute_eef")
    print("\n[step] 双臂 eef (gripper=0 全开)")
    print("  left_eef_pos:", obs["left_eef_pos"])
    _time.sleep(3.0)

    # ========== 3. step — 单臂 + gripper 半闭合 ==========
    obs = env.step({
        "left":  np.array([0.03, 0, 0.04, 0, 0, 0, 0.5], dtype=np.float32),
        "right": None,
        "base": None,
        "lift": None,
    }, action_mode="absolute_eef")
    print("\n[step] 单臂左 (gripper=0.5 半闭合)")
    print("  left gripper:", obs["left_joint_pos"][6])
    _time.sleep(3.0)

    # ========== 4. step — gripper 全闭 ==========
    obs = env.step({
        "left":  np.array([0.03, 0, 0.04, 0, 0, 0, 1.0], dtype=np.float32),
        "right": None,
        "base": None,
        "lift": None,
    }, action_mode="absolute_eef")
    print("\n[step] gripper=1.0 全闭")
    print("  left gripper:", obs["left_joint_pos"][6])
    _time.sleep(3.0)

    # ========== 5. step_base_lift — 只控制升降台 ==========
    env.step_base_lift(height=3.0)
    obs = env.get_observation(include_camera=False)
    print("\n[step_base_lift] height=3")
    print("  base_height:", obs["base_height"])
    _time.sleep(3.0)

    # ========== 6. step_base_lift — 只控制底盘前进 ==========
    env.step_base_lift(vx=0.1, vy=0, vz=0)
    _time.sleep(1.0)
    env.step_base_lift(vx=0, vy=0, vz=0)  # 停
    print("\n[step_base_lift] 前进 1s 后停止")
    _time.sleep(3.0)

    # ========== 7. step_base_lift — 联合控制 ==========
    env.step_base_lift(vx=0, vy=0, vz=0.15, height=2.0)
    _time.sleep(1.0)
    env.step_base_lift(vx=0, vy=0, vz=0, height=2.0)  # 停旋转
    print("\n[step_base_lift] 旋转 + 升降到 2")
    _time.sleep(3.0)

    # ========== 8. step — 底盘 + 升降 via unified action ==========
    obs = env.step({
        "left": None,
        "right": None,
        "base": np.array([0.0, 0.1, 0.0], dtype=np.float32),
        "lift": 1.0,
    }, action_mode="absolute_joint")
    _time.sleep(1.0)
    env.step_base_lift(vx=0, vy=0, vz=0)  # 停底盘
    print("\n[step] 底盘横移 + 升降到 1")
    print("  base_height:", obs["base_height"])
    _time.sleep(3.0)

    # ========== 9. step — 全部 None (纯取观测) ==========
    obs = env.step({
        "left": None, "right": None, "base": None, "lift": None,
    }, action_mode="absolute_joint")
    print("\n[step] 全 None (纯观测)")
    print("  obs keys:", sorted(obs.keys()))
    _time.sleep(3.0)

    # ========== 10. get_observation — 局部获取 ==========
    obs_arm = env.get_observation(include_camera=False, include_base=False)
    print("\n[get_observation] arm only keys:", sorted(obs_arm.keys()))

    obs_cam = env.get_observation(include_arm=False, include_base=False)
    print("[get_observation] camera only keys:", sorted(obs_cam.keys()))
    _time.sleep(3.0)

    # ========== 11. set_mode ==========
    env.set_mode(3, side="left")   # 左臂重力补偿
    print("\n[set_mode] 左臂 gravity 模式")
    _time.sleep(3.0)

    env.set_mode(1, side="left")   # 左臂回零
    print("[set_mode] 左臂 home")
    _time.sleep(3.0)

    # ========== 12. close ==========
    env.close()
    print("\n[close] 完成")
