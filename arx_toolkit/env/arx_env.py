"""Unified ARX LIFT2 robot environment.

Design philosophy: one ``step(action)`` controls the entire robot.

Robot
-----
ARX LIFT2 dual-arm mobile manipulator:
- 2 × 6-DOF arms (left / right), each with 1 gripper
- 3-wheel omnidirectional base chassis
- Lift (vertical linear stage)
- Up to 3 RealSense D405 cameras

Action — ``step(action) -> obs``
--------------------------------
``action`` is a dict with **4 required keys** (set value to ``None`` to skip):

.. code-block:: python

    action = {
        "left":  np.ndarray(7,) | None,
        "right": np.ndarray(7,) | None,
        "base":  np.ndarray(3,) | None,
        "lift":  float | None,
    }

**Arm action (left / right)** — 7D, semantics depend on ``action_mode``:

+-----------------+-------+--------------------------------------------------+
| action_mode     | dim   | meaning                                          |
+=================+=======+==================================================+
| delta_eef       | 7     | [dx, dy, dz, droll, dpitch, dyaw, gripper]     |
+-----------------+-------+--------------------------------------------------+
| absolute_eef    | 7     | [x, y, z, roll, pitch, yaw, gripper]             |
+-----------------+-------+--------------------------------------------------+
| absolute_joint  | 7     | [j0, j1, j2, j3, j4, j5, gripper]               |
+-----------------+-------+--------------------------------------------------+

- Position xyz: meters, in base frame.
- Orientation rpy: radians.
- Gripper: normalized [0, 1]. 0 = fully open, 1 = fully closed.
  Supports continuous values (e.g. 0.5 = half open).
  For delta_eef, gripper delta is also in [0, 1] space (positive = more closed).

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

Observation — ``obs = env.reset()`` / ``env.step(action)``
----------------------------------------------------------
``obs`` is a flat dict. Available keys depend on ``camera_type`` and
``camera_view``:

**Arm state** (per side, always present):

+---------------------+----------+----------------------------------------------+
| key                 | shape    | description                                  |
+=====================+==========+==============================================+
| {side}_eef_pos      | (6,)     | [x, y, z, roll, pitch, yaw] end-effector     |
+---------------------+----------+----------------------------------------------+
| {side}_joint_pos    | (7,)     | 6 joint angles + gripper [0,1] normalized     |
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

    env = ARXEnv(action_mode="absolute_eef", camera_type="rgbd")
    obs = env.reset()       # home arms, lift=0, base stop
    obs = env.step(action)  # send command, return new obs
    env.close()             # safe shutdown (also registered via atexit)

Convenience methods ``step_base()``, ``step_lift()``, ``step_base_lift()``
are available but the primary interface is ``step(action)``.
"""

from __future__ import annotations

import atexit
import time
from typing import Dict, Iterable, Literal, Optional, Tuple

import numpy as np

from arx_toolkit.utils.logger import get_logger
from arx_toolkit.utils.transforms import (
    quat_from_rpy,
    quat_multiply,
    rpy_from_quat,
)

logger = get_logger("arx_toolkit.env")

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ActionMode = Literal["delta_eef", "absolute_eef", "absolute_joint"]
Side = Literal["left", "right", "both"]

_VALID_ACTION_MODES: set[str] = {"delta_eef", "absolute_eef", "absolute_joint"}

# RobotCmd mode constants (from ARX firmware)
_MODE_EEF = 4
_MODE_JOINT = 5

# Gripper: hardware range is [-3.4, 0.0], we normalize to [0, 1]
#   0.0 = fully open,  1.0 = fully closed
#   hardware: -3.4 = open, 0.0 = closed
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
    action_mode : ActionMode
        ``"delta_eef"``  — 7D delta [dx,dy,dz,dr,dp,dy, gripper]
        ``"absolute_eef"``  — 7D absolute [x,y,z,r,p,y, gripper]
        ``"absolute_joint"``  — 7D absolute [j0..j5, gripper]
    camera_type : ``"rgb"`` or ``"rgbd"``
        ``"rgb"`` subscribes color only; ``"rgbd"`` subscribes color + depth.
    camera_view : Iterable[str]
        Camera names to subscribe, e.g. ``("camera_l", "camera_h", "camera_r")``.
    img_size : tuple[int, int] | None
        Resize images to this (W, H). None = no resize.
    """

    def __init__(
        self,
        action_mode: ActionMode = "delta_eef",
        camera_type: Literal["rgb", "rgbd"] = "rgbd",
        camera_view: Iterable[str] = ("camera_l", "camera_h", "camera_r"),
        img_size: Optional[Tuple[int, int]] = (640, 480),
    ):
        if action_mode not in _VALID_ACTION_MODES:
            raise ValueError(
                f"Invalid action_mode={action_mode!r}. "
                f"Choose from {_VALID_ACTION_MODES}"
            )
        if camera_type not in ("rgb", "rgbd"):
            raise ValueError(
                f"Invalid camera_type={camera_type!r}. Choose 'rgb' or 'rgbd'"
            )

        self.action_mode: ActionMode = action_mode
        self.camera_type = camera_type
        self.camera_view = list(camera_view)
        self.img_size = img_size
        self._closed = False

        # ---- Connect via ROS2 ----
        self._init_ros2()

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
        """Stop ROS2 node and executor."""
        import rclpy

        if getattr(self, "node", None) is not None:
            try:
                self.node.stop_saver()
            except Exception:
                pass
            self.node.destroy_node()
            self.node = None
        if getattr(self, "executor", None) is not None:
            self.executor.shutdown()
            self.executor = None
        if getattr(self, "_executor_thread", None) is not None:
            self._executor_thread.join(timeout=2.0)
            self._executor_thread = None
        if rclpy.ok():
            rclpy.shutdown()
        logger.info("ROS2 shutdown.")

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
        """Send absolute EEF targets (mode=4). Gripper is normalized [0,1]."""
        for side, target in action.items():
            self._send_arm_cmd(
                side, _MODE_EEF,
                end_pos=[float(x) for x in target[:6]],
                gripper=gripper_denormalize(target[6]),
            )

    def _apply_absolute_joint(self, action: Dict[str, np.ndarray]):
        """Send absolute joint targets (mode=5). Gripper is normalized [0,1]."""
        for side, target in action.items():
            self._send_arm_cmd(
                side, _MODE_JOINT,
                joint_pos=[float(x) for x in target[:6]],
                gripper=gripper_denormalize(target[6]),
            )

    def _apply_delta_eef(self, action: Dict[str, np.ndarray]):
        """Compute absolute targets from deltas, then send (mode=4).

        Gripper delta is in normalized space [0,1]: positive = more closed.
        """
        obs = self.get_observation(include_camera=False, include_base=False)
        for side, delta in action.items():
            curr_end = obs.get(f"{side}_eef_pos")
            curr_joint = obs.get(f"{side}_joint_pos")
            if curr_end is None or curr_joint is None:
                raise RuntimeError(f"{side}: current state unavailable for delta_eef")

            curr_end = np.asarray(curr_end, dtype=np.float32).reshape(-1)
            curr_joint = np.asarray(curr_joint, dtype=np.float32).reshape(-1)

            # Position: simple add
            target_xyz = curr_end[:3] + delta[:3]

            # Orientation: quaternion multiply (base-frame delta)
            q_curr = quat_from_rpy(curr_end[3:6])
            q_delta = quat_from_rpy(delta[3:6])
            q_target = quat_multiply(q_delta, q_curr)
            target_rpy = rpy_from_quat(q_target)

            # Gripper: obs joint_pos[6] is already normalized [0,1], just add delta
            curr_gripper_normalized = float(curr_joint[6])
            target_gripper_normalized = np.clip(
                curr_gripper_normalized + float(delta[6]), 0.0, 1.0,
            )

            self._send_arm_cmd(
                side, _MODE_EEF,
                end_pos=[float(x) for x in np.concatenate([target_xyz, target_rpy])],
                gripper=gripper_denormalize(target_gripper_normalized),
            )

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
                    "left_eef_pos":    np.float32(6,),   # [x, y, z, roll, pitch, yaw]
                    "left_joint_pos":  np.float32(7,),   # [j0, j1, j2, j3, j4, j5, gripper]
                                                         #  gripper normalized [0,1] (0=open, 1=closed)

                    # ---- Right arm ----
                    "right_eef_pos":   np.float32(6,),
                    "right_joint_pos": np.float32(7,),

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

        # Normalize gripper in joint_pos[6]: raw [-3.4, 0] -> [0, 1]
        if include_arm:
            for side in ("left", "right"):
                key = f"{side}_joint_pos"
                if key in obs and obs[key].shape[0] >= 7:
                    obs[key][6] = gripper_normalize(obs[key][6])

        if not obs:
            raise RuntimeError("Empty observation — is the robot connected?")
        return obs

    # ------------------------------------------------------------------
    # Public API — Step (arm control)
    # ------------------------------------------------------------------

    def step(
        self,
        action: dict,
    ) -> Dict[str, np.ndarray]:
        """Execute one control step for the whole robot.

        Args:
            action: Unified action dict with 4 required keys::

                {
                    "left":  np.ndarray(7,) | None,
                    "right": np.ndarray(7,) | None,
                    "base":  np.ndarray(3,) | None,  # [vx, vy, vz]
                    "lift":  float | None,            # height 0~20
                }

                Arm action semantics depend on ``action_mode``:
                - ``"delta_eef"``: [dx, dy, dz, dr, dp, dy, gripper]
                - ``"absolute_eef"``: [x, y, z, r, p, y, gripper]
                - ``"absolute_joint"``: [j0, j1, j2, j3, j4, j5, gripper]

                ``None`` = don't move that part.

        Returns:
            Observation dict after commands are sent.
        """
        action = self._validate_action(action)

        # -- Arms --
        arm_action = {}
        for side in ("left", "right"):
            if action[side] is not None:
                arm_action[side] = action[side]

        if arm_action:
            if self.action_mode == "delta_eef":
                self._apply_delta_eef(arm_action)
            elif self.action_mode == "absolute_eef":
                self._apply_absolute_eef(arm_action)
            elif self.action_mode == "absolute_joint":
                self._apply_absolute_joint(arm_action)

        # -- Base & Lift --
        base_val = action["base"]
        lift_val = action["lift"]

        if base_val is not None or lift_val is not None:
            vx, vy, vz = (float(base_val[0]), float(base_val[1]), float(base_val[2])) if base_val is not None else (0.0, 0.0, 0.0)
            self.step_base_lift(vx=vx, vy=vy, vz=vz, height=lift_val)

        return self.get_observation()

    # ------------------------------------------------------------------
    # Public API — Base & Lift
    # ------------------------------------------------------------------

    def step_base_lift(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
        height: Optional[float] = None,
    ) -> None:
        """Send one combined base chassis + lift command.

        This is the preferred method for base/lift control.

        Args:
            vx: Forward/backward speed, range [-1.5, 1.5].
            vy: Left/right speed, range [-1.5, 1.5].
            vz: Rotation speed, range [-2.0, 2.0].
            height: Lift height, range [0, 20]. None = keep current.
        """
        from arm_control.msg._pos_cmd import PosCmd

        msg = PosCmd()
        msg.chx = float(vx)
        msg.chy = float(vy)
        msg.chz = float(vz)
        msg.mode1 = 1

        if height is not None:
            msg.height = float(np.clip(height, 0.0, 20.0))
        else:
            status = self.node.get_robot_status()
            base = status.get("base")
            msg.height = float(base.height) if base is not None else 0.0

        ok = self.node.send_base_msg(msg)
        if not ok:
            logger.warning("base_lift command not sent")

    def step_base(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
    ) -> None:
        """Send one base chassis velocity command (lift unchanged).

        Args:
            vx: Forward/backward speed, range [-1.5, 1.5].
            vy: Left/right speed, range [-1.5, 1.5].
            vz: Rotation speed, range [-2.0, 2.0].
        """
        self.step_base_lift(vx=vx, vy=vy, vz=vz, height=None)

    def step_lift(self, height: float) -> None:
        """Set lift height (base velocity = 0).

        Args:
            height: Target height, range [0, 20].
        """
        self.step_base_lift(vx=0, vy=0, vz=0, height=height)

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

        # mode=1 (home) is special: move to initial pose
        if mode == 1:
            self._go_home(side=side)
            return

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
        """Move arm(s) to initial pose [0, 0, 0, 0, 0, 0, 0]."""
        home = np.zeros(7, dtype=np.float32)
        if side == "both":
            action = {"left": home.copy(), "right": home.copy()}
        else:
            action = {side: home.copy()}

        # Use absolute_eef to send home pose regardless of current action_mode
        self._apply_absolute_eef(action)
        logger.info("%s arm(s) homed", side)

    # ------------------------------------------------------------------
    # Public API — Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset robot: home both arms, lift to 0, stop base.

        Returns:
            Observation dict after reset.
        """
        logger.info("Resetting ...")
        time.sleep(2.0)

        self._go_home(side="both")
        self.step_lift(0.0)
        self.step_base(0.0, 0.0, 0.0)

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
            self.step_base(0.0, 0.0, 0.0)
            time.sleep(1.0)
            self._go_home(side="both")
            self.step_lift(0.0)
        except Exception as e:
            logger.warning("Error during close cleanup: %s", e)

        self._shutdown_ros2()
        logger.info("Closed.")


if __name__ == "__main__":
    import time as _time

    env = ARXEnv(
        action_mode="absolute_eef",
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

    # ========== 2. step — 双臂 absolute_eef ==========
    obs = env.step({
        "left":  np.array([0.1, 0, 0.15, 0, 0, 0, 0.0], dtype=np.float32),
        "right": np.array([0.1, 0, 0.15, 0, 0, 0, 0.0], dtype=np.float32),
        "base": None,
        "lift": None,
    })
    print("\n[step] 双臂 absolute_eef (gripper=0 全开)")
    print("  left_eef_pos:", obs["left_eef_pos"])

    # ========== 3. step — 单臂 + gripper 半闭合 ==========
    obs = env.step({
        "left":  np.array([0.15, 0, 0.18, 0, 0, 0, 0.5], dtype=np.float32),
        "right": None,
        "base": None,
        "lift": None,
    })
    print("\n[step] 单臂左 (gripper=0.5 半闭合)")
    print("  left gripper:", obs["left_joint_pos"][6])

    # ========== 4. step — gripper 全闭 ==========
    obs = env.step({
        "left":  np.array([0.15, 0, 0.18, 0, 0, 0, 1.0], dtype=np.float32),
        "right": None,
        "base": None,
        "lift": None,
    })
    print("\n[step] gripper=1.0 全闭")
    print("  left gripper:", obs["left_joint_pos"][6])

    # ========== 5. step_lift — 升降台 ==========
    env.step_lift(18.0)
    obs = env.get_observation(include_camera=False)
    print("\n[step_lift] height=18")
    print("  base_height:", obs["base_height"])

    # ========== 6. step_base — 底盘前进 ==========
    env.step_base(vx=0.3, vy=0, vz=0)
    _time.sleep(1.0)
    env.step_base(vx=0, vy=0, vz=0)  # 停
    print("\n[step_base] 前进 1s 后停止")

    # ========== 7. step_base_lift — 联合控制 ==========
    env.step_base_lift(vx=0, vy=0, vz=0.5, height=10.0)
    _time.sleep(1.0)
    env.step_base_lift(vx=0, vy=0, vz=0, height=10.0)  # 停旋转
    print("\n[step_base_lift] 旋转 + 升降到 10")

    # ========== 8. step — 底盘 + 升降 via unified action ==========
    obs = env.step({
        "left": None,
        "right": None,
        "base": np.array([0.0, 0.3, 0.0], dtype=np.float32),
        "lift": 5.0,
    })
    _time.sleep(1.0)
    env.step_base(0, 0, 0)  # 停底盘
    print("\n[step] 底盘横移 + 升降到 5")
    print("  base_height:", obs["base_height"])

    # ========== 9. step — 全部 None (纯取观测) ==========
    obs = env.step({
        "left": None, "right": None, "base": None, "lift": None,
    })
    print("\n[step] 全 None (纯观测)")
    print("  obs keys:", sorted(obs.keys()))

    # ========== 10. get_observation — 局部获取 ==========
    obs_arm = env.get_observation(include_camera=False, include_base=False)
    print("\n[get_observation] arm only keys:", sorted(obs_arm.keys()))

    obs_cam = env.get_observation(include_arm=False, include_base=False)
    print("[get_observation] camera only keys:", sorted(obs_cam.keys()))

    # ========== 11. set_mode ==========
    env.set_mode(3, side="left")   # 左臂重力补偿
    print("\n[set_mode] 左臂 gravity 模式")
    _time.sleep(2.0)

    env.set_mode(1, side="left")   # 左臂回零
    print("[set_mode] 左臂 home")

    # ========== 12. close ==========
    env.close()
    print("\n[close] 完成")
