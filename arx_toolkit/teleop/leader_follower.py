"""Leader-follower single-arm teleoperation.

One arm (leader) is put into gravity-compensation mode so the operator can
freely drag it by hand.  A background thread reads the leader's joint
positions at a fixed rate and mirrors them to the follower arm after applying
low-pass filtering and dead-band suppression.

Usage::

    from arx_toolkit.env import ARXEnv
    from arx_toolkit.teleop import LeaderFollowerTeleop

    env = ARXEnv(action_mode="absolute_joint", camera_type="rgb",
                 camera_view=())
    teleop = LeaderFollowerTeleop(env, leader_side="left")
    teleop.run_interactive()   # blocks until Enter
    env.close()
"""

from __future__ import annotations

import sys
import termios
import threading
import time
import tty
from typing import Optional

import numpy as np

from arx_toolkit.env.arx_env import ARXEnv
from arx_toolkit.utils.logger import get_logger

logger = get_logger("arx_toolkit.teleop")

# ---------------------------------------------------------------------------
# Smoothing helpers
# ---------------------------------------------------------------------------

def _lowpass(target: np.ndarray, previous: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential low-pass filter.

    ``alpha = 1.0`` means no filtering (use raw target).
    Smaller alpha ⇒ smoother / more lag.
    """
    return alpha * target + (1.0 - alpha) * previous


def _deadband(filtered: np.ndarray, previous: np.ndarray, threshold: float) -> np.ndarray:
    """Per-dimension dead-band: keep previous value when change is tiny."""
    diff = np.abs(filtered - previous)
    out = filtered.copy()
    out[diff < threshold] = previous[diff < threshold]
    return out


def _read_key() -> str:
    """Read a single keypress from stdin (blocking, raw mode).

    Returns the character read.  For Enter returns ``'\\n'``,
    for Space returns ``' '``.
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def _print_status(state: str, leader: str, follower: str, rate: float) -> None:
    """Print the current teleop status banner."""
    if state == "RUNNING":
        print(
            f"\n=== Leader-Follower Teleop [{state}] ===\n"
            f"  Leader : {leader} (gravity mode — drag freely)\n"
            f"  Follower: {follower} (mirrors leader)\n"
            f"  Rate   : {rate} Hz\n"
            f"\n  [Enter] pause   [Ctrl+C] quit\n"
        )
    else:  # PAUSED
        print(
            f"\n=== Leader-Follower Teleop [{state}] ===\n"
            f"  Follower holds current position.\n"
            f"\n  [Enter] resume   [Space] go home & quit   [Ctrl+C] quit\n"
        )


# ---------------------------------------------------------------------------
# LeaderFollowerTeleop
# ---------------------------------------------------------------------------

class LeaderFollowerTeleop:
    """Mirror one arm (leader) to the other (follower) in joint space.

    Parameters
    ----------
    env : ARXEnv
        An already-initialised environment instance.
    leader_side : str
        ``"left"`` or ``"right"`` — the arm the human drags.
    control_rate : float
        Target loop frequency in Hz (default 50).
    lowpass_alpha : float
        Low-pass coefficient in (0, 1]. 1.0 = no filtering.
    deadband : float
        Dead-band threshold (rad / normalised gripper units).
    """

    def __init__(
        self,
        env: ARXEnv,
        leader_side: str = "left",
        control_rate: float = 50.0,
        lowpass_alpha: float = 0.5,
        deadband: float = 0.004,
    ):
        if leader_side not in ("left", "right"):
            raise ValueError(f"leader_side must be 'left' or 'right', got {leader_side!r}")

        self.env = env
        self.leader_side = leader_side
        self.follower_side = "right" if leader_side == "left" else "left"
        self.control_rate = control_rate
        self.lowpass_alpha = lowpass_alpha
        self.deadband = deadband

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._prev_cmd: Optional[np.ndarray] = None  # last command sent
        self._last_command: Optional[dict] = None  # exposed for Collector

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def last_command(self) -> Optional[dict]:
        """Last action dict sent to the follower (for Collector to read).

        Returns a dict in the same format as ``ARXEnv.step(action)``::

            {
                "left":  np.ndarray(7,) | None,
                "right": np.ndarray(7,) | None,
                "base":  None,
                "lift":  None,
            }

        Returns ``None`` if teleop has not yet produced any command.
        """
        return self._last_command

    def start(self) -> None:
        """Set leader arm to gravity mode and launch the control thread."""
        if self._running:
            logger.warning("Already running — call stop() first.")
            return

        # Put leader into gravity-compensation mode (mode=3)
        logger.info("Setting %s arm to gravity mode …", self.leader_side)
        self.env.set_mode(3, side=self.leader_side)
        time.sleep(0.3)  # brief settle

        # Seed the filter with the current follower position
        obs = self.env.get_observation(include_camera=False, include_base=False)
        self._prev_cmd = obs[f"{self.follower_side}_joint_pos"].copy()

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(
            "Teleop started: %s (leader/gravity) → %s (follower)",
            self.leader_side, self.follower_side,
        )

    def stop(self) -> None:
        """Stop the control thread.  Follower holds its last position."""
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Teleop stopped.")

    def run_interactive(self) -> None:
        """Interactive loop with pause / resume / home-and-quit.

        Key bindings (single keypress, no need to press Enter):

        - **Enter** while running  → pause (follower holds position)
        - **Enter** while paused   → resume teleop
        - **Space** while paused   → go home (both arms) and exit
        - **Ctrl+C**               → immediate stop and exit
        """
        self.start()
        _print_status("RUNNING", self.leader_side, self.follower_side,
                       self.control_rate)
        try:
            while True:
                key = _read_key()

                if key == '\x03':          # Ctrl+C
                    break

                if key in ('\r', '\n'):    # Enter
                    if self._running:
                        # --- pause ---
                        self.stop()
                        _print_status("PAUSED", self.leader_side,
                                       self.follower_side, self.control_rate)
                    else:
                        # --- resume ---
                        self.start()
                        _print_status("RUNNING", self.leader_side,
                                       self.follower_side, self.control_rate)

                elif key == ' ':           # Space
                    if not self._running:
                        # --- go home & quit ---
                        logger.info("Homing both arms …")
                        self.env.set_mode(1, side="both")
                        print("\n  ✓ Both arms homed.\n")
                        return
                    # Space while running: ignore (must pause first)

        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            self.stop()

    # ------------------------------------------------------------------
    # Control loop (runs in background thread)
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        dt = 1.0 / self.control_rate
        while self._running:
            t0 = time.monotonic()
            try:
                self._tick()
            except Exception:
                logger.exception("Error in teleop loop")
            elapsed = time.monotonic() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _tick(self) -> None:
        """One control cycle: read leader → filter → send to follower."""
        obs = self.env.get_observation(include_camera=False, include_base=False)
        leader_joints = obs[f"{self.leader_side}_joint_pos"].copy()  # (7,) normalised gripper

        # Low-pass filter
        if self._prev_cmd is not None:
            filtered = _lowpass(leader_joints, self._prev_cmd, self.lowpass_alpha)
        else:
            filtered = leader_joints

        # Dead-band
        if self._prev_cmd is not None:
            cmd = _deadband(filtered, self._prev_cmd, self.deadband)
        else:
            cmd = filtered

        # Send to follower via the env internal method (bypasses validate + obs)
        self.env._apply_absolute_joint({self.follower_side: cmd})
        self._prev_cmd = cmd.copy()

        # Expose for Collector
        self._last_command = {
            "left": cmd.copy() if self.follower_side == "left" else None,
            "right": cmd.copy() if self.follower_side == "right" else None,
            "base": None,
            "lift": None,
        }
