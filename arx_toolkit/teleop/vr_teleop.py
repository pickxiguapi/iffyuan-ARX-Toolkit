"""VR dual-arm teleoperation via Quest 3.

Uses WebXR (A-Frame) on the Quest 3 browser to capture controller
position/orientation, sends data over WSS to this Python server, which
converts hand movements into **delta_eef** actions for ARXEnv.

Architecture::

    Quest 3 browser (A-Frame WebXR)
        |  vr_app.js: per-frame position/quaternion/trigger/grip
        |  sent as JSON over WSS
        v
    VRTeleop (this module)
        +-- HTTPS server (serves web-ui static files, required by WebXR)
        +-- WebSocket server (receives controller data)
        +-- Control loop (50 Hz): hand state -> delta_eef -> EMA smooth -> env.step

Control mapping (delta_eef):
    - Grip pressed: record origin position/quaternion
    - Grip held:    delta_xyz = (current - origin) * scale
                    delta_roll/pitch from relative quaternion
    - Grip released: arm action = None (hold position)
    - Trigger > 0.5: gripper open (1.0 target)
    - Trigger <= 0.5: gripper closed (0.0 target)
    - X button (left controller): speed up
    - Y button (left controller): speed down
    - 5 speed levels: [0.2, 0.4, 0.6, 0.8, 1.0]

Usage::

    from arx_toolkit.env import ARXEnv
    from arx_toolkit.teleop import VRTeleop

    env = ARXEnv(action_mode="delta_eef", camera_type="rgb", camera_view=())
    vr = VRTeleop(env)
    vr.run()  # blocks, Ctrl+C to quit
"""

from __future__ import annotations

import asyncio
import http.server
import json
import logging
import math
import os
import socket
import ssl
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from arx_toolkit.env.arx_env import ARXEnv
from arx_toolkit.utils.logger import get_logger

logger = get_logger("arx_toolkit.teleop.vr")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPEED_SCALES = [0.2, 0.4, 0.6, 0.8, 1.0]

# Smoothing defaults
DEFAULT_EMA_ALPHA = 0.3     # lower = smoother but more latent (range 0~1)
DEFAULT_DEADZONE = 0.003    # meters — ignore hand jitter below this threshold


# ---------------------------------------------------------------------------
# SSL certificate utilities
# ---------------------------------------------------------------------------

def _ensure_ssl_certificates(certfile: str, keyfile: str) -> bool:
    """Generate self-signed SSL certificates if they don't already exist.

    Quest 3 WebXR requires HTTPS, so we need a TLS certificate even for
    LAN-only use.  The generated cert is valid for 365 days.
    """
    if os.path.exists(certfile) and os.path.exists(keyfile):
        logger.info("SSL certificates already exist: %s, %s", certfile, keyfile)
        return True

    logger.info("Generating self-signed SSL certificates ...")
    try:
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", keyfile,
                "-out", certfile,
                "-sha256", "-days", "365", "-nodes",
                "-subj", "/C=CN/ST=Dev/L=Dev/O=ARXToolkit/OU=VR/CN=localhost",
            ],
            check=True,
            capture_output=True,
        )
        os.chmod(keyfile, 0o600)
        os.chmod(certfile, 0o644)
        logger.info("SSL certificates generated: %s, %s", certfile, keyfile)
        return True
    except Exception as e:
        logger.error("Failed to generate SSL certificates: %s", e)
        return False


def _get_local_ip() -> str:
    """Best-effort detection of the machine's LAN IP address."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "localhost"


# ---------------------------------------------------------------------------
# VR controller state
# ---------------------------------------------------------------------------

@dataclass
class _ControllerState:
    """Per-hand VR controller tracking state."""

    hand: str  # "left" or "right"

    # Grip activation
    grip_active: bool = False

    # Origin recorded when grip is first pressed
    origin_position: Optional[np.ndarray] = None
    origin_quaternion: Optional[np.ndarray] = None

    # Latest position / quaternion from VR
    current_position: Optional[np.ndarray] = None
    current_quaternion: Optional[np.ndarray] = None

    # Trigger state (gripper)
    trigger_value: float = 0.0

    def reset(self) -> None:
        self.grip_active = False
        self.origin_position = None
        self.origin_quaternion = None
        self.current_position = None
        self.current_quaternion = None
        self.trigger_value = 0.0


# ---------------------------------------------------------------------------
# Quaternion rotation helpers
# ---------------------------------------------------------------------------

def _extract_axis_rotation(
    current_quat: np.ndarray,
    origin_quat: np.ndarray,
    axis_index: int,
) -> float:
    """Extract rotation (degrees) around a specific axis from relative quaternion.

    Parameters
    ----------
    current_quat, origin_quat : (4,) arrays  [x, y, z, w]
    axis_index : 0=X (pitch), 1=Y (yaw), 2=Z (roll)
    """
    try:
        origin_rot = R.from_quat(origin_quat)
        current_rot = R.from_quat(current_quat)
        relative_rot = current_rot * origin_rot.inv()
        rotvec = relative_rot.as_rotvec()  # [rx, ry, rz] in radians
        return np.degrees(rotvec[axis_index])
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# HTTPS static file server
# ---------------------------------------------------------------------------

class _StaticHandler(http.server.BaseHTTPRequestHandler):
    """Minimal HTTPS handler that serves the VR web-ui static files.

    When ``swap_buttons`` is set on the server, accessing ``/`` without
    ``?swap=1`` will be redirected to ``/?swap=1``.
    """

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        try:
            super().end_headers()
        except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
            pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress noisy HTTP logs

    def do_GET(self):
        raw_path = self.path
        path = raw_path.split("?")[0]  # strip query string
        query = raw_path.split("?")[1] if "?" in raw_path else ""

        # Redirect to /?swap=1 if swap_buttons is enabled and not already set
        swap_buttons: bool = getattr(self.server, "swap_buttons", False)
        if swap_buttons and path in ("/", "/index.html") and "swap=1" not in query:
            location = "/?swap=1"
            self.send_response(302)
            self.send_header("Location", location)
            self.end_headers()
            return

        # Route to files
        if path in ("/", "/index.html"):
            self._serve("index.html", "text/html")
        elif path.endswith(".js"):
            self._serve(path.lstrip("/"), "application/javascript")
        elif path.endswith(".css"):
            self._serve(path.lstrip("/"), "text/css")
        elif path.endswith((".jpg", ".jpeg", ".png", ".gif")):
            ct = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                  ".png": "image/png", ".gif": "image/gif"}
            ext = os.path.splitext(path)[1]
            self._serve(path.lstrip("/"), ct.get(ext, "application/octet-stream"))
        else:
            self.send_error(404)

    def _serve(self, relpath: str, content_type: str):
        web_root: str = getattr(self.server, "web_root", "")
        fpath = os.path.join(web_root, relpath)
        if os.path.isfile(fpath):
            with open(fpath, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404, f"Not found: {relpath}")


# ---------------------------------------------------------------------------
# VRTeleop
# ---------------------------------------------------------------------------

class VRTeleop:
    """VR dual-arm teleoperation for ARX LIFT2.

    Parameters
    ----------
    env : ARXEnv
        Must be created with ``action_mode="delta_eef"``.
    https_port : int
        Port for the HTTPS server serving the WebXR page.
    ws_port : int
        Port for the secure WebSocket server.
    host : str
        Bind address.  ``"0.0.0.0"`` = all interfaces.
    control_rate : float
        Frequency (Hz) at which ``env.step`` is called.
    vr_to_robot_scale : float
        Multiplier applied to VR position deltas before sending to robot.
    axis_mapping : tuple[int, int, int]
        Index mapping from VR (x, y, z) to robot (x, y, z).
        Default ``(2, 0, 1)`` maps VR→robot as:
        robot_x ← vr_z, robot_y ← vr_x, robot_z ← vr_y.
    axis_sign : tuple[float, float, float]
        Sign flip per axis.  Default ``(-1.0, 1.0, 1.0)`` gives:
        robot_x = -vr_z (push forward → +X),
        robot_y = +vr_x (move left → +Y),
        robot_z = +vr_y (raise hand → +Z).
    rot_scale : float
        Multiplier applied to wrist rotation deltas (degrees -> radians).
    ema_alpha : float
        Exponential moving average smoothing factor (0~1).
        Lower = smoother but more lag.  Default 0.3.
    deadzone : float
        Position deadzone in meters.  Deltas with norm below this are
        zeroed out to suppress hand jitter.  Default 0.003 (3 mm).
    swap_buttons : bool
        If True, swap trigger/grip roles: trigger=arm activate, grip=gripper.
        Default False: grip=arm activate, trigger=gripper.
    certfile, keyfile : str or None
        Paths to SSL cert/key.  ``None`` = auto-generate in cwd.
    """

    def __init__(
        self,
        env: ARXEnv,
        https_port: int = 8443,
        ws_port: int = 8442,
        host: str = "0.0.0.0",
        control_rate: float = 50.0,
        vr_to_robot_scale: float = 1.0,
        axis_mapping: Tuple[int, int, int] = (2, 0, 1),
        axis_sign: Tuple[float, float, float] = (-1.0, 1.0, 1.0),
        rot_scale: float = 1.0,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        deadzone: float = DEFAULT_DEADZONE,
        swap_buttons: bool = False,
        certfile: Optional[str] = None,
        keyfile: Optional[str] = None,
    ):
        self.env = env
        self.https_port = https_port
        self.ws_port = ws_port
        self.host = host
        self.control_rate = control_rate
        self.vr_to_robot_scale = vr_to_robot_scale
        self.axis_mapping = axis_mapping
        self.axis_sign = np.array(axis_sign, dtype=np.float64)
        self.rot_scale = rot_scale
        self.ema_alpha = ema_alpha
        self.deadzone = deadzone
        self.swap_buttons = swap_buttons

        # SSL
        self.certfile = certfile or "cert.pem"
        self.keyfile = keyfile or "key.pem"

        # Controller states
        self._left = _ControllerState(hand="left")
        self._right = _ControllerState(hand="right")
        self._lock = threading.Lock()

        # Speed level (index into SPEED_SCALES)
        self._speed_level: int = 0  # default: slowest (0.2x)
        self._prev_speed_level: int = 0

        # Track first-time arm activation for terminal hints
        self._left_activated_once: bool = False
        self._right_activated_once: bool = False

        # EMA smoothed action per hand (7D: xyz + rpy + gripper)
        self._ema_left: Optional[np.ndarray] = None
        self._ema_right: Optional[np.ndarray] = None

        # Servers
        self._httpd: Optional[http.server.HTTPServer] = None
        self._httpd_thread: Optional[threading.Thread] = None
        self._ws_server = None  # websockets.serve handle

        # Web-UI directory
        self._web_ui_dir = str(Path(__file__).parent / "vr_web_ui")

        # Running flag
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start HTTPS + WebSocket servers and the control loop."""
        # 1. SSL certs
        if not _ensure_ssl_certificates(self.certfile, self.keyfile):
            raise RuntimeError("Cannot start without SSL certificates.")

        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(certfile=self.certfile, keyfile=self.keyfile)

        # 2. HTTPS static server (runs in a thread)
        self._httpd = http.server.HTTPServer(
            (self.host, self.https_port), _StaticHandler
        )
        self._httpd.web_root = self._web_ui_dir  # type: ignore[attr-defined]
        self._httpd.swap_buttons = self.swap_buttons  # type: ignore[attr-defined]

        https_ssl = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        https_ssl.load_cert_chain(certfile=self.certfile, keyfile=self.keyfile)
        self._httpd.socket = https_ssl.wrap_socket(
            self._httpd.socket, server_side=True
        )

        self._httpd_thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True
        )
        self._httpd_thread.start()

        # 3. WebSocket server
        import websockets  # lazy import

        self._ws_server = await websockets.serve(
            self._ws_handler,
            self.host,
            self.ws_port,
            ssl=ssl_ctx,
        )

        self._running = True

        host_display = _get_local_ip() if self.host == "0.0.0.0" else self.host
        btn_mode = "trigger=臂, grip=夹爪" if self.swap_buttons else "grip=臂, trigger=夹爪"
        print(
            f"\n{'='*60}\n"
            f"  VR Teleop started\n"
            f"  HTTPS : https://{host_display}:{self.https_port}\n"
            f"  WSS   : wss://{host_display}:{self.ws_port}\n"
            f"  Rate  : {self.control_rate} Hz\n"
            f"  Scale : {self.vr_to_robot_scale}\n"
            f"  Axis  : mapping={self.axis_mapping}, sign={tuple(self.axis_sign)}\n"
            f"  Buttons: {btn_mode}\n"
            f"  Speed : {SPEED_SCALES[self._speed_level]} (level {self._speed_level+1}/5)\n"
            f"  Smooth: EMA alpha={self.ema_alpha}, deadzone={self.deadzone*1000:.1f}mm\n"
            f"\n  Open the HTTPS URL on Quest 3 browser.\n"
            f"  X = speed up, Y = speed down.\n"
            f"  Ctrl+C to quit.\n"
            f"{'='*60}\n"
        )

        # 4. Launch control loop as async task
        await self._control_loop()

    async def stop(self) -> None:
        """Gracefully shut down all services."""
        self._running = False
        if self._ws_server is not None:
            self._ws_server.close()
            await self._ws_server.wait_closed()
            self._ws_server = None
        if self._httpd is not None:
            self._httpd.shutdown()
            if self._httpd_thread is not None:
                self._httpd_thread.join(timeout=5)
            self._httpd = None
            self._httpd_thread = None
        print("\n\033[1m\033[96m⏹  VR Teleop 已停止\033[0m")
        logger.info("VR Teleop stopped.")

    def run(self) -> None:
        """Blocking entry point — runs the event loop until Ctrl+C."""
        try:
            asyncio.run(self.start())
        except KeyboardInterrupt:
            print("\nStopping VR Teleop ...")
            # stop is called within the finally of _control_loop
        except Exception as e:
            logger.error("VR Teleop error: %s", e, exc_info=True)

    # ------------------------------------------------------------------
    # WebSocket handler
    # ------------------------------------------------------------------

    async def _ws_handler(self, websocket, path=None):
        """Handle a single VR client WebSocket connection."""
        addr = websocket.remote_address
        logger.info("VR client connected: %s", addr)
        print(
            f"\n\033[1m\033[92m"
            f"{'='*60}\n"
            f"  🎮  VR 已连接 — 开始操控!\n"
            f"{'='*60}"
            f"\033[0m\n"
        )

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    self._process_vr_data(data)
                except json.JSONDecodeError:
                    logger.warning("Non-JSON message from VR client")
                except Exception as e:
                    logger.error("Error processing VR data: %s", e, exc_info=True)
        except Exception:
            pass  # connection closed
        finally:
            # On disconnect, release grips
            with self._lock:
                self._left.reset()
                self._right.reset()
            logger.info("VR client disconnected: %s", addr)
            print(
                f"\n\033[1m\033[93m⚠️  VR 已断开\033[0m\n"
            )

    def _process_vr_data(self, data: Dict) -> None:
        """Parse incoming VR JSON and update controller states.

        Expected format (sent every frame by vr_app.js)::

            {
              "timestamp": ...,
              "leftController":  { position, quaternion, gripActive, trigger, ... },
              "rightController": { position, quaternion, gripActive, trigger, ... },
              "headset": { ... },
              "speedLevel": 0|1|2
            }
        """
        with self._lock:
            if "leftController" in data:
                self._update_hand(self._left, data["leftController"])
            if "rightController" in data:
                self._update_hand(self._right, data["rightController"])

            # Speed level
            new_speed = data.get("speedLevel", self._speed_level)
            if isinstance(new_speed, int) and 0 <= new_speed <= 4:
                if new_speed != self._speed_level:
                    self._speed_level = new_speed
                    print(
                        f"\033[1m\033[94m[SPEED] 档位: {self._speed_level+1}/5 "
                        f"(scale={SPEED_SCALES[self._speed_level]})\033[0m"
                    )

    def _update_hand(self, state: _ControllerState, d: Dict) -> None:
        """Update a single controller state from VR data."""
        pos = d.get("position")
        quat = d.get("quaternion")
        grip = d.get("gripActive", False)
        trigger = d.get("trigger", 0.0)

        # Update trigger value
        state.trigger_value = float(trigger)

        # Update position
        if pos and all(k in pos for k in ("x", "y", "z")):
            state.current_position = np.array(
                [pos["x"], pos["y"], pos["z"]], dtype=np.float64
            )

        # Update quaternion
        if quat and all(k in quat for k in ("x", "y", "z", "w")):
            state.current_quaternion = np.array(
                [quat["x"], quat["y"], quat["z"], quat["w"]], dtype=np.float64
            )

        # Grip state machine
        if grip and not state.grip_active:
            # Grip just pressed — record origin
            state.grip_active = True
            state.origin_position = (
                state.current_position.copy() if state.current_position is not None else None
            )
            state.origin_quaternion = (
                state.current_quaternion.copy() if state.current_quaternion is not None else None
            )
            logger.info(
                "%s grip ACTIVATED — origin recorded", state.hand.upper()
            )
            # First-time activation hint
            if state.hand == "left" and not self._left_activated_once:
                self._left_activated_once = True
                print(f"\033[1m\033[92m[LEFT] 臂激活 ✓\033[0m")
            elif state.hand == "right" and not self._right_activated_once:
                self._right_activated_once = True
                print(f"\033[1m\033[92m[RIGHT] 臂激活 ✓\033[0m")
        elif not grip and state.grip_active:
            # Grip released
            state.grip_active = False
            state.origin_position = None
            state.origin_quaternion = None
            logger.info("%s grip RELEASED", state.hand.upper())

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    async def _control_loop(self) -> None:
        """Fixed-rate loop that converts VR state to env.step actions."""
        dt = 1.0 / self.control_rate
        logger.info("Control loop started at %.1f Hz", self.control_rate)

        try:
            while self._running:
                t0 = time.monotonic()
                self._tick()
                elapsed = time.monotonic() - t0
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    def _tick(self) -> None:
        """One control cycle: read VR states → build action → EMA smooth → env.step."""
        with self._lock:
            left_raw = self._compute_arm_action(self._left)
            right_raw = self._compute_arm_action(self._right)

        # EMA smoothing
        left_action = self._apply_ema(left_raw, "left")
        right_action = self._apply_ema(right_raw, "right")

        action = {
            "left": left_action,
            "right": right_action,
            "base": None,   # VR does not control base
            "lift": None,   # VR does not control lift
        }

        # Only step if at least one arm is active
        if left_action is not None or right_action is not None:
            try:
                self.env.step(action)
            except Exception as e:
                logger.error("env.step error: %s", e)

    def _apply_ema(
        self, raw: Optional[np.ndarray], hand: str
    ) -> Optional[np.ndarray]:
        """Apply exponential moving average smoothing to a 7D action.

        When the hand is released (raw=None), reset the EMA state so the
        next activation starts fresh.
        """
        if raw is None:
            # Reset EMA state on release
            if hand == "left":
                self._ema_left = None
            else:
                self._ema_right = None
            return None

        prev = self._ema_left if hand == "left" else self._ema_right
        alpha = self.ema_alpha

        if prev is None:
            smoothed = raw.copy()
        else:
            smoothed = alpha * raw + (1.0 - alpha) * prev

        # Store
        if hand == "left":
            self._ema_left = smoothed
        else:
            self._ema_right = smoothed

        return smoothed

    def _compute_arm_action(self, state: _ControllerState) -> Optional[np.ndarray]:
        """Convert a single controller state to a 7D delta_eef action.

        Returns None if the grip is not active (arm should hold position).

        Action layout: [dx, dy, dz, droll, dpitch, dyaw, gripper_target]
        """
        if not state.grip_active:
            return None

        if (
            state.current_position is None
            or state.origin_position is None
        ):
            return None

        # --- Position delta ---
        raw_delta = (state.current_position - state.origin_position)

        # Apply axis mapping + sign + scale + speed
        speed_scale = SPEED_SCALES[self._speed_level]
        delta_xyz = np.zeros(3)
        for robot_i in range(3):
            vr_i = self.axis_mapping[robot_i]
            delta_xyz[robot_i] = raw_delta[vr_i] * self.axis_sign[robot_i]
        delta_xyz *= self.vr_to_robot_scale * speed_scale

        # Deadzone — suppress hand jitter
        if np.linalg.norm(delta_xyz) < self.deadzone:
            delta_xyz[:] = 0.0

        # --- Rotation delta ---
        droll = 0.0
        dpitch = 0.0
        dyaw = 0.0

        if (
            state.current_quaternion is not None
            and state.origin_quaternion is not None
        ):
            # Z-axis rotation -> wrist roll
            roll_deg = _extract_axis_rotation(
                state.current_quaternion, state.origin_quaternion, 2
            )
            # X-axis rotation -> wrist pitch
            pitch_deg = _extract_axis_rotation(
                state.current_quaternion, state.origin_quaternion, 0
            )

            # Convert degrees to radians, apply scale
            droll = math.radians(-roll_deg) * self.rot_scale
            dpitch = math.radians(-pitch_deg) * self.rot_scale

        # --- Gripper ---
        # trigger > 0.5 => open (0.0).  Push gripper toward 0 with negative delta.
        # trigger <= 0.5 => closed (1.0).  Push gripper toward 1 with positive delta.
        if state.trigger_value > 0.5:
            gripper_delta = -0.1
        else:
            gripper_delta = 0.1

        return np.array(
            [delta_xyz[0], delta_xyz[1], delta_xyz[2],
             droll, dpitch, dyaw, gripper_delta],
            dtype=np.float64,
        )
