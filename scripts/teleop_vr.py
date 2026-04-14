"""VR dual-arm teleop entry point.

Usage (on machine B with hardware):

    python scripts/teleop_vr.py
    python scripts/teleop_vr.py --scale 0.5 --rate 30
    python scripts/teleop_vr.py --swap-buttons
    python scripts/teleop_vr.py --https-port 9443 --ws-port 9442
"""

import argparse

from arx_toolkit.env import ARXEnv
from arx_toolkit.teleop import VRTeleop


def main():
    parser = argparse.ArgumentParser(
        description="VR dual-arm teleoperation for ARX LIFT2"
    )
    parser.add_argument(
        "--https-port", type=int, default=8443,
        help="HTTPS port for WebXR page. Default: 8443",
    )
    parser.add_argument(
        "--ws-port", type=int, default=8442,
        help="WebSocket port for VR data. Default: 8442",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Bind address. Default: 0.0.0.0",
    )
    parser.add_argument(
        "--rate", type=float, default=50.0,
        help="Control loop frequency in Hz. Default: 50",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="VR position delta -> robot delta scale factor. Default: 1.0",
    )
    parser.add_argument(
        "--rot-scale", type=float, default=1.0,
        help="Rotation delta scale factor. Default: 1.0",
    )
    parser.add_argument(
        "--ema-alpha", type=float, default=0.3,
        help="EMA smoothing factor (0~1). Lower = smoother. Default: 0.3",
    )
    parser.add_argument(
        "--deadzone", type=float, default=0.003,
        help="Position deadzone in meters. Default: 0.003 (3mm)",
    )
    parser.add_argument(
        "--swap-buttons", action="store_true",
        help="Swap trigger/grip roles: trigger=arm activate, grip=gripper.",
    )
    parser.add_argument(
        "--certfile", type=str, default=None,
        help="SSL cert file path. Default: auto-generate cert.pem",
    )
    parser.add_argument(
        "--keyfile", type=str, default=None,
        help="SSL key file path. Default: auto-generate key.pem",
    )
    args = parser.parse_args()

    # delta_eef mode, no camera needed for pure teleop
    env = ARXEnv(
        action_mode="delta_eef",
        camera_type="rgb",
        camera_view=(),
    )

    vr = VRTeleop(
        env,
        https_port=args.https_port,
        ws_port=args.ws_port,
        host=args.host,
        control_rate=args.rate,
        vr_to_robot_scale=args.scale,
        rot_scale=args.rot_scale,
        ema_alpha=args.ema_alpha,
        deadzone=args.deadzone,
        swap_buttons=args.swap_buttons,
        certfile=args.certfile,
        keyfile=args.keyfile,
    )

    try:
        vr.run()
    finally:
        print("\n正在回到初始位置...")
        try:
            env._go_home()
        except Exception as e:
            print(f"go_home 失败: {e}")
        env.close()


if __name__ == "__main__":
    main()
