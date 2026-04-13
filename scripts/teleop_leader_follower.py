"""Leader-follower teleop entry point.

Usage (on machine B with hardware):

    python scripts/teleop_leader_follower.py
    python scripts/teleop_leader_follower.py --leader right
    python scripts/teleop_leader_follower.py --rate 30 --alpha 0.3
"""

import argparse

from arx_toolkit.env import ARXEnv
from arx_toolkit.teleop import LeaderFollowerTeleop


def main():
    parser = argparse.ArgumentParser(description="Single-arm leader-follower teleop")
    parser.add_argument(
        "--leader", type=str, default="left", choices=["left", "right"],
        help="Which arm is the leader (human drags). Default: left",
    )
    parser.add_argument(
        "--rate", type=float, default=50.0,
        help="Control loop frequency in Hz. Default: 50",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Low-pass filter coefficient (0,1]. 1.0 = no filter. Default: 0.5",
    )
    parser.add_argument(
        "--deadband", type=float, default=0.004,
        help="Dead-band threshold. Default: 0.004",
    )
    args = parser.parse_args()

    # No camera needed for pure teleop
    env = ARXEnv(
        action_mode="absolute_joint",
        camera_type="rgb",
        camera_view=(),
    )

    teleop = LeaderFollowerTeleop(
        env,
        leader_side=args.leader,
        control_rate=args.rate,
        lowpass_alpha=args.alpha,
        deadband=args.deadband,
    )

    try:
        teleop.run_interactive()
    finally:
        env.close()


if __name__ == "__main__":
    main()
