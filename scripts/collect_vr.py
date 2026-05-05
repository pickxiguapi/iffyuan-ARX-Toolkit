#!/usr/bin/env python3
"""VR Zarr data collection entry point.

This script records the robot while the ROS2 VR stack is controlling it.
Start the hardware/VR nodes first:

    bash scripts/collect_VR.sh

Then record Zarr episodes:

    python scripts/collect_vr.py \
        --dataset datasets/take_part_bricks_vr.zarr \
        --arm-mode dual \
        --action-mode absolute_joint \
        --cameras camera_h camera_l camera_r \
        --task "take apart the building bricks"
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ARX LIFT2 VR Zarr data collection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default="datasets/vr.zarr",
        help="Zarr dataset path.",
    )
    parser.add_argument(
        "--arm-mode",
        choices=["dual", "single"],
        default="dual",
        help="Record both arms or one selected arm.",
    )
    parser.add_argument(
        "--leader-side",
        choices=["left", "right"],
        default="left",
        help="Arm to keep when --arm-mode single.",
    )
    parser.add_argument(
        "--action-mode",
        choices=["absolute_joint", "absolute_eef"],
        default="absolute_joint",
        help="Action fields saved to action_left/action_right.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Frame rate for Zarr sampling.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop each episode after N frames. 0 means no limit.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Stop after saving N new episodes. 0 means interactive unlimited.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["camera_h", "camera_l", "camera_r"],
        help="Camera names to save.",
    )
    parser.add_argument(
        "--with-depth",
        action="store_true",
        help="Subscribe and save aligned depth frames.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[640, 480],
        metavar=("W", "H"),
        help="Saved image size.",
    )
    parser.add_argument(
        "--task",
        default="",
        help="Task description stored in Zarr metadata.",
    )
    parser.add_argument(
        "--include-base",
        action="store_true",
        help="Record VR base command fields into action_base/action_lift.",
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=5,
        help="Countdown seconds before each recording starts.",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable spd-say countdown prompts.",
    )
    args = parser.parse_args()

    from arx_toolkit.collect import collect_vr_episode
    from arx_toolkit.env import ARXEnv

    camera_type = "rgbd" if args.with_depth else "rgb"
    print(f"[INFO] Init ARXEnv(camera_type={camera_type}, cameras={args.cameras})")
    env = ARXEnv(
        camera_type=camera_type,
        camera_view=tuple(args.cameras),
        img_size=tuple(args.image_size),
    )

    try:
        collect_vr_episode(
            env=env,
            arm_mode=args.arm_mode,
            dataset_path=args.dataset,
            frame_rate=args.fps,
            max_frames=args.max_frames,
            max_episodes=args.max_episodes,
            action_mode=args.action_mode,
            camera_names=tuple(args.cameras),
            with_depth=args.with_depth,
            img_size=tuple(args.image_size),
            task=args.task,
            leader_side=args.leader_side,
            include_base=args.include_base,
            countdown_sec=args.countdown,
            use_tts=not args.no_tts,
        )
    finally:
        print("[INFO] Closing environment...")
        env.close()


if __name__ == "__main__":
    main()
