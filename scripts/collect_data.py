#!/usr/bin/env python3
"""ARX LIFT2 数据采集入口脚本.

用法::

    # 主从遥操作采集
    python scripts/collect_data.py \
        --dataset datasets/pick_cup.zarr \
        --episodes 50 \
        --teleop leader_follower \
        --leader-side left \
        --hz 50 \
        --cam-mode rgbd \
        --image-size 640 480 \
        --task "pick up the cup"

    # 断点续采（自动识别已有 episodes）
    python scripts/collect_data.py \
        --dataset datasets/pick_cup.zarr \
        --episodes 50
"""

from __future__ import annotations

import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description="ARX LIFT2 数据采集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", "-d", default="datasets/demo.zarr",
        help="Zarr 数据集路径（默认 datasets/demo.zarr）",
    )
    parser.add_argument(
        "--episodes", "-n", type=int, default=3,
        help="目标 episode 总数（含已有，默认 3）",
    )
    parser.add_argument(
        "--teleop", choices=["leader_follower"], default="leader_follower",
        help="遥操作方式（默认 leader_follower）",
    )
    parser.add_argument(
        "--action-mode",
        choices=["delta_eef", "absolute_eef", "absolute_joint"],
        default="absolute_joint",
        help="Action mode for ARXEnv. 当前 leader_follower 只支持 absolute_joint.",
    )
    parser.add_argument(
        "--leader-side", choices=["left", "right"], default="left",
        help="Leader 臂（默认 left）",
    )
    parser.add_argument(
        "--hz", type=float, default=30.0,
        help="采集频率 Hz（默认 30）",
    )
    parser.add_argument(
        "--cam-mode", choices=["rgb", "rgbd"], default="rgbd",
        help="相机模式（默认 rgbd）",
    )
    parser.add_argument(
        "--image-size", type=int, nargs=2, default=[640, 480],
        metavar=("W", "H"),
        help="图像尺寸 W H（默认 640 480）",
    )
    parser.add_argument(
        "--task", default="",
        help="任务描述（写入 metadata）",
    )
    parser.add_argument(
        "--save-video", action="store_true",
        help="保存回放视频",
    )
    parser.add_argument(
        "--control-rate", type=float, default=50.0,
        help="Teleop 控制频率 Hz（默认 50）",
    )
    parser.add_argument(
        "--lowpass-alpha", type=float, default=0.5,
        help="低通滤波系数 (0,1]（默认 0.5）",
    )
    parser.add_argument(
        "--deadband", type=float, default=0.004,
        help="死区阈值（默认 0.004）",
    )

    args = parser.parse_args()

    # TODO: 后续接入 VR、策略回放等采集方式时，在这里扩展 teleop/action_mode 的合法组合。
    if args.teleop == "leader_follower" and args.action_mode != "absolute_joint":
        parser.error(
            "--teleop leader_follower 当前只支持 --action-mode absolute_joint"
        )

    # ------------------------------------------------------------------
    # 1. Init environment
    # ------------------------------------------------------------------
    from arx_toolkit.env import ARXEnv
    from arx_toolkit.collect import Collector

    print(f"[INFO] 当前动作模式 action_mode={args.action_mode}")
    print(f"[INFO] 初始化 ARXEnv (action_mode={args.action_mode}, camera_type={args.cam_mode})")
    env = ARXEnv(
        action_mode=args.action_mode,
        camera_type=args.cam_mode,
        camera_view=("camera_l", "camera_h", "camera_r"),
        img_size=tuple(args.image_size),
    )

    # ------------------------------------------------------------------
    # 2. Init teleop
    # ------------------------------------------------------------------
    teleop = None
    action_source = None

    if args.teleop == "leader_follower":
        from arx_toolkit.teleop import LeaderFollowerTeleop

        print(f"[INFO] 初始化 LeaderFollowerTeleop (leader={args.leader_side}, "
              f"rate={args.control_rate}Hz, alpha={args.lowpass_alpha})")
        teleop = LeaderFollowerTeleop(
            env,
            leader_side=args.leader_side,
            control_rate=args.control_rate,
            lowpass_alpha=args.lowpass_alpha,
            deadband=args.deadband,
        )

        def _get_teleop_action() -> dict:
            """Read latest command from teleop thread."""
            cmd = teleop.last_command
            if cmd is None:
                # Teleop hasn't produced a command yet — return idle action
                return {"left": None, "right": None, "base": None, "lift": None}
            return cmd

        action_source = _get_teleop_action

    # ------------------------------------------------------------------
    # 3. Start teleop
    # ------------------------------------------------------------------
    if teleop is not None:
        print("[INFO] 启动遥操作...")
        teleop.start()
        time.sleep(0.5)  # let teleop thread settle

    # ------------------------------------------------------------------
    # 4. Run collector
    # ------------------------------------------------------------------
    try:
        collector = Collector(
            env=env,
            action_source=action_source,
            dataset_path=args.dataset,
            num_episodes=args.episodes,
            hz=args.hz,
            cam_mode=args.cam_mode,
            image_size=tuple(args.image_size),
            task=args.task,
            action_mode=args.action_mode,
            save_video=args.save_video,
        )
        collector.run()

    finally:
        # ------------------------------------------------------------------
        # 5. Cleanup
        # ------------------------------------------------------------------
        if teleop is not None:
            print("[INFO] 停止遥操作...")
            teleop.stop()

        print("[INFO] 关闭环境...")
        env.close()
        print("[INFO] 完成。")


if __name__ == "__main__":
    main()
