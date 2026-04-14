"""模拟 VR 手柄操作 → 真机测试方向。

不需要 VR 手柄，脚本模拟手柄在 VR 空间中的移动/旋转，
经过 VRTeleop 的坐标映射后发给真机，观察实际方向是否正确。

每个动作按 Enter 执行，观察后自动回原位。

Usage (on machine B):
    python scripts/test_vr_direction.py
    python scripts/test_vr_direction.py --step 0.03
    python scripts/test_vr_direction.py --skip-rot
"""

import argparse
import time
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from arx_toolkit.env import ARXEnv
from arx_toolkit.teleop.vr_teleop import (
    _ControllerState,
    _extract_axis_rotation,
    SPEED_SCALES,
)
import math


# ---------------------------------------------------------------------------
# 模拟 VR 手柄 → 经过映射 → 得到 robot action
# ---------------------------------------------------------------------------

def simulate_vr_action(
    vr_delta_pos,
    vr_rot_axis=None,
    vr_rot_deg=0.0,
    trigger=0.0,
    axis_mapping=(2, 0, 1),
    axis_sign=(-1.0, -1.0, 1.0),
    rot_scale=1.0,
):
    """模拟 VRTeleop._compute_arm_action 的映射逻辑。

    Parameters
    ----------
    vr_delta_pos : 3-array, VR 空间的位移 [vr_x, vr_y, vr_z]
    vr_rot_axis : "x"/"y"/"z" or None
    vr_rot_deg : float, 旋转角度（度）
    """
    vr_delta_pos = np.array(vr_delta_pos, dtype=np.float64)
    axis_sign = np.array(axis_sign, dtype=np.float64)

    # --- 位置映射 ---
    delta_xyz = np.zeros(3)
    for robot_i in range(3):
        vr_i = axis_mapping[robot_i]
        delta_xyz[robot_i] = vr_delta_pos[vr_i] * axis_sign[robot_i]

    # --- 旋转映射 ---
    droll = 0.0
    dpitch = 0.0
    dyaw = 0.0

    if vr_rot_axis is not None and abs(vr_rot_deg) > 0.01:
        # 构造原点四元数和旋转后四元数
        origin_quat = np.array([0.0, 0.0, 0.0, 1.0])  # identity
        axis_map = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
        rotvec = np.array(axis_map[vr_rot_axis]) * np.radians(vr_rot_deg)
        r = Rot.from_rotvec(rotvec)
        current_quat = r.as_quat()  # [x,y,z,w]

        roll_deg = _extract_axis_rotation(current_quat, origin_quat, 2)
        pitch_deg = _extract_axis_rotation(current_quat, origin_quat, 0)
        # 当前代码没有 yaw 映射
        droll = math.radians(-roll_deg) * rot_scale
        dpitch = math.radians(-pitch_deg) * rot_scale

    # --- 夹爪 ---
    if trigger > 0.5:
        gripper_delta = -0.1
    else:
        gripper_delta = 0.1

    return np.array([
        delta_xyz[0], delta_xyz[1], delta_xyz[2],
        droll, dpitch, dyaw, gripper_delta
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# 测试项目
# ---------------------------------------------------------------------------

VR_POS_TESTS = [
    # (描述, 期望机器人方向, VR空间位移 [x, y, z])
    ("手向前推 (VR -Z)", "机器人应向前(+X)", [0, 0, -1]),
    ("手向后拉 (VR +Z)", "机器人应向后(-X)", [0, 0, +1]),
    ("手向左移 (VR -X)", "机器人应向左(+Y)", [-1, 0, 0]),
    ("手向右移 (VR +X)", "机器人应向右(-Y)", [+1, 0, 0]),
    ("手向上抬 (VR +Y)", "机器人应向上(+Z)", [0, +1, 0]),
    ("手向下压 (VR -Y)", "机器人应向下(-Z)", [0, -1, 0]),
]

VR_ROT_TESTS = [
    # (描述, 期望, VR旋转轴, 角度)
    ("手腕绕VR-X正转(前俯)", "观察 pitch 方向", "x", +15),
    ("手腕绕VR-X反转(后仰)", "观察 pitch 方向", "x", -15),
    ("手腕绕VR-Y正转(左偏)", "观察 yaw 方向",   "y", +15),
    ("手腕绕VR-Y反转(右偏)", "观察 yaw 方向",   "y", -15),
    ("手腕绕VR-Z正转(逆时针)", "观察 roll 方向", "z", +15),
    ("手腕绕VR-Z反转(顺时针)", "观察 roll 方向", "z", -15),
]


def run_test(env, name, expected, action, step_size, hold_time, steps):
    """分步发 action，观察，然后反向回原位。"""
    # 缩放位置分量
    scaled = action.copy()
    scaled[:3] *= step_size
    # 旋转和夹爪不额外缩放

    reverse = -scaled.copy()

    print(f"\n  >>> {name}")
    print(f"      期望: {expected}")
    print(f"      mapped action = [{', '.join(f'{x:+.4f}' for x in scaled)}]")
    input("      按 Enter 执行...")

    # 正向
    per_step = scaled / steps
    for i in range(steps):
        env.step({
            "left": None,
            "right": per_step,
            "base": None,
            "lift": None,
        })
        time.sleep(0.02)

    print(f"      → 观察... ({hold_time}s)")
    time.sleep(hold_time)

    # 反向回原位
    per_step_rev = reverse / steps
    for i in range(steps):
        env.step({
            "left": None,
            "right": per_step_rev,
            "base": None,
            "lift": None,
        })
        time.sleep(0.02)

    print(f"      ← 回到原位")
    time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description="模拟VR手柄→真机方向测试")
    parser.add_argument("--step", type=float, default=0.03,
                        help="平移步幅(米). Default: 0.03")
    parser.add_argument("--rot-deg", type=float, default=10.0,
                        help="旋转角度(度). Default: 10")
    parser.add_argument("--hold", type=float, default=2.0,
                        help="观察时间(秒). Default: 2.0")
    parser.add_argument("--steps", type=int, default=5,
                        help="每个方向分几步. Default: 5")
    parser.add_argument("--skip-rot", action="store_true",
                        help="跳过旋转测试")
    parser.add_argument("--skip-pos", action="store_true",
                        help="跳过平移测试")
    args = parser.parse_args()

    env = ARXEnv(action_mode="delta_eef", camera_type="rgb", camera_view=())

    print("=" * 60)
    print("  模拟 VR 手柄 → 真机方向测试 (右臂)")
    print(f"  axis_mapping = (2, 0, 1)")
    print(f"  axis_sign    = (-1.0, -1.0, 1.0)")
    print(f"  平移步幅: {args.step}m, 旋转角度: {args.rot_deg}°")
    print("=" * 60)
    print("\n  每个测试按 Enter 执行。Ctrl+C 随时退出。")

    try:
        if not args.skip_pos:
            print("\n" + "=" * 40)
            print("【平移测试】模拟 VR 手柄移动")
            print("=" * 40)

            for name, expected, vr_delta in VR_POS_TESTS:
                action = simulate_vr_action(
                    vr_delta_pos=vr_delta,
                )
                run_test(env, name, expected, action, args.step,
                         args.hold, args.steps)

        if not args.skip_rot:
            print("\n" + "=" * 40)
            print("【旋转测试】模拟 VR 手腕旋转")
            print("=" * 40)

            for name, expected, axis, deg in VR_ROT_TESTS:
                action = simulate_vr_action(
                    vr_delta_pos=[0, 0, 0],
                    vr_rot_axis=axis,
                    vr_rot_deg=deg,
                )
                # 旋转不需要额外缩放 step
                run_test(env, name, expected, action, 1.0,
                         args.hold, args.steps)

        print("\n\n✅ 全部测试完成!")
        print("\n请记录结果，格式如:")
        print("  手向前推 → 机器人实际向__")
        print("  手向左移 → 机器人实际向__")
        print("  手腕绕VR-Z正转 → 机器人实际__")
        print("反馈给我，我来调映射参数。")

    except KeyboardInterrupt:
        print("\n\n⏹ 测试中断")
    finally:
        print("\n正在回到初始位置...")
        try:
            env._go_home()
        except Exception as e:
            print(f"go_home 失败: {e}")
        env.close()


if __name__ == "__main__":
    main()
