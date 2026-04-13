"""最小环境示例 — ARX LIFT2"""

from arx_toolkit.env import ARXEnv
import numpy as np

env = ARXEnv(
    action_mode="absolute_eef",
    camera_type="rgbd",
    camera_view=("camera_h",),
    img_size=(640, 480),
)

# reset：双臂回零，升降归零，底盘停止
obs = env.reset()
print("obs keys:", sorted(obs.keys()))
print("left_eef_pos:", obs["left_eef_pos"])
print("left gripper:", obs["left_joint_pos"][6])  # 0=开, 1=闭

# step：左臂移动 + gripper 全开
obs = env.step({
    "left":  np.array([0.1, 0, 0.15, 0, 0, 0, 0.0]),
    "right": None,
    "base":  None,
    "lift":  None,
})

env.close()
