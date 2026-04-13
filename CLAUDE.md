# ARX Toolkit

ARX LIFT2 双臂移动操作机器人的模仿学习工具包。

- 所有分析和交流都用中文。任何操作都要称呼我为 iff
- 机械臂型号是 ARX LIFT2
- 注意开发约束

## 开发约束

- **机器 A**（本机）仅有代码，无硬件 → 修改代码后可以用 mock 数据做本地验证
- **机器 B** 有 ARX LIFT2 + 3 × RealSense D405 → 真机测试结果由 iff 反馈
- 不要自行运行依赖真实硬件的脚本，会直接报错
- 不要改动参考老工具包 `/Users/iffyuan/Documents/Website/TJUDRLLAB-ARX_LIFT2s` 的任何内容，需要参考时 iff 会告知

## 项目目标

| 模块 | 状态 | 说明 |
|------|------|------|
| **Env** | ✅ 完成 | 真机环境，统一 `step(action) -> obs` 接口 |
| **Teleop** | 🔲 待开发 | 1. 左臂 leader → 右臂 follower  2. VR 双臂控制 |
| **Collect** | 🔲 待开发 | env + teleop → Zarr 存储 → LeRobot v3 格式 |
| **Deploy** | 🔲 待开发 | Pi0.5 等 VLA 模型推理 + 真机部署与评测 |

## 硬件参数

| 组件 | 型号 | 备注 |
|------|------|------|
| 机械臂 | ARX LIFT2 双臂 (2 × 6-DOF) | 左臂 + 右臂 |
| 夹爪 | 内置 | 归一化 [0, 1]，硬件范围 [-3.4, 0.0] |
| 底盘 | 3 轮全向 | vx/vy: [-1.5, 1.5]，vz: [-2.0, 2.0] |
| 升降台 | 线性升降 | 高度 [0, 20] |
| 相机 | 3 × Intel RealSense D405 | camera_l (409122272587), camera_h (409122274317), camera_r (409122272707) |

## 项目结构

```
iffyuan-ARX-Toolkit/
├── arx_toolkit/                  # Python 包
│   ├── env/
│   │   ├── arx_env.py            # ARXEnv 主类
│   │   └── _ros2_io.py           # ROS2 通信层（内部）
│   └── utils/
│       ├── logger.py
│       └── transforms.py         # 四元数 / RPY 工具
├── scripts/
│   └── mini_env_example.py       # 最小环境示例
├── ros_scripts/
│   └── all.sh                    # 启动所有 ROS2 节点
├── docs/
│   ├── 1.启动和测试环境.md
│   └── 参考代码分析报告.md
├── src/
│   ├── LIFT/                     # ARX 官方 ROS2 驱动（submodule）
│   └── urdf_lift2/               # LIFT2 URDF 模型
└── pyproject.toml
```

## Env 模块概要

核心接口：

```python
env = ARXEnv(action_mode="absolute_eef", camera_type="rgbd")
obs = env.reset()
obs = env.step({
    "left":  np.ndarray(7,) | None,   # [pose(6) + gripper(1)]
    "right": np.ndarray(7,) | None,
    "base":  np.ndarray(3,) | None,   # [vx, vy, vz]
    "lift":  float | None,            # height [0, 20]
})
env.close()
```

详细文档见 `docs/1.启动和测试环境.md` 和 `arx_toolkit/env/arx_env.py` 文件头 docstring。
