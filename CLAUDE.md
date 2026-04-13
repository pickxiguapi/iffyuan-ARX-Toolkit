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
| **Collect** | ✅ 完成 | env + teleop → Zarr 存储 → LeRobot v3 转换 |
| **Deploy** | 🔲 待开发 | Pi0.5 等 VLA 模型推理 + 真机部署与评测 |

## 硬件参数

> 完整参数见 `docs/0.硬件参数手册.md`

| 参数 | 值 |
|------|-----|
| 系统类型 | 双臂移动操作机器人（底盘 + 升降 + 双臂） |
| 单臂 | 6 DOF + 1 夹爪，左右对称 |
| 底盘 | 3 全向轮，vx/vy: [-1.5, 1.5]，vz: [-2.0, 2.0] |
| 升降 | 高度 [0, 20]，行程 ~460 mm |
| 夹爪 | 归一化 [0, 1]，硬件范围 [-3.4, 0.0]，行程 44 mm |
| 相机 ×3 | D405: camera_l (409122272587), camera_h (409122274317), camera_r (409122272707) |
| 通信 | CANBus via SocketCAN，500 Hz |

## 项目结构

```
iffyuan-ARX-Toolkit/
├── arx_toolkit/                  # Python 包
│   ├── env/
│   │   ├── arx_env.py            # ARXEnv 主类
│   │   └── _ros2_io.py           # ROS2 通信层（内部）
│   ├── teleop/
│   │   └── leader_follower.py    # 主从遥操作（leader→follower）
│   ├── collect/
│   │   └── collector.py          # Collector — env + action_source → Zarr
│   └── utils/
│       ├── logger.py
│       └── transforms.py         # 四元数 / RPY 工具
├── scripts/
│   ├── mini_env_example.py       # 最小环境示例
│   ├── collect_data.py           # 数据采集入口
│   └── convert_to_lerobot_v3.py  # Zarr → LeRobot v3 转换
├── ros_scripts/
│   └── all.sh                    # 启动所有 ROS2 节点
├── docs/
│   ├── 0.硬件参数手册.md
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

## Collect 模块概要

核心接口：

```python
from arx_toolkit.collect import Collector

collector = Collector(
    env=env,                              # ARXEnv 实例
    action_source=lambda: teleop.last_command,  # 任意返回 action dict 的 callable
    dataset_path="datasets/demo.zarr",
    num_episodes=50,                      # 目标总数（含已有）
    hz=30.0,                              # 采集频率
    cam_mode="rgbd",                      # rgb | rgbd
    image_size=(640, 480),
    task="pick up the cup",
)
collector.run()  # 阻塞式采集循环
```

### Zarr 存储格式

```
dataset.zarr/
├── data/
│   ├── rgb_camera_l       (N, 3, H, W) uint8
│   ├── rgb_camera_h       (N, 3, H, W) uint8
│   ├── rgb_camera_r       (N, 3, H, W) uint8
│   ├── depth_camera_l     (N, 1, H, W) uint16  [rgbd 模式]
│   ├── depth_camera_h     (N, 1, H, W) uint16
│   ├── depth_camera_r     (N, 1, H, W) uint16
│   ├── left_eef_pos       (N, 7) float32   # 含归一化 gripper
│   ├── left_joint_pos     (N, 7) float32   # 含归一化 gripper
│   ├── right_eef_pos      (N, 7) float32   # 含归一化 gripper
│   ├── right_joint_pos    (N, 7) float32
│   ├── base_height        (N, 1) float32
│   ├── action_left        (N, 7) float32   # 7D = 6 joint + gripper
│   ├── action_right       (N, 7) float32
│   ├── action_base        (N, 3) float32
│   ├── action_lift        (N, 1) float32
│   ├── timestamp          (N,) float64
│   └── episode            (N,) uint16
└── meta/
    ├── episode_ends       (M,) uint32
    └── config             (JSON attrs)
```

### LeRobot v3 特征映射

```
observation.state   = 14D  — left_joint_pos(7) + right_joint_pos(7)
observation.eef_pos = 14D  — left_eef_pos(7) + right_eef_pos(7)   (可选)
observation.base_state = 1D — base_height
observation.images.camera_l/h/r — RGB 图像
action = 18D — action_left(7) + action_right(7) + action_base(3) + action_lift(1)
```

### 数据采集流程

```
LeaderFollower (后台线程, 50Hz)
    leader_joint → lowpass → deadband → follower
              ↓
Collector (前台, --hz 限速)
    1. obs = env.get_observation()
    2. action = teleop.last_command
    3. buffer.append(obs, action)
    4. 按键检测: Space=开始, Enter=结束, Ctrl+C=退出
              ↓
    Zarr 数据集 (Blosc 压缩)
              ↓
    convert_to_lerobot_v3.py → LeRobot v3 格式
```
