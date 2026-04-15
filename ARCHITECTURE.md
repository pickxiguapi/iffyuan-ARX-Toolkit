# ARX LIFT2 工具包 - 详细架构分析

## 📋 项目总览

**机器人**: ARX LIFT2 双臂移动操作机器人
- 2 × 6-DOF + 1 gripper 手臂
- 3 轮全向底盘
- 升降台 (linear stage)
- 3 × RealSense D405 相机

**当前完成模块**:
- ✅ **Env**: 统一环境接口 (`ARXEnv`)
- ✅ **Teleop**: 主从遥操作 (`LeaderFollowerTeleop`)
- ✅ **Collect**: 数据采集 (`Collector`)
- 🔲 **Deploy**: VLA 推理部署

---

## 🏗️ 核心架构

### 代码结构树

```
arx_toolkit/
├── __init__.py                     # 空文件
├── env/
│   ├── __init__.py                 # 导出 ARXEnv
│   ├── arx_env.py                  # 核心环境类 (700+ 行)
│   └── _ros2_io.py                 # ROS2 通信层 (390 行)
├── teleop/
│   ├── __init__.py                 # 导出 LeaderFollowerTeleop, VRTeleop
│   ├── leader_follower.py          # 主从遥操作 (310 行)
│   ├── vr_teleop.py                # VR 遥操作 (待查)
│   └── vr_web_ui/                  # Web UI 资源
├── utils/
│   ├── __init__.py                 # 空
│   ├── logger.py                   # 统一日志 (20 行)
│   └── transforms.py               # 四元数/RPY 工具 (63 行)
└── collect/
    └── collector.py                # 数据采集器 (待查)

scripts/
├── mini_env_example.py             # 最小示例
├── teleop_leader_follower.py       # 遥操作入口
├── collect_data.py                 # 采集脚本 (主脚本)
├── teleop_vr.py                    # VR 采集
├── convert_to_lerobot_v3.py        # Zarr → LeRobot
└── test_vr_direction.py            # VR 方向测试
```

---

## 🎮 关键接口详解

### 1. ARXEnv (环境)

#### 初始化参数

```python
env = ARXEnv(
    action_mode: ActionMode = "delta_eef",           # 控制模式
    camera_type: Literal["rgb", "rgbd"] = "rgbd",    # 相机类型
    camera_view: Iterable[str] = ("camera_l", ...),  # 相机列表
    img_size: Optional[Tuple[int, int]] = (640, 480) # 图像分辨率
)
```

**action_mode 有 3 种** (重点!):

| 模式 | 维度 | 含义 | 用途 |
|------|------|------|------|
| `"absolute_eef"` | 7D | [x, y, z, roll, pitch, yaw, gripper] | EEF 末端执行器绝对位置 |
| `"delta_eef"` | 7D | [dx, dy, dz, dr, dp, dy, gripper] | EEF 相对位移（四元数乘法融合旋转） |
| `"absolute_joint"` | 7D | [j0, j1, j2, j3, j4, j5, gripper] | 关节空间绝对位置 **← 遥操作用** |

**单位和范围**:
- 位置 xyz: 米 (m)，基座坐标系
- 旋转 rpy: 弧度 (rad)
- 夹爪: 归一化 [0, 1]
  - 0 = 全开 (硬件: -3.4)
  - 1 = 全闭 (硬件: 0.0)
  - 支持连续值，如 0.5 = 半开
- 底盘速度 [vx, vy, vz]: [-1.5~1.5], [-1.5~1.5], [-2.0~2.0]
- 升降高度: [0, 20] (单位不明，约 460mm 行程)

#### 核心方法

```python
# 生命周期
obs = env.reset()                   # 双臂回零 + 升降=0 + 底盘停止
obs = env.step(action)              # 执行单步动作
env.close()                         # 安全关闭（自动注册 atexit）

# 观测
obs = env.get_observation(
    include_arm=True,               # 包含臂状态
    include_camera=True,            # 包含相机
    include_base=True               # 包含底盘/升降
)

# 控制（低级）
env.step_base(vx, vy, vz)           # 底盘速度
env.step_lift(height)               # 升降高度
env.step_base_lift(vx, vy, vz, height)  # 联合控制

# 模式切换
env.set_mode(mode, side="both")     # 0=soft, 1=home, 2=protect, 3=gravity
```

#### 统一 Action 字典格式

```python
action = {
    "left":  np.ndarray(7,) | None,   # 左臂 (action_mode 决定语义)
    "right": np.ndarray(7,) | None,   # 右臂
    "base":  np.ndarray(3,) | None,   # 底盘 [vx, vy, vz]
    "lift":  float | None,             # 升降 height [0, 20]
}
```

**None 表示 "不动"**

#### 观测输出 (flat dict)

```python
{
    # 左臂
    "left_eef_pos":    np.float32(7,),   # [x, y, z, roll, pitch, yaw, gripper_norm]
    "left_joint_pos":  np.float32(7,),   # [j0..j5, gripper_norm]
    
    # 右臂（对称）
    "right_eef_pos":   np.float32(7,),
    "right_joint_pos": np.float32(7,),
    
    # 底盘/升降
    "base_height":     np.float32(1,),   # [height]
    
    # 相机（按配置）
    "camera_l_color":              np.uint8(H, W, 3),      # RGB
    "camera_l_aligned_depth_to_color":  np.uint16(H, W),   # 深度 (mm)
    "camera_h_color":              np.uint8(H, W, 3),
    "camera_h_aligned_depth_to_color":  np.uint16(H, W),
    "camera_r_color":              np.uint8(H, W, 3),
    "camera_r_aligned_depth_to_color":  np.uint16(H, W),
}

# 夹爪归一化: obs 中的 joint_pos[6] 始终在 [0, 1]
```

---

### 2. LeaderFollowerTeleop (遥操作)

#### 设计原理

```
Leader Arm (人手工拖动)
    ↓ gravity mode (mode=3)
    ↓ read joint_pos @ 50Hz (后台线程)
    ↓ low-pass filter
    ↓ dead-band suppression
    ↓ send to Follower (绝对关节命令)
Follower Arm (镜像跟随)
```

#### 初始化

```python
teleop = LeaderFollowerTeleop(
    env: ARXEnv,                    # 已初始化的环境
    leader_side: str = "left",      # "left" 或 "right"
    control_rate: float = 50.0,     # 控制循环 Hz
    lowpass_alpha: float = 0.5,     # 低通滤波系数 ∈ (0,1]
    deadband: float = 0.004,        # 死区阈值 (rad/gripper_norm)
)
```

**参数详解**:

| 参数 | 默认 | 范围 | 说明 |
|------|------|------|------|
| `lowpass_alpha` | 0.5 | (0, 1] | α=1.0 无滤波 / α=0.5 中等平滑 / α<0.1 高延迟 |
| `deadband` | 0.004 | > 0 | 阈值内的变化被忽略（减少抖动） |
| `control_rate` | 50 | Hz | 后台线程循环频率 |

**信号处理流程**:

```python
# 每个控制周期 (50Hz)
leader_joints = env.get_observation()["left_joint_pos"]  # (7,)

# 低通滤波: y = α * target + (1-α) * previous
filtered = α * leader_joints + (1-α) * prev_cmd

# 死区: |delta| < threshold 时保持前值
cmd = filtered.copy()
for i in range(7):
    if |filtered[i] - prev_cmd[i]| < deadband:
        cmd[i] = prev_cmd[i]

# 送往 Follower (via env._apply_absolute_joint())
env._apply_absolute_joint({"right": cmd})
prev_cmd = cmd.copy()
```

#### 公开 API

```python
# 启动/停止
teleop.start()              # 设置 leader 为 gravity 模式，启动后台线程
teleop.stop()               # 停止线程，follower 保持当前位置

# 交互式模式（阻塞）
teleop.run_interactive()    # 进入 REPL:
                            #   Enter (运行中) → 暂停
                            #   Enter (暂停) → 恢复
                            #   Space (暂停) → 回零并退出
                            #   Ctrl+C → 立即退出

# 获取最新命令 (Collector 用)
cmd = teleop.last_command   # dict | None
    # {
    #     "left":  np.ndarray(7,) | None,
    #     "right": np.ndarray(7,) | None,
    #     "base": None,
    #     "lift": None,
    # }
```

#### 后台线程详解

```python
# _loop() 运行在 daemon 线程中
def _loop(self):
    dt = 1.0 / self.control_rate  # e.g., 0.02s @ 50Hz
    while self._running:
        t0 = time.monotonic()
        self._tick()                # 一个控制周期
        elapsed = time.monotonic() - t0
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)  # 补偿处理时间
```

**关键实现细节**:

```python
def _tick(self) -> None:
    # 1. 读取 leader 关节位置 (non-blocking)
    obs = self.env.get_observation(include_camera=False, include_base=False)
    leader_joints = obs[f"{self.leader_side}_joint_pos"]  # (7,)
    
    # 2. 低通 + 死区处理
    filtered = _lowpass(leader_joints, self._prev_cmd, self.lowpass_alpha)
    cmd = _deadband(filtered, self._prev_cmd, self.deadband)
    
    # 3. **直接调用内部方法**，绕过 validate + 完整 step()
    self.env._apply_absolute_joint({self.follower_side: cmd})
    
    # 4. 记录供 Collector 读取
    self._last_command = {
        "left": cmd if self.follower_side == "left" else None,
        "right": cmd if self.follower_side == "right" else None,
        "base": None,
        "lift": None,
    }
```

#### 初始化序列

```python
# _wait_for_ros2_ready() → 等待 DDS 发现完成
#   轮询 cmd_pub_l / cmd_pub_r 的 subscription_count
#   直到 ≥1 订阅者或超时 (10s)

# env.set_mode(3, side=leader_side) → gravity mode
#   leader 进入重力补偿模式（可自由拖动）

# obs = env.get_observation(...) → 初始化 filter
#   _prev_cmd = obs[f"{follower_side}_joint_pos"]
```

---

### 3. 信号处理工具函数

#### Low-pass Filter

```python
def _lowpass(target: np.ndarray, previous: np.ndarray, alpha: float) -> np.ndarray:
    """指数低通滤波.
    
    y[n] = α * target[n] + (1-α) * y[n-1]
    
    α = 1.0  → 无滤波 (raw target)
    α = 0.5  → 中等平滑
    α < 0.1  → 高延迟，平滑度高
    """
    return alpha * target + (1.0 - alpha) * previous
```

**参数映射**:
- α=1.0: 立即跟随，无延迟，易抖动
- α=0.5: 折中（推荐）
- α=0.1: 延迟大，但非常平滑

#### Dead-band

```python
def _deadband(filtered: np.ndarray, previous: np.ndarray, threshold: float) -> np.ndarray:
    """逐维死区：小变化被忽略.
    
    if |filtered[i] - previous[i]| < threshold:
        output[i] = previous[i]  # 保持前值
    else:
        output[i] = filtered[i]  # 采用新值
    """
    diff = np.abs(filtered - previous)
    out = filtered.copy()
    out[diff < threshold] = previous[diff < threshold]
    return out
```

**作用**: 减少通信负荷、消除微小抖动

#### 四元数变换

```python
# arx_toolkit/utils/transforms.py
quat_from_rpy(rpy: np.ndarray) -> np.ndarray
    # RPY [roll, pitch, yaw] (rad) → Quaternion [x, y, z, w]

rpy_from_quat(q: np.ndarray) -> np.ndarray
    # Quaternion [x, y, z, w] → RPY [roll, pitch, yaw]

quat_multiply(q0, q1) -> np.ndarray
    # Hamilton 四元数乘法: q0 ⊗ q1
    # 用于 delta_eef 模式旋转叠加

quat_normalize(q) -> np.ndarray
    # 四元数归一化
```

**在 delta_eef 中的应用**:

```python
# delta_eef: [dx, dy, dz, droll, dpitch, dyaw, gripper_delta]
curr_end = obs["left_eef_pos"]         # [x, y, z, r, p, y, g]
delta = action["left"]                 # [dx, dy, dz, dr, dp, dy, dg]

# 位置叠加（简单）
target_xyz = curr_end[:3] + delta[:3]

# 旋转叠加（四元数乘法）
q_curr = quat_from_rpy(curr_end[3:6])
q_delta = quat_from_rpy(delta[3:6])
q_target = quat_multiply(q_delta, q_curr)  # ← 基座系增量旋转
target_rpy = rpy_from_quat(q_target)
```

---

### 4. ROS2 通信层 (_ros2_io.py)

#### RobotIO Node

```python
class RobotIO(Node):
    """单一 ROS2 节点处理所有 ARM / BASE / CAMERA 通信"""
```

**发布者 (Publishers)**:

| 发布者 | 主题 | 消息类型 | 功能 |
|--------|------|---------|------|
| `cmd_pub_l` | `arm_cmd_l` | `RobotCmd` | 左臂命令 |
| `cmd_pub_r` | `arm_cmd_r` | `RobotCmd` | 右臂命令 |
| `cmd_pub_base` | `/ARX_VR_L` | `PosCmd` | 底盘 / 升降命令 |

**订阅者 (Subscribers)**:

| 订阅者 | 主题 | 消息类型 | 功能 |
|--------|------|---------|------|
| 状态左 | `arm_status_l` | `RobotStatus` | 左臂反馈 |
| 状态右 | `arm_status_r` | `RobotStatus` | 右臂反馈 |
| 底盘状态 | `body_information` | `PosCmd` | 底盘/升降反馈 |
| 相机同步 | 多个 | `Image` | 相机流（ApproximateTimeSynchronizer） |

**相机同步**:

```python
# camera_type = "rgbd" 时
topics = [
    "/camera_l_namespace/camera_l/color/image_rect_raw",
    "/camera_l_namespace/camera_l/aligned_depth_to_color/image_raw",
    "/camera_h_namespace/camera_h/color/image_rect_raw",
    "/camera_h_namespace/camera_h/aligned_depth_to_color/image_raw",
    "/camera_r_namespace/camera_r/color/image_rect_raw",
    "/camera_r_namespace/camera_r/aligned_depth_to_color/image_raw",
]
# 使用 ApproximateTimeSynchronizer (slop=0.02) 同步
```

**视频保存** (可选):

```python
# save_video=True + save_dir 时启用后台线程录制
# H.264 MP4 格式，连续保存所有相机流
```

#### RobotCmd 消息格式

```python
RobotCmd:
  mode: int              # 0=soft, 1=home, 2=protect, 3=gravity, 4=eef, 5=joint
  end_pos: list[float]   # [x, y, z, r, p, y] (EEF 目标)
  joint_pos: list[float] # [j0, j1, j2, j3, j4, j5] (关节目标)
  gripper: float         # 夹爪硬件值 [-3.4, 0.0]
```

#### 内部方法

```python
# 发送
send_control_msg(side: str, cmd: RobotCmd) -> bool
send_base_msg(cmd: PosCmd) -> bool

# 接收
get_robot_status() -> Dict[str, RobotStatus | PosCmd]
get_camera(target_size, return_status) -> (frames_dict, status_snapshot)

# 状态快照（用于观测构建）
status_snapshot  # 当 _on_images() 时更新，确保相机和状态同步
```

---

## 🔄 完整数据流

### Teleop + Env 流程

```
┌─────────────────────────────────────────┐
│     LeaderFollowerTeleop.run_interactive()     │
└────────────────┬────────────────────────┘
                 │
                 ├─→ start()
                 │   ├─→ _wait_for_ros2_ready()  [轮询 DDS]
                 │   ├─→ env.set_mode(3, side=leader)  [gravity mode]
                 │   └─→ _thread = Thread(_loop)  [daemon]
                 │
                 ├─→ _loop() [后台线程, 50Hz]
                 │   └─→ _tick():
                 │       1. obs = env.get_observation(...)
                 │       2. leader_j = obs[f"{leader}_joint_pos"]  (7,)
                 │       3. filtered = _lowpass(...)
                 │       4. cmd = _deadband(...)
                 │       5. env._apply_absolute_joint({follower: cmd})
                 │       6. self._last_command = {...}  [Collector 读取]
                 │
                 ├─→ _read_key()  [主线程, 阻塞]
                 │   ├─ Enter (运行) → stop() → pause
                 │   ├─ Enter (暂停) → start() → resume
                 │   ├─ Space (暂停) → go_home() + return
                 │   └─ Ctrl+C → break
                 │
                 └─→ stop()  [最后]
                     └─→ _running = False
                         _thread.join()
```

### Collector 读取 Teleop 命令

```python
def _get_teleop_action():
    cmd = teleop.last_command
    return cmd or {"left": None, "right": None, "base": None, "lift": None}

# Collector 每次 step() 时调用
action = action_source()  # → last_command
```

---

## 📊 数据采集脚本流程 (collect_data.py)

```
1. Parse args
   ├─ dataset path
   ├─ num_episodes
   ├─ teleop type (leader_follower)
   ├─ control params (rate, alpha, deadband)
   └─ collection params (hz, cam_mode, image_size)

2. ARXEnv init
   └─ action_mode = "absolute_joint"
      camera_type = args.cam_mode
      camera_view = ("camera_l", "camera_h", "camera_r")

3. LeaderFollowerTeleop init + start()

4. Collector init
   ├─ env = ...
   ├─ action_source = teleop.last_command
   ├─ dataset_path = ...
   ├─ num_episodes = ...
   ├─ hz = ... (采集频率)
   └─ ...

5. collector.run()
   ├─ 循环采集 episodes
   ├─ 每个 episode:
   │   ├─ 等待 Space 键开始
   │   ├─ obs_list = []
   │   ├─ 循环 step():
   │   │   ├─ obs = env.step(action_source())
   │   │   ├─ 采样到 obs_list
   │   │   └─ 检查 Enter 键结束
   │   └─ 写入 Zarr 数据集
   └─ 输出数据集统计

6. Cleanup
   ├─ teleop.stop()
   ├─ env.close()
   └─ Done
```

---

## ⚙️ 关键配置文件

### arx_toolkit/env/arx_env.py (700+ 行)

**常数**:

```python
_MODE_EEF = 4        # RobotCmd mode for absolute_eef
_MODE_JOINT = 5      # RobotCmd mode for absolute_joint

GRIPPER_OPEN_RAW = -3.4
GRIPPER_CLOSE_RAW = 0.0

ActionMode = Literal["delta_eef", "absolute_eef", "absolute_joint"]
```

**内部方法**:

```python
_send_arm_cmd(side, mode, end_pos, joint_pos, gripper)
    ↓ 构建 RobotCmd 并发送

_apply_absolute_eef(action)      # mode=4
_apply_absolute_joint(action)    # mode=5
_apply_delta_eef(action)         # 计算后 mode=4

_validate_action(action)         # 检查格式
```

---

## 🔍 使用示例

### 最小化环境示例

```python
from arx_toolkit.env import ARXEnv
import numpy as np

# 初始化环境
env = ARXEnv(action_mode="absolute_eef", camera_type="rgbd")

# Reset
obs = env.reset()

# Step - 左臂移动
obs = env.step({
    "left": np.array([0.02, 0, 0.03, 0, 0, 0, 0.0]),
    "right": None,
    "base": None,
    "lift": None,
})

# 查看观测
print(obs["left_eef_pos"])      # [x, y, z, r, p, y, gripper]
print(obs["left_joint_pos"])    # [j0..j5, gripper]
print(obs["base_height"])       # [height]

env.close()
```

### 遥操作示例

```python
from arx_toolkit.env import ARXEnv
from arx_toolkit.teleop import LeaderFollowerTeleop

env = ARXEnv(action_mode="absolute_joint", camera_type="rgb", camera_view=())

teleop = LeaderFollowerTeleop(
    env,
    leader_side="left",
    control_rate=50.0,
    lowpass_alpha=0.5,
    deadband=0.004,
)

# 交互式模式（阻塞）
teleop.run_interactive()  # Enter 暂停/恢复, Space 回零退出, Ctrl+C 退出

env.close()
```

### 数据采集示例

```bash
# 完整采集
python scripts/collect_data.py \
    --dataset datasets/pick_cup.zarr \
    --episodes 50 \
    --hz 30 \
    --cam-mode rgbd \
    --image-size 640 480 \
    --leader-side left \
    --control-rate 50 \
    --lowpass-alpha 0.5 \
    --deadband 0.004 \
    --task "pick up the cup"

# 断点续采（自动识别 episodes）
python scripts/collect_data.py --dataset datasets/pick_cup.zarr --episodes 50
```

---

## 🔌 消息格式总结

### RobotStatus (从硬件来)

```python
{
    "left": RobotStatus(
        end_pos=[x, y, z, r, p, y],    # EEF 位置
        joint_pos=[j0, j1, j2, j3, j4, j5, gripper_raw],  # 关节 + 原始夹爪
    ),
    "right": RobotStatus(...),
    "base": PosCmd(
        chx=vx, chy=vy, chz=vz,
        height=lift_h,
        ...
    ),
}
```

### 夹爪转换

```
硬件: [-3.4 (open), 0.0 (closed)]
↓↑
归一化: [0.0 (open), 1.0 (closed)]

obs 中 joint_pos[6] 和 eef_pos[6] 始终是归一化值 [0,1]
action 中指定的夹爪也是 [0,1]
```

---

## 🎯 总结 - 关键架构决策

| 决策 | 原因 |
|------|------|
| **统一 action 字典** | 支持多部件 (arm_l, arm_r, base, lift) 独立控制 |
| **action_mode 参数化** | 支持 EEF / Joint / Delta 多种控制模式，灵活选择 |
| **后台线程 Teleop** | 50Hz 固定频率遥操作线程，解耦采集 (30Hz) 和控制 (50Hz) |
| **low-pass + deadband** | 减少抖动、降低通信频率、提升平稳性 |
| **gravity mode (mode=3)** | 人工拖动 leader 时无须用力（重力补偿） |
| **绝对关节镜像** | 遥操作用 joint 空间，不受 IK 失败影响 |
| **ROS2 同步** | ApproximateTimeSynchronizer 确保多相机和状态帧对齐 |

---

## ✅ 文件清单

| 文件 | 行数 | 功能 |
|------|------|------|
| `arx_env.py` | 833 | 核心环境 (3 action_mode, 所有接口) |
| `_ros2_io.py` | 391 | ROS2 节点 (PubSub, 相机同步, 视频保存) |
| `leader_follower.py` | 312 | 遥操作 (滤波, 死区, 交互) |
| `transforms.py` | 63 | 四元数/RPY 变换 |
| `logger.py` | 20 | 统一日志 |
| `collect_data.py` | 170 | 采集脚本入口 |
| `teleop_leader_follower.py` | 59 | 遥操作脚本 |
| `mini_env_example.py` | 28 | 最小环境示例 |

---

## 📝 还需查看的文件

1. **arx_toolkit/collect/collector.py** — 数据采集器核心
2. **arx_toolkit/teleop/vr_teleop.py** — VR 遥操作（可选）
3. **scripts/convert_to_lerobot_v3.py** — Zarr → LeRobot 转换
4. **scripts/collect_data.py 中 Collector 部分** — 采集逻辑

