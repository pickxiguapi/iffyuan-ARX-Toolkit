# ARX Toolkit

ARX LIFT2 双臂移动操作机器人的模仿学习工具包。

覆盖 **环境控制 → 遥操作 → 数据采集 → 格式转换 → 模型部署** 全流程。

| 模块 | 状态 | 说明 |
|------|:----:|------|
| **Env** | ✅ | 统一 `step(action) → obs`，支持 3 种控制模式 |
| **Teleop** | ✅ | 主从遥操作 + Quest 3 VR 双臂遥操作 |
| **Collect** | ✅ | 定频采集 → Zarr → LeRobot v3 |
| **Deploy** | 🔲 | VLA 模型推理 + 真机部署（开发中） |

---

## 目录

- [硬件](#硬件)
- [环境设计 (ARXEnv)](#环境设计-arxenv)
- [安装](#安装)
- [Step 1：启动 ROS2 节点](#step-1启动-ros2-节点)
- [Step 2：测试环境](#step-2测试环境)
- [Step 3：遥操作](#step-3遥操作)
- [Step 4：数据采集](#step-4数据采集)
- [Step 5：转换为 LeRobot v3](#step-5转换为-lerobot-v3)
- [API 参考](#api-参考)
- [项目结构](#项目结构)
- [文档](#文档)

---

## 硬件

| 组件 | 规格 |
|------|------|
| 双臂 | 6 DOF + 1 夹爪 ×2，左右对称 |
| 底盘 | 3 全向轮，vx/vy ∈ [-1.5, 1.5]，vz ∈ [-2.0, 2.0] |
| 升降台 | 高度 [0, 20]，行程 ~460 mm |
| 夹爪 | 归一化 [0, 1]（0 = 全开，1 = 全闭），行程 44 mm |
| 相机 ×3 | Intel RealSense D405 — camera_l / camera_h / camera_r |
| 通信 | CANBus via SocketCAN，500 Hz |

---

## 环境设计 (ARXEnv)

ARXEnv 是整个工具包的核心：一个 `step(action) → obs` 就能控制双臂 + 底盘 + 升降 + 相机的统一环境。遥操作、数据采集、模型部署全部基于这个接口，确保训练和部署时数据格式完全一致。

### 设计哲学

> **One `step(action)` controls the entire robot.**

- 所有子系统（双臂、底盘、升降）通过一个 action dict 统一控制
- 不想动的部分设为 `None`，不会发送任何指令
- observation 是扁平 dict，所有值都是 numpy 数组，方便直接喂给模型

### 初始化

```python
from arx_toolkit.env import ARXEnv
import numpy as np

env = ARXEnv(
    action_mode="absolute_eef",      # delta_eef | absolute_eef | absolute_joint
    camera_type="rgbd",               # rgb | rgbd
    camera_view=("camera_l", "camera_h", "camera_r"),  # 订阅哪些相机
    img_size=(640, 480),              # (W, H)，None = 不缩放
)
```

| 参数 | 可选值 | 说明 |
|------|--------|------|
| `action_mode` | `delta_eef` / `absolute_eef` / `absolute_joint` | 臂的控制模式 |
| `camera_type` | `rgb` / `rgbd` | `rgb` 只订阅彩色；`rgbd` 额外订阅深度 |
| `camera_view` | 任意子集 | 要用哪几个相机，如只用头部 `("camera_h",)` |
| `img_size` | `(W, H)` 或 `None` | 输出图像尺寸 |

### Action 格式

```python
action = {
    "left":  np.ndarray(7,) | None,   # 左臂 7D 命令
    "right": np.ndarray(7,) | None,   # 右臂 7D 命令
    "base":  np.ndarray(3,) | None,   # [vx, vy, vz] 底盘速度
    "lift":  float | None,            # 升降高度 [0, 20]
}
obs = env.step(action)
```

**4 个 key 都必须写**，不想动的设 `None`。

#### 臂 Action — 7D

语义取决于 `action_mode`：

| action_mode | 7D 含义 | 适用场景 |
|-------------|---------|---------|
| `delta_eef` | `[dx, dy, dz, droll, dpitch, dyaw, gripper_delta]` | 遥操作增量控制 |
| `absolute_eef` | `[x, y, z, roll, pitch, yaw, gripper]` | **VLA 推理首选** |
| `absolute_joint` | `[j0, j1, j2, j3, j4, j5, gripper]` | 关节级控制，**VLA 推理首选** |

**单位约定：**

- 位置 xyz：**米 (m)**，基坐标系
- 姿态 rpy：**弧度 (rad)**
- Gripper：归一化 **[0, 1]**，0 = 全开，1 = 全闭。支持连续值（如 0.5 = 半开）
  - `delta_eef` 下 gripper 是增量，正值 = 更闭合
  - 内部自动处理硬件值 `[-3.4, 0.0]` ↔ 归一化 `[0, 1]` 的转换

#### 底盘 Action — 3D 速度

| 索引 | 名称 | 范围 | 说明 |
|------|------|------|------|
| 0 | vx | [-1.5, 1.5] | 前进 / 后退 |
| 1 | vy | [-1.5, 1.5] | 左移 / 右移 |
| 2 | vz | [-2.0, 2.0] | 左转 / 右转 |

#### 升降 Action — 标量

- `height ∈ [0, 20]`，0 = 最低，20 = 最高

### Observation 格式

`obs` 是扁平 dict，所有值为 numpy 数组：

**臂状态**（左右各一组，始终返回）：

| Key | Shape | Dtype | 说明 |
|-----|-------|-------|------|
| `{side}_eef_pos` | (7,) | float32 | `[x, y, z, roll, pitch, yaw, gripper]` 末端位姿 |
| `{side}_joint_pos` | (7,) | float32 | 6 个关节角 + gripper，gripper 归一化 [0,1] |

其中 `{side}` = `left` 或 `right`。

**底盘/升降状态**（始终返回）：

| Key | Shape | Dtype | 说明 |
|-----|-------|-------|------|
| `base_height` | (1,) | float32 | 当前升降高度 [0, 20] |

**相机图像**（取决于 `camera_type` 和 `camera_view`）：

| Key | Shape | Dtype | 说明 |
|-----|-------|-------|------|
| `{cam}_color` | (H, W, 3) | uint8 | RGB 彩色图 |
| `{cam}_aligned_depth_to_color` | (H, W) | uint16 | 深度图 (mm)，仅 `rgbd` 模式 |

其中 `{cam}` ∈ camera_view，如 `camera_l`、`camera_h`、`camera_r`。

**完整 obs 示例**（`camera_type="rgbd"`, `camera_view=("camera_l", "camera_h", "camera_r")`, `img_size=(640, 480)`）：

```python
obs = {
    # ---- 左臂 ----
    "left_eef_pos":    np.float32(7,),   # [x, y, z, roll, pitch, yaw, gripper]
    "left_joint_pos":  np.float32(7,),   # [j0, j1, j2, j3, j4, j5, gripper]

    # ---- 右臂 ----
    "right_eef_pos":   np.float32(7,),
    "right_joint_pos": np.float32(7,),

    # ---- 底盘 / 升降 ----
    "base_height":     np.float32(1,),   # [height]

    # ---- 相机 (RGB) ----
    "camera_l_color":  np.uint8(480, 640, 3),
    "camera_h_color":  np.uint8(480, 640, 3),
    "camera_r_color":  np.uint8(480, 640, 3),

    # ---- 相机 (深度, 仅 rgbd 模式) ----
    "camera_l_aligned_depth_to_color": np.uint16(480, 640),
    "camera_h_aligned_depth_to_color": np.uint16(480, 640),
    "camera_r_aligned_depth_to_color": np.uint16(480, 640),
}
```

### 生命周期

```python
env = ARXEnv(action_mode="absolute_eef", camera_type="rgbd")

obs = env.reset()       # 双臂回零 + 夹爪关闭 + 升降归零 + 底盘停止
obs = env.step(action)  # 发送指令，返回新 observation

env.close()             # 安全关闭（也通过 atexit 自动注册）
```

- `reset()` 会让双臂回到初始位姿并关闭夹爪
- `close()` 会停底盘、回零双臂、关闭夹爪、释放 ROS2 节点
- 即使程序异常退出，`atexit` 也会自动调用 `close()`

### 便捷方法

除了 `step(action)` 主接口外，还提供独立控制底盘/升降的便捷方法：

```python
env.step_base(vx=0.5, vy=0, vz=0)      # 仅控制底盘
env.step_lift(height=10)                 # 仅控制升降
env.step_base_lift(vx=0, vy=0, vz=0.15, height=2.0)  # 底盘+升降联合

env.get_observation(                     # 获取观测（可过滤）
    include_arm=True,
    include_camera=True,
    include_base=True,
)
```

### 特殊模式

```python
env.set_mode(0, side="both")   # 柔顺模式 (soft)
env.set_mode(1, side="both")   # 回零 (home)
env.set_mode(2, side="left")   # 保护模式 (protect)
env.set_mode(3, side="right")  # 重力补偿 / 拖动示教 (gravity)
```

### 内部架构

```
┌───────────────────────────────────────┐
│  用户代码 (VLA / Teleop / Collector)   │
├───────────────────────────────────────┤
│  ARXEnv                               │  ← 统一 API
│  · action 验证 & 模式转换              │
│  · gripper 归一化 [0,1] ↔ [-3.4,0]   │
│  · delta_eef 四元数旋转（避免万向锁）  │
├───────────────────────────────────────┤
│  RobotIO (ROS2 Node)                  │  ← 内部通信层
│  · Pub: arm_cmd_{l,r}, base_cmd      │
│  · Sub: arm_status_{l,r}, cameras     │
│  · 3 相机 ApproximateTimeSynchronizer  │
│  · 后台异步视频保存                    │
├───────────────────────────────────────┤
│  ROS2 / CANBus / 硬件                 │
└───────────────────────────────────────┘
```

### 典型使用模式

**遥操作 + 数据采集：**

```python
env = ARXEnv(action_mode="absolute_joint", camera_type="rgbd")
obs = env.reset()

while collecting:
    action = teleop.read()    # 从遥操作设备读取 action
    obs = env.step(action)    # 执行 + 获取观测
    recorder.push(obs, action)  # 存储
```

**VLA 推理部署：**

```python
env = ARXEnv(action_mode="absolute_eef", camera_type="rgb")
obs = env.reset()

policy = load_vla_model("pi0.5.pt")
for step in range(max_steps):
    images = {k: v for k, v in obs.items() if "color" in k}
    state = np.concatenate([obs["left_joint_pos"], obs["right_joint_pos"]])  # 14D
    action = policy.predict(images, state)  # → 18D
    obs = env.step({
        "left":  action[:7],
        "right": action[7:14],
        "base":  action[14:17],
        "lift":  action[17],
    })
```

> 详细实现见 `arx_toolkit/env/arx_env.py` 文件头 docstring 和 `docs/1.启动和测试环境.md`。

---

## 安装

### 前置条件

- Ubuntu 22.04 + ROS2 Humble
- CAN 接口已配置（CAN1 / CAN3 / CAN5）
- [uv](https://docs.astral.sh/uv/) 已安装（`curl -LsSf https://astral.sh/uv/install.sh | sh`）

### 克隆仓库

```bash
git clone https://github.com/pickxiguapi/iffyuan-ARX-Toolkit.git
cd iffyuan-ARX-Toolkit
```

### 创建虚拟环境 & 安装

> **⚠️ 必须使用uv虚拟环境，不允许使用conda和原生pip。** ROS2 绑定系统 Python，直接 pip install 会污染系统环境或产生版本冲突。用 uv 创建的虚拟环境可以继承系统 Python 的 ROS2 包（`rclpy`、`cv_bridge` 等），同时隔离项目依赖。  
> uv下面所有的命令都在pip前加uv即可，例如uv pip install

```bash
# 1. 创建虚拟环境（--system-site-packages 继承 ROS2 包）
uv venv --system-site-packages

# 2. 激活虚拟环境
source .venv/bin/activate

# 3. 安装
uv pip install -e .
```

### 激活uv环境

```bash
cd /path/to/iffyuan-ARX-Toolkit
source .venv/bin/activate
```

---

## Step 1：启动 ROS2 节点

所有后续操作（环境测试、遥操作、数据采集）都需要先启动底层 ROS2 节点。

```bash
cd /home/arx/Arx_Lift2s/Script
bash all.sh
```

启动后会打开 6 个终端窗口（CAN 总线、底盘+升降、双臂、3 个相机），等待 5–10 秒就绪。

**验证节点是否正常：**

```bash
# 检查 topic
ros2 topic list
# 应该看到 /arm_status_l, /arm_status_r, /body_information, /camera_*_namespace/...

# 检查臂状态
ros2 topic echo /arm_status_l --once

# 检查相机帧率
ros2 topic hz /camera_h_namespace/camera_h/color/image_rect_raw
```

---

## Step 2：测试环境

### 最小可用示例

```bash
python scripts/mini_env_example.py
```

```python
from arx_toolkit.env import ARXEnv
import numpy as np

env = ARXEnv(
    action_mode="absolute_eef",
    camera_type="rgbd",
    camera_view=("camera_h",),   # 只用头部相机
    img_size=(640, 480),
)

obs = env.reset()   # 双臂回零，升降归零，底盘停止
print("obs keys:", sorted(obs.keys()))
print("left_eef_pos:", obs["left_eef_pos"])     # (7,) [x,y,z,r,p,y,gripper]
print("left gripper:", obs["left_joint_pos"][6]) # 0=开, 1=闭

# 左臂移动 + gripper 半闭合
obs = env.step({
    "left":  np.array([0.1, 0, 0.15, 0, 0, 0, 0.5]),
    "right": None,    # 不动右臂
    "base":  None,    # 不动底盘
    "lift":  None,    # 不动升降
})

env.close()
```

### 完整测试

运行内置测试脚本，依次测试双臂控制、gripper、升降、底盘、模式切换等全部功能：

```bash
python -m arx_toolkit.env.arx_env
```

---

## Step 3：遥操作

### 方式 A：主从遥操作（Leader-Follower）

人手拖动一只臂（leader），另一只臂（follower）实时镜像跟随。

```bash
python scripts/teleop_leader_follower.py
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--leader` | `left` | 主臂（`left` / `right`） |
| `--rate` | `50` | 控制频率 Hz |
| `--alpha` | `0.5` | 低通滤波 (0,1]，越小越平滑 |
| `--deadband` | `0.004` | 死带阈值（rad），抑制微抖 |

**按键控制（单键触发，无需回车）：**

| 状态 | 按键 | 效果 |
|------|------|------|
| 运行中 | Enter | 暂停 |
| 运行中 | Ctrl+C | 退出 |
| 暂停中 | Enter | 恢复 |
| 暂停中 | Space | 双臂回零 & 退出 |

**其他参数：**

```bash
# 右臂当 leader
python scripts/teleop_leader_follower.py --leader right

# 更平滑
python scripts/teleop_leader_follower.py --alpha 0.3 --rate 30

# 不滤波，直接透传
python scripts/teleop_leader_follower.py --alpha 1.0 --deadband 0.0
```

### 方式 B：VR 遥操作（Quest 3）

> 注意：已经完全接通，但是不够丝滑，单臂操作优先使用Leader-Follower。  

Quest 3 头显双臂遥操作，左手柄控制左臂，右手柄控制右臂。

```bash
python scripts/teleop_vr.py
```

Quest 3 浏览器打开 `https://<机器人IP>:8443` → 点击 "Start Controller Tracking" → 握住 grip 移动手柄 → 按 trigger 控制 gripper。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--https-port` | `8443` | HTTPS 端口 |
| `--ws-port` | `8442` | WebSocket 端口 |
| `--rate` | `20` | 控制频率 Hz |
| `--scale` | `1.0` | 位移缩放 |
| `--rot-scale` | `1.0` | 旋转缩放 |

> 首次运行自动生成自签名 SSL 证书，Quest 3 浏览器会提示不安全，点"继续"即可。

---

## Step 4：数据采集

通过遥操作采集数据，存为 Zarr 格式。

```bash
python scripts/collect_data.py \
    --dataset datasets/pick_cup.zarr \
    --episodes 5 \
    --teleop leader_follower \
    --hz 30 \
    --cam-mode rgbd \
    --image-size 640 480 \
    --task "pick up the cup"
```

### 操作流程

```
启动 → 等待就绪
  │
  ├── Space    → 开始录制当前 episode
  │   └── 拖动 leader 臂，follower 跟随，采集中...
  │       └── Enter → 结束当前 episode，保存数据
  │           └── 自动进入下一个 episode 的等待
  │
  └── Ctrl+C  → 退出采集，输出最终总结
```

每个 episode 结束后会打印摘要（步数、FPS、磁盘占用、剩余 episode 数）。

### 断点续采

使用同样的 **dataset文件名** ，已有的 episode 会被自动识别。`--episodes` 是**目标总数**，不是新增数：

```bash
# 第一次采了 20 个，中断了
# 第二次运行同样的命令，会自动从 episode 20 继续采到 50
python scripts/collect_data.py \
    --dataset datasets/pick_cup.zarr \
    --episodes 50
```

### 采集参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--dataset, -d` | `datasets/demo.zarr` | Zarr 路径 |
| `--episodes, -n` | `3` | 目标 episode **总数** |
| `--teleop` | `leader_follower` | 遥操作方式 |
| `--leader-side` | `left` | Leader 臂 |
| `--hz` | `30` | 采集频率 Hz |
| `--cam-mode` | `rgbd` | `rgb` 或 `rgbd` |
| `--image-size` | `640 480` | 图像 W H |
| `--task` | `""` | 任务描述（写入 metadata） |
| `--save-video` | off | 保存每个 episode 的 MP4 回放 |
| `--control-rate` | `50` | Teleop 控制频率 Hz |
| `--lowpass-alpha` | `0.5` | 低通滤波系数 |
| `--deadband` | `0.004` | 死区阈值 |

### Zarr 数据格式

```
dataset.zarr/
├── data/
│   ├── rgb_camera_{l,h,r}     (N, 3, H, W) uint8       3 相机 RGB
│   ├── depth_camera_{l,h,r}   (N, 1, H, W) uint16      3 相机 深度 [rgbd 模式]
│   ├── left_eef_pos           (N, 7) float32            state: 末端位姿 + gripper
│   ├── left_joint_pos         (N, 7) float32            state: 6 关节 + gripper
│   ├── right_eef_pos          (N, 7) float32            state: 末端位姿 + gripper
│   ├── right_joint_pos        (N, 7) float32            state: 6 关节 + gripper
│   ├── base_height            (N, 1) float32            升降高度
│   ├── action_left            (N, 7) float32            动作: 6 joint + gripper
│   ├── action_right           (N, 7) float32            动作: 6 joint + gripper
│   ├── action_base            (N, 3) float32            底盘速度 vx, vy, vz
│   ├── action_lift            (N, 1) float32            升降
│   ├── timestamp              (N,) float64
│   └── episode                (N,) uint16
└── meta/
    ├── episode_ends           (M,) uint32
    └── config                 (JSON attrs)              采集参数快照
```

---

## Step 5：转换为 LeRobot v3

将 Zarr 数据集转换为 [LeRobot](https://github.com/huggingface/lerobot) v3 格式，用于 VLA 模型训练。

转换脚本支持**交互式**和**非交互式**两种模式，可以自由选择哪些字段作为 `observation.state`、`action`，以及使用哪些相机。

### 交互模式

```bash
python scripts/convert_to_lerobot_v3.py \
    --zarr datasets/pick_cup.zarr \
    --output lerobot_datasets/pick_cup \
    --repo-id iffyuan/arx_pick_cup \
    --task "pick up the cup" \
    --fps 30
```

运行后按提示分三步选择：

```
=== Zarr 数据集: datasets/pick_cup.zarr ===
Episodes: 50, 总帧数: 15000

  可用数组:
    [0] action_base               (15000, 3)            float32
    [1] action_left               (15000, 7)            float32
    [2] action_lift               (15000, 1)            float32
    [3] action_right              (15000, 7)            float32
    [4] base_height               (15000, 1)            float32
    [5] left_eef_pos              (15000, 7)            float32
    [6] left_joint_pos            (15000, 7)            float32
    [7] right_eef_pos             (15000, 7)            float32
    [8] right_joint_pos           (15000, 7)            float32

  相机:
    [A] camera_h                  (15000, 3, 480, 640)
    [B] camera_l                  (15000, 3, 480, 640)
    [C] camera_r                  (15000, 3, 480, 640)

Step 1: 选择 observation.state 的组成
  输入编号（逗号分隔）: 6,8
  → state = left_joint_pos(7) + right_joint_pos(7) = 14D

Step 2: 选择 action 的组成
  输入编号（逗号分隔）: 1,3,0,2
  → action = action_left(7) + action_right(7) + action_base(3) + action_lift(1) = 18D

Step 3: 选择相机（直接回车=全选）
  输入编号（逗号分隔）: ↵
  → cameras = camera_h, camera_l, camera_r

确认:
  observation.state = 14D (left_joint_pos + right_joint_pos)
  action            = 18D (action_left + action_right + action_base + action_lift)
  images            = camera_h, camera_l, camera_r
  继续? [Y/n] y
```

### 非交互模式

通过 CLI 参数直传字段名，跳过交互提示：

```bash
python scripts/convert_to_lerobot_v3.py \
    --zarr datasets/pick_cup.zarr \
    --output lerobot_datasets/pick_cup \
    --repo-id iffyuan/arx_pick_cup \
    --task "pick up the cup" --fps 30 \
    --state left_joint_pos,right_joint_pos \
    --action action_left,action_right,action_base,action_lift \
    --cameras camera_l,camera_h,camera_r
```

### 其他选项

```bash
# 用视频格式存储图像（大数据集推荐，体积缩小 5-10 倍）
python scripts/convert_to_lerobot_v3.py ... --use-videos

# 只看 Zarr 内容，不转换
python scripts/convert_to_lerobot_v3.py --zarr datasets/pick_cup.zarr --dry-run

# 只转换前 10 个 episode
python scripts/convert_to_lerobot_v3.py ... --episodes 10
```

### 转换参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--zarr, -i` | — | 输入 Zarr 路径（必需） |
| `--output, -o` | — | LeRobot 输出目录 |
| `--repo-id` | — | 数据集 repo ID |
| `--fps` | `30` | 帧率 |
| `--task` | zarr 文件名 | 任务描述 |
| `--state` | — | 非交互: state 字段（逗号分隔） |
| `--action` | — | 非交互: action 字段（逗号分隔） |
| `--cameras` | 全选 | 非交互: 相机名（逗号分隔） |
| `--use-videos` | off | 视频格式存储图像 |
| `--episodes` | 全部 | 只转换前 N 个 episode |
| `--dry-run` | off | 只列出数组信息 |

---

## API 参考

### ARXEnv

详见上方 [环境设计 (ARXEnv)](#环境设计-arxenv) 章节。

### LeaderFollowerTeleop

```python
from arx_toolkit.teleop import LeaderFollowerTeleop

teleop = LeaderFollowerTeleop(env, leader_side="left", control_rate=50)
teleop.start()                   # 启动后台控制线程
cmd = teleop.last_command        # 最新 action dict（供 Collector 读取）
teleop.stop()                    # 停止
teleop.run_interactive()         # 或：交互式运行（阻塞）
```

### VRTeleop

```python
from arx_toolkit.teleop import VRTeleop

vr = VRTeleop(env, https_port=8443, ws_port=8442, control_rate=20)
vr.run()  # 阻塞：启动 HTTPS + WebSocket + 控制循环
```

### Collector

```python
from arx_toolkit.collect import Collector

collector = Collector(
    env=env,
    action_source=lambda: teleop.last_command,
    dataset_path="datasets/demo.zarr",
    num_episodes=50,
    hz=30.0,
    cam_mode="rgbd",
    image_size=(640, 480),
    task="pick up the cup",
)
collector.run()  # 阻塞式采集循环
```

`action_source` 是一个无参 callable，返回标准 action dict。Collector 不关心 action 从哪来——主从遥操作、VR、策略推理都行。

---

## 项目结构

```
iffyuan-ARX-Toolkit/
├── arx_toolkit/                  # Python 包
│   ├── env/
│   │   ├── arx_env.py            # ARXEnv 主类
│   │   └── _ros2_io.py           # ROS2 通信层
│   ├── teleop/
│   │   ├── leader_follower.py    # 主从遥操作
│   │   ├── vr_teleop.py          # VR 双臂遥操作
│   │   └── vr_web_ui/            # WebXR 前端
│   ├── collect/
│   │   └── collector.py          # Collector
│   └── utils/
│       ├── logger.py
│       └── transforms.py
├── scripts/
│   ├── collect_data.py           # 数据采集 CLI
│   ├── convert_to_lerobot_v3.py  # Zarr → LeRobot v3
│   ├── teleop_leader_follower.py # 主从遥操作
│   ├── teleop_vr.py              # VR 遥操作
│   ├── web_control_demo.py       # Web 键盘控制 Demo
│   └── mini_env_example.py       # 最小环境示例
├── ros_scripts/
│   └── all.sh                    # 启动所有 ROS2 节点
├── docs/                         # 详细文档
├── src/
│   ├── LIFT/                     # ARX 官方 ROS2 驱动（submodule）
│   └── urdf_lift2/               # LIFT2 URDF 模型
└── pyproject.toml
```

## 文档

| 文档 | 内容 |
|------|------|
| [0. 硬件参数手册](docs/0.硬件参数手册.md) | LIFT2 完整硬件规格 |
| [1. 启动和测试环境](docs/1.启动和测试环境.md) | ROS2 节点启动 + Env 验证 + 常见问题 |
| [2-1. 主从遥操作](docs/2-1.主从遥操作.md) | Leader-follower 调参与技术细节 |
| [2-2. VR 遥操作](docs/2-2.VR遥操作.md) | Quest 3 配置与坐标校准 |
| [3. 数据采集](docs/3.数据采集.md) | 完整采集流程 + LeRobot 转换 |

## License

Apache2.0
