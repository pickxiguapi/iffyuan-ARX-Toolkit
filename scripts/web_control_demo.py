#!/usr/bin/env python3
"""ARX LIFT2 键盘控制 Web Demo — 三相机实时画面 + 双臂/底盘/升降 键盘遥操作.

启动 (真机):  python scripts/web_control_demo.py
启动 (调试):  python scripts/web_control_demo.py --mock
打开:         http://localhost:8080

键盘映射 (网页内按键):
  ── 左臂 (默认) / 右臂 (按 Tab 切换) ──
  W/S — Y 前/后      A/D — X 左/右      Q/E — Z 上/下
  I/K — Roll ±       J/L — Pitch ±      U/O — Yaw ±
  Space — 切换夹爪

  ── 底盘 ──
  ↑/↓ — 前进/后退    ←/→ — 左移/右移    ,/. — 左转/右转

  ── 升降 ──
  =/- — 升/降

  ── 系统 ──
  Tab — 切换控制臂    R — 复位到 Home     Esc — 急停
  G — 进入重力补偿    1/2/3 — 步长档位

--mock 模式下不连接真机 / 相机，用于本地 UI 调试。
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Mock env for local testing (Machine A, no hardware)
# ---------------------------------------------------------------------------

class MockARXEnv:
    """Fake env that mimics ARXEnv interface for UI development."""

    def __init__(self, **_kw):
        self._left_eef = np.array([0.15, 0.0, 0.20, 0.0, 0.0, 0.0], dtype=np.float64)
        self._right_eef = np.array([0.15, 0.0, 0.20, 0.0, 0.0, 0.0], dtype=np.float64)
        self._left_joint = np.zeros(7, dtype=np.float64)
        self._right_joint = np.zeros(7, dtype=np.float64)
        self._base_height = 0.0
        print("[MockARXEnv] 初始化完成 (模拟模式, 无真机连接)")

    def reset(self):
        self._left_eef = np.array([0.15, 0.0, 0.20, 0.0, 0.0, 0.0], dtype=np.float64)
        self._right_eef = np.array([0.15, 0.0, 0.20, 0.0, 0.0, 0.0], dtype=np.float64)
        self._left_joint = np.zeros(7, dtype=np.float64)
        self._right_joint = np.zeros(7, dtype=np.float64)
        self._base_height = 0.0
        return self._obs()

    def step(self, action: dict):
        left_a = action.get("left")
        right_a = action.get("right")
        base_a = action.get("base")
        lift_a = action.get("lift")

        if left_a is not None:
            a = np.asarray(left_a, dtype=np.float64)
            self._left_eef += a[:6]
            self._left_joint[6] = np.clip(self._left_joint[6] + a[6], 0.0, 1.0)
        if right_a is not None:
            a = np.asarray(right_a, dtype=np.float64)
            self._right_eef += a[:6]
            self._right_joint[6] = np.clip(self._right_joint[6] + a[6], 0.0, 1.0)
        if lift_a is not None:
            self._base_height = np.clip(self._base_height + lift_a, 0.0, 20.0)
        return self._obs()

    def set_mode(self, mode, side="both"):
        print(f"[MockARXEnv] set_mode({mode}, side={side})")

    def _obs(self):
        return {
            "left_eef_pos": self._left_eef.copy().astype(np.float32),
            "left_joint_pos": self._left_joint.copy().astype(np.float32),
            "right_eef_pos": self._right_eef.copy().astype(np.float32),
            "right_joint_pos": self._right_joint.copy().astype(np.float32),
            "base_height": np.array([self._base_height], dtype=np.float32),
        }

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Camera background capture thread
# ---------------------------------------------------------------------------

cam_frames: dict[str, bytes | None] = {"camera_l": None, "camera_h": None, "camera_r": None}
cam_lock = threading.Lock()


def _make_placeholder_frame(label: str) -> bytes:
    """Generate a 640×480 dark placeholder JPEG with centered label."""
    img = np.full((480, 640, 3), 30, dtype=np.uint8)
    cv2.putText(img, label, (140, 230), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (100, 100, 100), 2, cv2.LINE_AA)
    cv2.putText(img, "MOCK MODE", (220, 290), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (80, 80, 80), 1, cv2.LINE_AA)
    _, jpeg = cv2.imencode(".jpg", img)
    return jpeg.tobytes()


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

env = None
obs: dict | None = None
lock = threading.Lock()

# 控制参数
STEP_PRESETS = {
    1: {"pos": 0.0004, "rot": 0.002, "base": 0.08, "lift": 0.2},   # 精细
    2: {"pos": 0.0005, "rot": 0.003, "base": 0.10, "lift": 0.3},   # 中等
    3: {"pos": 0.002,  "rot": 0.010, "base": 0.30, "lift": 0.8},   # 粗调
}
current_step_level = 1  # 默认精细档，避免急停
active_side = "left"  # 当前控制的臂
gravity_mode = False

# Action table: key → 7D delta [dx, dy, dz, droll, dpitch, dyaw, dgripper]
def _build_arm_action_map(sp, sr):
    return {
        "a": np.array([-sp, 0, 0, 0, 0, 0, 0]),   # X-
        "d": np.array([sp, 0, 0, 0, 0, 0, 0]),     # X+
        "w": np.array([0, sp, 0, 0, 0, 0, 0]),      # Y+
        "s": np.array([0, -sp, 0, 0, 0, 0, 0]),     # Y-
        "q": np.array([0, 0, sp, 0, 0, 0, 0]),      # Z+
        "e": np.array([0, 0, -sp, 0, 0, 0, 0]),     # Z-
        "i": np.array([0, 0, 0, sr, 0, 0, 0]),      # Roll+
        "k": np.array([0, 0, 0, -sr, 0, 0, 0]),     # Roll-
        "j": np.array([0, 0, 0, 0, sr, 0, 0]),      # Pitch+
        "l": np.array([0, 0, 0, 0, -sr, 0, 0]),     # Pitch-
        "u": np.array([0, 0, 0, 0, 0, sr, 0]),      # Yaw+
        "o": np.array([0, 0, 0, 0, 0, -sr, 0]),     # Yaw-
    }


def handle_key(key: str) -> dict:
    """Process a single keypress and return updated status."""
    global obs, active_side, current_step_level, gravity_mode

    with lock:
        sp = STEP_PRESETS[current_step_level]
        arm_map = _build_arm_action_map(sp["pos"], sp["rot"])

        # --- 臂控制 ---
        if key in arm_map:
            action = {"left": None, "right": None, "base": None, "lift": None}
            action[active_side] = arm_map[key]
            obs = env.step(action)
            return _status("ok", f"{active_side[0].upper()} {key}")

        # --- 夹爪切换 ---
        if key == " ":
            side_joint = f"{active_side}_joint_pos"
            curr_g = float(obs[side_joint][6]) if obs else 0.0
            # 切换: <0.5 → 闭合(1.0), ≥0.5 → 打开(0.0)
            target_delta = (1.0 - curr_g) if curr_g < 0.5 else (-curr_g)
            action = {"left": None, "right": None, "base": None, "lift": None}
            grip_cmd = np.array([0, 0, 0, 0, 0, 0, target_delta])
            action[active_side] = grip_cmd
            obs = env.step(action)
            return _status("ok", f"{active_side[0].upper()} 夹爪{'关闭' if curr_g < 0.5 else '打开'}")

        # --- 底盘控制 ---
        base_step = sp["base"]
        base_map = {
            "arrowup":    np.array([base_step, 0, 0]),
            "arrowdown":  np.array([-base_step, 0, 0]),
            "arrowleft":  np.array([0, base_step, 0]),
            "arrowright": np.array([0, -base_step, 0]),
            ",":          np.array([0, 0, base_step]),
            ".":          np.array([0, 0, -base_step]),
        }
        if key in base_map:
            obs = env.step({"left": None, "right": None, "base": base_map[key], "lift": None})
            return _status("ok", f"底盘 {key}")

        # --- 升降控制 ---
        lift_step = sp["lift"]
        if key == "=":
            obs = env.step({"left": None, "right": None, "base": None, "lift": lift_step})
            return _status("ok", "升降 ↑")
        if key == "-":
            obs = env.step({"left": None, "right": None, "base": None, "lift": -lift_step})
            return _status("ok", "升降 ↓")

        # --- 切换控制臂 ---
        if key == "tab":
            active_side = "right" if active_side == "left" else "left"
            return _status("ok", f"切换到 {'左臂' if active_side == 'left' else '右臂'}")

        # --- 步长档位 ---
        if key in ("1", "2", "3"):
            current_step_level = int(key)
            labels = {1: "精细", 2: "中等", 3: "粗调"}
            return _status("ok", f"步长: {labels[current_step_level]}")

        # --- 复位 ---
        if key == "r":
            obs = env.reset()
            gravity_mode = False
            return _status("ok", "已复位")

        # --- 重力补偿 ---
        if key == "g":
            gravity_mode = not gravity_mode
            env.set_mode(3 if gravity_mode else 1, side="both")
            return _status("ok", "重力补偿 " + ("开启" if gravity_mode else "关闭"))

        # --- 急停 ---
        if key == "escape":
            obs = env.step({"left": None, "right": None, "base": np.array([0, 0, 0]), "lift": None})
            return _status("warn", "急停!")

    return _status("ignore", "")


def _status(level: str, msg: str) -> dict:
    """Build JSON-serializable status dict."""
    if obs is None:
        return {"level": level, "msg": msg}

    def _eef(arr):
        return {
            "x": round(float(arr[0]) * 1000, 1),  # m → mm for display
            "y": round(float(arr[1]) * 1000, 1),
            "z": round(float(arr[2]) * 1000, 1),
            "roll": round(float(math.degrees(arr[3])), 1),
            "pitch": round(float(math.degrees(arr[4])), 1),
            "yaw": round(float(math.degrees(arr[5])), 1),
        }

    def _joints(arr):
        return [round(float(math.degrees(arr[i])), 1) for i in range(6)]

    return {
        "level": level,
        "msg": msg,
        "left_eef": _eef(obs["left_eef_pos"]),
        "right_eef": _eef(obs["right_eef_pos"]),
        "left_joints": _joints(obs["left_joint_pos"]),
        "right_joints": _joints(obs["right_joint_pos"]),
        "left_gripper": round(float(obs["left_joint_pos"][6]), 3),
        "right_gripper": round(float(obs["right_joint_pos"][6]), 3),
        "base_height": round(float(obs["base_height"][0]), 1),
        "active_side": active_side,
        "step_level": current_step_level,
        "gravity_mode": gravity_mode,
    }


# ---------------------------------------------------------------------------
# HTML — 三相机 + 双臂/底盘/升降 键盘控制面板
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ARX LIFT2 键盘控制</title>
<style>
  :root { --bg: #0a0a0a; --card: #151515; --card2: #1a1a1a; --accent: #3b82f6;
          --green: #22c55e; --red: #ef4444; --yellow: #eab308; --orange: #f97316;
          --purple: #a855f7; --cyan: #06b6d4;
          --text: #e4e4e7; --dim: #71717a; --border: #262626; }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: 'SF Mono', 'Cascadia Code', 'JetBrains Mono', monospace;
         background: var(--bg); color: var(--text); min-height:100vh;
         display:flex; flex-direction:column; align-items:center;
         padding: 16px 12px; user-select:none; }

  /* Header */
  .header { display:flex; align-items:center; gap:12px; margin-bottom:12px; flex-wrap:wrap; justify-content:center; }
  h1 { font-size: 15px; font-weight: 600; letter-spacing: 2px; color: var(--accent); }
  .badge { display:inline-block; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; }
  .badge.mock { background:#854d0e; color:#fef08a; }
  .badge.side-l { background:#1e3a5f; color:#93c5fd; }
  .badge.side-r { background:#3b1f5e; color:#d8b4fe; }
  .badge.step { background:#1a2e1a; color:#86efac; }
  .badge.gravity { background:#5c1a1a; color:#fca5a5; }
  #status { font-size: 12px; color: var(--dim); padding: 3px 10px;
            background: var(--card); border-radius: 4px; transition: all .15s; }
  #status.warn { color: var(--yellow); background: #422006; }

  /* Camera row */
  .cam-row { display:flex; gap:8px; margin-bottom:12px; width:100%; max-width:1200px; }
  .cam-box { flex:1; position:relative; }
  .cam-box img { width:100%; aspect-ratio:4/3; border-radius:6px;
                 background:var(--card); object-fit:cover; display:block;
                 border:1px solid var(--border); }
  .cam-label { position:absolute; top:6px; left:8px;
               font-size:9px; color:#fff; background:rgba(0,0,0,0.65);
               padding:2px 7px; border-radius:3px; letter-spacing:1px; }

  /* Info panel */
  .info-panel { width:100%; max-width:1200px; display:flex; gap:8px;
                margin-bottom:12px; flex-wrap:wrap; }
  .info-section { background:var(--card); border-radius:6px; padding:10px 12px;
                  border:1px solid var(--border); flex:1; min-width:200px; }
  .info-section.active-arm { border-color: var(--accent); box-shadow: 0 0 8px rgba(59,130,246,0.15); }
  .info-title { font-size:10px; font-weight:600; letter-spacing:1px;
                margin-bottom:6px; text-transform:uppercase; display:flex; align-items:center; gap:6px; }
  .info-title .dot { width:6px; height:6px; border-radius:50%; }

  /* Pose grid */
  .pose-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:4px; }
  .pose-cell { background:var(--card2); border-radius:4px; padding:5px 6px; text-align:center; }
  .pose-label { font-size:9px; color:var(--dim); margin-bottom:1px; }
  .pose-val { font-size:15px; font-weight:700; }

  /* Joint row */
  .joint-grid { display:grid; grid-template-columns:repeat(6,1fr); gap:3px; margin-top:6px; }
  .joint-cell { background:var(--card2); border-radius:3px; padding:3px 2px; text-align:center; }
  .joint-label { font-size:8px; color:var(--dim); }
  .joint-val { font-size:11px; font-weight:600; }

  /* Gripper bar */
  .gripper-row { display:flex; align-items:center; gap:8px; margin-top:8px; }
  .gripper-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
  .gripper-dot.open { background:var(--green); box-shadow:0 0 5px var(--green); }
  .gripper-dot.close { background:var(--red); box-shadow:0 0 5px var(--red); }
  .gripper-bar-bg { flex:1; height:6px; background:var(--card2); border-radius:3px; overflow:hidden; }
  .gripper-bar-fill { height:100%; background:var(--accent); border-radius:3px; transition:width .1s; }
  .gripper-pct { font-size:11px; font-weight:600; min-width:36px; text-align:right; }

  /* Extra info (base/lift) */
  .extra-row { display:flex; gap:8px; }
  .extra-cell { background:var(--card2); border-radius:4px; padding:6px 10px;
                text-align:center; flex:1; }
  .extra-label { font-size:9px; color:var(--dim); }
  .extra-val { font-size:18px; font-weight:700; }

  /* Controls */
  .controls { width:100%; max-width:1200px; background:var(--card);
              border-radius:6px; padding:12px; border:1px solid var(--border); }
  .ctrl-title { font-size:10px; color:var(--accent); font-weight:600;
                letter-spacing:1px; margin-bottom:10px; text-transform:uppercase; text-align:center; }
  .ctrl-groups { display:flex; justify-content:center; gap:24px; flex-wrap:wrap; }
  .ctrl-group { display:flex; flex-direction:column; align-items:center; }
  .ctrl-group-label { font-size:9px; color:var(--dim); margin-bottom:5px;
                      letter-spacing:1px; text-transform:uppercase; }
  .key-hint { display:grid; gap:3px; }
  .key-row { display:flex; justify-content:center; gap:3px; }
  .key { width:40px; height:40px; border-radius:5px; display:flex;
         align-items:center; justify-content:center; font-size:12px;
         font-weight:600; background:var(--card2); border:1px solid var(--border);
         transition:all .08s; cursor:pointer; flex-direction:column; line-height:1.1; }
  .key:hover { border-color:#444; }
  .key.active { background:var(--accent); border-color:var(--accent);
                color:#fff; transform:scale(0.93); }
  .key .key-sub { font-size:7px; color:var(--dim); font-weight:400; }
  .key.active .key-sub { color:rgba(255,255,255,0.7); }
  .key.wide { width:72px; }
  .key.action-btn { background:#1a1a2e; border-color:#2a2a4e; }
  .key.action-btn:hover { border-color:var(--accent); }
  .key.action-btn.danger { border-color:#7f1d1d; }
  .key.action-btn.danger:hover { border-color:var(--red); }
  .key.action-btn.gravity-btn { border-color:#3b1f5e; }
  .key.action-btn.gravity-btn:hover { border-color:var(--purple); }

  /* Footer */
  .footer { margin-top:10px; font-size:9px; color:var(--dim); text-align:center; }
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <h1>ARX LIFT2 KEYBOARD CTRL</h1>
  <div id="mode-badge"></div>
  <div class="badge side-l" id="side-badge">LEFT 左臂</div>
  <div class="badge step" id="step-badge">步长 2 中等</div>
  <div class="badge gravity" id="gravity-badge" style="display:none;">重力补偿</div>
  <div id="status">按任意控制键开始 ...</div>
</div>

<!-- Cameras -->
<div class="cam-row">
  <div class="cam-box">
    <div class="cam-label">LEFT · D405</div>
    <img id="cam-l" src="/stream/camera_l" alt="left camera">
  </div>
  <div class="cam-box">
    <div class="cam-label">HIGH · D405</div>
    <img id="cam-h" src="/stream/camera_h" alt="high camera">
  </div>
  <div class="cam-box">
    <div class="cam-label">RIGHT · D405</div>
    <img id="cam-r" src="/stream/camera_r" alt="right camera">
  </div>
</div>

<!-- Info panel -->
<div class="info-panel">

  <!-- Left arm -->
  <div class="info-section" id="left-section">
    <div class="info-title" style="color:#93c5fd;">
      <span class="dot" style="background:#3b82f6;"></span> 左臂 · Left Arm
    </div>
    <div class="pose-grid">
      <div class="pose-cell"><div class="pose-label">X (mm)</div><div class="pose-val" id="lx">—</div></div>
      <div class="pose-cell"><div class="pose-label">Y (mm)</div><div class="pose-val" id="ly">—</div></div>
      <div class="pose-cell"><div class="pose-label">Z (mm)</div><div class="pose-val" id="lz">—</div></div>
      <div class="pose-cell"><div class="pose-label">Roll°</div><div class="pose-val" id="lr">—</div></div>
      <div class="pose-cell"><div class="pose-label">Pitch°</div><div class="pose-val" id="lp">—</div></div>
      <div class="pose-cell"><div class="pose-label">Yaw°</div><div class="pose-val" id="lw">—</div></div>
    </div>
    <div class="joint-grid">
      <div class="joint-cell"><div class="joint-label">J1</div><div class="joint-val" id="lj0">—</div></div>
      <div class="joint-cell"><div class="joint-label">J2</div><div class="joint-val" id="lj1">—</div></div>
      <div class="joint-cell"><div class="joint-label">J3</div><div class="joint-val" id="lj2">—</div></div>
      <div class="joint-cell"><div class="joint-label">J4</div><div class="joint-val" id="lj3">—</div></div>
      <div class="joint-cell"><div class="joint-label">J5</div><div class="joint-val" id="lj4">—</div></div>
      <div class="joint-cell"><div class="joint-label">J6</div><div class="joint-val" id="lj5">—</div></div>
    </div>
    <div class="gripper-row">
      <span class="gripper-dot open" id="lg-dot"></span>
      <div class="gripper-bar-bg"><div class="gripper-bar-fill" id="lg-bar" style="width:0%"></div></div>
      <span class="gripper-pct" id="lg-pct">0%</span>
    </div>
  </div>

  <!-- Right arm -->
  <div class="info-section" id="right-section">
    <div class="info-title" style="color:#d8b4fe;">
      <span class="dot" style="background:#a855f7;"></span> 右臂 · Right Arm
    </div>
    <div class="pose-grid">
      <div class="pose-cell"><div class="pose-label">X (mm)</div><div class="pose-val" id="rx">—</div></div>
      <div class="pose-cell"><div class="pose-label">Y (mm)</div><div class="pose-val" id="ry">—</div></div>
      <div class="pose-cell"><div class="pose-label">Z (mm)</div><div class="pose-val" id="rz">—</div></div>
      <div class="pose-cell"><div class="pose-label">Roll°</div><div class="pose-val" id="rr">—</div></div>
      <div class="pose-cell"><div class="pose-label">Pitch°</div><div class="pose-val" id="rp">—</div></div>
      <div class="pose-cell"><div class="pose-label">Yaw°</div><div class="pose-val" id="rw">—</div></div>
    </div>
    <div class="joint-grid">
      <div class="joint-cell"><div class="joint-label">J1</div><div class="joint-val" id="rj0">—</div></div>
      <div class="joint-cell"><div class="joint-label">J2</div><div class="joint-val" id="rj1">—</div></div>
      <div class="joint-cell"><div class="joint-label">J3</div><div class="joint-val" id="rj2">—</div></div>
      <div class="joint-cell"><div class="joint-label">J4</div><div class="joint-val" id="rj3">—</div></div>
      <div class="joint-cell"><div class="joint-label">J5</div><div class="joint-val" id="rj4">—</div></div>
      <div class="joint-cell"><div class="joint-label">J6</div><div class="joint-val" id="rj5">—</div></div>
    </div>
    <div class="gripper-row">
      <span class="gripper-dot open" id="rg-dot"></span>
      <div class="gripper-bar-bg"><div class="gripper-bar-fill" id="rg-bar" style="width:0%"></div></div>
      <span class="gripper-pct" id="rg-pct">0%</span>
    </div>
  </div>

  <!-- Base + Lift -->
  <div class="info-section" style="min-width:160px; max-width:220px;">
    <div class="info-title" style="color:var(--cyan);">
      <span class="dot" style="background:var(--cyan);"></span> 底盘 · 升降
    </div>
    <div class="extra-row">
      <div class="extra-cell">
        <div class="extra-label">升降高度</div>
        <div class="extra-val" id="lift-val" style="color:var(--cyan);">0.0</div>
      </div>
    </div>
    <div style="margin-top:8px;">
      <div class="extra-row" style="gap:4px;">
        <div class="extra-cell">
          <div class="extra-label">步长·位移</div>
          <div class="extra-val" id="step-pos" style="font-size:13px; color:var(--green);">5.0mm</div>
        </div>
        <div class="extra-cell">
          <div class="extra-label">步长·旋转</div>
          <div class="extra-val" id="step-rot" style="font-size:13px; color:var(--green);">1.7°</div>
        </div>
      </div>
    </div>
  </div>

</div>

<!-- Controls -->
<div class="controls">
  <div class="ctrl-title">键盘控制 · Keyboard Controls</div>
  <div class="ctrl-groups">

    <!-- Arm Translation -->
    <div class="ctrl-group">
      <div class="ctrl-group-label">臂·平移 XYZ</div>
      <div class="key-hint">
        <div class="key-row">
          <div class="key" id="kq">Q<span class="key-sub">Z+</span></div>
          <div class="key" id="kw">W<span class="key-sub">Y+</span></div>
          <div class="key" id="ke">E<span class="key-sub">Z-</span></div>
        </div>
        <div class="key-row">
          <div class="key" id="ka">A<span class="key-sub">X-</span></div>
          <div class="key" id="ks">S<span class="key-sub">Y-</span></div>
          <div class="key" id="kd">D<span class="key-sub">X+</span></div>
        </div>
      </div>
    </div>

    <!-- Arm Rotation -->
    <div class="ctrl-group">
      <div class="ctrl-group-label">臂·旋转 RPY</div>
      <div class="key-hint">
        <div class="key-row">
          <div class="key" id="ku">U<span class="key-sub">Yaw+</span></div>
          <div class="key" id="ki">I<span class="key-sub">Roll+</span></div>
          <div class="key" id="ko">O<span class="key-sub">Yaw-</span></div>
        </div>
        <div class="key-row">
          <div class="key" id="kj">J<span class="key-sub">Pit+</span></div>
          <div class="key" id="kk">K<span class="key-sub">Roll-</span></div>
          <div class="key" id="kl">L<span class="key-sub">Pit-</span></div>
        </div>
      </div>
    </div>

    <!-- Base -->
    <div class="ctrl-group">
      <div class="ctrl-group-label">底盘</div>
      <div class="key-hint">
        <div class="key-row">
          <div class="key" id="kcomma">,<span class="key-sub">左转</span></div>
          <div class="key" id="kup">↑<span class="key-sub">前进</span></div>
          <div class="key" id="kdot">.<span class="key-sub">右转</span></div>
        </div>
        <div class="key-row">
          <div class="key" id="kleft">←<span class="key-sub">左移</span></div>
          <div class="key" id="kdown">↓<span class="key-sub">后退</span></div>
          <div class="key" id="kright">→<span class="key-sub">右移</span></div>
        </div>
      </div>
    </div>

    <!-- Lift + Actions -->
    <div class="ctrl-group">
      <div class="ctrl-group-label">升降 / 操作</div>
      <div class="key-hint">
        <div class="key-row">
          <div class="key" id="keq">=<span class="key-sub">升</span></div>
          <div class="key" id="kminus">-<span class="key-sub">降</span></div>
          <div class="key wide action-btn" id="kspace">Space<span class="key-sub">夹爪</span></div>
        </div>
        <div class="key-row">
          <div class="key action-btn" id="ktab" style="width:46px;">Tab<span class="key-sub">切臂</span></div>
          <div class="key action-btn gravity-btn" id="kg">G<span class="key-sub">重力</span></div>
          <div class="key action-btn" id="kr">R<span class="key-sub">复位</span></div>
          <div class="key action-btn danger" id="kesc">Esc<span class="key-sub">急停</span></div>
        </div>
        <div class="key-row" style="margin-top:2px;">
          <div class="key" id="k1" style="width:32px;font-size:11px;">1<span class="key-sub">精细</span></div>
          <div class="key" id="k2" style="width:32px;font-size:11px;">2<span class="key-sub">中</span></div>
          <div class="key" id="k3" style="width:32px;font-size:11px;">3<span class="key-sub">粗</span></div>
        </div>
      </div>
    </div>

  </div>
</div>

<div class="footer">
  ARX LIFT2 · 2×6-DOF · delta_eef · 3×RealSense D405
</div>

<script>
const KEY_IDS = {
  'q':'kq','w':'kw','e':'ke','a':'ka','s':'ks','d':'kd',
  'i':'ki','k':'kk','j':'kj','l':'kl','u':'ku','o':'ko',
  'r':'kr',' ':'kspace','escape':'kesc','tab':'ktab','g':'kg',
  'arrowup':'kup','arrowdown':'kdown','arrowleft':'kleft','arrowright':'kright',
  ',':'kcomma','.':'kdot','=':'keq','-':'kminus',
  '1':'k1','2':'k2','3':'k3'
};
const pressedKeys = new Set();
const REPEAT_KEYS = new Set('qweasdikjluo'.split('').concat(
  ['arrowup','arrowdown','arrowleft','arrowright',',','.','=','-']
));
const STEP_INFO = {
  1: {pos:'0.4mm', rot:'0.11°'},
  2: {pos:'0.5mm', rot:'0.17°'},
  3: {pos:'2.0mm', rot:'0.57°'},
};

function send(key) {
  fetch('/cmd', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({key})
  })
  .then(r => r.json())
  .then(update)
  .catch(() => {
    document.getElementById('status').textContent = '⚠ 通信失败';
    document.getElementById('status').className = 'warn';
  });
}

function update(d) {
  if (d.level === 'ignore') return;
  const st = document.getElementById('status');
  st.textContent = d.msg;
  st.className = d.level === 'warn' ? 'warn' : '';

  if (!d.left_eef) return;

  // Left arm
  document.getElementById('lx').textContent = d.left_eef.x;
  document.getElementById('ly').textContent = d.left_eef.y;
  document.getElementById('lz').textContent = d.left_eef.z;
  document.getElementById('lr').textContent = d.left_eef.roll;
  document.getElementById('lp').textContent = d.left_eef.pitch;
  document.getElementById('lw').textContent = d.left_eef.yaw;
  for (let i=0;i<6;i++) document.getElementById('lj'+i).textContent = d.left_joints[i];
  const lg = Math.round(d.left_gripper * 100);
  document.getElementById('lg-dot').className = 'gripper-dot ' + (d.left_gripper < 0.5 ? 'open' : 'close');
  document.getElementById('lg-bar').style.width = lg + '%';
  document.getElementById('lg-pct').textContent = lg + '%';

  // Right arm
  document.getElementById('rx').textContent = d.right_eef.x;
  document.getElementById('ry').textContent = d.right_eef.y;
  document.getElementById('rz').textContent = d.right_eef.z;
  document.getElementById('rr').textContent = d.right_eef.roll;
  document.getElementById('rp').textContent = d.right_eef.pitch;
  document.getElementById('rw').textContent = d.right_eef.yaw;
  for (let i=0;i<6;i++) document.getElementById('rj'+i).textContent = d.right_joints[i];
  const rg = Math.round(d.right_gripper * 100);
  document.getElementById('rg-dot').className = 'gripper-dot ' + (d.right_gripper < 0.5 ? 'open' : 'close');
  document.getElementById('rg-bar').style.width = rg + '%';
  document.getElementById('rg-pct').textContent = rg + '%';

  // Base & Lift
  document.getElementById('lift-val').textContent = d.base_height;

  // Active side highlight
  const ls = document.getElementById('left-section');
  const rs = document.getElementById('right-section');
  ls.className = 'info-section' + (d.active_side === 'left' ? ' active-arm' : '');
  rs.className = 'info-section' + (d.active_side === 'right' ? ' active-arm' : '');

  const sb = document.getElementById('side-badge');
  sb.className = 'badge ' + (d.active_side === 'left' ? 'side-l' : 'side-r');
  sb.textContent = d.active_side === 'left' ? 'LEFT 左臂' : 'RIGHT 右臂';

  // Step level
  const labels = {1:'精细',2:'中等',3:'粗调'};
  document.getElementById('step-badge').textContent = '步长 ' + d.step_level + ' ' + labels[d.step_level];
  const si = STEP_INFO[d.step_level];
  document.getElementById('step-pos').textContent = si.pos;
  document.getElementById('step-rot').textContent = si.rot;

  // Gravity mode
  const gb = document.getElementById('gravity-badge');
  gb.style.display = d.gravity_mode ? 'inline-block' : 'none';
}

// --- Continuous key repeat while held ---
let repeatTimers = {};

document.addEventListener('keydown', e => {
  const k = e.key.toLowerCase();
  const mapped = k === ' ' ? ' ' : k;

  if (mapped === 'escape') { send('escape'); return; }
  if (mapped === 'tab') { e.preventDefault(); send('tab'); return; }
  if (!(mapped in KEY_IDS)) return;
  e.preventDefault();

  const el = KEY_IDS[mapped];
  if (el) document.getElementById(el)?.classList.add('active');

  if (pressedKeys.has(mapped)) return;
  pressedKeys.add(mapped);

  send(mapped);
  if (REPEAT_KEYS.has(mapped)) {
    repeatTimers[mapped] = setTimeout(() => {
      repeatTimers[mapped] = setInterval(() => send(mapped), 80);
    }, 120);
  }
});

document.addEventListener('keyup', e => {
  const k = e.key.toLowerCase();
  const mapped = k === ' ' ? ' ' : k;

  const el = KEY_IDS[mapped];
  if (el) document.getElementById(el)?.classList.remove('active');

  pressedKeys.delete(mapped);
  if (repeatTimers[mapped]) {
    clearTimeout(repeatTimers[mapped]);
    clearInterval(repeatTimers[mapped]);
    delete repeatTimers[mapped];
  }
});

// Click support
document.querySelectorAll('.key').forEach(el => {
  el.addEventListener('mousedown', () => {
    const keyMap = {};
    for (const [k, v] of Object.entries(KEY_IDS)) { keyMap[v] = k; }
    const k = keyMap[el.id];
    if (k) send(k);
  });
});

// Fetch initial state
fetch('/cmd', {method:'POST', headers:{'Content-Type':'application/json'},
  body: JSON.stringify({key:'_init'})}).then(r=>r.json()).then(d => {
    if (d.level !== 'ignore') update(d);
  });
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class Handler(SimpleHTTPRequestHandler):
    """Serve HTML page, MJPEG camera streams, and /cmd POST."""

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/stream/"):
            cam_name = self.path.split("/")[-1]
            if cam_name in cam_frames:
                self._stream_mjpeg(cam_name)
            else:
                self.send_error(404)
        elif self.path == "/":
            self._serve_html()
        else:
            self.send_error(404)

    def _serve_html(self):
        page = HTML_PAGE
        if args_global.mock:
            page = page.replace(
                '<div id="mode-badge"></div>',
                '<div id="mode-badge"><span class="badge mock">MOCK 模式</span></div>'
            )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(page.encode())

    def _stream_mjpeg(self, cam_name: str):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        try:
            while True:
                with cam_lock:
                    frame = cam_frames.get(cam_name)
                if frame is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n".encode())
                self.wfile.write(b"\r\n")
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                time.sleep(0.033)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        key = body.get("key", "")

        if key == "_init":
            result = _status("ok", "已连接")
        else:
            result = handle_key(key)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result, ensure_ascii=False).encode())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

args_global = None


def main():
    global env, obs, args_global

    parser = argparse.ArgumentParser(description="ARX LIFT2 键盘控制 Web Demo")
    parser.add_argument("--mock", action="store_true",
                        help="使用模拟环境 (无需真机)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Web 服务端口 (默认 8080)")
    args = parser.parse_args()
    args_global = args

    # --- Init robot env ---
    if args.mock:
        env = MockARXEnv()
    else:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from arx_toolkit.env import ARXEnv
        env = ARXEnv(
            action_mode="delta_eef",
            camera_type="rgb",
            camera_view=("camera_l", "camera_h", "camera_r"),
            img_size=(640, 480),
        )

    obs = env.reset()

    # --- Init cameras ---
    if args.mock:
        cam_frames["camera_l"] = _make_placeholder_frame("LEFT CAM")
        cam_frames["camera_h"] = _make_placeholder_frame("HIGH CAM")
        cam_frames["camera_r"] = _make_placeholder_frame("RIGHT CAM")
    else:
        # 真机模式: 相机由 ARXEnv 内部 ROS2 订阅管理
        # 启动后台线程定期从 env 获取相机帧
        import threading

        def _camera_loop():
            while True:
                try:
                    cam_obs = env.get_observation(
                        include_arm=False, include_camera=True, include_base=False
                    )
                    for cam_key in ("camera_l_color", "camera_h_color", "camera_r_color"):
                        if cam_key in cam_obs:
                            rgb = np.asarray(cam_obs[cam_key])
                            # obs 已经是 RGB, imencode 需要 BGR
                            # 但 D405 经 passthrough 解码后 _decode 做了 [::-1]
                            # 实测通道已反，这里不再转换，直接 encode
                            ok, jpeg = cv2.imencode(
                                ".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 70]
                            )
                            if ok:
                                # camera_l_color → camera_l
                                name = cam_key.replace("_color", "")
                                with cam_lock:
                                    cam_frames[name] = jpeg.tobytes()
                except Exception as e:
                    print(f"[CameraLoop] {e}")
                    time.sleep(0.1)
                time.sleep(0.033)  # ~30 fps cap

        t = threading.Thread(target=_camera_loop, daemon=True, name="cam-loop")
        t.start()

    # --- Start server ---
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"\n  🤖 ARX LIFT2 键盘控制 Demo 已启动")
    print(f"  📡 打开浏览器: http://localhost:{args.port}")
    print(f"  {'⚠️  MOCK 模式 — 不连接真机' if args.mock else '✅ 真机模式'}")
    print(f"  按 Ctrl+C 退出\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭 ...")
        server.shutdown()
        env.close()


if __name__ == "__main__":
    main()
