from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from arx_toolkit.env.arx_env import ARXEnv


def make_env() -> ARXEnv:
    env = ARXEnv.__new__(ARXEnv)
    env.img_size = None
    env._closed = False
    return env


def test_step_dispatches_arm_and_base_lift() -> None:
    env = make_env()
    calls = []

    def fake_step_arm(*, left=None, right=None, action_mode, return_observation=False):
        calls.append(("arm", left, right, action_mode, return_observation))

    def fake_step_base_lift(*, vx=None, vy=None, vz=None, height=None):
        calls.append(("base_lift", vx, vy, vz, height))

    env.step_arm = fake_step_arm  # type: ignore[method-assign]
    env.step_base_lift = fake_step_base_lift  # type: ignore[method-assign]
    env.get_observation = lambda: {"ok": np.array([1], dtype=np.float32)}  # type: ignore[method-assign]

    left = np.zeros(7, dtype=np.float32)
    obs = env.step(
        {"left": left, "right": None, "base": np.array([1.0, 2.0, 3.0]), "lift": 4.0},
        action_mode="absolute_eef",
    )

    assert obs["ok"][0] == 1
    assert calls == [
        ("arm", left, None, "absolute_eef", False),
        ("base_lift", 1.0, 2.0, 3.0, 4.0),
    ]


def test_step_lift_only_preserves_base_velocity_with_none_components() -> None:
    env = make_env()
    calls = []

    env.step_base_lift = lambda **kwargs: calls.append(kwargs)  # type: ignore[method-assign]

    env.step(
        {"left": None, "right": None, "base": None, "lift": 10.0},
        action_mode="absolute_joint",
        return_observation=False,
    )

    assert calls == [{"vx": None, "vy": None, "vz": None, "height": 10.0}]


def test_step_arm_is_wrapper_for_arm_dispatch_only() -> None:
    env = make_env()
    calls = []

    env._apply_absolute_joint = lambda action: calls.append(action)  # type: ignore[method-assign]

    right = np.ones(7, dtype=np.float32)
    env.step_arm(right=right, action_mode="absolute_joint", return_observation=False)

    assert len(calls) == 1
    assert set(calls[0]) == {"right"}
    np.testing.assert_array_equal(calls[0]["right"], right)


def test_step_base_lift_none_components_keep_previous_command(monkeypatch) -> None:
    class FakePosCmd:
        def __init__(self):
            self.chx = 0.0
            self.chy = 0.0
            self.chz = 0.0
            self.height = 0.0
            self.mode1 = 0

    sent = []
    fake_node = SimpleNamespace(
        get_robot_status=lambda: {"base": SimpleNamespace(height=2.0)},
        send_base_msg=lambda msg: sent.append(msg) or True,
    )
    env = make_env()
    env.node = fake_node
    env._init_base_lift_state()
    env._ensure_base_lift_smoother = lambda: None  # type: ignore[method-assign]

    import sys

    monkeypatch.setitem(sys.modules, "arm_control", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "arm_control.msg", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "arm_control.msg._pos_cmd",
        SimpleNamespace(PosCmd=FakePosCmd),
    )

    env.step_base_lift(vx=0.5, vy=0.0, vz=0.1, height=None)
    env.step_base_lift(vx=None, vy=None, vz=None, height=8.0)

    assert sent[-1].chx == 0.5
    assert sent[-1].chy == 0.0
    assert sent[-1].chz == 0.1
    assert sent[-1].height == 2.0
    assert env._lift_target == 8.0


def test_step_base_lift_lift_only_does_not_publish_old_height(monkeypatch) -> None:
    sent = []
    env = make_env()
    env.node = SimpleNamespace(get_robot_status=lambda: {"base": SimpleNamespace(height=2.0)})
    env._init_base_lift_state()
    env._publish_base_lift_once = lambda *args: sent.append(args) or True  # type: ignore[method-assign]
    env._ensure_base_lift_smoother = lambda: None  # type: ignore[method-assign]

    env.step_base_lift(height=8.0)

    assert sent == []
    assert env._lift_target == 8.0


def test_reset_waits_for_lift_target_before_observation() -> None:
    env = make_env()
    calls = []
    env._safe_stop_robot = lambda: calls.append(("safe_stop", None))  # type: ignore[method-assign]
    env.get_observation = lambda: calls.append(("obs", None)) or {"ok": np.array([1], dtype=np.float32)}  # type: ignore[method-assign]

    env.reset()

    assert calls == [
        ("safe_stop", None),
        ("obs", None),
    ]


def test_close_uses_safe_stop_without_observation() -> None:
    env = make_env()
    calls = []
    env._safe_stop_robot = lambda: calls.append(("safe_stop", None))  # type: ignore[method-assign]
    env.get_observation = lambda: calls.append(("obs", None)) or {}  # type: ignore[method-assign]
    env._stop_base_lift_smoother = lambda: calls.append(("stop_smoother", None))  # type: ignore[method-assign]
    env._shutdown_ros2 = lambda: calls.append(("shutdown", None))  # type: ignore[method-assign]

    env.close()

    assert calls == [
        ("safe_stop", None),
        ("stop_smoother", None),
        ("shutdown", None),
    ]
