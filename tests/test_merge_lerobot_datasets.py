from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


class FakeSeries:
    def __init__(self, values):
        self.values = list(values)

    def astype(self, _dtype):
        return self

    def unique(self):
        return list(dict.fromkeys(self.values))

    def min(self):
        return min(self.values)

    def max(self):
        return max(self.values)

    def __add__(self, other):
        return FakeSeries([value + other for value in self.values])


class FakeFrame:
    def __init__(self, rows):
        self.rows = [dict(row) for row in rows]

    @property
    def columns(self):
        keys = set()
        for row in self.rows:
            keys.update(row)
        return keys

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        return FakeSeries([row[key] for row in self.rows])

    def __setitem__(self, key, values):
        if isinstance(values, FakeSeries):
            values = values.values
        if isinstance(values, list):
            assert len(self.rows) == len(values)
            for row, value in zip(self.rows, values):
                row[key] = value
            return
        for row in self.rows:
            row[key] = values


def load_merge_module(monkeypatch):
    fake_numpy = types.SimpleNamespace(int64=int)
    fake_pandas = types.SimpleNamespace(
        DataFrame=lambda *args, **kwargs: None,
        concat=lambda frames, ignore_index=True: FakeFrame(
            [row for frame in frames for row in frame.rows]
        ),
    )
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "pandas", fake_pandas)

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "merge_lerobot_datasets.py"
    )
    spec = importlib.util.spec_from_file_location("merge_lerobot_datasets", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_merge_episodes_rewrites_frame_and_data_file_metadata(monkeypatch):
    module = load_merge_module(monkeypatch)
    episode_tables = {
        "first": FakeFrame(
            [
                {
                    "episode_index": 0,
                    "length": 2,
                    "dataset_from_index": 0,
                    "dataset_to_index": 2,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "videos/observation.images.cam/file_index": 0,
                    "videos/observation.images.cam/chunk_index": 0,
                }
            ]
        ),
        "second": FakeFrame(
            [
                {
                    "episode_index": 0,
                    "length": 3,
                    "dataset_from_index": 0,
                    "dataset_to_index": 3,
                    "data/chunk_index": 7,
                    "data/file_index": 9,
                    "videos/observation.images.cam/file_index": 0,
                    "videos/observation.images.cam/chunk_index": 0,
                }
            ]
        ),
    }
    monkeypatch.setattr(
        module, "_read_episodes", lambda path: episode_tables[path.name]
    )

    merged = module._merge_episodes(
        [(Path("first"), "first"), (Path("second"), "second")],
        episode_offsets=[0, 1],
        frame_offsets=[0, 2],
        total_episodes=2,
        all_video_keys=["observation.images.cam"],
        video_file_offsets={
            "first": {"observation.images.cam": 0},
            "second": {"observation.images.cam": 1},
        },
    )

    assert merged.rows == [
        {
            "episode_index": 0,
            "length": 2,
            "dataset_from_index": 0,
            "dataset_to_index": 2,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "videos/observation.images.cam/file_index": 0,
            "videos/observation.images.cam/chunk_index": 0,
        },
        {
            "episode_index": 1,
            "length": 3,
            "dataset_from_index": 2,
            "dataset_to_index": 5,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "videos/observation.images.cam/file_index": 1,
            "videos/observation.images.cam/chunk_index": 0,
        },
    ]
