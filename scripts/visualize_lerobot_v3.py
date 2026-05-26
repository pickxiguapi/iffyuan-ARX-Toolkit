#!/usr/bin/env python3
"""Visualize converted LeRobot v3 datasets locally."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


def _import_lerobot_viz():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.scripts.lerobot_dataset_viz import visualize_dataset

    return LeRobotDataset, visualize_dataset


def _parse_episode_selector(selector: int | str, total_episodes: int) -> list[int]:
    if total_episodes <= 0:
        raise ValueError("Dataset has no episodes.")

    if isinstance(selector, int):
        indices = [selector]
    else:
        raw = str(selector).strip().lower()
        if raw in {"", "all"}:
            indices = list(range(total_episodes))
        elif raw == "random":
            indices = [random.randrange(total_episodes)]
        elif raw.isdigit():
            indices = [int(raw)]
        elif raw.startswith("[") and raw.endswith("]"):
            body = raw[1:-1]
            parts = [part.strip() for part in body.split(",")]
            if len(parts) != 2 or not all(part.lstrip("-").isdigit() for part in parts):
                raise ValueError("Episode selector range must look like '[1,5]'.")
            start, end = (int(parts[0]), int(parts[1]))
            if start > end:
                raise ValueError("Episode selector range start must be <= end.")
            indices = list(range(start, end + 1))
        else:
            raise ValueError(
                "Unsupported episode selector. Use 'all', 'random', "
                "an integer like '3', or a range like '[1,5]'."
            )

    normalized: list[int] = []
    for idx in indices:
        if idx < 0 or idx >= total_episodes:
            raise IndexError(
                f"Episode index {idx} is out of range for "
                f"total_episodes={total_episodes}."
            )
        normalized.append(idx)
    return normalized


def _dataset_num_episodes(meta_root: Path) -> int:
    info_path = meta_root / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing dataset info file: {info_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    total_episodes = int(info.get("total_episodes", 0))
    if total_episodes <= 0:
        raise ValueError(f"Invalid total_episodes in {info_path}: {total_episodes}")
    return total_episodes


def visualize_lerobot_v3(
    repo_id: str,
    dataset_root: Path | str,
    episode_selector: int | str = "all",
    video_backend: str = "pyav",
) -> None:
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    os.environ["HF_HUB_OFFLINE"] = "1"

    total_episodes = _dataset_num_episodes(dataset_root / "meta")
    selected_indices = _parse_episode_selector(episode_selector, total_episodes)
    print(
        f"[INFO] Visualizing {len(selected_indices)} / {total_episodes} episodes "
        f"from {dataset_root}"
    )

    LeRobotDataset, visualize_dataset = _import_lerobot_viz()

    for episode_index in selected_indices:
        print(f"[INFO] Episode {episode_index}")
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=dataset_root,
            episodes=[episode_index],
            video_backend=video_backend,
        )
        visualize_dataset(
            dataset,
            episode_index=episode_index,
            mode="local",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a local LeRobot v3 dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Dataset repo id used when the LeRobot dataset was created.",
    )
    parser.add_argument(
        "--dataset-root",
        "-d",
        required=True,
        help="Local LeRobot dataset root, e.g. lerobot_datasets/pick_cup.",
    )
    parser.add_argument(
        "--episodes",
        default="all",
        help="Episode selector: all, random, an integer, or a range like [1,5].",
    )
    parser.add_argument(
        "--video-backend",
        default="pyav",
        help="LeRobot video backend.",
    )
    args = parser.parse_args()

    visualize_lerobot_v3(
        repo_id=args.repo_id,
        dataset_root=args.dataset_root,
        episode_selector=args.episodes,
        video_backend=args.video_backend,
    )


if __name__ == "__main__":
    main()
