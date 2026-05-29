#!/usr/bin/env python3
"""Merge multiple LeRobot v3 datasets for the same task into one dataset.

Usage::

    source .venv/bin/activate
    python data_collection/merge_lerobot_datasets.py \\
        --datasets lerobot_datasets/add_bottom_bread_vr_20260526_202336 \\
                   lerobot_datasets/add_bottom_bread_vr_20260526_181056 \\
        --output lerobot_datasets/add_bottom_bread_merged \\
        --repo-id bluecontra/arx_add_bottom_bread

    # --dry-run prints what would happen without copying files
    python data_collection/merge_lerobot_datasets.py \\
        --datasets lerobot_datasets/add_bottom_bread_vr_* \\
        --output lerobot_datasets/add_bottom_bread_merged \\
        --repo-id bluecontra/arx_add_bottom_bread --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_info(dataset_path: Path) -> dict:
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing info.json in {dataset_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def _read_tasks(dataset_path: Path) -> pd.DataFrame:
    tasks_path = dataset_path / "meta" / "tasks.parquet"
    if tasks_path.is_file():
        return pd.read_parquet(tasks_path)
    return pd.DataFrame()


def _read_data(dataset_path: Path) -> pd.DataFrame:
    data_dir = dataset_path / "data"
    dfs = []
    for chunk_dir in sorted(data_dir.iterdir()):
        if chunk_dir.is_dir():
            for parquet_file in sorted(chunk_dir.glob("file-*.parquet")):
                dfs.append(pd.read_parquet(parquet_file))
    if not dfs:
        raise FileNotFoundError(f"No data parquet files in {data_dir}")
    return pd.concat(dfs, ignore_index=True)


def _read_episodes(dataset_path: Path) -> pd.DataFrame:
    eps_dir = dataset_path / "meta" / "episodes"
    dfs = []
    for chunk_dir in sorted(eps_dir.iterdir()):
        if chunk_dir.is_dir():
            for parquet_file in sorted(chunk_dir.glob("file-*.parquet")):
                dfs.append(pd.read_parquet(parquet_file))
    if not dfs:
        raise FileNotFoundError(f"No episode parquet files in {eps_dir}")
    return pd.concat(dfs, ignore_index=True)


def _validate_compatible(datasets: list[tuple[Path, str]], infos: list[dict]) -> None:
    """Check all datasets have the same feature keys and shapes."""
    ref_features = infos[0]["features"]
    ref_video_keys = {k for k in ref_features if ref_features[k].get("dtype") == "video"}
    ref_image_keys = {k for k in ref_features if ref_features[k].get("dtype") == "image"}

    for i, (path, label) in enumerate(datasets[1:], 1):
        feats = infos[i]["features"]
        cur_video_keys = {k for k in feats if feats[k].get("dtype") == "video"}
        cur_image_keys = {k for k in feats if feats[k].get("dtype") == "image"}

        if cur_video_keys != ref_video_keys:
            raise ValueError(
                f"Video key mismatch: {label} has {sorted(cur_video_keys)}, "
                f"expected {sorted(ref_video_keys)}"
            )
        if cur_image_keys != ref_image_keys:
            raise ValueError(
                f"Image key mismatch: {label} has {sorted(cur_image_keys)}, "
                f"expected {sorted(ref_image_keys)}"
            )

        # Check state/action shapes match
        for key in ("observation.state", "action"):
            if key in ref_features and key in feats:
                if ref_features[key]["shape"] != feats[key]["shape"]:
                    raise ValueError(
                        f"{key} shape mismatch in {label}: "
                        f"{feats[key]['shape']} vs {ref_features[key]['shape']}"
                    )

        # Check all feature names match
        ref_keys = set(ref_features)
        cur_keys = set(feats)
        if ref_keys != cur_keys:
            extra = cur_keys - ref_keys
            missing = ref_keys - cur_keys
            parts = []
            if extra:
                parts.append(f"extra: {sorted(extra)}")
            if missing:
                parts.append(f"missing: {sorted(missing)}")
            raise ValueError(
                f"Feature key mismatch in {label}: {', '.join(parts)}"
            )

    print(f"[OK] All {len(datasets)} datasets have compatible features")


def _compute_video_file_offsets(
    datasets: list[tuple[Path, str]],
    all_video_keys: list[str],
) -> dict[str, dict[str, int]]:
    """Pre-compute video file index offsets for merge_episodes and copy_videos."""
    offsets = {k: 0 for k in all_video_keys}
    result: dict[str, int] = {}
    for i, (path, label) in enumerate(datasets):
        eps = _read_episodes(path)
        result[label] = {}
        for vk in all_video_keys:
            fi_col = f"videos/{vk}/file_index"
            if fi_col not in eps.columns:
                result[label][vk] = offsets[vk]
                continue
            result[label][vk] = offsets[vk]
            src_indices = eps[fi_col].unique()
            if len(src_indices):
                offsets[vk] += int(max(src_indices)) + 1
    return result


def _merge_data(
    datasets: list[tuple[Path, str]],
    infos: list[dict],
    episode_offsets: list[int],
    task_offsets: list[int],
    total_frames: int,
) -> pd.DataFrame:
    """Concat data parquets with re-indexed episode_index, task_index, and index."""
    merged_parts = []
    frame_offset = 0
    for i, (path, label) in enumerate(datasets):
        df = _read_data(path)
        ep_offset = episode_offsets[i]
        task_offset = task_offsets[i]

        df["episode_index"] = df["episode_index"].astype(np.int64) + ep_offset
        if "task_index" in df.columns and task_offset:
            df["task_index"] = df["task_index"].astype(np.int64) + task_offset
        df["index"] = np.arange(frame_offset, frame_offset + len(df), dtype=np.int64)
        frame_offset += len(df)

        merged_parts.append(df)
        ep_min, ep_max = int(df["episode_index"].min()), int(df["episode_index"].max())
        print(f"  [{label}] {len(df)} frames, re-indexed episodes → "
              f"[{ep_min}, {ep_max}]")

    result = pd.concat(merged_parts, ignore_index=True)
    assert len(result) == total_frames
    return result


def _merge_episodes(
    datasets: list[tuple[Path, str]],
    episode_offsets: list[int],
    frame_offsets: list[int],
    total_episodes: int,
    all_video_keys: list[str],
    video_file_offsets: dict[str, dict[str, int]],
) -> pd.DataFrame:
    """Concat episodes parquets with re-indexed episode, frame, data, and video metadata."""
    merged_parts = []

    for i, (path, label) in enumerate(datasets):
        eps = _read_episodes(path)
        ep_offset = episode_offsets[i]
        frame_offset = frame_offsets[i]

        # Re-index episode_index
        eps["episode_index"] = eps["episode_index"].astype(np.int64) + ep_offset

        # All frame rows are rewritten into data/chunk-000/file-000.parquet.
        if "dataset_from_index" in eps.columns:
            eps["dataset_from_index"] = (
                eps["dataset_from_index"].astype(np.int64) + frame_offset
            )
        if "dataset_to_index" in eps.columns:
            eps["dataset_to_index"] = (
                eps["dataset_to_index"].astype(np.int64) + frame_offset
            )
        if "data/chunk_index" in eps.columns:
            eps["data/chunk_index"] = 0
        if "data/file_index" in eps.columns:
            eps["data/file_index"] = 0

        # Re-index video file_index and chunk_index for each camera
        for vk in all_video_keys:
            fi_col = f"videos/{vk}/file_index"
            ci_col = f"videos/{vk}/chunk_index"
            if fi_col in eps.columns:
                eps[fi_col] = eps[fi_col].astype(np.int64) + video_file_offsets[label][vk]
            if ci_col in eps.columns:
                eps[ci_col] = 0

        merged_parts.append(eps)
        print(f"  [{label}] {len(eps)} episodes")

    result = pd.concat(merged_parts, ignore_index=True)
    assert len(result) == total_episodes
    return result


def _copy_videos(
    datasets: list[tuple[Path, str]],
    output_path: Path,
    all_video_keys: list[str],
    video_file_offsets: dict[str, dict[str, int]],
) -> None:
    """Copy video files from source datasets to merged output with pre-computed offsets."""
    copied: dict[str, int] = {}

    for i, (path, label) in enumerate(datasets):
        eps = _read_episodes(path)
        for vk in all_video_keys:
            src_video_dir = path / "videos" / vk
            dst_video_dir = output_path / "videos" / vk / "chunk-000"
            dst_video_dir.mkdir(parents=True, exist_ok=True)

            fi_col = f"videos/{vk}/file_index"
            if fi_col not in eps.columns:
                continue

            offset = video_file_offsets[label][vk]
            src_indices = sorted(eps[fi_col].unique())
            for src_fi in src_indices:
                src_file = src_video_dir / "chunk-000" / f"file-{int(src_fi):03d}.mp4"
                dst_fi = int(src_fi) + offset
                dst_file = dst_video_dir / f"file-{dst_fi:03d}.mp4"
                if src_file.is_file() and not dst_file.exists():
                    shutil.copy2(src_file, dst_file)
                    copied.setdefault(vk, 0)
                    copied[vk] += 1

    print(f"  Copied video files: {dict(copied)}")


def _merge_stats(
    datasets: list[tuple[Path, str]],
    infos: list[dict],
    output_path: Path,
) -> None:
    """Merge stats.json from multiple datasets using weighted formulas."""
    import math

    stats_list: list[dict] = []
    for path, label in datasets:
        stats_path = path / "meta" / "stats.json"
        if stats_path.is_file():
            stats_list.append(json.loads(stats_path.read_text(encoding="utf-8")))
        else:
            print(f"  [WARN] {label}: no stats.json, skipping")
    if not stats_list:
        return

    # Check which features are present in all datasets
    all_keys = set(stats_list[0])
    for s in stats_list[1:]:
        all_keys &= set(s)

    # Get frame counts from info.json for weighting
    frame_counts = [info["total_frames"] for info in infos]
    total_count = sum(frame_counts)

    merged: dict[str, Any] = {}
    for key in sorted(all_keys):
        ref = stats_list[0][key]
        # Determine if this feature has array or scalar values
        sample = ref.get("mean")
        is_array = isinstance(sample, list)

        def _get(val: Any) -> Any:
            """Get value as np.array for arrays, or scalar for scalars."""
            if is_array:
                return np.array(val, dtype=np.float64)
            return float(val)

        # merge count
        counts = []
        for i, s in enumerate(stats_list):
            c = s[key].get("count")
            if is_array:
                counts.append(np.array(c if c is not None else [frame_counts[i]], dtype=np.float64))
            else:
                counts.append(float(c if c is not None else frame_counts[i]))
        if is_array:
            merged_count = sum(counts)  # type: ignore[arg-type]
        else:
            merged_count = sum(counts)  # type: ignore[assignment]

        # merge min / max
        mins = [_get(s[key]["min"]) for s in stats_list]
        maxs = [_get(s[key]["max"]) for s in stats_list]
        if is_array:
            merged_min = np.minimum.reduce(mins).tolist()  # type: ignore[arg-type]
            merged_max = np.maximum.reduce(maxs).tolist()  # type: ignore[arg-type]
        else:
            merged_min = min(mins)
            merged_max = max(maxs)

        # merge mean (weighted)
        means = [_get(s[key]["mean"]) for s in stats_list]
        if is_array:
            weights = [np.full_like(m, c_i) for m, c_i in zip(means, counts)]  # type: ignore[arg-type]
            total_w = sum(weights)  # type: ignore[arg-type]
            pooled_mean = sum(m * w for m, w in zip(means, weights)) / total_w
            merged_mean = pooled_mean.tolist()
        else:
            pooled_mean = sum(m * c_i for m, c_i in zip(means, counts)) / total_count  # type: ignore[operator]
            merged_mean = float(pooled_mean)

        # merge std (pooled)
        stds = [_get(s[key]["std"]) for s in stats_list]
        if is_array:
            var_parts = []
            for m_i, s_i, c_i in zip(means, stds, counts):  # type: ignore[arg-type]
                n_i = c_i  # type: ignore[assignment]
                var_parts.append((n_i - 1) * s_i ** 2 + n_i * (m_i - pooled_mean) ** 2)
            total_var = sum(var_parts) / (total_count - 1)
            merged_std = np.sqrt(total_var).tolist()
        else:
            var_sum = 0.0
            for m_i, s_i, c_i in zip(means, stds, counts):
                n_i = float(c_i)
                var_sum += (n_i - 1.0) * s_i ** 2 + n_i * (m_i - merged_mean) ** 2
            merged_std = math.sqrt(var_sum / (total_count - 1))

        # merge quantiles (weighted average — approximation)
        q_keys = ["q01", "q10", "q50", "q90", "q99"]
        merged_q: dict[str, Any] = {}
        for qk in q_keys:
            if qk not in ref:
                continue
            q_vals = [_get(s[key][qk]) for s in stats_list]
            if is_array:
                total_w = sum(counts)  # type: ignore[arg-type]
                pooled_q = sum(v * w for v, w in zip(q_vals, counts)) / total_w  # type: ignore[arg-type]
                merged_q[qk] = pooled_q.tolist()
            else:
                pooled_q = sum(float(v) * float(c_i) for v, c_i in zip(q_vals, counts)) / total_count  # type: ignore[operator]
                merged_q[qk] = pooled_q

        merged[key] = {
            "min": merged_min,
            "max": merged_max,
            "mean": merged_mean,
            "std": merged_std,
            "count": merged_count.tolist() if is_array else merged_count,
            **merged_q,
        }

    stats_out = output_path / "meta" / "stats.json"
    stats_out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"  Merged stats ({len(merged)} features) → {stats_out}")


def merge_datasets(
    dataset_paths: list[Path],
    output_path: Path,
    repo_id: str,
    dry_run: bool = False,
) -> None:
    """Merge multiple LeRobot v3 datasets into one."""
    if len(dataset_paths) < 2:
        raise ValueError("Need at least 2 datasets to merge")

    # Validate all paths exist
    labels = [p.name for p in dataset_paths]
    datasets = []
    for p, label in zip(dataset_paths, labels):
        if not p.is_dir():
            raise NotADirectoryError(f"Dataset not found: {p}")
        datasets.append((p, label))

    # Read metadata
    infos = [_read_info(p) for p in dataset_paths]
    _validate_compatible(datasets, infos)

    n_eps = [info["total_episodes"] for info in infos]
    total_episodes = sum(n_eps)
    total_frames = sum(info["total_frames"] for info in infos)
    episode_offsets = [0]
    for n in n_eps[:-1]:
        episode_offsets.append(episode_offsets[-1] + n)
    frame_counts = [info["total_frames"] for info in infos]
    frame_offsets = [0]
    for n in frame_counts[:-1]:
        frame_offsets.append(frame_offsets[-1] + n)

    ref_info = infos[0]
    all_video_keys = [k for k in ref_info["features"]
                      if ref_info["features"][k].get("dtype") == "video"]
    all_image_keys = [k for k in ref_info["features"]
                      if ref_info["features"][k].get("dtype") == "image"]

    print(f"\nMerging {len(datasets)} datasets:")
    for (p, label), n, offset in zip(datasets, n_eps, episode_offsets):
        print(f"  {label}: {n} episodes (→ offset {offset})")
    print(f"  Total: {total_episodes} episodes, {total_frames} frames")
    print(f"  Video keys: {all_video_keys}")
    print(f"  Image keys: {all_image_keys}")

    if dry_run:
        print("\n[Dry run — no files written]")
        return

    # Remove existing output
    if output_path.exists():
        shutil.rmtree(output_path)

    # Tasks need to be known before data merge so per-frame task_index can be re-indexed.
    task_tables = [_read_tasks(p) for p, _ in datasets]
    all_same_task = all(t.equals(task_tables[0]) for t in task_tables[1:])
    task_offsets = [0]
    if not all_same_task:
        next_task_offset = 0
        task_offsets = []
        for t in task_tables:
            task_offsets.append(next_task_offset)
            if "task_index" in t.columns and not t.empty:
                next_task_offset += int(t["task_index"].max()) + 1

    # Pre-compute video file offsets (shared by _merge_episodes and _copy_videos)
    video_file_offsets = _compute_video_file_offsets(datasets, all_video_keys)

    # Create directory structure
    (output_path / "data" / "chunk-000").mkdir(parents=True)
    (output_path / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    for vk in all_video_keys:
        (output_path / "videos" / vk / "chunk-000").mkdir(parents=True)

    # Merge and write data
    print("\n[1/4] Merging data parquets ...")
    merged_data = _merge_data(
        datasets, infos, episode_offsets, task_offsets, total_frames,
    )
    merged_data.to_parquet(
        output_path / "data" / "chunk-000" / "file-000.parquet",
        index=False,
    )

    # Merge and write episodes
    print("\n[2/4] Merging episode metadata ...")
    merged_episodes = _merge_episodes(
        datasets, episode_offsets, frame_offsets, total_episodes, all_video_keys,
        video_file_offsets,
    )
    merged_episodes.to_parquet(
        output_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        index=False,
    )

    # Copy videos
    print("\n[3/4] Copying video files ...")
    _copy_videos(datasets, output_path, all_video_keys, video_file_offsets)

    # Write metadata
    print("\n[4/4] Writing metadata ...")
    new_info = dict(ref_info)
    new_info["repo_id"] = repo_id
    new_info["total_episodes"] = total_episodes
    new_info["total_frames"] = total_frames
    new_info["splits"] = {"train": f"0:{total_episodes}"}
    (output_path / "meta" / "info.json").write_text(
        json.dumps(new_info, indent=2), encoding="utf-8",
    )

    if all_same_task:
        task_tables[0].to_parquet(output_path / "meta" / "tasks.parquet", index=False)
        new_info["total_tasks"] = int(len(task_tables[0]))
    else:
        # Multiple different tasks — merge with re-indexing
        task_dfs = []
        for t, task_offset in zip(task_tables, task_offsets):
            if "task_index" in t.columns:
                t = t.copy()
                t["task_index"] = t["task_index"].astype(np.int64) + task_offset
            task_dfs.append(t)
        merged_tasks = pd.concat(task_dfs, ignore_index=True)
        merged_tasks.to_parquet(
            output_path / "meta" / "tasks.parquet", index=False,
        )
        new_info["total_tasks"] = int(len(merged_tasks))

    (output_path / "meta" / "info.json").write_text(
        json.dumps(new_info, indent=2), encoding="utf-8",
    )

    # Merge stats.json with weighted formulas
    _merge_stats(datasets, infos, output_path)

    print(f"\nDone. Merged dataset: {output_path}")
    print(f"  Episodes: {total_episodes}, Frames: {total_frames}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple LeRobot v3 datasets for the same task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets", "-d", nargs="+", required=True,
        help="Paths to LeRobot v3 dataset directories (at least 2).",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output merged dataset path.",
    )
    parser.add_argument(
        "--repo-id", default="merged/unknown",
        help="HF repo ID for the merged dataset.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print merge plan without writing files.",
    )
    args = parser.parse_args()

    if len(args.datasets) < 2:
        print("[ERROR] Need at least 2 datasets to merge")
        sys.exit(1)

    dataset_paths = [Path(p).resolve() for p in args.datasets]
    output_path = Path(args.output).resolve()

    merge_datasets(
        dataset_paths=dataset_paths,
        output_path=output_path,
        repo_id=args.repo_id,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
