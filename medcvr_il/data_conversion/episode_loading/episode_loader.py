from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from medcvr_il.common.config_schema import ActionSourceConfig, ImagePreprocessorConfig
from medcvr_il.data_conversion.episode_loading.csv_loader import (
    load_all_joints_from_joint_state,
    load_jaw_csv,
    load_jaw_from_joint_state,
    load_timestamp,
)
from medcvr_il.data_conversion.episode_loading.frame_loader import get_cam_dirs, natural_key


episodes = lambda root: sorted((p for p in root.iterdir() if p.is_dir()), key=lambda x: natural_key(x))


@dataclass
class EpisodeData:
    cam_frames: list[list[np.ndarray]]
    eef_position_list: list[np.ndarray]
    timestamps: np.ndarray
    jaw_data_list: list[np.ndarray | None]
    all_joints_data: np.ndarray | None
    length: int


def load_episode_data(
    ep: Path,
    image_loader: Callable[[Path, ImagePreprocessorConfig | None], list[np.ndarray]],
    eef_csv_loader: Callable[[Path], np.ndarray],
    action_sources: list[ActionSourceConfig],
    cam_configs: dict[str, ImagePreprocessorConfig],
    camera_filter: list[int] | None = None,
    cam_names: list[str] | None = None,
) -> EpisodeData:
    """Load and align all per-episode data needed by convert()."""
    cam_dirs = get_cam_dirs(ep, camera_filter, cam_names)

    all_cam_frames: list[list[np.ndarray]] = []
    for cam_dir in cam_dirs:
        frames = image_loader(cam_dir, image_config=cam_configs.get(cam_dir.name))
        all_cam_frames.append(frames)

    frame_counts = [len(frames) for frames in all_cam_frames]
    timestamps = load_timestamp(ep / "timestamp.txt")

    eef_position_list: list[np.ndarray] = []
    jaw_data_list: list[np.ndarray | None] = []
    all_joints_data = None

    for source in action_sources:
        eef_position_list.append(eef_csv_loader(ep / source.eef_file))

        jaw_mode = source.jaw.value if hasattr(source.jaw, "value") else str(source.jaw)

        if jaw_mode == "none" or source.jaw_file is None or source.jaw_file == "none":
            jaw_data_list.append(None)

        elif jaw_mode == "joint_state":
            source_path = ep / source.jaw_file
            jaw_data_list.append(load_jaw_from_joint_state(source_path))

            if all_joints_data is None:
                all_joints_data = load_all_joints_from_joint_state(source_path)

        elif jaw_mode in {"joy", "joint"}:
            source_path = ep / source.jaw_file
            jaw_data_list.append(load_jaw_csv(source_path))

        else:
            raise ValueError(f"Unsupported jaw mode: {jaw_mode}")

    lengths = frame_counts + [timestamps.shape[0]]
    lengths += [eef_position.shape[0] for eef_position in eef_position_list]
    lengths += [jaw_data.shape[0] for jaw_data in jaw_data_list if jaw_data is not None]

    if all_joints_data is not None:
        lengths.append(all_joints_data.shape[0])

    min_len = min(lengths)

    if any(l != min_len for l in lengths):
        print(f"⚠️ {ep.name}: Data length mismatch {lengths}, truncating all to {min_len}")
        all_cam_frames = [frames[:min_len] for frames in all_cam_frames]
        timestamps = timestamps[:min_len]
        eef_position_list = [eef_position[:min_len] for eef_position in eef_position_list]
        jaw_data_list = [
            jaw_data[:min_len] if jaw_data is not None else None
            for jaw_data in jaw_data_list
        ]

        if all_joints_data is not None:
            all_joints_data = all_joints_data[:min_len]

    return EpisodeData(
        cam_frames=all_cam_frames,
        eef_position_list=eef_position_list,
        timestamps=timestamps,
        jaw_data_list=jaw_data_list,
        all_joints_data=all_joints_data,
        length=min_len,
    )
