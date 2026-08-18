#!/usr/bin/env python3
"""
Build a LeRobot dataset from recorded multi-camera robot episodes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import inspect
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import tyro
from omegaconf import OmegaConf
from medcvr_il.common.config_schema import DataConversionConfig, ImagePreprocessorConfig, TopicConfig, ActionSourceConfig
from medcvr_il.common.image_preprocessing import resolve_camera_preprocessor_configs

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from medcvr_il.data_conversion.dataset_schema import build_dataset_schema
from medcvr_il.data_conversion.episode_loading import (
    IMAGE_LOADERS_DICT,
    episodes,
    get_cam_dirs,
    load_eef_csv,
    load_eef_unity_csv,
    load_episode_data,
)
from medcvr_il.data_conversion.pose_processing import (
    build_action,
    build_multi_source_observation_state,
    prepare_pose_data,
)


@dataclass
class DataConverter:
    topics: dict[str, TopicConfig] = field(default_factory=dict)
    global_image_pre_processor: ImagePreprocessorConfig = field(default_factory=ImagePreprocessorConfig)
    data_conversion: DataConversionConfig = field(default_factory=DataConversionConfig)


def flush(ds):
    for method_name in ("consolidate", "finalize", "close"):
        if hasattr(ds, method_name):
            getattr(ds, method_name)()
            return
    print("[warn] no flush method on LeRobotDataset!")


def write_episode_frames(
    ds,
    all_cam_frames: list[list[np.ndarray]],
    obs_state: np.ndarray,
    action: np.ndarray,
    num_steps: int,
    task: str,
):
    """Write one episode to the LeRobot dataset."""
    for t in range(num_steps - 1):
        frame_data = {}
        for i, frames in enumerate(all_cam_frames):
            frame_data[f"observation.image.cam{i}"] = frames[t]

        frame_data["observation.state"] = obs_state[t]
        frame_data["action"] = action[t]
        frame_data["task"] = task
        ds.add_frame(frame_data)

    ds.save_episode()


def create_dataset(cfg: DataConverter, features: dict):
    dc = cfg.data_conversion

    kwargs = {
        "repo_id": dc.repo_id,
        "robot_type": dc.robot_env.lower(),
        "fps": dc.fps,
        "features": features,
    }

    create_parameters = inspect.signature(LeRobotDataset.create).parameters
    if "root" in create_parameters:
        kwargs["root"] = dc.out_dir
    elif "cache_dir" in create_parameters:
        kwargs["cache_dir"] = dc.out_dir
    else:
        os.environ["LEROBOT_HOME"] = str(dc.out_dir.parent.resolve())

    return LeRobotDataset.create(**kwargs)


def convert(cfg: DataConverter):
    """Convert multi-camera episodes to LeRobot dataset format."""
    dc = cfg.data_conversion

    if dc.raw_dir is None:
        raise ValueError("data_conversion.raw_dir must be provided")
    if dc.out_dir is None:
        raise ValueError("data_conversion.out_dir must be provided")

    camera_filter = [int(c.strip()) for c in dc.cameras.split(",")] if dc.cameras else None
    cam_names_override = [c.strip() for c in dc.cam_names.split(",")] if dc.cam_names else None
    action_sources = dc.action_sources if dc.action_sources is not None else [dc.action_source]
    state_mode = dc.state.value if hasattr(dc.state, "value") else str(dc.state)
    state_source_indices = dc.state_source_indices or [0]
    if state_mode in {"zero", "tool_joint"} and len(state_source_indices) != 1:
        raise ValueError(
            f"data_conversion.state='{state_mode}' supports exactly one state source"
        )
    for index in state_source_indices:
        if index < 0 or index >= len(action_sources):
            raise IndexError(
                f"state_source_indices contains {index}, but there are "
                f"only {len(action_sources)} action sources"
            )
    state_source_count = 1 if state_mode in {"zero", "tool_joint"} else len(state_source_indices)

    if dc.out_dir.exists():
        shutil.rmtree(dc.out_dir)

    first_episode = episodes(dc.raw_dir)[0]
    cam_dirs = get_cam_dirs(first_episode, camera_filter, cam_names_override)
    cam_names = [cam_dir.name for cam_dir in cam_dirs]
    cam_configs = resolve_camera_preprocessor_configs(
        cam_names=cam_names,
        topics=cfg.topics,
        global_pre=cfg.global_image_pre_processor,
    )

    features = build_dataset_schema(
        cam_configs,
        dc.state,
        action_sources,
        state_source_count=state_source_count,
    )
    dataset = create_dataset(cfg, features)

    image_loader = IMAGE_LOADERS_DICT[dc.image_type]
    eef_loader = load_eef_unity_csv if dc.robot_env.lower() == "unity" else load_eef_csv

    for episode_path in episodes(dc.raw_dir):
        episode_data = load_episode_data(
            ep=episode_path,
            image_loader=image_loader,
            eef_csv_loader=eef_loader,
            action_sources=action_sources,
            cam_configs=cam_configs,
            camera_filter=camera_filter,
            cam_names=cam_names_override,
        )

        per_source_actions = []
        xyz_by_source = []
        quat_by_source = []

        for i, source in enumerate(action_sources):
            xyz, quat = prepare_pose_data(
                eef=episode_data.eef_position_list[i],
                transform_matrix=source.transform_matrix,
            )

            base_to_cam_rot = np.array(source.base_to_cam_rot) if source.base_to_cam_rot is not None else None

            xyz_by_source.append(xyz)
            quat_by_source.append(quat)

            action_i = build_action(
                xyz=xyz,
                quat=quat,
                action_mode=source.action,
                jaw_data=episode_data.jaw_data_list[i],
                absolute_xyz=source.absolute_xyz,
                absolute_quat=source.absolute_quat,
                absolute_jaw=source.absolute_jaw,
                base_to_cam_rot=base_to_cam_rot,
            )
            per_source_actions.append(action_i)

        action = np.concatenate(per_source_actions, axis=1)

        obs_state = build_multi_source_observation_state(
            xyz_list=xyz_by_source,
            quat_list=quat_by_source,
            jaw_data_list=episode_data.jaw_data_list,
            T=episode_data.length,
            state_mode=dc.state,
            source_indices=state_source_indices,
            all_joints_data=episode_data.all_joints_data,
        )

        write_episode_frames(
            ds=dataset,
            all_cam_frames=episode_data.cam_frames,
            obs_state=obs_state,
            action=action,
            num_steps=episode_data.length,
            task=dc.task,
        )
        print(
            f"[LeRobot] {episode_path.name}: {episode_data.length - 1} frames "
            f"(action-aligned), {len(episode_data.cam_frames)} cameras"
        )

    flush(dataset)
    print(f"\n✓ Dataset ready → {dc.out_dir}")
    return dataset


def push_dataset(ds, cfg: DataConverter):
    if cfg.data_conversion.push:
        ds.push_to_hub(push_videos=True)


def load_conversion_config(argv: list[str] | None = None) -> DataConverter:
    argv = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config file with defaults (CLI args override YAML).")
    known_args, remaining_args = parser.parse_known_args(argv)

    if known_args.config is None:
        yaml_cfg = DataConverter()
    else:
        loaded_cfg = OmegaConf.load(known_args.config)
        converter_cfg = OmegaConf.masked_copy(
            loaded_cfg,
            ["topics", "global_image_pre_processor", "data_conversion"],
        )
        yaml_cfg = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(DataConverter), converter_cfg)
        )

    yaml_cfg.data_conversion = tyro.cli(DataConversionConfig, args=remaining_args, default=yaml_cfg.data_conversion)

    return yaml_cfg


def main(argv: list[str] | None = None):
    cfg = load_conversion_config(argv)
    dataset = convert(cfg)
    push_dataset(dataset, cfg)


if __name__ == "__main__":
    main()
