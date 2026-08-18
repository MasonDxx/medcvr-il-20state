from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from omegaconf import MISSING


@dataclass
class ImagePreprocessorConfig:
    """Image preprocessing controls applied before model input or dataset write."""
    rectangular_crop_hw: list[int] | None = None
    image_resize_hw: list[int] = field(default_factory=lambda: [240, 320])
    align: list[str] | None = None


@dataclass
class TopicRecorderConfig:
    file: str | None = None
    folder: str | None = None


@dataclass
class TopicInferenceConfig:
    observation_name: str = MISSING
    policy_targets: list[str] | None = None
    """
    Optional policy names to receive this topic.
    None means all policies.
    """


@dataclass
class TopicConfig:
    """Per-topic ROS metadata and optional recorder/inference/image settings."""

    name: str = MISSING
    type: str = MISSING
    """ROS message type string, e.g. "sensor_msgs/Image"."""
    recorder: TopicRecorderConfig | None = None
    inference: TopicInferenceConfig | None = None
    pre_processor: ImagePreprocessorConfig | None = None


@dataclass
class SynchSubscriberNodeConfig:
    slop_sec: float = 0.1


@dataclass
class DataWriterConfig:
    saving_folder: str = "."
    timestamp_file: str = "timestamps.txt"


@dataclass
class RecorderConfig:
    frequency: float = 30.0
    episode: int = 0
    data_writer: DataWriterConfig = field(default_factory=DataWriterConfig)


class TopicType(str, Enum):
    joy = "joy"
    joint = "joint"
    joint_state = "joint_state"
    twist_stamped = "twist_stamped"
    pose_stamped = "pose_stamped"
    none = "none"

class StateActionMode(str, Enum):
    """Action/state encoding modes used by conversion and inference configs.

    Note: `scripts/lerobot_eval_ros2.py` currently publishes orientations only for
    `xyz_quat`, `xyz_rotmat`, and `xyz_6d`. 
    """

    xyz = "xyz"
    xyz_quat = "xyz_quat"
    xyz_6d = "xyz_6d"
    zero = "zero"
    tool_joint = "tool_joint"
    xyz_jaw = "xyz_jaw"
    xyz_quat_jaw = "xyz_quat_jaw"
    xyz_6d_jaw = "xyz_6d_jaw"


def mode_uses_quat(mode: str) -> bool:
    return "quat" in mode

def mode_uses_rotmat(mode: str) -> bool:
    return "rotmat" in mode

def mode_uses_6d(mode: str) -> bool:
    return "6d" in mode

def mode_uses_jaw(mode: str) -> bool:
    return "jaw" in mode

def mode_uses_orientation(mode: str) -> bool:
    return mode_uses_quat(mode) or mode_uses_rotmat(mode) or mode_uses_6d(mode)

def get_state_dim(state_mode: str, source_count: int = 1) -> int:
    if source_count <= 0:
        raise ValueError("source_count must be > 0")

    if state_mode == "tool_joint":
        return 4
    if state_mode == "zero":
        return 3

    state_dim = 3
    if mode_uses_quat(state_mode):
        state_dim += 4
    elif mode_uses_rotmat(state_mode):
        state_dim += 9
    elif mode_uses_6d(state_mode):
        state_dim += 6

    if mode_uses_jaw(state_mode):
        state_dim += 1

    return state_dim * source_count

def get_action_dim(action_mode: str, jaw: str) -> int:
    action_dim = 3

    if mode_uses_quat(action_mode):
        action_dim += 4
    elif mode_uses_rotmat(action_mode):
        action_dim += 9
    elif mode_uses_6d(action_mode):
        action_dim += 6

    if jaw != "none":
        action_dim += 1

    return action_dim

@dataclass
class RobotProcessorConfig:
    """ROS output mapping and scaling for policy actions."""

    pose_topic: str = ""
    pose_topic_type: TopicType = TopicType.none
    """Supported pose topic types: twist_stamped, pose_stamped"""
    
    jaw_topic: str = ""
    jaw_topic_type: TopicType = TopicType.none
    """Supported jaw topic types: joy, joint, joint_state"""

    position_increment_scale: float = 1.0
    max_position_increment_norm: float | None = None
    """Optional Euclidean-norm limit applied to one published xyz increment."""
    rotation_increment_scale: float = 1.0
    jaw_action_scale: float = 1.0
    """Multiplier applied only to the policy jaw action before publishing."""
    max_jaw_increment_abs: float | None = None
    """Optional absolute limit applied after scaling the jaw action."""
    action_type: StateActionMode = StateActionMode.xyz_quat
    base_to_cam_rotation: list[float] | None = None
    """Legacy name: row-major 3x3 matrix applied directly to xyz before publishing. None to skip."""
    """
    Valid types and resulting sizes: xyz_quat->7, xyz_rotmat->12, xyz_6d->9.
    Jaw variants add +1 and place jaw on the last action dimension."""

@dataclass
class ActionOutputConfig:
    """One output route for a policy action."""

    name: str = "default"
    action_slice: list[int] | None = None
    """
    Optional [start, end) slice into the action.
    None means the full action.
    """

    robot_processor: RobotProcessorConfig = field(default_factory=RobotProcessorConfig)


@dataclass
class ObservationStateSourceConfig:
    """ROS topic pair used to construct one robot's observation.state block."""

    pose_topic_key: str = ""
    jaw_topic_key: str | None = None
    jaw_position_index: int = -1
    """Index into JointState.position. -1 selects the final position value."""


@dataclass
class PolicyInferenceConfig:
    """Runtime settings for one LeRobot policy."""

    device: str = "cuda"
    observation_state_size: int = -1
    zero_observation_state: bool = True
    observation_state_mode: StateActionMode = StateActionMode.zero
    observation_state_sources: list[ObservationStateSourceConfig] = field(default_factory=list)
    """Ordered robot state sources concatenated into observation.state."""

    policy_type: str = ""
    """
    Supported Policies:
    - diffusion
    - act
    - dit_flow
    """

    pretrained_policy_path: str = ""

    action_outputs: list[ActionOutputConfig] = field(
        default_factory=lambda: [ActionOutputConfig()]
    )
    """
    One or more output routes.
    Use multiple outputs for action slicing.
    """


@dataclass
class InferenceConfig:
    """Unified eval/inference config."""

    frequency_hz: int = 10
    policy_inferences: dict[str, PolicyInferenceConfig] = field(default_factory=dict)

@dataclass
class ActionSourceConfig:
    """Per-action source and transform configuration."""

    eef_file: str = "eef_position.txt"
    jaw_file: str | None = None
    jaw: TopicType = TopicType.none
    action: StateActionMode = StateActionMode.xyz
    transform_matrix: list[list[float]] | None = None
    absolute_xyz: bool = False
    absolute_quat: bool = False
    absolute_jaw: bool = False
    base_to_cam_rot: list[list[float]] | None = None
    """Optional 3x3 rotation matrix to transform xyz action from base frame to camera frame (inverse applied)."""

@dataclass
class DataConversionConfig:
    """Converter options for building LeRobot datasets from recorded episodes."""

    class ImageType(str, Enum):
        png = "png"
        jpg = "jpg"
        mp4 = "mp4"
    class RobotEnv(str, Enum):
        Unity = "Unity"
        franka = "franka"

    raw_dir: Path | None = None
    out_dir: Path | None = None
    image_type: ImageType = ImageType.png
    state: StateActionMode = StateActionMode.zero
    state_source_indices: list[int] | None = None
    """Ordered action-source indices used to build a multi-robot observation.state."""
    robot_env: RobotEnv = RobotEnv.Unity
    repo_id: str = "local/franka_pushblock"
    push: bool = False
    fps: int = 20
    cameras: str | None = None
    """Comma-separated camera indices to include (e.g. "0,2")."""
    cam_names: str | None = None
    """Comma-separated camera folder names (e.g. "cam0,cam2")."""
    task: str = "pushblock"
    action_source: ActionSourceConfig = field(default_factory=ActionSourceConfig)
    """Single-action configuration. Overridden when action_sources is provided."""
    action_sources: list[ActionSourceConfig] | None = None
    """Optional list of multiple actions to concatenate (overrides single action configs)."""
