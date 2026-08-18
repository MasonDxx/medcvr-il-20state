#!/usr/bin/env python3
"""

Image Processing Pipeline
-------------------------
1. Subscribe to topics listed under topics.* that have an inference key
2. For each image topic, look up pre_processor (per-topic, else global_image_pre_processor)
3. If rectangular_crop_hw present: center-crop to [h, w]
4. If image_resize_hw present: resize to [h, w]
5. Convert to float32, normalize to [0, 1], transpose to (C, H, W) tensor

Policy Output
-------------
Actions are published as:
  • PoseStamped for position increments (Δx, Δy, Δz)
  • Joy or JointState for gripper/jaw commands

"""

import argparse
import json
from dataclasses import dataclass, field
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import rclpy
import time
import torch
import tyro
from huggingface_hub import hf_hub_download

from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState, Joy 
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.dit_flow.modeling_dit_flow import DiTFlowPolicy
from lerobot.policies.act.modeling_act import ACTPolicy

from medcvr_il.common.config_schema import (
    ImagePreprocessorConfig,
    InferenceConfig,
    PolicyInferenceConfig,
    RobotProcessorConfig,
    SynchSubscriberNodeConfig,
    TopicConfig,
    get_state_dim,
    mode_uses_jaw,
)
from medcvr_il.common.image_preprocessing import preprocess_image, resolve_camera_preprocessor_configs
from medcvr_il.common.rotation_math import gram_schmidt_6d_to_rotation_matrix
from medcvr_il.data_conversion.pose_processing import build_multi_source_observation_state
from medcvr_il.synchronizer import SynchSubscriberNode

@dataclass
class CombinedInferenceConfig:
    """Combined configuration for both inference and synchronizer."""
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    synchronizer: SynchSubscriberNodeConfig = field(default_factory=SynchSubscriberNodeConfig)
    topics: dict[str, TopicConfig] = field(default_factory=dict)
    global_image_pre_processor: ImagePreprocessorConfig | None = None

def load_combined_inference_config(yaml_path: str, remaining_args: list) -> CombinedInferenceConfig:
    """Load both inference and synchronizer configs from YAML and CLI args."""
    loaded_cfg = OmegaConf.load(yaml_path)
    inference_cfg = OmegaConf.masked_copy(
        loaded_cfg,
        ["inference", "synchronizer", "topics", "global_image_pre_processor"],
    )
    yaml_defaults = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(CombinedInferenceConfig), inference_cfg)
    )
    return tyro.cli(CombinedInferenceConfig, default=yaml_defaults, args=remaining_args)

class RobotOutputProcessor:
    """
    Process policy actions and publish to ROS2 topics.
    
    Publishes PoseStamped messages for position commands and Joy/JointState
    messages for gripper/jaw control based on configuration.
    """
    class JoyJawOutput:
        """Publish jaw commands as Joy messages (buttons[0])."""
        def __init__(self, node: rclpy.node, jaw_topic: str):
            self.jaw_msg = Joy()
            self.jaw_msg.buttons = [0]
            self.joy_publisher = node.create_publisher(Joy, jaw_topic, 10)
        def __call__(self, numpy_value):
            self.jaw_msg.buttons[0] = int(numpy_value)
            self.joy_publisher.publish(self.jaw_msg)

    class JointStateJawOutput:
        """Publish jaw commands as JointState messages (position[0])."""
        def __init__(self, node: rclpy.node, jaw_topic: str):
            self.jaw_msg = JointState()
            self.jaw_msg.name = ["jaw"]
            self.jaw_msg.position = [0.0]
            self.jaw_publisher = node.create_publisher(JointState, jaw_topic, 10)
        def __call__(self, numpy_value):
            self.jaw_msg.position[0] = numpy_value
            self.jaw_publisher.publish(self.jaw_msg)

    def __init__(self, config: RobotProcessorConfig, node: rclpy.node):
        """
        Initialize robot output processor.
        
        Args:
            node: ROS2 node for creating publishers
            config: Configuration specifying topics and message types
        """

        if config.pose_topic_type == "twist_stamped":
            self.pose_msg = TwistStamped()
            self.pose_publisher = node.create_publisher(TwistStamped, config.pose_topic, 10)
        elif config.pose_topic_type == "pose_stamped":
            self.pose_msg = PoseStamped()
            self.pose_publisher = node.create_publisher(PoseStamped, config.pose_topic, 10)
        else:
            raise ValueError(f"Unknown pose topic type: {config.pose_topic_type}")

        if config.jaw_topic_type == "joy":
            self.process_jaw_output = RobotOutputProcessor.JoyJawOutput(node, config.jaw_topic)
        elif config.jaw_topic_type == "joint_state":  
            self.process_jaw_output = RobotOutputProcessor.JointStateJawOutput(node, config.jaw_topic)
        elif config.jaw_topic_type == "none" or config.jaw_topic_type == "":
            self.process_jaw_output = None
        #TODO: add support for "joint" topic type that publishes to joint state
        # elif config.jaw_topic_type == "joint":
        #     raise NotImplementedError("joint topic type is not implemented yet")
        else:
            raise ValueError(f"Unknown jaw topic type: {config.jaw_topic_type}")
        self.config = config

        if self.config.action_type == "xyz":
            self.action_size = 3
        elif self.config.action_type == "xyz_quat":
            self.action_size = 7
        elif self.config.action_type == "xyz_rotmat":
            self.action_size = 12
        elif self.config.action_type == "xyz_6d":
            self.action_size = 9
        else:
            raise ValueError(f"Unsupported action type: {self.config.action_type}")
    
    def process_output(self, action: np.ndarray):
        """
        Publish action to ROS2 topics.
        
        Args:
            action: NumPy array where the last element is jaw command.
        """

        action = np.asarray(action, dtype=float).copy()
        if not np.all(np.isfinite(action)):
            raise ValueError(f"Policy produced a non-finite action: {action}")

        if self.config.base_to_cam_rotation is not None:
            world_to_base_rotation = np.array(
                self.config.base_to_cam_rotation, dtype=float
            ).reshape(3, 3)
            action[0:3] = world_to_base_rotation @ action[0:3]

        pos = action[0:3] * self.config.position_increment_scale
        max_pos_norm = self.config.max_position_increment_norm
        if max_pos_norm is not None:
            if max_pos_norm <= 0:
                raise ValueError("max_position_increment_norm must be > 0")
            pos_norm = np.linalg.norm(pos)
            if pos_norm > max_pos_norm:
                pos = pos * (max_pos_norm / pos_norm)
        action_type = self.config.action_type

        # Decode rotation from action into a scaled quaternion [qx, qy, qz, qw] (scipy convention),
        # or None for xyz-only actions.
        if action_type == "xyz_quat":
            quat = action[3:7]  # [qw, qx, qy, qz]
            quat_norm = np.linalg.norm(quat)
            if quat_norm < 1e-6:
                print(f"Warning: Invalid quaternion with near-zero magnitude: {quat_norm}")
                quat = np.array([1.0, 0.0, 0.0, 0.0])
            else:
                quat = quat / quat_norm
            # Reorder to scipy [qx, qy, qz, qw] and apply scale via rotvec
            rotation = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            scaled_rotation = R.from_rotvec(rotation.as_rotvec() * self.config.rotation_increment_scale)
            quat_out = scaled_rotation.as_quat()  # [qx, qy, qz, qw]

        elif action_type == "xyz_rotmat":
            R_mat = action[3:12].reshape(3, 3)
            identity_check = R_mat.T @ R_mat
            is_orthogonal = np.allclose(identity_check, np.eye(3), atol=1e-3)
            det = np.linalg.det(R_mat)
            is_proper_rotation = np.abs(det - 1.0) < 1e-3
            if not is_orthogonal or not is_proper_rotation:
                print(f"Warning: Invalid rotation matrix detected!")
                print(f"  Orthogonality error: {np.max(np.abs(identity_check - np.eye(3))):.6f}")
                print(f"  Determinant: {det:.6f} (should be 1.0)")
                print(f"  Recovering using Gram-Schmidt on first two columns...")
                a1 = R_mat[:, 0]
                a2 = R_mat[:, 1]
                R_mat = gram_schmidt_6d_to_rotation_matrix(a1, a2)
            rotation = R.from_matrix(R_mat)
            scaled_rotation = R.from_rotvec(rotation.as_rotvec() * self.config.rotation_increment_scale)
            quat_out = scaled_rotation.as_quat()  # [qx, qy, qz, qw]

        elif action_type == "xyz_6d":
            a1 = action[3:6]
            a2 = action[6:9]
            R_mat = gram_schmidt_6d_to_rotation_matrix(a1, a2)
            rotation = R.from_matrix(R_mat)
            scaled_rotation = R.from_rotvec(rotation.as_rotvec() * self.config.rotation_increment_scale)
            quat_out = scaled_rotation.as_quat()  # [qx, qy, qz, qw]

        elif action_type == "xyz":
            quat_out = None

        else:
            raise ValueError(f"Unsupported action type: {action_type}")

        # Populate message fields based on the configured topic type.
        pose_topic_type = self.config.pose_topic_type
        if pose_topic_type == "twist_stamped":
            self.pose_msg.twist.linear.x = float(pos[0])
            self.pose_msg.twist.linear.y = float(pos[1])
            self.pose_msg.twist.linear.z = float(pos[2])
            if quat_out is not None:
                rotvec = R.from_quat(quat_out).as_rotvec()
                self.pose_msg.twist.angular.x = float(rotvec[0])
                self.pose_msg.twist.angular.y = float(rotvec[1])
                self.pose_msg.twist.angular.z = float(rotvec[2])
        elif pose_topic_type == "pose_stamped":
            self.pose_msg.pose.position.x = float(pos[0])
            self.pose_msg.pose.position.y = float(pos[1])
            self.pose_msg.pose.position.z = float(pos[2])
            if quat_out is not None:
                self.pose_msg.pose.orientation.x = float(quat_out[0])
                self.pose_msg.pose.orientation.y = float(quat_out[1])
                self.pose_msg.pose.orientation.z = float(quat_out[2])
                self.pose_msg.pose.orientation.w = float(quat_out[3])

        self.pose_publisher.publish(self.pose_msg)
        if self.process_jaw_output is not None:
            jaw_output = float(action[-1]) * self.config.jaw_action_scale
            max_jaw_abs = self.config.max_jaw_increment_abs
            if max_jaw_abs is not None:
                if max_jaw_abs <= 0:
                    raise ValueError("max_jaw_increment_abs must be > 0")
                jaw_output = float(np.clip(jaw_output, -max_jaw_abs, max_jaw_abs))
            self.process_jaw_output(jaw_output)
        return pos

def parse_image_to_tensor(img, processor_config: ImagePreprocessorConfig):
    if processor_config is None:
        raise ValueError("Missing image processor config")

    img = preprocess_image(img, config=processor_config)

    if img.ndim != 3 or img.shape[0] not in (1, 3, 4):
        raise ValueError(
            "Expected a CHW image after preprocessing, "
            f"but received shape {img.shape}."
        )

    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).unsqueeze(0)
    return img_tensor

def parse_tf_to_tensor(tf_stamped_msg):
    tf_msg = tf_stamped_msg.transform
    translation = np.array([tf_msg.translation.x, tf_msg.translation.y, tf_msg.translation.z], dtype=np.float32)
    rotation = np.array([tf_msg.rotation.x, tf_msg.rotation.y, tf_msg.rotation.z, tf_msg.rotation.w], dtype=np.float32)
    return torch.from_numpy(np.concatenate([translation, rotation])).unsqueeze(0)

def parse_pose_to_tensor(pose_stamped_msg):
    pose_msg = pose_stamped_msg.pose
    translation = np.array([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z], dtype=np.float32)
    rotation = np.array([pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w], dtype=np.float32)
    return torch.from_numpy(np.concatenate([translation, rotation])).unsqueeze(0)

def parse_joint_state_to_tensor(joint_state_msg, expected_size=None):
    values = np.array(joint_state_msg.position, dtype=np.float32)
    if expected_size is not None:
        if values.size >= expected_size:
            values = values[:expected_size]
        else:
            padded = np.zeros(expected_size, dtype=np.float32)
            padded[:values.size] = values
            values = padded
    return torch.from_numpy(values).unsqueeze(0)

TOPIC_PARSERS = {
    "sensor_msgs/Image": parse_image_to_tensor,
    "geometry_msgs/TransformStamped": parse_tf_to_tensor,
    "geometry_msgs/PoseStamped": parse_pose_to_tensor,
    "sensor_msgs/JointState": parse_joint_state_to_tensor,
}


def enum_value(value) -> str:
    return value.value if hasattr(value, "value") else str(value)

@dataclass
class ActionOutput:
    name: str
    action_slice: list[int] | None = None
    robot_output_processor: RobotOutputProcessor = field(default_factory=RobotOutputProcessor)

@dataclass
class PolicyInference:
    cfg: PolicyInferenceConfig
    policy: object
    pre_processor: object | None
    post_processor: object | None
    name_to_action_outputs: dict[str, ActionOutput] = field(default_factory=dict)

def load_policy_and_processors(policy_inference_cfg: PolicyInferenceConfig):
    pre_processor = None
    post_processor = None
    policy_type = enum_value(policy_inference_cfg.policy_type)

    if policy_type == "diffusion":
        policy = DiffusionPolicy.from_pretrained(policy_inference_cfg.pretrained_policy_path)

        from lerobot.policies.factory import make_pre_post_processors

        preprocessor_overrides = {
            "device_processor": {"device": str(policy.config.device)},
        }

        pre_processor, post_processor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=policy_inference_cfg.pretrained_policy_path,
            preprocessor_overrides=preprocessor_overrides,
        )

    elif policy_type == "act":
        policy = ACTPolicy.from_pretrained(policy_inference_cfg.pretrained_policy_path)

    elif policy_type == "dit":
        policy = DiTFlowPolicy.from_pretrained(policy_inference_cfg.pretrained_policy_path)

    else:
        raise NotImplementedError(f"policy_type='{policy_inference_cfg.policy_type}' is not supported")

    policy.reset()
    return policy, pre_processor, post_processor


def state_topic_keys(policy_inference_cfg: PolicyInferenceConfig) -> list[str]:
    """Return ordered, de-duplicated ROS topic keys needed for policy state."""
    keys = []
    for source in policy_inference_cfg.observation_state_sources:
        if not source.pose_topic_key:
            raise ValueError("observation_state_sources.pose_topic_key cannot be empty")
        keys.append(source.pose_topic_key)
        if source.jaw_topic_key is not None:
            keys.append(source.jaw_topic_key)
    return list(dict.fromkeys(keys))


def resolve_inference_topics(
    topics: dict[str, TopicConfig],
    policy_configs: dict[str, PolicyInferenceConfig],
) -> dict[str, TopicConfig]:
    topic_keys = {
        key for key, topic in topics.items() if topic.inference is not None
    }

    for policy_name, policy_cfg in policy_configs.items():
        for topic_key in state_topic_keys(policy_cfg):
            if topic_key not in topics:
                raise KeyError(
                    f"Policy '{policy_name}' references unknown state topic '{topic_key}'"
                )
            topic_keys.add(topic_key)

    inference_topics_config = {
        key: topic for key, topic in topics.items() if key in topic_keys
    }

    if not inference_topics_config:
        raise ValueError("No topics with inference config found.")

    return inference_topics_config


def resolve_topic_observation_names(
    inference_topics_config: dict[str, TopicConfig],
) -> dict[str, str]:
    topic_to_observation = {}

    for topic_key, topic_info in inference_topics_config.items():
        if topic_info.inference is None:
            continue
        observation_name = topic_info.inference.observation_name

        if observation_name is None:
            raise KeyError(f"Missing inference.observation_name for topic '{topic_key}'")

        topic_to_observation[topic_key] = observation_name

    return topic_to_observation


def resolve_policy_topic_routing(
    policy_configs: dict[str, PolicyInferenceConfig],
    inference_topics_config: dict[str, TopicConfig],
) -> dict[str, list[str]]:
    policy_names = list(policy_configs)
    routing = {policy_name: [] for policy_name in policy_names}

    for topic_key, topic_info in inference_topics_config.items():
        if topic_info.inference is None:
            continue
        target_policies = topic_info.inference.policy_targets

        if target_policies is None:
            target_policies = policy_names
        elif isinstance(target_policies, str):
            target_policies = [target_policies]

        for policy_name in target_policies:
            if policy_name not in routing:
                raise KeyError(
                    f"Topic '{topic_key}' targets unknown policy '{policy_name}'. "
                    f"Known policies: {policy_names}"
                )

            routing[policy_name].append(topic_key)

    for policy_name, policy_cfg in policy_configs.items():
        for topic_key in state_topic_keys(policy_cfg):
            if topic_key not in routing[policy_name]:
                routing[policy_name].append(topic_key)

    return routing


def build_runtime_observation_state(
    data_dict: dict,
    inference_topics_config: dict[str, TopicConfig],
    policy_inference_cfg: PolicyInferenceConfig,
) -> torch.Tensor:
    """Encode and concatenate all configured robot states exactly as conversion does."""
    sources = policy_inference_cfg.observation_state_sources
    if not sources:
        raise ValueError(
            "zero_observation_state is false, but observation_state_sources is empty"
        )

    xyz_list = []
    quat_list = []
    jaw_data_list = []

    for source in sources:
        pose_key = source.pose_topic_key
        pose_type = inference_topics_config[pose_key].type
        if pose_type == "geometry_msgs/PoseStamped":
            pose_tensor = parse_pose_to_tensor(data_dict[pose_key])
        elif pose_type == "geometry_msgs/TransformStamped":
            pose_tensor = parse_tf_to_tensor(data_dict[pose_key])
        else:
            raise TypeError(
                f"State pose topic '{pose_key}' must be PoseStamped or "
                f"TransformStamped, got '{pose_type}'"
            )

        pose_values = pose_tensor.squeeze(0).cpu().numpy()
        xyz_list.append(pose_values[:3][None, :])
        quat_list.append(pose_values[3:7][None, :])

        if source.jaw_topic_key is None:
            jaw_data_list.append(None)
            continue

        jaw_key = source.jaw_topic_key
        jaw_type = inference_topics_config[jaw_key].type
        if jaw_type != "sensor_msgs/JointState":
            raise TypeError(
                f"State jaw topic '{jaw_key}' must be sensor_msgs/JointState, "
                f"got '{jaw_type}'"
            )
        jaw_positions = data_dict[jaw_key].position
        jaw_index = source.jaw_position_index
        resolved_index = jaw_index if jaw_index >= 0 else len(jaw_positions) + jaw_index
        if resolved_index < 0 or resolved_index >= len(jaw_positions):
            raise IndexError(
                f"jaw_position_index={jaw_index} is invalid for topic '{jaw_key}' "
                f"with {len(jaw_positions)} positions"
            )
        jaw_data_list.append(
            np.array([[jaw_positions[resolved_index]]], dtype=np.float32)
        )

    state_values = build_multi_source_observation_state(
        xyz_list=xyz_list,
        quat_list=quat_list,
        jaw_data_list=jaw_data_list,
        T=1,
        state_mode=policy_inference_cfg.observation_state_mode,
        source_indices=list(range(len(sources))),
    )
    expected_size = policy_inference_cfg.observation_state_size
    if state_values.shape != (1, expected_size):
        raise ValueError(
            f"Constructed observation.state shape {state_values.shape}; "
            f"expected (1, {expected_size})"
        )
    return torch.from_numpy(state_values).to(policy_inference_cfg.device)

def build_observation(
    data_dict: dict,
    topic_keys: list[str],
    inference_topics_config: dict[str, TopicConfig],
    topic_to_observation: dict[str, str],
    image_preprocessor_configs: dict[str, ImagePreprocessorConfig],
    policy_inference_cfg: PolicyInferenceConfig,
) -> dict[str, torch.Tensor]:
    observation = {}
    state_keys = set(state_topic_keys(policy_inference_cfg))

    for topic_key in topic_keys:
        if topic_key in state_keys:
            continue
        data = data_dict[topic_key]
        topic_type = inference_topics_config[topic_key].type
        parse_fn = TOPIC_PARSERS[topic_type]

        if topic_type == "sensor_msgs/Image":
            parsed_tensor = parse_fn(
                data,
                processor_config=image_preprocessor_configs[topic_key],
            )
        elif topic_type == "sensor_msgs/JointState":
            parsed_tensor = parse_fn(
                data,
                expected_size=policy_inference_cfg.observation_state_size,
            )
        else:
            parsed_tensor = parse_fn(data)

        if topic_key not in topic_to_observation:
            raise KeyError(f"Topic '{topic_key}' has no observation mapping")
        obs_name = topic_to_observation[topic_key]
        observation[obs_name] = parsed_tensor.to(policy_inference_cfg.device)

    if policy_inference_cfg.zero_observation_state:
        observation["observation.state"] = torch.zeros(
            policy_inference_cfg.observation_state_size
        ).unsqueeze(0).to(policy_inference_cfg.device)
    else:
        observation["observation.state"] = build_runtime_observation_state(
            data_dict=data_dict,
            inference_topics_config=inference_topics_config,
            policy_inference_cfg=policy_inference_cfg,
        )

    return observation


def validate_policy_observation_shapes(policy, observation: dict[str, torch.Tensor]):
    """Fail before actuation when runtime observations do not match training shapes."""
    for observation_name, feature in policy.config.input_features.items():
        if observation_name not in observation:
            raise KeyError(f"Missing policy observation '{observation_name}'.")

        tensor = observation[observation_name]
        expected_shape = tuple(feature.shape)
        actual_shape = tuple(tensor.shape[1:])
        if actual_shape != expected_shape:
            raise ValueError(
                f"Observation '{observation_name}' has shape {actual_shape}; "
                f"the policy expects {expected_shape}."
            )


def validate_policy_state_config(
    policy_name: str,
    policy_inference_cfg: PolicyInferenceConfig,
):
    """Validate state configuration without loading a local or Hub policy."""
    state_mode = enum_value(policy_inference_cfg.observation_state_mode)
    state_sources = policy_inference_cfg.observation_state_sources
    if policy_inference_cfg.zero_observation_state:
        if state_sources:
            raise ValueError(
                f"Policy '{policy_name}' enables zero_observation_state but also "
                "defines observation_state_sources"
            )
    else:
        if not state_sources:
            raise ValueError(
                f"Policy '{policy_name}' requires observation_state_sources when "
                "zero_observation_state is false"
            )
        if mode_uses_jaw(state_mode):
            missing_jaw_sources = [
                index
                for index, source in enumerate(state_sources)
                if source.jaw_topic_key is None
            ]
            if missing_jaw_sources:
                raise ValueError(
                    f"Policy '{policy_name}' state mode '{state_mode}' requires a "
                    f"jaw topic for every source; missing at indices {missing_jaw_sources}"
                )
        expected_size = get_state_dim(state_mode, source_count=len(state_sources))
        if policy_inference_cfg.observation_state_size != expected_size:
            raise ValueError(
                f"Policy '{policy_name}' observation_state_size is "
                f"{policy_inference_cfg.observation_state_size}, but mode "
                f"'{state_mode}' with {len(state_sources)} sources requires {expected_size}"
            )


def validate_policy_outputs(policy_name: str, policy_inference_cfg: PolicyInferenceConfig):
    validate_policy_state_config(policy_name, policy_inference_cfg)

    if not policy_inference_cfg.action_outputs:
        raise ValueError(f"Policy '{policy_name}' must define at least one output.")

    action_dim = load_action_dim_from_pretrained(policy_inference_cfg.pretrained_policy_path)

    for output_cfg in policy_inference_cfg.action_outputs:
        if output_cfg.action_slice is None:
            continue

        if len(output_cfg.action_slice) != 2:
            raise ValueError(
                f"Policy '{policy_name}' output '{output_cfg.name}' has invalid "
                f"action_slice={output_cfg.action_slice}. Expected [start, end]."
            )

        start_idx, end_idx = output_cfg.action_slice

        if start_idx < 0 or end_idx <= start_idx or start_idx >= action_dim or end_idx > action_dim:
            raise ValueError(
                f"Policy '{policy_name}' output '{output_cfg.name}' has invalid "
                f"action_slice={output_cfg.action_slice}."
            )


def load_action_dim_from_pretrained(pretrained_policy_path: str) -> int:
    local_path = Path(pretrained_policy_path)
    config_path = local_path / "config.json" if local_path.exists() else hf_hub_download(
        repo_id=pretrained_policy_path,
        filename="config.json",
    )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return int(config["output_features"]["action"]["shape"][0])


def main(args=None):
    """
    Main inference loop: subscribe to ROS2 topics, run policy, publish actions.
    
    Processing pipeline:
      1. Load config and initialize policy from pretrained_policy_path
      2. Start SynchSubscriber in background thread for topic synchronization
      3. In control loop:
         - Get synchronized observations (images + state)
         - Apply cropping and resizing to images
         - Convert to torch tensors
         - Run policy.select_action(observation)
         - Publish action to robot via RobotOutputProcessor
      4. Loop at specified frequency_hz
    
    Args:
        args: Command-line arguments (passed to rclpy.init)
    """
    rclpy.init(args=args)
    np.set_printoptions(precision=4, suppress=True)

    # add_help=False so that --help is forwarded to tyro.cli, which knows
    # about all nested config fields and prints their attribute docstrings.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, default=None, help="Path to the YAML configuration file")
    args, remaining_args = parser.parse_known_args()

    if args.config is None:
        # No YAML supplied — let tyro show --help (or error on missing fields).
        tyro.cli(CombinedInferenceConfig, args=remaining_args)
        return

    # Load both configs together using a single tyro.cli call
    combined_cfg = load_combined_inference_config(args.config, remaining_args=remaining_args)

    inference_cfg = combined_cfg.inference
    policy_name_to_policy_inference_config = inference_cfg.policy_inferences
    if not policy_name_to_policy_inference_config:
        raise ValueError("No policies found. Expected inference.policy_inferences.<name> entries.")

    synchronizer_cfg = combined_cfg.synchronizer
    topics_cfg = combined_cfg.topics
    global_image_pre_processor_config = combined_cfg.global_image_pre_processor

    print(f"Loaded policies: {list(policy_name_to_policy_inference_config.keys())}")
    print("Loaded SynchronizerConfig:", synchronizer_cfg)

    for policy_name, policy_inference_cfg in policy_name_to_policy_inference_config.items():
        validate_policy_outputs(policy_name, policy_inference_cfg)

    inference_topics_config = resolve_inference_topics(
        topics_cfg,
        policy_name_to_policy_inference_config,
    )
    print("Only using filtered inference topics for synchronization:", inference_topics_config.keys())

    image_topic_keys = [
        topic_key
        for topic_key, topic_info in inference_topics_config.items()
        if topic_info.type == "sensor_msgs/Image"
    ]

    image_preprocessor_configs = resolve_camera_preprocessor_configs(
        cam_names=image_topic_keys,
        topics=inference_topics_config,
        global_pre=global_image_pre_processor_config,
    )

    synchronizer_node = SynchSubscriberNode(inference_topics_config, synchronizer_cfg)

    topic_to_observation = resolve_topic_observation_names(inference_topics_config)

    policy_topic_routing = resolve_policy_topic_routing(
        policy_configs=policy_name_to_policy_inference_config,
        inference_topics_config=inference_topics_config,
    )

    for policy_name, topic_keys in policy_topic_routing.items():
        print(f"Topics for policy '{policy_name}': {topic_keys}")

    policy_name_to_policy_inference: dict[str, PolicyInference] = {}

    for policy_name, policy_inference_cfg in policy_name_to_policy_inference_config.items():
        policy, pre_processor, post_processor = load_policy_and_processors(policy_inference_cfg)

        name_to_action_outputs = {}
        for output_cfg in policy_inference_cfg.action_outputs:
            name_to_action_outputs[output_cfg.name] = ActionOutput(
                name=output_cfg.name,
                action_slice=output_cfg.action_slice,
                robot_output_processor=RobotOutputProcessor(
                       output_cfg.robot_processor,
                        synchronizer_node))
                # {
                #     "name": output_cfg.name,
                #     "action_slice": output_cfg.action_slice,
                #     "processor": RobotOutputProcessor(
                #         output_cfg.robot_processor,
                #         synchronizer_node,
                #     ),
                # }

        policy_name_to_policy_inference[policy_name] = PolicyInference(
            cfg=policy_inference_cfg,
            policy=policy,
            pre_processor=pre_processor,
            post_processor=post_processor,
            name_to_action_outputs=name_to_action_outputs)

        print(
            f"Loaded policy '{policy_name}': "
            f"{policy_inference_cfg.policy_type} @ {policy_inference_cfg.pretrained_policy_path}"
        )

        for output in name_to_action_outputs.values():
            processor = output.robot_output_processor
            print(
                f"  Output '{output.name}': "
                f"slice={output.action_slice} "
                f"pose_topic={processor.config.pose_topic} "
                f"jaw_topic={processor.config.jaw_topic}"
            )

    if inference_cfg.frequency_hz <= 0:
        raise ValueError("frequency_hz must be > 0.")

    rate = synchronizer_node.create_rate(inference_cfg.frequency_hz)
    loop_count = 0
    last_status_time = time.time()
    logged_policy_states = set()

    while rclpy.ok():
        data_dict = synchronizer_node.get_latest_data_dict()
        if data_dict is None:
            print("No data received yet. Waiting for synchronized data...")
            rate.sleep()
            continue

        latest_status = []

        for policy_name, policy_inference in policy_name_to_policy_inference.items():
            policy_cfg = policy_inference.cfg

            observation = build_observation(
                data_dict=data_dict,
                topic_keys=policy_topic_routing[policy_name],
                inference_topics_config=inference_topics_config,
                topic_to_observation=topic_to_observation,
                image_preprocessor_configs=image_preprocessor_configs,
                policy_inference_cfg=policy_cfg,
            )

            if policy_name not in logged_policy_states:
                state_values = observation["observation.state"].detach().cpu().numpy()
                print(
                    f"Policy '{policy_name}' observation.state "
                    f"shape={state_values.shape}: {state_values[0]}"
                )
                logged_policy_states.add(policy_name)

            input_observation = observation
            if policy_inference.pre_processor is not None:
                input_observation = policy_inference.pre_processor(observation)

            validate_policy_observation_shapes(
                policy_inference.policy,
                input_observation,
            )

            with torch.inference_mode():
                action = policy_inference.policy.select_action(input_observation)

            if policy_inference.post_processor is not None:
                action = policy_inference.post_processor(action)

            numpy_action = action.squeeze(0).detach().to("cpu").numpy().astype(float)

            for output_name, action_output in policy_inference.name_to_action_outputs.items():
                action_slice = action_output.action_slice
                robot_output_processor = action_output.robot_output_processor

                if action_slice is None:
                    routed_action = numpy_action
                else:
                    start_idx, end_idx = action_slice
                    routed_action = numpy_action[start_idx:end_idx]

                published_xyz = robot_output_processor.process_output(routed_action)

                latest_status.append(
                    f"{policy_name}.{output_name}:"
                    f"published_xyz=({published_xyz[0]:+.4f},"
                    f"{published_xyz[1]:+.4f},{published_xyz[2]:+.4f})"
                )

        loop_count += 1
        now = time.time()

        if now - last_status_time >= 2.0:
            print(f"[loop {loop_count}] " + " | ".join(latest_status))

            for policy_name, policy_inference in policy_name_to_policy_inference.items():
                for output_name, action_output in policy_inference.name_to_action_outputs.items():
                    processor = action_output.robot_output_processor
                    pose_subscribers = processor.pose_publisher.get_subscription_count()

                    if pose_subscribers == 0:
                        print(
                            f"[WARN] No subscribers for output "
                            f"'{policy_name}.{output_name}' "
                            f"on {processor.config.pose_topic}."
                        )

            last_status_time = now

        rate.sleep()
    
    synchronizer_node.join()

if __name__ == '__main__':
    main()
