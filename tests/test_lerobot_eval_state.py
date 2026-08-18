from types import SimpleNamespace

import numpy as np

from medcvr_il.common.config_schema import (
    ObservationStateSourceConfig,
    PolicyInferenceConfig,
    TopicConfig,
)
from scripts.lerobot_eval_ros2 import (
    build_runtime_observation_state,
    resolve_inference_topics,
    resolve_policy_topic_routing,
    validate_policy_state_config,
)


def _pose(x, y, z):
    return SimpleNamespace(
        pose=SimpleNamespace(
            position=SimpleNamespace(x=x, y=y, z=z),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
    )


def _joint_state(value):
    return SimpleNamespace(position=[value])


def _policy_config():
    return PolicyInferenceConfig(
        device="cpu",
        observation_state_size=20,
        zero_observation_state=False,
        observation_state_mode="xyz_6d_jaw",
        observation_state_sources=[
            ObservationStateSourceConfig("psm1_eef", "psm1_jaw", -1),
            ObservationStateSourceConfig("psm2_eef", "psm2_jaw", -1),
        ],
    )


def test_runtime_state_topics_are_routed_and_encoded_to_twenty_values():
    topics = {
        "psm1_eef": TopicConfig(
            name="/PSM1/command_end_tip_tf",
            type="geometry_msgs/PoseStamped",
        ),
        "psm1_jaw": TopicConfig(
            name="/PSM1/jaw/measured_js",
            type="sensor_msgs/JointState",
        ),
        "psm2_eef": TopicConfig(
            name="/PSM2/command_end_tip_tf",
            type="geometry_msgs/PoseStamped",
        ),
        "psm2_jaw": TopicConfig(
            name="/PSM2/jaw/measured_js",
            type="sensor_msgs/JointState",
        ),
    }
    policy_cfg = _policy_config()
    policy_configs = {"main": policy_cfg}

    inference_topics = resolve_inference_topics(topics, policy_configs)
    routing = resolve_policy_topic_routing(policy_configs, inference_topics)

    assert routing["main"] == ["psm1_eef", "psm1_jaw", "psm2_eef", "psm2_jaw"]
    validate_policy_state_config("main", policy_cfg)

    state = build_runtime_observation_state(
        data_dict={
            "psm1_eef": _pose(1.0, 2.0, 3.0),
            "psm1_jaw": _joint_state(0.1),
            "psm2_eef": _pose(4.0, 5.0, 6.0),
            "psm2_jaw": _joint_state(0.2),
        },
        inference_topics_config=inference_topics,
        policy_inference_cfg=policy_cfg,
    ).numpy()

    assert state.shape == (1, 20)
    np.testing.assert_allclose(state[0, 0:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(state[0, 10:13], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(state[0, [9, 19]], [0.1, 0.2])
