import numpy as np
import pytest

from medcvr_il.common.config_schema import ActionSourceConfig, ImagePreprocessorConfig, get_state_dim
from medcvr_il.data_conversion.dataset_schema import build_dataset_schema
from medcvr_il.data_conversion.pose_processing import (
    build_multi_source_observation_state,
    build_observation_state,
)


def test_xyz_6d_jaw_identity_quaternion_layout():
    xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    quat = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    jaw = np.array([[0.25]], dtype=np.float32)

    state = build_observation_state(
        xyz=xyz,
        quat=quat,
        T=1,
        state_mode="xyz_6d_jaw",
        all_joints_data=None,
        jaw_data=jaw,
    )

    np.testing.assert_allclose(
        state,
        [[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.25]],
    )


def test_bimanual_state_is_psm1_then_psm2_with_jaws_at_9_and_19():
    xyz_list = [
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
    ]
    quat_list = [
        np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    ]
    jaw_list = [
        np.array([[0.1]], dtype=np.float32),
        np.array([[0.2]], dtype=np.float32),
    ]

    state = build_multi_source_observation_state(
        xyz_list=xyz_list,
        quat_list=quat_list,
        jaw_data_list=jaw_list,
        T=1,
        state_mode="xyz_6d_jaw",
        source_indices=[0, 1],
    )

    assert state.shape == (1, 20)
    np.testing.assert_allclose(state[0, 0:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(state[0, 10:13], [4.0, 5.0, 6.0])
    assert state[0, 9] == pytest.approx(0.1)
    assert state[0, 19] == pytest.approx(0.2)
    assert get_state_dim("xyz_6d_jaw", source_count=2) == 20


def test_jaw_state_mode_rejects_missing_jaw_data():
    with pytest.raises(ValueError, match="requires jaw data"):
        build_observation_state(
            xyz=np.zeros((1, 3), dtype=np.float32),
            quat=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            T=1,
            state_mode="xyz_6d_jaw",
            all_joints_data=None,
            jaw_data=None,
        )


def test_legacy_zero_state_remains_three_dimensional():
    state = build_multi_source_observation_state(
        xyz_list=[np.zeros((2, 3), dtype=np.float32)],
        quat_list=[np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)).astype(np.float32)],
        jaw_data_list=[None],
        T=2,
        state_mode="zero",
        source_indices=[0],
    )

    assert state.shape == (2, 3)
    assert np.count_nonzero(state) == 0


def test_dataset_schema_declares_twenty_state_and_twenty_action_values():
    schema = build_dataset_schema(
        cam_configs={"cam0": ImagePreprocessorConfig(image_resize_hw=[240, 360])},
        state_mode="xyz_6d_jaw",
        action_sources=[
            ActionSourceConfig(jaw="joint_state", action="xyz_6d", jaw_file="jaw1.txt"),
            ActionSourceConfig(jaw="joint_state", action="xyz_6d", jaw_file="jaw2.txt"),
        ],
        state_source_count=2,
    )

    assert schema["observation.state"]["shape"] == (20,)
    assert schema["action"]["shape"] == (20,)
