from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from medcvr_il.common.config_schema import (
    StateActionMode,
    mode_uses_6d,
    mode_uses_jaw,
    mode_uses_orientation,
    mode_uses_rotmat,
)
from medcvr_il.common.rotation_math import quat_multiply, quat_to_rotation_matrix


def apply_transform(eef: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Transform EEF poses into a new coordinate frame.

    For each timestep, builds a 4×4 homogeneous matrix from (xyz, quat),
    left-multiplies by *transform*, then extracts the new (xyz, quat).

    Args:
        eef:       (T, 7) array  [x, y, z, qx, qy, qz, qw]
        transform: (4, 4) homogeneous transformation matrix

    Returns:
        (T, 7) transformed [x, y, z, qx, qy, qz, qw]  float32
    """
    T_len = eef.shape[0]
    result = np.empty((T_len, 7), dtype=np.float32)

    for t in range(T_len):
        T_curr = np.eye(4)
        T_curr[:3, :3] = R.from_quat(eef[t, 3:7]).as_matrix()
        T_curr[:3, 3]  = eef[t, :3]

        T_new = transform @ T_curr

        result[t, :3]  = T_new[:3, 3]
        result[t, 3:7] = R.from_matrix(T_new[:3, :3]).as_quat()

    return result


# -------- action / state builders ----------------------------------

def build_action(
    xyz: np.ndarray,
    quat: np.ndarray,
    action_mode: str,
    jaw_data: np.ndarray | None,
    absolute_xyz: bool = False,
    absolute_quat: bool = False,
    absolute_jaw: bool = False,
    base_to_cam_rot: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build action array from position/orientation data and optional jaw data.

    Returns:
        action: (T, D) float32
    """
    action = (
        np.concatenate([xyz[1:], xyz[-1:]], axis=0).astype(np.float32)
        if absolute_xyz
        else np.diff(xyz, axis=0, append=xyz[-1:]).astype(np.float32)
    )

    # Change xyz frame to another camera frame
    if base_to_cam_rot is not None:
        if not absolute_xyz:
            rot = base_to_cam_rot.T  # inverse of rotation matrix
            action = (rot @ action.T).T.astype(np.float32)
        else:
            raise NotImplementedError("Absolute XYZ in camera frame is unsupported. set base_to_cam_rot to None or set absolute_xyz to False.")

    #TODO test if this works correctly
    if mode_uses_orientation(action_mode):
        orient_source = (
            np.concatenate([quat[1:], quat[-1:]], axis=0).astype(np.float32)
            if absolute_quat 
            else _compute_relative_quat(quat)
        ) 
        
        if mode_uses_rotmat(action_mode) or mode_uses_6d(action_mode):
            orient = np.array( [quat_to_rotation_matrix(q, return_6d=mode_uses_6d(action_mode)) for q in orient_source],
            dtype=np.float32,
            )
        else:
            orient = orient_source.astype(np.float32)
        action = np.concatenate((action, orient), axis=1)

    if jaw_data is not None:
        jaw_values = (
            np.concatenate([jaw_data[1:], jaw_data[-1:]], axis=0)
            if absolute_jaw 
            else np.diff(jaw_data, axis=0, append=jaw_data[-1:])
        ).astype(np.float32)
        
        action = np.concatenate((action, jaw_values), axis=1)

    return action

def _compute_relative_quat(quats: np.ndarray) -> np.ndarray:
    quat_delta = np.zeros((quats.shape[0], 4), dtype=np.float32)

    for t in range(quats.shape[0] - 1):
        q_curr = quats[t]
        q_next = quats[t + 1]

        q_curr_inv = np.array(
            [-q_curr[0], -q_curr[1], -q_curr[2], q_curr[3]],
            dtype=np.float32,
        )
        quat_delta[t] = quat_multiply(q_curr_inv, q_next) # body frame rotation 

    quat_delta[-1] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    quat_delta_norm = np.linalg.norm(quat_delta, axis=1, keepdims=True)
    return quat_delta / quat_delta_norm

def build_observation_state(
    xyz: np.ndarray,
    quat: np.ndarray,
    T: int,
    state_mode: StateActionMode,
    all_joints_data: np.ndarray | None,
    jaw_data: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build observation.state from pose data or joint-state data.

    Returns:
        (T, D) float32 observation state array.
    """
    if xyz.shape[0] != T or quat.shape[0] != T:
        raise ValueError(
            f"Pose length mismatch while building observation state: "
            f"xyz={xyz.shape[0]}, quat={quat.shape[0]}, expected={T}"
        )

    if state_mode == "tool_joint":
        if all_joints_data is not None and all_joints_data.shape[0] == T:
            obs_state = all_joints_data
        else:
            print(f"  ⚠ Warning: joint_state.txt not found or length mismatch, using zeros")
            obs_state = np.zeros((T, 4), np.float32)

    elif state_mode in {"xyz", "xyz_jaw"}:
        obs_state = xyz.astype(np.float32)

    elif state_mode in {"xyz_quat", "xyz_quat_jaw"}:
        obs_state = np.concatenate((xyz, quat), axis=1).astype(np.float32)

    elif state_mode in {"xyz_6d", "xyz_6d_jaw"}:
        obs_state = np.zeros((T, 9), dtype=np.float32)
        obs_state[:, :3] = xyz.astype(np.float32)
        for t in range(T):
            obs_state[t, 3:9] = quat_to_rotation_matrix(quat[t], return_6d=True)

    elif state_mode == "zero":
        obs_state = np.zeros((T, 3), np.float32)
    else: 
        raise ValueError(f"Unsupported observation state mode: {state_mode}")

    if mode_uses_jaw(state_mode):
        if jaw_data is None:
            raise ValueError(
                f"Observation state mode '{state_mode}' requires jaw data"
            )
        jaw_values = np.asarray(jaw_data, dtype=np.float32)
        if jaw_values.ndim == 1:
            jaw_values = jaw_values[:, None]
        if jaw_values.shape != (T, 1):
            raise ValueError(
                f"Jaw state must have shape ({T}, 1), got {jaw_values.shape}"
            )
        obs_state = np.concatenate((obs_state, jaw_values), axis=1)

    return obs_state


def build_multi_source_observation_state(
    xyz_list: list[np.ndarray],
    quat_list: list[np.ndarray],
    jaw_data_list: list[np.ndarray | None],
    T: int,
    state_mode: StateActionMode,
    source_indices: list[int],
    all_joints_data: np.ndarray | None = None,
) -> np.ndarray:
    """Build one observation.state by concatenating ordered robot sources."""
    if state_mode in {"zero", "tool_joint"}:
        if len(source_indices) != 1:
            raise ValueError(
                f"State mode '{state_mode}' supports exactly one source index"
            )
        index = source_indices[0]
        return build_observation_state(
            xyz=xyz_list[index],
            quat=quat_list[index],
            T=T,
            state_mode=state_mode,
            all_joints_data=all_joints_data,
            jaw_data=jaw_data_list[index],
        )

    per_source_states = []
    source_count = len(xyz_list)
    for index in source_indices:
        if index < 0 or index >= source_count:
            raise IndexError(
                f"State source index {index} is out of range for {source_count} sources"
            )
        per_source_states.append(
            build_observation_state(
                xyz=xyz_list[index],
                quat=quat_list[index],
                T=T,
                state_mode=state_mode,
                all_joints_data=None,
                jaw_data=jaw_data_list[index],
            )
        )

    if not per_source_states:
        raise ValueError("At least one state source index is required")
    return np.concatenate(per_source_states, axis=1).astype(np.float32)

def prepare_pose_data(
    eef: np.ndarray,
    transform_matrix: np.ndarray | None,
):
    """
    Extract xyz/quaternion pose data from EEF poses, normalize quaternions,
    and optionally apply a coordinate-frame transform.

    Returns:
        xyz:  (T, 3)
        quat: (T, 4)
    """
    xyz = eef[:, :3]
    quat = eef[:, 3:]

    quat_norm = np.linalg.norm(quat, axis=1, keepdims=True)
    quat = quat / quat_norm

    if transform_matrix is not None:
        if quat.shape[1] == 0:
            quat = np.tile([0.0, 0.0, 0.0, 1.0], (eef.shape[0], 1)).astype(np.float32)

        eef_full = np.concatenate((xyz, quat), axis=1)
        eef_full = apply_transform(eef_full, transform_matrix)
        xyz = eef_full[:, :3]
        quat = eef_full[:, 3:]

    return xyz, quat
