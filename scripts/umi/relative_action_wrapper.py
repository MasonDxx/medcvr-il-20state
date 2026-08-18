"""
UMI-Style Relative Action Wrapper for LeRobot Datasets

Transforms absolute pose actions into relative actions where ALL actions 
in the horizon are relative to the LAST observation EEF pose.

This matches the Universal Manipulation Interface (UMI) approach:
    T_relative[i] = T_ref^(-1) @ T_absolute[i]

Key insight: The action data already contains EEF poses across the horizon,
and we use the action at the LAST observation timestep as the reference.

Usage:
    from lerobot.datasets import LeRobotDataset
    from relative_action_wrapper import UMIDatasetWrapper, UMIRelativeActionTransform
    
    dataset = LeRobotDataset(repo_id=...)
    transform = UMIRelativeActionTransform(
        action_key="action",
        n_obs_steps=2,
        pose_dim=9,  # xyz (3) + 6D rotation (6)
        has_gripper=True,  # +1 for jaw
    )
    wrapped_dataset = UMIDatasetWrapper(dataset, transform)
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from typing import Optional


def gram_schmidt_6d_to_rotation_matrix(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation representation to 3x3 rotation matrix using Gram-Schmidt.
    
    Args:
        a1: First column vector, shape (3,)
        a2: Second column vector, shape (3,)
    
    Returns:
        R: 3x3 rotation matrix, or identity matrix if inputs are invalid
    
    The 6D representation encodes the first two columns of a rotation matrix.
    This function orthonormalizes them and computes the third column via cross product.
    """
    # Validate vectors are not zero or near-zero
    norm_a1 = np.linalg.norm(a1)
    norm_a2 = np.linalg.norm(a2)
    
    if norm_a1 < 1e-6 or norm_a2 < 1e-6:
        print(f"Warning: Invalid 6D representation with near-zero vectors!")
        print(f"  ||a1||: {norm_a1:.6f}, ||a2||: {norm_a2:.6f}")
        return np.eye(3)
    
    # Gram-Schmidt orthogonalization
    b1 = a1 / norm_a1
    b2 = a2 - np.dot(a2, b1) * b1
    norm_b2 = np.linalg.norm(b2)
    
    if norm_b2 < 1e-6:
        print(f"Warning: Vectors a1 and a2 are nearly collinear!")
        print(f"  ||b2|| after orthogonalization: {norm_b2:.6f}")
        return np.eye(3)
    
    b2 = b2 / norm_b2
    b3 = np.cross(b1, b2)
    R_mat = np.stack((b1, b2, b3), axis=1)
    
    # Validate the resulting rotation matrix
    det = np.linalg.det(R_mat)
    if np.abs(det - 1.0) > 1e-3:
        print(f"Warning: 6D→rotation matrix has bad determinant: {det:.6f}")
        return np.eye(3)
    
    return R_mat


def rotation_matrix_to_6d(R_mat: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to 6D representation.
    
    Args:
        R_mat: 3x3 rotation matrix
    
    Returns:
        6D vector: [r00, r10, r20, r01, r11, r21] (first two columns, column-major)
    """
    return np.array([
        R_mat[0, 0], R_mat[1, 0], R_mat[2, 0],  # col0
        R_mat[0, 1], R_mat[1, 1], R_mat[2, 1],  # col1
    ], dtype=np.float32)


def pose_6d_to_matrix(pose: np.ndarray) -> np.ndarray:
    """
    Convert pose with 6D rotation to 4x4 transformation matrix.
    
    Args:
        pose: shape (9,) as [x, y, z, r00, r10, r20, r01, r11, r21]
              where the 6 rotation values are first two columns of rotation matrix
    
    Returns:
        T: 4x4 homogeneous transformation matrix
    """
    T = np.eye(4)
    T[:3, 3] = pose[:3]  # xyz
    
    # Extract 6D rotation (column-major: col0, col1)
    col0 = pose[3:6]   # [r00, r10, r20]
    col1 = pose[6:9]   # [r01, r11, r21]
    
    # Use Gram-Schmidt to get valid rotation matrix
    T[:3, :3] = gram_schmidt_6d_to_rotation_matrix(col0, col1)
    
    return T


def pose_quat_to_matrix(pose: np.ndarray) -> np.ndarray:
    """
    Convert pose with quaternion to 4x4 transformation matrix.
    
    Args:
        pose: shape (7,) as [x, y, z, qx, qy, qz, qw]
    
    Returns:
        T: 4x4 homogeneous transformation matrix
    """
    T = np.eye(4)
    T[:3, 3] = pose[:3]  # xyz
    quat = pose[3:7]  # qx, qy, qz, qw
    T[:3, :3] = R.from_quat(quat).as_matrix()
    return T


def matrix_to_pose_6d(T: np.ndarray) -> np.ndarray:
    """
    Convert 4x4 transformation matrix to pose with 6D rotation.
    
    Args:
        T: 4x4 homogeneous transformation matrix
    
    Returns:
        pose: shape (9,) as [x, y, z, r00, r10, r20, r01, r11, r21]
    """
    xyz = T[:3, 3]
    R_mat = T[:3, :3]
    rot_6d = rotation_matrix_to_6d(R_mat)
    return np.concatenate([xyz, rot_6d]).astype(np.float32)


def matrix_to_pose_quat(T: np.ndarray) -> np.ndarray:
    """
    Convert 4x4 transformation matrix to pose with quaternion.
    
    Args:
        T: 4x4 homogeneous transformation matrix
    
    Returns:
        pose: shape (7,) as [x, y, z, qx, qy, qz, qw]
    """
    xyz = T[:3, 3]
    R_mat = T[:3, :3]
    quat = R.from_matrix(R_mat).as_quat()  # qx, qy, qz, qw
    return np.concatenate([xyz, quat]).astype(np.float32)


def compute_relative_actions(
    action_poses: np.ndarray, 
    ref_pose: np.ndarray,
    pose_format: str = "6d"
) -> np.ndarray:
    """
    Compute UMI-style relative actions.
    
    All action poses are transformed to be relative to the reference pose.
    
    Args:
        action_poses: shape (horizon, pose_dim) - absolute action poses
                      pose_dim is 9 for 6D format, 7 for quaternion format
        ref_pose: shape (pose_dim,) - reference pose (last observation EEF pose)
        pose_format: "6d" for xyz+6D rotation (dim=9), "quat" for xyz+quaternion (dim=7)
    
    Returns:
        relative_actions: shape (horizon, pose_dim) - poses relative to ref_pose
    """
    # Select conversion functions based on format
    if pose_format == "6d":
        pose_to_mat = pose_6d_to_matrix
        mat_to_pose = matrix_to_pose_6d
    else:  # quaternion
        pose_to_mat = pose_quat_to_matrix
        mat_to_pose = matrix_to_pose_quat
    
    # Get reference transformation and its inverse
    T_ref = pose_to_mat(ref_pose)
    T_ref_inv = np.linalg.inv(T_ref)
    
    # Compute relative for each action in horizon
    relative_actions = []
    for action_pose in action_poses:
        T_action = pose_to_mat(action_pose)
        T_relative = T_ref_inv @ T_action  # UMI formula
        relative_actions.append(mat_to_pose(T_relative))
    
    return np.stack(relative_actions)


class UMIRelativeActionTransform:
    """
    Transform to apply UMI-style relative action representation.
    
    At each training sample, transforms all actions in the horizon to be 
    relative to the EEF pose at the LAST observation timestep.
    
    For inference, the robot's current EEF pose becomes the reference,
    and predicted relative actions are converted back to absolute poses.
    
    Args:
        action_key: Key in data dict for action array (default: "action")
        n_obs_steps: Number of observation steps (default: 2)
        pose_dim: Dimension of the EEF pose (9 for xyz+6D, 7 for xyz+quat)
        pose_format: "6d" or "quat" - must match pose_dim
        has_gripper: If True, action has additional gripper/jaw dimension at the end
    
    Data Flow:
        - observation.state: 4 joint angles (NOT modified by this transform)
        - action: EEF pose + optional gripper
            - Before: absolute EEF poses, shape (horizon, pose_dim + gripper)
            - After:  relative EEF poses to last obs, same shape
    """
    
    def __init__(
        self, 
        action_key: str = "action",
        n_obs_steps: int = 2,
        pose_dim: int = 9,
        pose_format: str = "6d",
        has_gripper: bool = True
    ):
        self.action_key = action_key
        self.n_obs_steps = n_obs_steps
        self.pose_dim = pose_dim
        self.pose_format = pose_format
        self.has_gripper = has_gripper
        
        # Validate
        if pose_format == "6d" and pose_dim != 9:
            raise ValueError(f"6D format requires pose_dim=9, got {pose_dim}")
        if pose_format == "quat" and pose_dim != 7:
            raise ValueError(f"Quaternion format requires pose_dim=7, got {pose_dim}")
        
        # The reference index in the action array
        # action_delta_indices = [-1, 0, 1, ..., 14] for n_obs_steps=2, horizon=16
        # Index 0 corresponds to delta=-1 (t-1), index 1 corresponds to delta=0 (t=0, last obs)
        self.ref_index = n_obs_steps - 1  # Index 1 for n_obs_steps=2
    
    def __call__(self, item: dict) -> dict:
        """
        Transform a single data item.
        
        Args:
            item: Dictionary from LeRobot dataset __getitem__
                  Contains action with shape (horizon, action_dim)
                  where action_dim = pose_dim (+ 1 if has_gripper)
        
        Returns:
            item with action transformed to relative representation
        """
        # Get action - shape: (horizon, action_dim)
        action = item[self.action_key]
        if isinstance(action, torch.Tensor):
            action = action.numpy()
        
        # Split action into EEF pose and gripper
        action_poses = action[:, :self.pose_dim]  # (horizon, pose_dim)
        gripper = action[:, self.pose_dim:] if self.has_gripper else None
        
        # Get reference pose from the action at the last observation timestep
        # This is the EEF pose at t=0 (current pose)
        ref_pose = action_poses[self.ref_index]  # (pose_dim,)
        
        # Compute relative actions
        relative_poses = compute_relative_actions(
            action_poses, 
            ref_pose, 
            self.pose_format
        )
        
        # Recombine with gripper if present
        if gripper is not None:
            relative_action = np.concatenate([relative_poses, gripper], axis=1)
        else:
            relative_action = relative_poses
        
        # Update item
        item[self.action_key] = torch.from_numpy(relative_action.astype(np.float32))
        
        return item


class UMIDatasetWrapper(torch.utils.data.Dataset):
    """
    Wrapper that applies UMI-style relative action transformation to a dataset.
    
    This is the simplest integration approach (Option 4A) - just wrap your 
    existing LeRobotDataset.
    
    Example:
        from lerobot.datasets import LeRobotDataset
        
        # Load dataset with absolute poses
        base_dataset = LeRobotDataset(
            repo_id="cagedBirdy/push_absolute_poses",
            ...
        )
        
        # Create transform
        transform = UMIRelativeActionTransform(
            action_key="action",
            n_obs_steps=2,
            pose_dim=9,      # xyz + 6D rotation
            pose_format="6d",
            has_gripper=True  # jaw angle
        )
        
        # Wrap dataset
        dataset = UMIDatasetWrapper(base_dataset, transform)
        
        # Use in training
        dataloader = DataLoader(dataset, batch_size=32, ...)
    """
    
    def __init__(self, dataset, transform: UMIRelativeActionTransform):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self.transform(item)
    
    # Forward common dataset attributes
    @property
    def fps(self):
        return self.dataset.fps
    
    @property
    def features(self):
        return self.dataset.features
    
    @property
    def meta(self):
        return self.dataset.meta


# ============== Inference Utilities ==============

def relative_to_absolute_pose(
    relative_poses: np.ndarray,
    ref_pose: np.ndarray,
    pose_format: str = "6d"
) -> np.ndarray:
    """
    Convert relative poses back to absolute poses (for inference).
    
    This is the inverse operation: T_absolute[i] = T_ref @ T_relative[i]
    
    Args:
        relative_poses: shape (horizon, pose_dim) - relative poses from model output
        ref_pose: shape (pose_dim,) - current robot EEF pose
        pose_format: "6d" or "quat"
    
    Returns:
        absolute_poses: shape (horizon, pose_dim) - absolute target poses
    """
    if pose_format == "6d":
        pose_to_mat = pose_6d_to_matrix
        mat_to_pose = matrix_to_pose_6d
    else:
        pose_to_mat = pose_quat_to_matrix
        mat_to_pose = matrix_to_pose_quat
    
    T_ref = pose_to_mat(ref_pose)
    
    absolute_poses = []
    for rel_pose in relative_poses:
        T_relative = pose_to_mat(rel_pose)
        T_absolute = T_ref @ T_relative  # Inverse of training transform
        absolute_poses.append(mat_to_pose(T_absolute))
    
    return np.stack(absolute_poses)


# ============== Unit Tests ==============

def test_roundtrip():
    """Test that relative -> absolute recovers original pose."""
    # Create random test poses
    np.random.seed(42)
    
    # Reference pose
    ref_xyz = np.random.randn(3) * 0.1
    ref_rot = R.random().as_matrix()
    ref_6d = rotation_matrix_to_6d(ref_rot)
    ref_pose = np.concatenate([ref_xyz, ref_6d])
    
    # Target pose
    target_xyz = ref_xyz + np.random.randn(3) * 0.05
    target_rot = R.random().as_matrix()
    target_6d = rotation_matrix_to_6d(target_rot)
    target_pose = np.concatenate([target_xyz, target_6d])
    
    # Convert to relative
    relative = compute_relative_actions(
        target_pose[None, :], ref_pose, pose_format="6d"
    )[0]
    
    # Convert back to absolute
    recovered = relative_to_absolute_pose(
        relative[None, :], ref_pose, pose_format="6d"
    )[0]
    
    # Check
    assert np.allclose(recovered[:3], target_pose[:3], atol=1e-5), \
        f"Position mismatch: {recovered[:3]} vs {target_pose[:3]}"
    
    # For rotation, compare the matrices
    T_target = pose_6d_to_matrix(target_pose)
    T_recovered = pose_6d_to_matrix(recovered)
    assert np.allclose(T_target[:3, :3], T_recovered[:3, :3], atol=1e-5), \
        "Rotation mismatch"
    
    print("✓ Roundtrip test passed!")


def test_identity_at_reference():
    """Test that the action at reference index becomes identity-like."""
    np.random.seed(42)
    
    # Create a horizon of poses
    horizon = 4
    poses = []
    for i in range(horizon):
        xyz = np.random.randn(3) * 0.1
        rot = R.random().as_matrix()
        rot_6d = rotation_matrix_to_6d(rot)
        poses.append(np.concatenate([xyz, rot_6d]))
    poses = np.stack(poses)
    
    # Use index 1 as reference (like n_obs_steps=2)
    ref_idx = 1
    ref_pose = poses[ref_idx]
    
    # Compute relative
    relative = compute_relative_actions(poses, ref_pose, pose_format="6d")
    
    # The pose at reference index should be near identity
    # Identity: xyz=[0,0,0], rot=I → 6d=[1,0,0,0,1,0]
    identity_6d = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)
    assert np.allclose(relative[ref_idx], identity_6d, atol=1e-5), \
        f"Reference pose should be identity, got {relative[ref_idx]}"
    
    print("✓ Identity at reference test passed!")


if __name__ == "__main__":
    test_roundtrip()
    test_identity_at_reference()
    print("\n✓ All tests passed!")
