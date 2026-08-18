#!/usr/bin/env python
"""
Test script to verify UMI relative action wrapper works correctly.

Run inside the container:
    pip install -e /workspace/lerobot --no-deps --quiet
    export PYTHONPATH=/workspace/lerobot/src:$PYTHONPATH
    python /workspace/medcvr-il/test_umi_wrapper.py
"""

import numpy as np
import torch

# Setup paths
import sys
sys.path.insert(0, "/workspace/lerobot/src")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.relative_action_wrapper import (
    UMIDatasetWrapper, 
    UMIRelativeActionTransform,
    pose_6d_to_matrix,
)

# Config matching your DiT policy
N_OBS_STEPS = 2
HORIZON = 16
FPS = 15
REPO_ID = "cagedBirdy/peg_01_12_left_12_abs"

print("=" * 60)
print("Testing UMI Relative Action Wrapper")
print("=" * 60)

# Step 1: Load dataset WITHOUT wrapper
print("\n1. Loading dataset without wrapper...")
delta_timestamps = {
    "action": [i / FPS for i in range(-N_OBS_STEPS + 1, -N_OBS_STEPS + 1 + HORIZON)]
}
print(f"   action_delta_indices: {list(range(-N_OBS_STEPS + 1, -N_OBS_STEPS + 1 + HORIZON))}")
print(f"   delta_timestamps: {delta_timestamps['action'][:5]}... (showing first 5)")

base_dataset = LeRobotDataset(
    repo_id=REPO_ID,
    delta_timestamps=delta_timestamps,
)
print(f"   Dataset loaded: {len(base_dataset)} frames")

# Step 2: Get a sample without wrapper
print("\n2. Getting sample WITHOUT wrapper...")
sample_idx = 0  # Pick a frame in the middle
item_before = base_dataset[sample_idx]

action_before = item_before["action"]
print(f"   action shape: {action_before.shape}")
print(f"   action[0] (t=-1): {action_before[0].numpy()[:6]}...")  # First 6 dims
print(f"   action[1] (t=0, ref): {action_before[1].numpy()[:6]}...")
print(f"   action[2] (t=1): {action_before[2].numpy()[:6]}...")

# Step 3: Apply wrapper
print("\n3. Applying UMI wrapper...")
action_dim = action_before.shape[1]
has_gripper = action_dim > 9
pose_dim = 9

print(f"   action_dim={action_dim}, pose_dim={pose_dim}, has_gripper={has_gripper}")

transform = UMIRelativeActionTransform(
    action_key="action",
    n_obs_steps=N_OBS_STEPS,
    pose_dim=pose_dim,
    pose_format="6d",
    has_gripper=has_gripper
)
wrapped_dataset = UMIDatasetWrapper(base_dataset, transform)

# Step 4: Get the same sample WITH wrapper
print("\n4. Getting same sample WITH wrapper...")
item_after = wrapped_dataset[sample_idx]
action_after = item_after["action"]

print(f"   action shape: {action_after.shape} (should be same as before)")
print(f"   action[0] (t=-1): {action_after[0].numpy()[:6]}...")
print(f"   action[1] (t=0, ref): {action_after[1].numpy()[:6]}...")
print(f"   action[2] (t=1): {action_after[2].numpy()[:6]}...")

# Step 5: Verify action at reference index is near identity
print("\n5. Verifying reference action is near identity...")
ref_idx = N_OBS_STEPS - 1  # Index 1 for n_obs_steps=2
ref_action = action_after[ref_idx].numpy()

# Identity in 6D: xyz=[0,0,0], rot=[1,0,0,0,1,0]
identity_pose = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0])
ref_pose_part = ref_action[:9]

print(f"   Reference action (pose part): {ref_pose_part}")
print(f"   Expected identity:            {identity_pose}")

pose_error = np.abs(ref_pose_part - identity_pose).max()
print(f"   Max error from identity: {pose_error:.6f}")

if pose_error < 1e-4:
    print("   ✓ PASS: Reference action is identity (as expected)")
else:
    print("   ✗ FAIL: Reference action is NOT identity - check implementation!")

# Step 6: Verify gripper (jaw) is unchanged
if has_gripper:
    print("\n6. Verifying gripper/jaw values are unchanged...")
    gripper_before = action_before[:, 9:].numpy()
    gripper_after = action_after[:, 9:].numpy()
    gripper_error = np.abs(gripper_before - gripper_after).max()
    print(f"   Max gripper change: {gripper_error:.6f}")
    if gripper_error < 1e-6:
        print("   ✓ PASS: Gripper values unchanged")
    else:
        print("   ✗ FAIL: Gripper values changed unexpectedly!")

# Step 7: Show action statistics
print("\n7. Action statistics comparison...")
print(f"   BEFORE wrapper - action xyz range:")
print(f"      x: [{action_before[:, 0].min():.4f}, {action_before[:, 0].max():.4f}]")
print(f"      y: [{action_before[:, 1].min():.4f}, {action_before[:, 1].max():.4f}]")
print(f"      z: [{action_before[:, 2].min():.4f}, {action_before[:, 2].max():.4f}]")

print(f"   AFTER wrapper - action xyz range:")
print(f"      x: [{action_after[:, 0].min():.4f}, {action_after[:, 0].max():.4f}]")
print(f"      y: [{action_after[:, 1].min():.4f}, {action_after[:, 1].max():.4f}]")
print(f"      z: [{action_after[:, 2].min():.4f}, {action_after[:, 2].max():.4f}]")

print("\n" + "=" * 60)
print("8. Verifying SE(3) transformation math...")
print("=" * 60)

# Get action[2] before and reference pose
ref_pose = action_before[1].numpy()  # t=0 reference
target_pose = action_before[2].numpy()  # t=1 target

print(f"\nReference pose (t=0):")
print(f"  xyz: {ref_pose[:3]}")
print(f"  6D:  {ref_pose[3:9]}")

print(f"\nTarget pose (t=1) - BEFORE wrapper:")
print(f"  xyz: {target_pose[:3]}")

# Method 1: Simple subtraction (WRONG for SE(3))
simple_diff = target_pose[:3] - ref_pose[:3]
print(f"\nSimple xyz subtraction (WRONG):")
print(f"  {simple_diff}")

# Method 2: Proper SE(3) relative transformation
# Build rotation matrices
from lerobot.datasets.relative_action_wrapper import gram_schmidt_6d_to_rotation_matrix

R_ref = gram_schmidt_6d_to_rotation_matrix(ref_pose[3:6], ref_pose[6:9])
R_target = gram_schmidt_6d_to_rotation_matrix(target_pose[3:6], target_pose[6:9])

# SE(3) relative: relative_pos = R_ref^T @ (pos_target - pos_ref)
pos_diff = target_pose[:3] - ref_pose[:3]
relative_pos_se3 = R_ref.T @ pos_diff

print(f"\nSE(3) relative position (R_ref^T @ delta_pos):")
print(f"  {relative_pos_se3}")

# Compare to wrapper output
wrapper_pos = action_after[2].numpy()[:3]
print(f"\nWrapper output for action[2]:")
print(f"  {wrapper_pos}")

print(f"\nDifference (wrapper - SE(3) manual):")
diff = wrapper_pos - relative_pos_se3
print(f"  {diff}")
print(f"  Max error: {np.abs(diff).max():.2e}")

if np.abs(diff).max() < 1e-6:
    print("  ✓ MATCH: SE(3) transformation is correct!")
else:
    print("  ✗ MISMATCH: Something is wrong!")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
