from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def load_eef_unity_csv(p: Path, drop: int = 2):
    """
    Load EEF pose data for Unity-style recordings.

    Args:
        p: Path to CSV file
        drop: Number of leading columns to drop (default 2 for sec,nsec)

    Returns:
        NumPy array (T,7): x,y,z,qx,qy,qz,qw

    Expected format:
        sec,nsec,x,y,z,qx,qy,qz,qw
    """
    if not p.exists():
        return None
    arr = np.genfromtxt(p, delimiter=",", dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if np.isnan(arr[:, -1]).all():
        arr = arr[:, :-1]
    return arr[:, drop:]

def load_jaw_csv(p: Path):
    """
    Load jaw/gripper scalar data for Unity-style recordings.

    Args:
        p: Path to CSV file

    Returns:
        NumPy array (T,1): value

    Expected format:
        sec,nsec,value
    """
    if not p.exists():
        return None
    arr = np.genfromtxt(p, delimiter=",", dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, 2:3]


def load_eef_csv(csv_path: Path) -> np.ndarray:
    """
    Robust loader for eef_position.txt.

    Accepts either layout:
        • sec,nsec,x,y,z,qx,qy,qz,qw[, …empty…]
        • x,y,z,qx,qy,qz,qw[, …empty…]
    and returns (T,7)     x,y,z,qx,qy,qz,qw   float32
    """
    data = np.genfromtxt(csv_path, delimiter=",", dtype=float, filling_values=np.nan)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    while data.shape[1] and np.isnan(data[:, -1]).all():
        data = data[:, :-1]

    n_cols = data.shape[1]
    if n_cols == 9:                               # sec,nsec,x,y,z,qx,qy,qz,qw
        xyz   = data[:, 2:5]
        quat  = data[:, 5:]
        data  = np.concatenate((xyz, quat), axis=1)
    elif n_cols == 7:                             # x,y,z,qx,qy,qz,qw
        pass                                      # already in desired order
    else:
        raise ValueError(f"{csv_path}: expected 7 or 9 columns after cleanup, got {n_cols}")

    return data.astype(np.float32)                # (T,7)


def _load_joint_state(csv_path: Path) -> np.ndarray:
    data = np.genfromtxt(csv_path, delimiter=",", dtype=float, filling_values=np.nan)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    while data.shape[1] and np.isnan(data[:, -1]).all():
        data = data[:, :-1]

    return data.astype(np.float32)


def load_jaw_from_joint_state(csv_path: Path) -> np.ndarray:
    """
    Load jaw angle from joint_state.txt.

    Format: sec,nsec,joint1,joint2,joint3,jaw_angle
    Returns: (T, 1) jaw angle only (last column)
    """
    data = _load_joint_state(csv_path)
    return data[:, -1:]


def load_all_joints_from_joint_state(csv_path: Path) -> np.ndarray:
    """
    Load all 4 joint values from joint_state.txt.

    Format: sec,nsec,joint1,joint2,joint3,jaw_angle
    Returns: (T, 4) all joint values (columns 2-5: joint1,joint2,joint3,jaw_angle)
    """
    data = _load_joint_state(csv_path)
    return data[:, 2:6]


def load_timestamp(p: Path):
    """
    Load timestamps from a file.
    Expected format: 'YYYY-MM-DD HH:MM:SS.ffffff' (datetime string).
    Returns a NumPy array of integers (nanoseconds since Unix epoch).
    """
    with open(p, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    nanoseconds = []
    for line in lines:
        dt = datetime.fromisoformat(line).replace(tzinfo=timezone.utc)
        epoch = int(dt.timestamp())
        ns = epoch * 1_000_000_000 + dt.microsecond * 1_000
        nanoseconds.append(ns)
    return np.array(nanoseconds)
