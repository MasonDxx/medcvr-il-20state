from __future__ import annotations

import numpy as np
import pytest

from medcvr_il.data_conversion.episode_loading.csv_loader import (
    load_eef_unity_csv,
    load_jaw_csv,
    load_eef_csv,
    load_jaw_from_joint_state,
    load_all_joints_from_joint_state,
    load_timestamp,
)


# -------- load_eef_unity_csv -----------------------------------------------

class TestLoadEefUnityCsv:
    def test_returns_none_for_missing_file(self, tmp_path):
        assert load_eef_unity_csv(tmp_path / "nope.txt") is None

    def test_unity_eef_9col(self, tmp_path):
        """sec,nsec,x,y,z,qx,qy,qz,qw -> drop 2 -> (T,7)"""
        p = tmp_path / "eef.txt"
        p.write_text(
            "1,0,0.1,0.2,0.3,0.0,0.0,0.0,1.0\n"
            "2,0,0.4,0.5,0.6,0.0,0.0,0.0,1.0\n"
        )
        arr = load_eef_unity_csv(p)
        assert arr.shape == (2, 7)
        assert arr.dtype == np.float32
        np.testing.assert_allclose(arr[0], [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], atol=1e-6)

    def test_single_row(self, tmp_path):
        """Single row should still return 2-D array."""
        p = tmp_path / "single.txt"
        p.write_text("1,0,0.1,0.2,0.3,0.0,0.0,0.0,1.0\n")
        arr = load_eef_unity_csv(p)
        assert arr.ndim == 2
        assert arr.shape == (1, 7)

    def test_trailing_nan_column_stripped(self, tmp_path):
        """Trailing empty column (parsed as NaN) should be dropped."""
        p = tmp_path / "trailing.txt"
        p.write_text(
            "1,0,0.1,0.2,0.3,0.0,0.0,0.0,1.0,\n"
            "2,0,0.4,0.5,0.6,0.0,0.0,0.0,1.0,\n"
        )
        arr = load_eef_unity_csv(p)
        assert arr.shape == (2, 7)

    def test_custom_drop(self, tmp_path):
        """drop=0 keeps all columns."""
        p = tmp_path / "nodrop.txt"
        p.write_text("1,0,0.1,0.2,0.3,0.0,0.0,0.0,1.0\n")
        arr = load_eef_unity_csv(p, drop=0)
        assert arr.shape == (1, 9)


# -------- load_jaw_csv ------------------------------------------------------

class TestLoadJawCsv:
    def test_returns_none_for_missing_file(self, tmp_path):
        assert load_jaw_csv(tmp_path / "nope.txt") is None

    def test_gripper_3col(self, tmp_path):
        """sec,nsec,value -> (T,1)"""
        p = tmp_path / "gripper.txt"
        p.write_text("1,0,0.5\n2,0,0.8\n3,0,1.0\n")
        arr = load_jaw_csv(p)
        assert arr.shape == (3, 1)
        assert arr.dtype == np.float32
        np.testing.assert_allclose(arr[:, 0], [0.5, 0.8, 1.0], atol=1e-6)

    def test_single_row(self, tmp_path):
        p = tmp_path / "single.txt"
        p.write_text("1,0,0.5\n")
        arr = load_jaw_csv(p)
        assert arr.ndim == 2
        assert arr.shape == (1, 1)

# -------- load_eef_csv ---------------------------------------------

class TestLoadEefCsv:
    def test_9col_layout(self, tmp_path):
        """sec,nsec,x,y,z,qx,qy,qz,qw → (T,7)"""
        p = tmp_path / "eef.txt"
        p.write_text("1,0,0.1,0.2,0.3,0.0,0.0,0.0,1.0\n"
                      "2,0,0.4,0.5,0.6,0.0,0.0,0.0,1.0\n")
        arr = load_eef_csv(p)
        assert arr.shape == (2, 7)
        assert arr.dtype == np.float32
        np.testing.assert_allclose(arr[0], [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], atol=1e-6)

    def test_7col_layout(self, tmp_path):
        """x,y,z,qx,qy,qz,qw → (T,7)"""
        p = tmp_path / "eef.txt"
        p.write_text("0.1,0.2,0.3,0.0,0.0,0.0,1.0\n")
        arr = load_eef_csv(p)
        assert arr.shape == (1, 7)
        np.testing.assert_allclose(arr[0], [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], atol=1e-6)

    def test_trailing_nan_stripped(self, tmp_path):
        p = tmp_path / "eef.txt"
        p.write_text("0.1,0.2,0.3,0.0,0.0,0.0,1.0,\n")
        arr = load_eef_csv(p)
        assert arr.shape == (1, 7)

    def test_bad_column_count_raises(self, tmp_path):
        p = tmp_path / "eef.txt"
        p.write_text("0.1,0.2,0.3,0.0,0.0\n")
        with pytest.raises(ValueError, match="expected 7 or 9"):
            load_eef_csv(p)


# -------- load_jaw_from_joint_state --------------------------------

class TestLoadJawFromJointState:
    def test_extracts_last_column(self, tmp_path):
        """sec,nsec,j1,j2,j3,jaw → (T,1) jaw only"""
        p = tmp_path / "joint_state.txt"
        p.write_text("1,0,0.1,0.2,0.3,0.9\n"
                      "2,0,0.4,0.5,0.6,0.8\n")
        jaw = load_jaw_from_joint_state(p)
        assert jaw.shape == (2, 1)
        assert jaw.dtype == np.float32
        np.testing.assert_allclose(jaw[:, 0], [0.9, 0.8], atol=1e-6)

    def test_single_row(self, tmp_path):
        p = tmp_path / "joint_state.txt"
        p.write_text("1,0,0.1,0.2,0.3,0.5\n")
        jaw = load_jaw_from_joint_state(p)
        assert jaw.shape == (1, 1)


# -------- load_all_joints_from_joint_state -------------------------

class TestLoadAllJointsFromJointState:
    def test_extracts_4_joints(self, tmp_path):
        """sec,nsec,j1,j2,j3,jaw → (T,4) columns 2-5"""
        p = tmp_path / "joint_state.txt"
        p.write_text("1,0,0.1,0.2,0.3,0.9\n"
                      "2,0,0.4,0.5,0.6,0.8\n")
        joints = load_all_joints_from_joint_state(p)
        assert joints.shape == (2, 4)
        assert joints.dtype == np.float32
        np.testing.assert_allclose(joints[0], [0.1, 0.2, 0.3, 0.9], atol=1e-6)


# -------- load_timestamp -------------------------------------------

class TestLoadTimestamp:
    def test_parses_iso_timestamps(self, tmp_path):
        p = tmp_path / "timestamp.txt"
        p.write_text("2024-01-15 10:30:00.000000\n"
                      "2024-01-15 10:30:01.000000\n")
        ts = load_timestamp(p)
        assert ts.shape == (2,)
        # 1 second apart = 1e9 nanoseconds
        assert ts[1] - ts[0] == 1_000_000_000

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "timestamp.txt"
        p.write_text("2024-01-15 10:30:00.000000\n\n\n2024-01-15 10:30:01.000000\n")
        ts = load_timestamp(p)
        assert ts.shape == (2,)

    def test_microsecond_precision(self, tmp_path):
        p = tmp_path / "timestamp.txt"
        p.write_text("2024-01-15 10:30:00.000000\n"
                      "2024-01-15 10:30:00.000001\n")
        ts = load_timestamp(p)
        # 1 microsecond = 1000 nanoseconds
        assert ts[1] - ts[0] == 1_000
