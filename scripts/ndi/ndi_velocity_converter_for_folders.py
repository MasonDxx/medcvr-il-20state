from scipy.spatial.transform import Rotation as R
import numpy as np
from pathlib import Path
import sys
import shutil

# Base directory containing episode folders
BASE_DIR = "/workspace/data/data_push_11_14"

# Apply rotations about x, then about z (in degrees)
ANGLE_X_DEG = 45.0
ANGLE_Z_DEG = 90.0

def normalize_quaternion(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0 or not np.isfinite(n):
        raise ValueError("Zero or invalid quaternion.")
    return q / n

def create_transformation_matrix(x, y, z, R_mat):
    """Create 4x4 transformation matrix from position and rotation matrix."""
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = [x, y, z]
    return T

def extract_pose_from_transformation(T):
    """Extract x, y, z, qx, qy, qz, qw from 4x4 transformation matrix."""
    x, y, z = T[:3, 3]
    R_mat = T[:3, :3]
    quat = R.from_matrix(R_mat).as_quat()  # Returns [qx, qy, qz, qw]
    return x, y, z, quat[0], quat[1], quat[2], quat[3]

def process_episode(episode_dir):
    """Process a single episode directory."""
    in_file = episode_dir / "ndi_joint_state.txt"
    out_file = episode_dir / "eef_position.txt"
    
    if not in_file.exists():
        print(f"  ✗ {episode_dir.name}: ndi_joint_state.txt not found - DELETING FOLDER")
        try:
            shutil.rmtree(episode_dir, ignore_errors=True)
            if episode_dir.exists():
                for item in episode_dir.iterdir():
                    try:
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item, ignore_errors=True)
                    except Exception:
                        pass
                try:
                    episode_dir.rmdir()
                except Exception:
                    pass
            
            if not episode_dir.exists():
                print(f"      Deleted: {episode_dir}")
            else:
                print(f"      Partially deleted: {episode_dir}")
        except Exception as e:
            print(f"      Error deleting folder: {e}")
        return False, True
    
    rows = []
    with open(in_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [float(p) for p in line.split(",")]
            rows.append(parts)

    if len(rows) == 0:
        print(f"  ✗ {episode_dir.name}: empty file - DELETING FOLDER")
        try:
            shutil.rmtree(episode_dir, ignore_errors=True)
            if episode_dir.exists():
                for item in episode_dir.iterdir():
                    try:
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item, ignore_errors=True)
                    except Exception:
                        pass
                try:
                    episode_dir.rmdir()
                except Exception:
                    pass
            
            if not episode_dir.exists():
                print(f"      Deleted: {episode_dir}")
            else:
                print(f"      Partially deleted: {episode_dir}")
        except Exception as e:
            print(f"      Error deleting folder: {e}")
        return False, True

    rows = np.asarray(rows, dtype=float)  # shape (N, 9)

    # Fixed correction: intrinsic rotation (Rx then Rz)
    Rxm = R.from_euler("x", ANGLE_X_DEG, degrees=True).as_matrix()
    Rzm = R.from_euler("z", ANGLE_Z_DEG, degrees=True).as_matrix()
    R_fix = Rxm @ Rzm           # rightmost (Rz) applied first

    # First pass: reorient all frames
    reoriented_transforms = []
    invalid_rows = []
    for row_idx, (sec, nsec, x, y, z, qx, qy, qz, qw) in enumerate(rows):
        try:
            q = normalize_quaternion([qx, qy, qz, qw])
            R_orig_mat = R.from_quat(q).as_matrix()
            R_total_mat = R_orig_mat @ R_fix      # post-multiply
            
            # Create transformation matrix
            T = create_transformation_matrix(x, y, z, R_total_mat)
            reoriented_transforms.append((int(sec), int(nsec), T))
        except (ValueError, np.linalg.LinAlgError) as e:
            # Record invalid quaternion details
            invalid_rows.append({
                'row': row_idx,
                'quaternion': [qx, qy, qz, qw],
                'error': str(e)
            })
            continue

    # If there are any invalid rows with NaN, mark folder for deletion
    has_nan = any(np.isnan(info['quaternion']).any() for info in invalid_rows)
    
    if has_nan:
        print(f"  ✗ {episode_dir.name}: Contains {len(invalid_rows)} NaN quaternions - DELETING FOLDER")
        for info in invalid_rows[:3]:  # Show first 3 invalid rows
            quat = info['quaternion']
            print(f"      Row {info['row']}: quat=[{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")
        if len(invalid_rows) > 3:
            print(f"      ... and {len(invalid_rows) - 3} more invalid rows")
        
        # Delete the entire episode folder
        try:
            # Use ignore_errors to handle macOS metadata files
            shutil.rmtree(episode_dir, ignore_errors=True)
            
            # Verify deletion - if folder still exists, try manual cleanup
            if episode_dir.exists():
                # Remove all files first
                for item in episode_dir.iterdir():
                    try:
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item, ignore_errors=True)
                    except Exception:
                        pass
                # Remove the directory itself
                try:
                    episode_dir.rmdir()
                except Exception:
                    pass
            
            if not episode_dir.exists():
                print(f"      Deleted: {episode_dir}")
            else:
                print(f"      Partially deleted (some files remain): {episode_dir}")
        except Exception as e:
            print(f"      Error deleting folder: {e}")
        
        return False, True  # Not successful, but deleted

    if len(reoriented_transforms) == 0:
        print(f"  ⚠ Skipping {episode_dir.name}: all rows have invalid quaternions")
        return False, False

    # Second pass: compute relative transformations T_prev^-1 * T_curr
    out_rows = []
    for i in range(len(reoriented_transforms)):
        sec, nsec, T_curr = reoriented_transforms[i]
        
        if i == 0:
            # For the first frame, use identity (no previous frame)
            T_relative = np.eye(4)
        else:
            _, _, T_prev = reoriented_transforms[i - 1]
            T_prev_inv = np.linalg.inv(T_prev)
            T_relative = T_prev_inv @ T_curr
        
        # Extract pose from relative transformation
        x_rel, y_rel, z_rel, qx_rel, qy_rel, qz_rel, qw_rel = extract_pose_from_transformation(T_relative)
        
        out_rows.append([sec, nsec, x_rel, y_rel, z_rel, qx_rel, qy_rel, qz_rel, qw_rel])

    out_rows = np.asarray(out_rows)

    np.savetxt(
        out_file,
        out_rows,
        fmt=["%d","%d","%.5f","%.5f","%.5f","%.5f","%.5f","%.5f","%.5f"],
        delimiter=","
    )

    print(f"  ✓ {episode_dir.name}: Converted {len(out_rows)} rows (relative transformations)")
    return True, False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert NDI joint state files to EEF position files')
    parser.add_argument('--base-dir', type=str, required=True,
                        help='Base directory containing episode folders (e0, e1, ...)')
    args = parser.parse_args()
    
    base_path = Path(args.base_dir)
    
    if not base_path.exists():
        print(f"Error: Base directory not found: {base_path}")
        sys.exit(1)
    
    print(f"Processing episodes in: {base_path}\n")
    
    # Find all episode directories (e0, e1, e2, ...)
    episode_dirs = sorted([d for d in base_path.iterdir() 
                          if d.is_dir() and d.name.startswith("e") and d.name[1:].isdigit()],
                         key=lambda x: int(x.name[1:]))
    
    if len(episode_dirs) == 0:
        print(f"Error: No episode directories (e*) found in {base_path}")
        sys.exit(1)
    
    print(f"Found {len(episode_dirs)} episode directories\n")
    
    success_count = 0
    deleted_count = 0
    for episode_dir in episode_dirs:
        success, deleted = process_episode(episode_dir)
        if success:
            success_count += 1
        if deleted:
            deleted_count += 1
    
    print(f"\n{'='*60}")
    print(f"Completed: {success_count}/{len(episode_dirs)} episodes processed successfully")
    if deleted_count > 0:
        print(f"Deleted: {deleted_count} episodes with NaN quaternions")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()