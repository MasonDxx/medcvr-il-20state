"""
FFT Noise Filter for Velocity Data

Apply a double FFT low-pass filter to reduce high-frequency noise 
in position and quaternion delta data.
"""
import os
import numpy as np
from pathlib import Path


def load_data(filepath):
    """Load tracking delta data from file.
    
    Format: timestamp_sec, timestamp_nsec, dx, dy, dz, dqx, dqy, dqz, dqw
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) >= 8:
                    timestamp = float(parts[0]) + float(parts[1]) / 1e9
                    dx, dy, dz = float(parts[2]), float(parts[3]), float(parts[4])
                    dqx, dqy, dqz = float(parts[5]), float(parts[6]), float(parts[7])
                    dqw = float(parts[8]) if len(parts) == 9 else 1.0
                    data.append([timestamp, dx, dy, dz, dqx, dqy, dqz, dqw])
    return np.array(data) if data else None


def double_fft_lowpass(signal, cutoff_ratio=0.1):
    """
    Apply FFT low-pass filter twice for stronger noise reduction.
    
    Args:
        signal: 1D numpy array of signal values
        cutoff_ratio: fraction of frequencies to keep (0.1 = keep lowest 10%)
    
    Returns:
        Filtered signal
    """
    n = len(signal)
    if n < 4:
        return signal
    
    # First pass
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n)
    
    # Create smooth low-pass mask (Gaussian-like rolloff instead of hard cutoff)
    mask = np.exp(-(freqs / cutoff_ratio) ** 2)
    
    filtered_fft = fft * mask
    filtered = np.fft.ifft(filtered_fft).real
    
    # Second pass (double filtering for stronger noise suppression)
    fft2 = np.fft.fft(filtered)
    filtered_fft2 = fft2 * mask
    result = np.fft.ifft(filtered_fft2).real
    
    return result


def filter_position_deltas(dx, dy, dz, cutoff_ratio=0.1):
    """Filter position deltas independently."""
    dx_filtered = double_fft_lowpass(dx, cutoff_ratio)
    dy_filtered = double_fft_lowpass(dy, cutoff_ratio)
    dz_filtered = double_fft_lowpass(dz, cutoff_ratio)
    return dx_filtered, dy_filtered, dz_filtered


def filter_quaternion_deltas(dqx, dqy, dqz, dqw, cutoff_ratio=0.1):
    """
    Filter quaternion deltas and renormalize.
    
    For small-angle delta quaternions (close to identity), we can filter
    the imaginary parts (qx, qy, qz) and recompute qw to maintain unit norm.
    """
    # Filter the imaginary components
    dqx_filtered = double_fft_lowpass(dqx, cutoff_ratio)
    dqy_filtered = double_fft_lowpass(dqy, cutoff_ratio)
    dqz_filtered = double_fft_lowpass(dqz, cutoff_ratio)
    
    # Recompute dqw to maintain unit quaternion constraint
    # qw = sqrt(1 - qx^2 - qy^2 - qz^2), clipped to avoid sqrt of negative
    sum_sq = dqx_filtered**2 + dqy_filtered**2 + dqz_filtered**2
    sum_sq = np.clip(sum_sq, 0, 1)  # Ensure we don't go negative
    dqw_filtered = np.sqrt(1 - sum_sq)
    
    return dqx_filtered, dqy_filtered, dqz_filtered, dqw_filtered


def save_filtered_data(filepath, data):
    """Save filtered data in the same format as original."""
    with open(filepath, 'w') as f:
        for row in data:
            timestamp = row[0]
            sec = int(timestamp)
            nsec = int((timestamp - sec) * 1e9)
            dx, dy, dz = row[1], row[2], row[3]
            dqx, dqy, dqz, dqw = row[4], row[5], row[6], row[7]
            f.write(f"{sec},{nsec},{dx:.5f},{dy:.5f},{dz:.5f},{dqx:.5f},{dqy:.5f},{dqz:.5f},{dqw:.5f}\n")


def process_episode(episode_dir, output_dir, cutoff_ratio=0.1):
    """Process one episode and save filtered data."""
    input_file = os.path.join(episode_dir, 'eef_position.txt')
    if not os.path.exists(input_file):
        return None
    
    # Load data
    data = load_data(input_file)
    if data is None or len(data) < 10:
        return None
    
    timestamps = data[:, 0]
    dx, dy, dz = data[:, 1], data[:, 2], data[:, 3]
    dqx, dqy, dqz, dqw = data[:, 4], data[:, 5], data[:, 6], data[:, 7]
    
    # Filter position deltas
    dx_f, dy_f, dz_f = filter_position_deltas(dx, dy, dz, cutoff_ratio)
    
    # Filter quaternion deltas
    dqx_f, dqy_f, dqz_f, dqw_f = filter_quaternion_deltas(dqx, dqy, dqz, dqw, cutoff_ratio)
    
    # Combine filtered data
    filtered_data = np.column_stack([
        timestamps, dx_f, dy_f, dz_f, dqx_f, dqy_f, dqz_f, dqw_f
    ])
    
    # Create output directory structure
    episode_name = os.path.basename(episode_dir)
    output_episode_dir = os.path.join(output_dir, episode_name)
    os.makedirs(output_episode_dir, exist_ok=True)
    
    # Save filtered data
    output_file = os.path.join(output_episode_dir, 'eef_position.txt')
    save_filtered_data(output_file, filtered_data)
    
    # Copy other files (ndi_joint_state.txt, timestamp.txt, cam folders)
    for item in os.listdir(episode_dir):
        src = os.path.join(episode_dir, item)
        dst = os.path.join(output_episode_dir, item)
        if item != 'eef_position.txt':
            if os.path.isfile(src) and not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)
            elif os.path.isdir(src) and not os.path.exists(dst):
                import shutil
                shutil.copytree(src, dst)
    
    # Return stats for reporting
    stats = {
        'episode': episode_name,
        'samples': len(data),
        'dx_std_original': dx.std(),
        'dx_std_filtered': dx_f.std(),
        'dqx_std_original': dqx.std(),
        'dqx_std_filtered': dqx_f.std(),
    }
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Apply FFT low-pass filter to reduce noise in velocity data'
    )
    parser.add_argument(
        '--base-dir', '-i',
        type=str,
        required=True,
        help='Input directory containing episode folders with velocity data'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for filtered data (default: <base_dir>_filtered)'
    )
    parser.add_argument(
        '--cutoff-ratio', '-c',
        type=float,
        default=0.1,
        help='Fraction of frequencies to keep (default: 0.1 = keep lowest 10%%)'
    )
    
    args = parser.parse_args()
    
    base_dir = args.base_dir
    output_dir = args.output_dir if args.output_dir else f"{base_dir.rstrip('/')}_filtered"
    cutoff_ratio = args.cutoff_ratio
    
    print(f"FFT Noise Filtering for Velocity Data")
    print(f"======================================")
    print(f"Input:  {base_dir}")
    print(f"Output: {output_dir}")
    print(f"Cutoff ratio: {cutoff_ratio} (keeping {cutoff_ratio*100:.0f}% of frequencies)\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all episode directories
    episodes = sorted([d for d in os.listdir(base_dir) 
                       if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('e')])
    
    print(f"Processing {len(episodes)} episodes...")
    
    all_stats = []
    for i, ep in enumerate(episodes):
        ep_dir = os.path.join(base_dir, ep)
        stats = process_episode(ep_dir, output_dir, cutoff_ratio)
        if stats:
            all_stats.append(stats)
            reduction = (1 - stats['dx_std_filtered'] / stats['dx_std_original']) * 100
            print(f"  [{i+1:2d}/{len(episodes)}] {ep}: Δx std {stats['dx_std_original']:.4f} -> {stats['dx_std_filtered']:.4f} ({reduction:.1f}% reduction)")
    
    print(f"\nProcessed {len(all_stats)} episodes successfully!")
    print(f"Filtered data saved to: {output_dir}")
    
    # Summary statistics
    if all_stats:
        avg_dx_orig = np.mean([s['dx_std_original'] for s in all_stats])
        avg_dx_filt = np.mean([s['dx_std_filtered'] for s in all_stats])
        avg_reduction = (1 - avg_dx_filt / avg_dx_orig) * 100
        print(f"\nAverage position delta std reduction: {avg_reduction:.1f}%")


if __name__ == "__main__":
    main()
