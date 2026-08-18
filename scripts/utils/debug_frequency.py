#!/usr/bin/env python3
"""
Debug script to analyze timestamps and frequency calculation for all episodes
"""
import numpy as np
import cv2
import re
from pathlib import Path
import argparse

def natural_key(p: Path):
    """Sort episodes naturally (episode_1, episode_2, ..., episode_10)"""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', p.stem)]

def get_mp4_info(episode_dir: Path):
    """Get MP4 file information (frame count, duration, fps)"""
    # Look for MP4 files in common locations
    mp4_locations = [
        episode_dir / "static_cam" / "video.mp4",
        # episode_dir / "static_cam" / "mp4" / "video.mp4",
        episode_dir / "wrist_cam" / "video.mp4", 
        # episode_dir / "wrist_cam" / "mp4" / "video.mp4",
    ]
    
    mp4_info = {}
    
    for location in mp4_locations:
        if location.exists():
            cap = cv2.VideoCapture(str(location))
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                metadata_fps = cap.get(cv2.CAP_PROP_FPS)
                metadata_duration = frame_count / metadata_fps if metadata_fps > 0 else 0
                
                cam_name = location.parent.name if location.parent.name != "mp4" else location.parent.parent.name
                mp4_info[cam_name] = {
                    'frame_count': frame_count,
                    'metadata_fps': metadata_fps,
                    'metadata_duration': metadata_duration,
                    'path': location
                }
                cap.release()
    
    return mp4_info

def debug_frequency_single(csv_path: Path):
    """Debug frequency calculation for a single episode"""
    
    if not csv_path.exists():
        return None
        
    # Load data
    data = np.genfromtxt(csv_path, delimiter=",", dtype=float, filling_values=np.nan)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    # Drop trailing NaN columns
    while data.shape[1] and np.isnan(data[:, -1]).all():
        data = data[:, :-1]
    
    if data.shape[1] < 2:
        return None
    
    # Extract timestamps
    sec = data[:, 0]
    nsec = data[:, 1]
    
    # Validate timestamps
    if np.any(sec < 1e9) or np.any(nsec >= 1e9) or np.any(nsec < 0):
        return None
    
    # Convert to continuous timestamps
    timestamps = sec + nsec / 1e9
    
    if len(timestamps) < 2:
        return None
    
    # Calculate time differences
    time_diffs = np.diff(timestamps)
    valid_diffs = time_diffs[time_diffs > 0]
    
    if len(valid_diffs) == 0:
        return None
    
    # Calculate frequencies
    median_dt = np.median(valid_diffs)
    mean_dt = valid_diffs.mean()
    
    median_freq = 1.0 / median_dt
    mean_freq = 1.0 / mean_dt
    expected_freq = len(timestamps) / (timestamps[-1] - timestamps[0])
    
    # Find most common interval
    unique_diffs, counts = np.unique(np.round(valid_diffs, 6), return_counts=True)
    most_common_idx = np.argmax(counts)
    most_common_dt = unique_diffs[most_common_idx]
    most_common_freq = 1.0 / most_common_dt
    most_common_pct = 100 * counts[most_common_idx] / len(valid_diffs)
    
    return {
        'total_samples': len(timestamps),
        'duration': timestamps[-1] - timestamps[0],
        'median_freq': median_freq,
        'mean_freq': mean_freq,
        'expected_freq': expected_freq,
        'most_common_freq': most_common_freq,
        'most_common_pct': most_common_pct,
        'timing_std': valid_diffs.std(),
        'valid_intervals': len(valid_diffs),
        'invalid_intervals': len(time_diffs) - len(valid_diffs)
    }

def debug_all_episodes(raw_dir: Path):
    """Debug frequency for all episodes in a directory"""
    
    # Find all episode directories - look for e0, e1, e2, etc.
    episode_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir() and re.match(r'^e\d+$', p.name)], 
                         key=natural_key)
    
    if not episode_dirs:
        print(f"No episode directories found in {raw_dir}")
        print(f"Looking for directories named: e0, e1, e2, etc.")
        return
    
    print(f"Found {len(episode_dirs)} episodes in {raw_dir}")
    print("=" * 80)
    
    all_freqs = []
    
    for episode_dir in episode_dirs:
        print(f"\n📁 {episode_dir.name}")
        print("-" * 40)
        
        # Analyze timestamps
        csv_path = episode_dir / "eef_position.txt"
        freq_info = debug_frequency_single(csv_path)
        
        if freq_info is None:
            print(f"❌ Could not analyze timestamps in {csv_path}")
        else:
            print(f"📊 Timestamp Analysis:")
            print(f"   Total samples:     {freq_info['total_samples']}")
            print(f"   Duration:          {freq_info['duration']:.3f}s")
            print(f"   Median frequency:  {freq_info['median_freq']:.2f} Hz")
            print(f"   Mean frequency:    {freq_info['mean_freq']:.2f} Hz")
            print(f"   Expected freq:     {freq_info['expected_freq']:.2f} Hz")
            print(f"   Most common freq:  {freq_info['most_common_freq']:.2f} Hz ({freq_info['most_common_pct']:.1f}% of intervals)")
            print(f"   Timing std dev:    {freq_info['timing_std']:.6f}s")
            print(f"   Invalid intervals: {freq_info['invalid_intervals']}")
            
            all_freqs.append(freq_info['median_freq'])
        
        # Analyze MP4 files
        mp4_info = get_mp4_info(episode_dir)
        if mp4_info:
            print(f"🎥 Video Analysis:")
            for cam_name, info in mp4_info.items():
                print(f"   {cam_name}:")
                print(f"     Frame count: {info['frame_count']}")
                print(f"     Metadata FPS: {info['metadata_fps']:.2f}")
                print(f"     Metadata duration: {info['metadata_duration']:.3f}s")
                
                # Calculate actual FPS based on timestamp duration
                if freq_info and freq_info['duration'] > 0:
                    actual_fps = info['frame_count'] / freq_info['duration']
                    print(f"     Actual FPS (from timestamps): {actual_fps:.2f}")
                    print(f"     Timestamp duration: {freq_info['duration']:.3f}s")
                    
                    # Check if synchronized
                    fps_diff = abs(actual_fps - 20.0)
                    if fps_diff < 1.0:
                        print(f"     ✅ Synchronized at ~20Hz (diff: {fps_diff:.2f})")
                    else:
                        print(f"     ⚠️  Not at 20Hz (diff: {fps_diff:.2f})")
                else:
                    print(f"     ❌ Cannot calculate actual FPS (no timestamp data)")
        else:
            print(f"🎥 No MP4 files found")
    
    # Summary
    if all_freqs:
        print(f"\n" + "=" * 80)
        print(f"📈 SUMMARY ACROSS ALL EPISODES")
        print(f"   Episodes analyzed: {len(all_freqs)}")
        print(f"   Frequency range:   {min(all_freqs):.2f} - {max(all_freqs):.2f} Hz")
        print(f"   Average frequency: {np.mean(all_freqs):.2f} Hz")
        print(f"   Median frequency:  {np.median(all_freqs):.2f} Hz")
        print(f"   Std deviation:     {np.std(all_freqs):.2f} Hz")
        
        # Check if frequencies are consistent
        if np.std(all_freqs) < 0.5:
            print(f"   ✅ Frequencies are consistent")
        else:
            print(f"   ⚠️  Frequencies vary significantly")
            
        # Check against expected 20Hz
        avg_freq = np.mean(all_freqs)
        if abs(avg_freq - 20.0) < 1.0:
            print(f"   ✅ Close to expected 20Hz")
        else:
            print(f"   ⚠️  Differs from expected 20Hz by {abs(avg_freq - 20.0):.2f}Hz")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, 
                       help="Path to directory containing episodes OR specific eef_position.txt file")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed analysis for each episode")
    args = parser.parse_args()
    
    if args.path.is_file() and args.path.name == "eef_position.txt":
        # Single file mode - show detailed analysis
        print(f"Analyzing single file: {args.path}")
        print("=" * 80)
        
        # Original detailed analysis function (renamed)
        data = np.genfromtxt(args.path, delimiter=",", dtype=float, filling_values=np.nan)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Drop trailing NaN columns
        while data.shape[1] and np.isnan(data[:, -1]).all():
            data = data[:, :-1]
        
        print(f"Data shape: {data.shape}")
        print(f"First few rows:")
        for i in range(min(5, len(data))):
            print(f"  {data[i]}")
        
        # Extract timestamps
        sec = data[:, 0]
        nsec = data[:, 1]
        
        print(f"\nTimestamp ranges:")
        print(f"  sec:  {sec.min():.0f} - {sec.max():.0f} (range: {sec.max() - sec.min():.3f}s)")
        print(f"  nsec: {nsec.min():.0f} - {nsec.max():.0f}")
        
        # Convert to continuous timestamps
        timestamps = sec + nsec / 1e9
        
        print(f"\nContinuous timestamps:")
        print(f"  First timestamp: {timestamps[0]:.9f}")
        print(f"  Last timestamp:  {timestamps[-1]:.9f}")
        print(f"  Total duration:  {timestamps[-1] - timestamps[0]:.3f} seconds")
        print(f"  Total samples:   {len(timestamps)}")
        
        # Calculate time differences
        time_diffs = np.diff(timestamps)
        valid_diffs = time_diffs[time_diffs > 0]
        
        print(f"\nTime differences analysis:")
        print(f"  All diffs count:   {len(time_diffs)}")
        print(f"  Valid diffs count: {len(valid_diffs)}")
        print(f"  Min diff:          {valid_diffs.min():.6f}s")
        print(f"  Max diff:          {valid_diffs.max():.6f}s")
        print(f"  Mean diff:         {valid_diffs.mean():.6f}s")
        print(f"  Median diff:       {np.median(valid_diffs):.6f}s")
        print(f"  Std diff:          {valid_diffs.std():.6f}s")
        
        # Show distribution of time differences
        print(f"\nTime difference distribution (first 20):")
        for i, diff in enumerate(valid_diffs[:20]):
            freq = 1.0 / diff
            print(f"  {i:2d}: {diff:.6f}s -> {freq:.2f}Hz")
        
        # Calculate frequencies
        median_dt = np.median(valid_diffs)
        mean_dt = valid_diffs.mean()
        
        median_freq = 1.0 / median_dt
        mean_freq = 1.0 / mean_dt
        expected_freq = len(timestamps) / (timestamps[-1] - timestamps[0])
        
        print(f"\nFrequency calculations:")
        print(f"  From median dt:     {median_freq:.2f} Hz")
        print(f"  From mean dt:       {mean_freq:.2f} Hz") 
        print(f"  From total samples: {expected_freq:.2f} Hz")
        
        # Check for any patterns in timing
        print(f"\nTiming pattern analysis:")
        unique_diffs, counts = np.unique(np.round(valid_diffs, 6), return_counts=True)
        print(f"  Most common intervals:")
        for diff, count in sorted(zip(unique_diffs, counts), key=lambda x: x[1], reverse=True)[:5]:
            freq = 1.0 / diff
            print(f"    {diff:.6f}s ({freq:.2f}Hz): {count} times ({100*count/len(valid_diffs):.1f}%)")
    
    else:
        # Directory mode - analyze all episodes
        debug_all_episodes(args.path)
