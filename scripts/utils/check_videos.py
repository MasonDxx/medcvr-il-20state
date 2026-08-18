#!/usr/bin/env python3
import argparse
import os
import sys
import cv2

def count_jpegs(folder):
    return sum(
        1 for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and f.lower().endswith(('.jpg', '.jpeg'))
    )

def find_video(folder):
    videos = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith('.mp4')
    ]

    for v in videos:
        if os.path.basename(v).lower() == 'video.mp4':
            return v

def get_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return -1
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()
    return n

def discover_camera_folders(root_dir):
    cams = set()
    for ep in sorted(os.listdir(root_dir)):
        if not ep.startswith('e'):
            continue
        ep_path = os.path.join(root_dir, ep)
        if not os.path.isdir(ep_path):
            continue
        for item in sorted(os.listdir(ep_path)):
            cam_path = os.path.join(ep_path, item)
            if not os.path.isdir(cam_path):
                continue
            has_mp4 = any(
                os.path.isfile(os.path.join(cam_path, f)) and f.lower().endswith(".mp4")
                for f in os.listdir(cam_path)
            )
            has_jpg = any(
                os.path.isfile(os.path.join(cam_path, f)) and f.lower().endswith((".jpg", ".jpeg"))
                for f in os.listdir(cam_path)
            )
            if has_mp4 or has_jpg:
                cams.add(item)
    return sorted(cams)

def check_root(root_dir, tolerance=0, cams=None, check_jpg_match=False):
    root_dir = os.path.expanduser(root_dir)
    if cams is None:
        cams = discover_camera_folders(root_dir)
        if not cams:
            cams = ['static_cam', 'wrist_cam']
            print("[WARN] Could not auto-detect camera folders; falling back to static_cam,wrist_cam")

    print(f"Checking cameras: {', '.join(cams)}")
    mismatches = []

    for ep in sorted(os.listdir(root_dir)):
        if not ep.startswith('e'):
            continue
        ep_path = os.path.join(root_dir, ep)
        if not os.path.isdir(ep_path):
            continue

        frame_counts = {}
        for cam in cams:
            cam_path = os.path.join(ep_path, cam)
            if not os.path.isdir(cam_path):
                mismatches.append(f"{ep}: missing folder '{cam}' at {cam_path}")
                continue

            jpgs = count_jpegs(cam_path)
            video_path = find_video(cam_path)
            if not video_path:
                mismatches.append(f"{ep}/{cam}: {jpgs} JPGs but NO .mp4 found in {cam_path}")
                continue

            frames = get_frame_count(video_path)
            if frames < 0:
                mismatches.append(f"{ep}/{cam}: failed to open video {video_path} (JPGs={jpgs})")
                continue

            frame_counts[cam] = frames
            if check_jpg_match and abs(frames - jpgs) > tolerance:
                mismatches.append(
                    f"{ep}/{cam}: JPGs={jpgs} vs video frames={frames} -> {video_path}"
                )

        if len(frame_counts) > 1:
            values = sorted(set(frame_counts.values()))
            if len(values) != 1:
                details = ", ".join(f"{k}={v}" for k, v in sorted(frame_counts.items()))
                mismatches.append(f"{ep}: camera frame-count mismatch ({details})")

        if frame_counts:
            details = ", ".join(f"{k}={v}" for k, v in sorted(frame_counts.items()))
            print(f"{ep}: {details}")

    if mismatches:
        print("Mismatches found:")
        for m in mismatches:
            print("  -", m)
        sys.exit(1)
    else:
        if check_jpg_match:
            print("All checks passed: per-camera frame counts are consistent and match JPG counts.")
        else:
            print("All checks passed: per-camera video frame counts are consistent.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check per-episode video frame counts across camera folders."
    )
    parser.add_argument("root_dir", nargs="?", default="~/medcvr/data")
    parser.add_argument(
        "--tolerance",
        type=int,
        default=0,
        help="Allowed absolute JPG-vs-video frame difference (used only with --check-jpg-match).",
    )
    parser.add_argument(
        "--cams",
        type=str,
        default="",
        help="Comma-separated camera folder names. Default: auto-detect from episodes.",
    )
    parser.add_argument(
        "--check-jpg-match",
        action="store_true",
        help="Also require JPG count to match video frame count for each camera.",
    )

    args = parser.parse_args()
    cams = [c.strip() for c in args.cams.split(",") if c.strip()] if args.cams else None
    check_root(args.root_dir, args.tolerance, cams=cams, check_jpg_match=args.check_jpg_match)
