"""
Hybrid keyframe extraction: uniform sampling + adaptive scene detection.
"""

import cv2
from pathlib import Path
from scenedetect import open_video, SceneManager
from scenedetect.detectors import AdaptiveDetector
from src.config import (
    KEYFRAMES_DIR,
    UNIFORM_FPS,
    ADAPTIVE_THRESHOLD,
    MIN_SCENE_LEN,
    BLUR_THRESHOLD,
)


def is_blurry(image, threshold=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < (threshold or BLUR_THRESHOLD)


def extract_keyframes(video_path, output_dir=None, video_id=None):
    video_path = str(video_path)
    video_id = video_id or Path(video_path).stem
    out = Path(output_dir or KEYFRAMES_DIR) / video_id
    out.mkdir(parents=True, exist_ok=True)

    # Scene detection
    video = open_video(video_path)
    sm = SceneManager()
    sm.add_detector(
        AdaptiveDetector(
            adaptive_threshold=ADAPTIVE_THRESHOLD, min_scene_len=MIN_SCENE_LEN
        )
    )
    sm.detect_scenes(video, show_progress=True)
    boundaries = {s.frame_num for s, _ in sm.get_scene_list()}
    print(f"[Frames] {len(boundaries)} scene boundaries detected")

    # Extract
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(fps / UNIFORM_FPS))

    frames, n, saved, blurry = [], 0, 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if n % interval == 0 or n in boundaries:
            if is_blurry(frame):
                blurry += 1
            else:
                p = out / f"frame_{saved:04d}.jpg"
                cv2.imwrite(str(p), frame)
                frames.append(
                    {
                        "frame_num": n,
                        "timestamp": n / fps,
                        "time_str": f"{int(n / fps) // 60}:{int(n / fps) % 60:02d}",
                        "path": str(p),
                        "scene_cut": n in boundaries,
                    }
                )
                saved += 1
        n += 1
    cap.release()

    print(f"[Frames] {saved} keyframes saved, {blurry} blurry skipped")
    return {
        "video_id": video_id,
        "fps": fps,
        "total_frames": total,
        "duration": total / fps,
        "keyframes": frames,
        "n_saved": saved,
        "n_blurry": blurry,
        "output_dir": str(out),
    }
