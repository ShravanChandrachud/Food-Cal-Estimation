"""
Downloads YouTube videos as 720p H.264 MP4 via yt-dlp.
"""

import yt_dlp, os
from pathlib import Path
from src.config import VIDEOS_DIR, MAX_VIDEO_HEIGHT


def download_video(url, output_dir=None):
    output_dir = Path(output_dir or VIDEOS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": f"bv*[height<={MAX_VIDEO_HEIGHT}][ext=mp4]+ba[ext=m4a]/b[height<={MAX_VIDEO_HEIGHT}]",
        "merge_output_format": "mp4",
        "format_sort": ["res", "codec:h264"],
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(os.path.splitext(ydl.prepare_filename(info))[0] + ".mp4")
        print(f"[Download] {info.get('title')} → {path}")
        return {
            "path": str(path),
            "title": info.get("title"),
            "duration": info.get("duration", 0),
        }
