"""
USAGE: python scripts/download_video.py <youtube_url>
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.video_downloader import download_video

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/download_video.py <url>")
        sys.exit(1)
    download_video(sys.argv[1])
