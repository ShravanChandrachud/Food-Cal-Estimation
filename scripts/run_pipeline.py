"""
V2 Pipeline: Video → Keyframes → CLIP embed → MLP multi-label detect
→ User confirms/provides quantities → LLM → Macro report
"""

import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    VIDEOS_DIR,
    OUTPUTS_DIR,
    COUNT_BASED,
    VOLUME_BASED,
    DEFAULT_WEIGHT_G,
)
from src.embedder import CLIPEmbedder
from src.classifier import MultiLabelClassifier
from src.frame_extractor import extract_keyframes
from src.llm_summarizer import USDALookup, summarize, format_report


def find_videos(path=None):
    if path:
        p = Path(path)
        if not p.exists():
            print(f"ERROR: {p} not found")
            sys.exit(1)
        return [p]
    exts = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    vids = [f for f in VIDEOS_DIR.iterdir() if f.suffix.lower() in exts]
    if not vids:
        print(f"No videos in {VIDEOS_DIR}")
        sys.exit(1)
    return vids


def aggregate_detections(all_frame_detections):
    """
    Merge per-frame multi-label detections into a final ingredient set.
    Returns dict: {ingredient_name: {"count": int, "max_prob": float, "n_frames": int}}
    """
    agg = {}
    for frame_dets in all_frame_detections:
        seen_this_frame = set()
        for name, prob in frame_dets:
            if name not in agg:
                agg[name] = {"count": 0, "max_prob": 0.0, "n_frames": 0}
            if name not in seen_this_frame:
                agg[name]["n_frames"] += 1
                seen_this_frame.add(name)
            agg[name]["max_prob"] = max(agg[name]["max_prob"], prob)
    return agg


def ask_quantities(detected, total_frames):
    """Interactive: show detections, let user confirm and provide quantities."""
    print(f"\n{'=' * 55}")
    print("  DETECTED INGREDIENTS (confirm and provide quantities)")
    print(f"{'=' * 55}")

    ingredient_lines = []
    for name, info in sorted(detected.items(), key=lambda x: -x[1]["max_prob"]):
        pct = info["n_frames"] / total_frames * 100
        typ = (
            "COUNT"
            if name in COUNT_BASED
            else "VOLUME"
            if name in VOLUME_BASED
            else "OTHER"
        )

        print(f"\n  [{typ}] {name}")
        print(f"    Detected in {info['n_frames']}/{total_frames} frames ({pct:.0f}%)")
        print(f"    Max confidence: {info['max_prob']:.3f}")

        keep = input("    Keep this ingredient? (Y/n): ").strip().lower()
        if keep == "n":
            print("    → Skipped")
            continue

        if name in COUNT_BASED:
            default_w = DEFAULT_WEIGHT_G.get(name, 100)
            count = input(f"    How many? (default 1): ").strip()
            count = int(count) if count.isdigit() else 1
            ingredient_lines.append(f"{count}x {name} (~{default_w}g each)")
        else:
            qty = input(f"    Quantity (e.g. '1 cup', '200g'): ").strip()
            if not qty:
                qty = "1 serving"
            ingredient_lines.append(f"{name}: {qty}")

    return ingredient_lines


def process_video(video_path, embedder, classifier, usda):
    vid_name = Path(video_path).stem
    out_dir = OUTPUTS_DIR / f"v2_{vid_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 55}")
    print(f"  PROCESSING: {Path(video_path).name}")
    print(f"{'#' * 55}")

    # Step 1: Extract keyframes
    print("\n[1/4] Extracting keyframes...")
    t0 = time.time()
    ext = extract_keyframes(str(video_path), video_id=vid_name)
    t_frames = time.time() - t0
    print(f"  → {ext['n_saved']} keyframes in {t_frames:.1f}s")

    if not ext["keyframes"]:
        print("  ERROR: No keyframes extracted.")
        return None

    # Step 2: Embed keyframes with CLIP
    print("\n[2/4] Embedding keyframes with CLIP ViT-L/14...")
    t0 = time.time()
    paths = [f["path"] for f in ext["keyframes"]]
    frame_embeddings = embedder.embed_images(paths)
    t_embed = time.time() - t0
    print(f"  → {frame_embeddings.shape} in {t_embed:.1f}s")

    # Step 3: Multi-label classify each frame
    print("\n[3/4] Multi-label classification...")
    all_detections = classifier.predict(frame_embeddings)

    # Show per-frame summary
    for i, (finfo, dets) in enumerate(zip(ext["keyframes"], all_detections)):
        if dets:
            det_str = ", ".join(f"{n}({p:.2f})" for n, p in dets)
            print(f"  {finfo['time_str']} → {det_str}")
        else:
            print(f"  {finfo['time_str']} → (nothing detected)")

    # Aggregate across all frames
    detected = aggregate_detections(all_detections)
    print(f"\n  Unique ingredients detected: {len(detected)}")

    # Save intermediate results
    with open(out_dir / "frame_detections.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {"frame": ext["keyframes"][i], "detections": d}
                for i, d in enumerate(all_detections)
            ],
            f,
            indent=2,
        )

    if not detected:
        print("  No ingredients found. Try a different video.")
        return None

    # Step 3b: User confirms and provides quantities
    ingredient_lines = ask_quantities(detected, ext["n_saved"])

    if not ingredient_lines:
        print("  No ingredients confirmed.")
        return None

    # USDA lookup
    usda_data = {}
    for name in detected:
        r = usda.lookup(name)
        if r:
            usda_data[name] = r

    # Step 4: LLM summary
    print(f"\n[4/4] Generating nutritional summary...")
    embedder.unload()  # free GPU for Ollama

    t0 = time.time()
    try:
        summary = summarize(ingredient_lines, usda_data)
        t_llm = time.time() - t0

        report = format_report(summary)
        try:
            print(report)
        except UnicodeEncodeError:
            print(report.encode("ascii", errors="replace").decode("ascii"))

        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary.model_dump(), f, indent=2, ensure_ascii=False)
        with open(out_dir / "report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n  LLM inference: {t_llm:.1f}s")

    except Exception as e:
        print(f"  LLM ERROR: {e}")
        print("  Is ollama serve running?")

    print(f"\n  Results → {out_dir}")
    return out_dir


def main():
    vid_path = sys.argv[1] if len(sys.argv) > 1 else None
    videos = find_videos(vid_path)

    print("[Init] Loading CLIP embedder...")
    embedder = CLIPEmbedder()

    print("[Init] Loading MLP classifier...")
    classifier = MultiLabelClassifier()
    classifier.load()

    print("[Init] Loading USDA...")
    usda = USDALookup()

    for v in videos:
        process_video(str(v), embedder, classifier, usda)
        # Reload embedder onto GPU for next video
        if embedder.model.device.type == "cpu":
            embedder.model = embedder.model.to(embedder.device)

    usda.close()
    print("\nAll done!")


if __name__ == "__main__":
    main()
