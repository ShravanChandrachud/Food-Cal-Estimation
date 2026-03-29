"""
End-to-end video demo with visualizations at every step.
Processes the first video in videos/ and generates report-ready PNGs.
"""

import sys, json, time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from collections import Counter

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path(".").resolve()
    if ROOT.name == "notebooks":
        ROOT = ROOT.parent
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

exts = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
videos = [f for f in VIDEOS_DIR.iterdir() if f.suffix.lower() in exts]
print(f"Found {len(videos)} video(s):")
for v in videos:
    print(f"  {v.name} ({v.stat().st_size / 1024 / 1024:.1f} MB)")

VIDEO = videos[2]
VID_NAME = VIDEO.stem
OUT = OUTPUTS_DIR / f"v2_demo_{VID_NAME}"
OUT.mkdir(parents=True, exist_ok=True)
print(f"\nProcessing: {VIDEO.name}")

print(f"\n{'=' * 50}")
print("  STEP 1: Keyframe extraction")
print(f"{'=' * 50}")

t0 = time.time()
ext = extract_keyframes(str(VIDEO), video_id=VID_NAME)
t_frames = time.time() - t0

print(f"  FPS: {ext['fps']:.1f}, Duration: {ext['duration']:.1f}s")
print(f"  Keyframes: {ext['n_saved']}, Blurry skipped: {ext['n_blurry']}")
print(f"  Time: {t_frames:.1f}s")

frames = ext["keyframes"]
samples = frames[:: max(1, len(frames) // 12)][:12]
n_cols = 4
n_rows = (len(samples) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
axes = axes.flatten()
for i, fi in enumerate(samples):
    try:
        axes[i].imshow(Image.open(fi["path"]))
        tag = " [CUT]" if fi["scene_cut"] else ""
        axes[i].set_title(f"t={fi['time_str']}{tag}", fontsize=10)
    except:
        pass
    axes[i].set_xticks([])
    axes[i].set_yticks([])
for i in range(len(samples), len(axes)):
    axes[i].set_visible(False)
plt.suptitle(f"Keyframes — {VIDEO.name}", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "keyframes_sample.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\n{'=' * 50}")
print("  STEP 2: CLIP embedding")
print(f"{'=' * 50}")

embedder = CLIPEmbedder()
t0 = time.time()
paths = [f["path"] for f in ext["keyframes"]]
frame_emb = embedder.embed_images(paths)
t_embed = time.time() - t0
print(f"  {frame_emb.shape} embeddings in {t_embed:.1f}s")

print(f"\n{'=' * 50}")
print("  STEP 3: Multi-label classification")
print(f"{'=' * 50}")

clf = MultiLabelClassifier()
clf.load()

all_dets = clf.predict(frame_emb)

for fi, dets in zip(ext["keyframes"], all_dets):
    if dets:
        s = ", ".join(f"{n}({p:.2f})" for n, p in dets)
        print(f"  {fi['time_str']} → {s}")
    else:
        print(f"  {fi['time_str']} → (none)")

probs_matrix = np.zeros((len(all_dets), len(clf.class_names)))
clf.model.eval()
with __import__("torch").no_grad():
    logits = clf.model(
        __import__("torch").tensor(frame_emb, dtype=__import__("torch").float32)
    )
    all_probs = __import__("torch").sigmoid(logits).numpy()
probs_matrix = all_probs

fig, ax = plt.subplots(figsize=(14, max(5, len(ext["keyframes"]) * 0.15)))
im = ax.imshow(probs_matrix.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
ax.set_yticks(range(len(clf.class_names)))
ax.set_yticklabels(clf.class_names, fontsize=9)
ax.set_xlabel("Frame index")
ax.set_title("Detection probabilities per frame", fontweight="bold")
plt.colorbar(im, ax=ax, label="Sigmoid probability")
plt.tight_layout()
plt.savefig(OUT / "detection_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

ingredient_counts = Counter()
for dets in all_dets:
    for name, prob in dets:
        ingredient_counts[name] += 1

print(f"\nAggregated detections across {len(all_dets)} frames:")
for name, count in ingredient_counts.most_common():
    pct = count / len(all_dets) * 100
    typ = "COUNT" if name in COUNT_BASED else "VOLUME"
    print(f"  [{typ}] {name:15s}: {count:>3d} frames ({pct:.0f}%)")

if ingredient_counts:
    fig, ax = plt.subplots(figsize=(10, max(4, len(ingredient_counts) * 0.5)))
    names = [n for n, _ in ingredient_counts.most_common()]
    counts = [c for _, c in ingredient_counts.most_common()]
    c = ["#1D9E75" if n in COUNT_BASED else "#BA7517" for n in names]
    ax.barh(names, counts, color=c)
    for i, (cnt, n) in enumerate(zip(counts, names)):
        ax.text(
            cnt + 0.5,
            i,
            f"{cnt}/{len(all_dets)} ({cnt / len(all_dets) * 100:.0f}%)",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Frames detected in")
    ax.set_title("Ingredient detection frequency", fontweight="bold")
    ax.invert_yaxis()
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color="#1D9E75", label="Count-based"),
            Patch(color="#BA7517", label="Volume-based"),
        ],
        loc="lower right",
    )
    plt.tight_layout()
    plt.savefig(OUT / "detection_frequency.png", dpi=150, bbox_inches="tight")
    plt.show()

print(f"\n{'=' * 50}")
print("  STEP 4: Building ingredient list")
print(f"{'=' * 50}")

ingredient_lines = []
for name, count in ingredient_counts.most_common():
    if name in COUNT_BASED:
        w = DEFAULT_WEIGHT_G.get(name, 100)
        est_count = max(1, min(5, count // (len(all_dets) // 3 + 1)))
        line = f"{est_count}x {name} (~{w}g each)"
        ingredient_lines.append(line)
        print(f"  {line}")
    elif name in VOLUME_BASED:
        defaults = {
            "oil": "1 tbsp",
            "arborio": "1 cup",
            "basmati": "1 cup",
            "ipsala": "1 cup",
            "jasmine": "1 cup",
            "karacadag": "1 cup",
        }
        qty = defaults.get(name, "1 serving")
        line = f"{name}: {qty}"
        ingredient_lines.append(line)
        print(f"  {line} (default)")
    else:
        line = f"1x {name} (~100g)"
        ingredient_lines.append(line)
        print(f"  {line}")

print(f"\n{'=' * 50}")
print("  STEP 5: USDA nutritional lookup")
print(f"{'=' * 50}")

usda = USDALookup()
usda_data = {}
for name in ingredient_counts:
    r = usda.lookup(name)
    if r and r.get("per_100g"):
        m = r["per_100g"]
        print(
            f"  {name:15s} → {r['matched']} ({m.get('calories_kcal', '?')} kcal/100g)"
        )
        usda_data[name] = r
    else:
        print(f"  {name:15s} → no match")

print(f"\n{'=' * 50}")
print("  STEP 6: LLM nutritional summary")
print(f"{'=' * 50}")

embedder.unload()

t0 = time.time()
try:
    summary = summarize(ingredient_lines, usda_data)
    t_llm = time.time() - t0

    report = format_report(summary)
    try:
        print(report)
    except UnicodeEncodeError:
        print(report.encode("ascii", errors="replace").decode("ascii"))

    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary.model_dump(), f, indent=2, ensure_ascii=False)
    with open(OUT / "report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  LLM time: {t_llm:.1f}s")

except Exception as e:
    print(f"  LLM ERROR: {e}")
    print("  Is ollama serve running?")
    summary = None

if summary:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    t = summary.total_macros
    macro_cals = {
        "Protein": t.protein_g * 4,
        "Carbs": t.carbs_g * 4,
        "Fat": t.fat_g * 9,
    }
    ax1.pie(
        macro_cals.values(),
        labels=macro_cals.keys(),
        autopct="%1.1f%%",
        colors=["#1D9E75", "#BA7517", "#D85A30"],
        startangle=90,
    )
    ax1.set_title(
        f"Macro calorie split\n({t.calories_kcal:.0f} kcal total)", fontweight="bold"
    )

    ing_names = [i.name for i in summary.ingredients]
    ing_cals = [i.macros.calories_kcal for i in summary.ingredients]
    ax2.barh(ing_names, ing_cals, color="#534AB7")
    for bar, cal in zip(ax2.patches, ing_cals):
        ax2.text(
            bar.get_width() + 2,
            bar.get_y() + bar.get_height() / 2,
            f"{cal:.0f} kcal",
            va="center",
            fontsize=9,
        )
    ax2.set_xlabel("Calories (kcal)")
    ax2.set_title("Calories per ingredient", fontweight="bold")
    ax2.invert_yaxis()

    plt.suptitle(
        f"Nutritional analysis — {summary.dish_name}", fontsize=15, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(OUT / "macro_breakdown.png", dpi=150, bbox_inches="tight")
    plt.show()

print(f"\n{'=' * 50}")
print("  TIMING SUMMARY")
print(f"{'=' * 50}")
print(f"  Keyframe extraction:  {t_frames:.1f}s")
print(f"  CLIP embedding:       {t_embed:.1f}s")
if summary:
    print(f"  LLM summarization:    {t_llm:.1f}s")
    print(f"  Total:                {t_frames + t_embed + t_llm:.1f}s")
print(f"\n  All outputs → {OUT}/")

timing = {"keyframes_s": round(t_frames, 1), "embedding_s": round(t_embed, 1)}
if summary:
    timing["llm_s"] = round(t_llm, 1)
with open(OUT / "timing.json", "w", encoding="utf-8") as f:
    json.dump(timing, f, indent=2)

usda.close()
print("\nDone!")
