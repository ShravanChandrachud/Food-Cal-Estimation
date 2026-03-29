"""
Download USDA SR Legacy CSVs and build offline SQLite database.
"""

import sys, csv, io, zipfile, sqlite3, requests
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import USDA_DB_PATH, DATA_DIR, MACRO_NUTRIENT_IDS

CSV_DIR = DATA_DIR / "usda_csv"
URL = "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_sr_legacy_food_csv_2018-04.zip"


def download():
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    if (CSV_DIR / "food.csv").exists():
        print("[USDA] CSVs exist, skipping download.")
        return
    print(f"[USDA] Downloading SR Legacy...")
    try:
        r = requests.get(URL, stream=True, timeout=120)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            for m in zf.namelist():
                bn = Path(m).name
                if bn in ("food.csv", "food_nutrient.csv"):
                    (CSV_DIR / bn).write_bytes(zf.read(m))
                    print(f"  Extracted {bn}")
    except Exception as e:
        print(f"Download failed: {e}")
        print(f"Manual: download from https://fdc.nal.usda.gov/download-datasets/")
        print(f"Extract food.csv + food_nutrient.csv to {CSV_DIR}/")
        sys.exit(1)


def build():
    if USDA_DB_PATH.exists():
        USDA_DB_PATH.unlink()
    conn = sqlite3.connect(str(USDA_DB_PATH))
    c = conn.cursor()
    c.execute("CREATE TABLE food (fdc_id INTEGER PRIMARY KEY, description TEXT)")
    c.execute(
        "CREATE TABLE food_nutrient (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "fdc_id INTEGER, nutrient_id INTEGER, amount REAL)"
    )

    fc = 0
    with open(CSV_DIR / "food.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fid, desc = (
                row.get("fdc_id", "").strip(),
                row.get("description", "").strip(),
            )
            if fid and desc:
                try:
                    c.execute(
                        "INSERT OR IGNORE INTO food VALUES (?,?)", (int(fid), desc)
                    )
                    fc += 1
                except:
                    pass
    print(f"  {fc} foods loaded")

    target = set(MACRO_NUTRIENT_IDS.keys())
    nc = 0
    with open(CSV_DIR / "food_nutrient.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                nid = int(row.get("nutrient_id", "").strip())
                if nid not in target:
                    continue
                fid = int(row.get("fdc_id", "").strip())
                amt = float(row.get("amount", "").strip())
                c.execute(
                    "INSERT INTO food_nutrient (fdc_id,nutrient_id,amount) VALUES (?,?,?)",
                    (fid, nid, amt),
                )
                nc += 1
            except:
                pass
    print(f"  {nc} nutrient records loaded")

    c.execute("CREATE INDEX idx_fn ON food_nutrient(fdc_id)")
    conn.commit()
    conn.close()
    sz = USDA_DB_PATH.stat().st_size / 1024 / 1024
    print(f"[USDA] Built {USDA_DB_PATH} ({sz:.1f} MB)")


if __name__ == "__main__":
    print("Building USDA database...")
    download()
    build()
    print("Done!")
