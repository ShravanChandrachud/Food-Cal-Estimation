from pathlib import Path
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASETS_DIR = PROJECT_ROOT / "datasets"
TRAIN_DIR = DATASETS_DIR / "train"
VAL_DIR = DATASETS_DIR / "val"
TEST_DIR = DATASETS_DIR / "test"

DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_TRAIN = DATA_DIR / "embeddings_train.npz"
EMBEDDINGS_VAL = DATA_DIR / "embeddings_val.npz"
EMBEDDINGS_TEST = DATA_DIR / "embeddings_test.npz"
MLP_MODEL_PATH = DATA_DIR / "mlp_classifier.pth"
THRESHOLDS_PATH = DATA_DIR / "thresholds.json"
CLASS_NAMES_PATH = DATA_DIR / "class_names.json"
USDA_DB_PATH = DATA_DIR / "usda_nutrition.db"

VIDEOS_DIR = PROJECT_ROOT / "videos"
KEYFRAMES_DIR = PROJECT_ROOT / "keyframes"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

for d in [DATA_DIR, VIDEOS_DIR, KEYFRAMES_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "laion2b_s32b_b82k"
CLIP_EMBED_DIM = 768
CLIP_BATCH_SIZE = 24
MLP_HIDDEN = 256
MLP_DROPOUT = 0.3
MLP_LR = 1e-3
MLP_EPOCHS = 50
MLP_BATCH = 256

UNIFORM_FPS = 0.5
ADAPTIVE_THRESHOLD = 3.0
MIN_SCENE_LEN = 15
BLUR_THRESHOLD = 100.0

MAX_VIDEO_HEIGHT = 720

COUNT_BASED = {
    "egg",
    "apple",
    "banana",
    "lemon",
    "strawberry",
    "orange",
    "onion",
    "tomato",
    "potato",
    "carrot",
    "broccoli",
    "shrimp",
    "salmon",
    "crab",
}
VOLUME_BASED = {
    "oil",
    "arborio",
    "basmati",
    "ipsala",
    "jasmine",
    "karacadag",
}
DEFAULT_WEIGHT_G = {
    "egg": 50,
    "apple": 182,
    "banana": 118,
    "lemon": 58,
    "strawberry": 12,
    "orange": 131,
    "onion": 110,
    "tomato": 123,
    "potato": 150,
    "carrot": 61,
    "broccoli": 91,
    "shrimp": 22,
    "salmon": 170,
    "crab": 135,
}

USDA_SEARCH = {
    "egg": "egg whole raw",
    "oil": "oil olive",
    "arborio": "rice white cooked",
    "basmati": "rice white cooked",
    "ipsala": "rice white cooked",
    "jasmine": "rice white cooked",
    "karacadag": "rice white cooked",
    "apple": "apple raw",
    "banana": "banana raw",
    "lemon": "lemon raw",
    "strawberry": "strawberry raw",
    "orange": "orange raw",
    "onion": "onion raw",
    "tomato": "tomato red raw",
    "potato": "potato raw",
    "carrot": "carrot raw",
    "broccoli": "broccoli raw",
    "shrimp": "shrimp raw",
    "salmon": "salmon raw",
    "crab": "crab raw",
}

OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_TEMPERATURE = 0
OLLAMA_NUM_CTX = 2048

USDA_API_KEY = os.getenv("USDA_API_KEY", "DEMO_KEY")
USDA_API_BASE = "https://api.nal.usda.gov/fdc/v1"
MACRO_NUTRIENT_IDS = {
    1008: "calories_kcal",
    1003: "protein_g",
    1004: "fat_g",
    1005: "carbs_g",
    1079: "fiber_g",
}
