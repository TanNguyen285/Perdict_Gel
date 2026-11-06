"""moitruong.py - Cấu hình chung: đường dẫn model, ngưỡng, màu, theme, cột CSV."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ---- Model ----
POSE_TRAINED_MODEL_PATH = os.path.abspath(os.path.join(
    BASE_DIR, "..", "runs", "pose", "runs", "pose", "ngontay_yolo26s_pose_train", "weights", "best.onnx"
))
POSE_FALLBACK_MODEL_PATH = os.path.join(MODEL_DIR, "best.onnx")
POSE_MODEL_PATH = (POSE_TRAINED_MODEL_PATH if os.path.isfile(POSE_TRAINED_MODEL_PATH)
                   else POSE_FALLBACK_MODEL_PATH)
POSE_KEYPOINT_CLASS = "cogel"
GEL_MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "gel_cnn1d_gru"))
GEL_MODEL_PATH = os.path.join(GEL_MODEL_DIR, "best_model.pth")
INPUT_SIZE = 640
NUM_KPT = 3
KPT_NAMES = ["P0 (gốc)", "P1", "P2 (đầu)"]

# ---- Ngưỡng mặc định ----
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_SMOOTH = 0.4          # 0 = không làm mượt, 0.9 = rất mượt
ORT_THREADS = max(1, min(8, os.cpu_count() or 4))

# ---- Góc ----
USE_ABS_ANGLE = True          # góc uốn tam giác: ngón thẳng = 0°, càng cong càng lớn

# ---- Lấy mẫu / ghi ----
DEFAULT_VIDEO_SAMPLE_FPS = 1.0
DEFAULT_CAM_SAMPLE_FPS = 1.0
CAM_RECORD_FPS = 20
CHART_MAX_POINTS = 1200       # tối đa điểm trên biểu đồ (video/camera)

# ---- Màu vẽ (BGR cho OpenCV) ----
LINE_COLOR = (255, 255, 255)
FILL_COLOR = (245, 175, 185)  # tím nhạt
POINT_COLOR = (245, 175, 185)
POINT_EDGE = (255, 255, 255)
LABEL_BG = (60, 40, 40)

# ---- Màu biểu đồ (RGB hex) ----
CHART_COLORS = {"angle": "#ffd166", "length": "#4cc9f0", "resistance": "#ff9f43"}

CSV_COLUMNS = ["stt", "nguon", "ten_anh", "thoi_gian_s", "goc_tam_giac", "do_dai_mm",
               "dien_tro_kohm", "conf", "x0", "y0", "x1", "y1", "x2", "y2"]


def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
