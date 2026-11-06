"""Triangle-angle calculation and connection to the trained CNN1D-GRU predictor."""
import math
from dataclasses import dataclass

import numpy as np

import moitruong as env
from gel_predict import GelCNNGRUPredictor

NAN = float("nan")


def compute_triangle_angle(pts):
    """Return the bend angle for ordered points P0, P1, P2; straight is 0 degrees."""
    p = np.asarray(pts, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 3 or p.shape[1] < 2:
        raise ValueError("Cần ít nhất 3 keypoint P0, P1, P2 để tính góc tam giác.")
    p = p[:3, :2]
    first, second = p[1] - p[0], p[2] - p[1]
    if np.linalg.norm(first) <= 1e-6 or np.linalg.norm(second) <= 1e-6:
        return NAN
    cross = first[0] * second[1] - first[1] * second[0]
    dot = first[0] * second[0] + first[1] * second[1]
    angle = math.degrees(math.atan2(cross, dot))
    return abs(angle) if env.USE_ABS_ANGLE else angle


class Smoother:
    """EMA cho mảng giá trị, bỏ qua NaN."""
    def __init__(self):
        self.prev, self.alpha = None, 1.0

    def reset(self):
        self.prev = None

    def __call__(self, arr):
        arr = np.asarray(arr, dtype=np.float64)
        if self.prev is None or self.alpha >= 1.0:
            self.prev = arr
        else:
            bad = np.isnan(self.prev) | np.isnan(arr)
            self.prev = np.where(bad, arr, self.prev * (1 - self.alpha) + arr * self.alpha)
        return self.prev


# ------------------------------------------------------------------ Kết nối
@dataclass
class Result:
    kpts: np.ndarray
    conf: float
    bbox: np.ndarray = None
    angle: float = NAN
    length: float = None
    resistance: float = None
    class_name: str = "man"

    def values(self):
        nn = lambda v: NAN if v is None else v
        return {"angle": self.angle, "length": nn(self.length), "resistance": nn(self.resistance)}


class FingerAnalyzer:
    def __init__(self, predictor: GelCNNGRUPredictor):
        self.predictor = predictor
        self.smoother = Smoother()

    def set_smooth(self, s):
        self.smoother.alpha = 1.0 - float(s)

    def reset(self):
        self.smoother.reset()
        self.predictor.reset()

    def analyze(self, kpts, score, bbox=None, class_name="man"):
        angle = float(self.smoother([compute_triangle_angle(kpts)])[0])
        r = Result(kpts=np.asarray(kpts)[:3], conf=score, bbox=bbox, angle=angle, class_name=class_name)
        if self.predictor.loaded and math.isfinite(angle):
            r.length, r.resistance = self.predictor.predict(angle)
        return r