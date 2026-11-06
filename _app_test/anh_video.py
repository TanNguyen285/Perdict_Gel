"""anh_video.py - Đọc/ghi ảnh & video, lấy mẫu video theo FPS, ghi CSV."""
import csv
import os
import re

import cv2
import numpy as np

import moitruong as env

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_images(folder):
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMG_EXT]
    return sorted(fs, key=natural_key)


def imread(path):
    data = np.fromfile(path, np.uint8)          # đọc được đường dẫn có dấu tiếng Việt
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


class VideoReader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Không mở được video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total / self.fps if self.total else 0.0

    def first_frame(self):
        ok, f = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return f if ok else None

    def frames(self, get_sample_fps):
        """Sinh (chỉ số, thời gian s, frame) theo FPS lấy mẫu (đọc được thay đổi lúc đang chạy).
           Frame không lấy mẫu chỉ grab() (không giải mã ra ảnh) nên nhanh."""
        i, next_t = 0, 0.0
        while True:
            if not self.cap.grab():
                break
            t = i / self.fps
            if t + 1e-6 >= next_t:
                ok, frame = self.cap.retrieve()
                if not ok:
                    break
                yield i, t, frame
                next_t = t + 1.0 / max(get_sample_fps(), 0.01)
            i += 1

    def release(self):
        self.cap.release()


class VideoRecorder:
    """Ghi video kết quả (mở khi có frame đầu tiên để biết kích thước)."""
    def __init__(self, path, fps):
        self.path, self.fps, self.vw = path, max(1.0, fps), None

    def write(self, frame):
        if self.vw is None:
            h, w = frame.shape[:2]
            self.vw = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        self.vw.write(frame)

    def close(self):
        if self.vw is not None:
            self.vw.release()


class CsvLogger:
    """Lưu số liệu theo khoá (ảnh: số thứ tự -> chạy lại cùng ảnh sẽ ghi đè, không bị trùng dòng)."""
    def __init__(self):
        self.rows = {}
        self.mode = None

    def clear(self):
        self.rows.clear()

    def upsert(self, key, row):
        self.rows[key] = row

    def __len__(self):
        return len(self.rows)

    def save(self, path):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=env.CSV_COLUMNS)
            w.writeheader()
            for n, (_, row) in enumerate(sorted(self.rows.items()), 1):
                w.writerow({"stt": n, **row})
