"""camera.py - Camera realtime: thread riêng, chỉ giữ frame mới nhất (không bị trễ/lag)."""
import sys
import threading

import cv2

BACKEND = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY


def list_cameras(n=5):
    found = []
    for i in range(n):
        cap = cv2.VideoCapture(i, BACKEND)
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found


class CameraStream:
    def __init__(self):
        self.cap = None
        self.frame = None
        self.seq = 0
        self.running = False
        self.cond = threading.Condition()
        self.thread = None

    def open(self, index=0, width=640, height=480):
        cap = cv2.VideoCapture(index, BACKEND)
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # MJPG nhanh hơn YUY2
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, _ = cap.read()
        if not ok:
            cap.release()
            return False
        self.cap, self.running = cap, True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return True

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                continue
            with self.cond:
                self.frame = f
                self.seq += 1
                self.cond.notify_all()

    def read(self, last_seq, timeout=0.2):
        """Chờ frame MỚI hơn last_seq. Trả (seq, frame) hoặc (last_seq, None) nếu hết thời gian chờ."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq or not self.running, timeout)
            if self.seq == last_seq:
                return last_seq, None
            return self.seq, self.frame

    def close(self):
        self.running = False
        with self.cond:
            self.cond.notify_all()
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        self.cap = None
