"""main.py - Giao diện PySide6: 3 tab Ảnh / Video / Camera, số liệu, biểu đồ, xuất CSV."""
import os
import sys
import time

import cv2
import anh_video as av
import moitruong as env
import ui_style
from camera import CameraStream, list_cameras
from dudoan import PoseONNX, draw_detections, draw_result
from gel_predict import GelCNNGRUPredictor
from logic import FingerAnalyzer

from PySide6.QtCore import QSignalBlocker, QThread, Signal
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox

from ui import MainWindowUI


# ============================ WORKERS ============================
class BaseWorker(QThread):
    # frame đã vẽ, kết quả, mốc (thời gian s / số thứ tự ảnh), có lấy mẫu?, ms suy luận, nhãn
    frame_ready = Signal(object, object, float, bool, float, str)
    progress = Signal(int)
    message = Signal(str)

    def __init__(self, pose, analyzer, cfg):
        super().__init__()
        self.pose, self.analyzer, self.cfg = pose, analyzer, cfg
        self._stop, self.paused, self.ui_busy = False, False, False

    def stop(self):
        self._stop = True

    def infer(self, frame):
        t0 = time.perf_counter()
        dets = self.pose.predict(frame, self.cfg["conf"], self.cfg["iou"], 1)
        draw_detections(frame, dets)
        finger = [det for det in dets if det[3] == env.POSE_KEYPOINT_CLASS]
        selected = max(finger, key=lambda det: det[1]) if finger else None
        res = None
        if selected is not None:
            res = self.analyzer.analyze(*selected)
            draw_result(frame, res, self.cfg["fill"])
        else:
            self.analyzer.reset()
        return res, (time.perf_counter() - t0) * 1000

    def wait_ui(self, timeout=3.0):
        t0 = time.perf_counter()
        while self.ui_busy and not self._stop and time.perf_counter() - t0 < timeout:
            self.msleep(1)
        self.ui_busy = False

    def emit_frame(self, frame, res, t, sample, ms, label=""):
        self.ui_busy = True
        self.frame_ready.emit(frame, res, float(t), sample, ms, label)


class ImageWorker(BaseWorker):
    def __init__(self, pose, analyzer, cfg, items, save_dir=None):
        super().__init__(pose, analyzer, cfg)
        self.items, self.save_dir = items, save_dir

    def run(self):
        n = len(self.items)
        for k, (idx, path) in enumerate(self.items):
            if self._stop:
                break
            try:
                img = av.imread(path)
                if img is None:
                    raise IOError("không đọc được ảnh")
                self.analyzer.reset()                      # mỗi ảnh độc lập, không làm mượt giữa các ảnh
                res, ms = self.infer(img)
                name = os.path.basename(path)
                if self.save_dir:
                    av.imwrite(os.path.join(self.save_dir, os.path.splitext(name)[0] + "_kq.png"), img)
                self.wait_ui()
                self.emit_frame(img, res, idx + 1, True, ms, name)
            except Exception as e:
                self.message.emit(f"Lỗi ảnh {os.path.basename(path)}: {e}")
            self.progress.emit(int(100 * (k + 1) / n))


class VideoWorker(BaseWorker):
    def __init__(self, pose, analyzer, cfg, path, rec_path=None):
        super().__init__(pose, analyzer, cfg)
        self.path, self.rec_path = path, rec_path

    def run(self):
        try:
            self._run()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.message.emit(f"Lỗi video: {e}")

    def _run(self):
        vr = av.VideoReader(self.path)
        smooth = self.cfg["smooth"]
        rec = av.VideoRecorder(self.rec_path, vr.fps if smooth else self.cfg["video_sample"]) if self.rec_path else None
        self.analyzer.reset()
        start, paused_total, last_sample = time.perf_counter(), 0.0, -1e9
        # smooth: đọc MỌI khung để phát liên tục; không smooth: chỉ đọc đúng khung lấy mẫu
        rate = lambda: 1e6 if self.cfg["smooth"] else self.cfg["video_sample"]
        try:
            for idx, t, frame in vr.frames(rate):
                if self._stop:
                    break
                if self.paused:
                    tp = time.perf_counter()
                    while self.paused and not self._stop:
                        self.msleep(30)
                    paused_total += time.perf_counter() - tp
                    if self._stop:
                        break
                sample = (t - last_sample) >= 1.0 / max(self.cfg["video_sample"], 0.01) - 1e-6
                if self.cfg["pace"]:                        # phát đúng nhịp thời gian của video
                    wait = t - (time.perf_counter() - start - paused_total)
                    while wait > 0 and not self._stop:
                        self.msleep(int(min(wait, 0.05) * 1000) + 1)
                        wait = t - (time.perf_counter() - start - paused_total)
                res, ms = self.infer(frame)
                if sample:
                    last_sample = t
                if rec:
                    rec.write(frame)
                self.wait_ui()
                self.emit_frame(frame, res, t, sample, ms)
                self.progress.emit(int(100 * idx / max(1, vr.total)))
        finally:
            vr.release()
            if rec:
                rec.close()
        if not self._stop:
            self.progress.emit(100)


class CameraWorker(BaseWorker):
    def __init__(self, pose, analyzer, cfg, index, rec_path=None):
        super().__init__(pose, analyzer, cfg)
        self.index, self.rec_path = index, rec_path

    def run(self):
        self.message.emit("Đang mở camera...")
        cam = CameraStream()
        if not cam.open(self.index):
            self.message.emit(f"Không mở được camera {self.index}")
            return
        self.message.emit(f"Camera {self.index} đang chạy")
        rec = av.VideoRecorder(self.rec_path, env.CAM_RECORD_FPS) if self.rec_path else None
        self.analyzer.reset()
        seq, t0 = 0, time.perf_counter()
        last_sample, last_rec = -1e9, -1e9
        try:
            while not self._stop:
                seq, frame = cam.read(seq)
                if frame is None:
                    continue
                frame = cv2.flip(frame, 1) if self.cfg["mirror"] else frame.copy()
                now = time.perf_counter()
                t = now - t0
                sample = (t - last_sample) >= 1.0 / max(self.cfg["cam_sample"], 0.01)
                res, ms = self.infer(frame)
                if rec and (t - last_rec) >= 1.0 / env.CAM_RECORD_FPS:
                    rec.write(frame)
                    last_rec = t
                if sample:
                    last_sample = t
                if self.ui_busy and not sample:
                    continue                                 # UI chưa kịp vẽ -> bỏ frame, không dồn hàng đợi
                self.wait_ui()
                self.emit_frame(frame, res, t, sample, ms)
        finally:
            cam.close()
            if rec:
                rec.close()


# ============================ MAIN WINDOW ============================
class MainWindow(QMainWindow, MainWindowUI):
    def __init__(self):
        super().__init__()
        env.ensure_dirs()
        self.setWindowTitle("Finger Bend Analyzer")
        self.resize(1480, 900)
        self.pose, self.gel = None, GelCNNGRUPredictor()
        self.analyzer = FingerAnalyzer(self.gel)
        self.logger = av.CsvLogger()
        self.cfg = dict(conf=env.DEFAULT_CONF, iou=env.DEFAULT_IOU, fill=True, pace=True, mirror=True,
                        video_sample=env.DEFAULT_VIDEO_SAMPLE_FPS, smooth=True, cam_sample=env.DEFAULT_CAM_SAMPLE_FPS)
        self.analyzer.set_smooth(env.DEFAULT_SMOOTH)
        self.worker, self.mode, self.running = None, None, False
        self.chart_on, self.counter = False, 0
        self.image_paths, self.video_path, self.last_frame = [], None, None
        self.fps_avg, self.ms_avg = 0.0, 0.0
        self._build()
        self._auto_load()
        self._update_buttons()

    # ---------------- Model ----------------
    def _auto_load(self):
        if os.path.exists(env.POSE_MODEL_PATH):
            self.load_pose(env.POSE_MODEL_PATH)
        if os.path.exists(env.GEL_MODEL_PATH):
            self.load_gel(env.GEL_MODEL_PATH)
        else:
            self.statusBar().showMessage(f"Không tìm thấy model CNN-GRU: {env.GEL_MODEL_PATH}")
        self._chips()

    def load_pose(self, path):
        try:
            self.pose = PoseONNX(path)
            self.statusBar().showMessage(f"Đã nạp Pose: {os.path.basename(path)} ({self.pose.w}x{self.pose.h}, CPU)")
        except Exception as e:
            self.pose = None
            QMessageBox.critical(self, "Lỗi nạp model Pose", str(e))
        self._chips()
        self._update_buttons()

    def load_gel(self, path):
        try:
            self.gel.load(path)
            self.statusBar().showMessage(f"Đã nạp CNN-GRU: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi nạp model CNN-GRU", str(e))
        self._chips()

    def pick_pose(self):
        p, _ = QFileDialog.getOpenFileName(self, "Chọn model Pose ONNX", env.MODEL_DIR, "ONNX (*.onnx)")
        if p:
            self.load_pose(p)

    def pick_gel(self):
        p, _ = QFileDialog.getOpenFileName(self, "Chọn model CNN-GRU", env.GEL_MODEL_DIR, "PyTorch (*.pth)")
        if p:
            self.load_gel(p)

    # ---------------- Điều khiển chạy ----------------
    def _need_pose(self):
        if self.pose is None:
            QMessageBox.warning(self, "Thiếu model", "Hãy nạp model Pose ONNX trước (nút ở góc phải trên).")
            return False
        return True

    def _launch(self, w, mode):
        self._stop_worker_wait()
        if mode != "anh" or self.logger.mode != "anh":      # đổi nguồn / chạy video-camera mới -> làm mới dữ liệu
            self.logger.clear()
            self.chart.clear()
            self.counter = 0
        self.logger.mode = mode
        self.chart.set_mode(mode == "anh")
        self.worker, self.mode, self.running = w, mode, True
        w.frame_ready.connect(self.on_frame)
        w.progress.connect(self._on_progress)
        w.message.connect(self.statusBar().showMessage)
        w.finished.connect(self.on_finished)
        w.start()
        self._update_buttons()

    def _stop_worker_wait(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.paused = False
            self.worker.wait(3000)

    def stop_worker(self):
        if self.worker:
            self.worker.stop()
            self.worker.paused = False

    def on_finished(self):
        self.running = False
        self.btn_pause.setText("⏸  Tạm dừng")
        self._update_buttons()
        self.statusBar().showMessage("Đã dừng / hoàn tất.")

    def _on_progress(self, v):
        (self.prog_img if self.mode == "anh" else self.prog_vid).setValue(v)

    def _update_buttons(self):
        r, ready = self.running, self.pose is not None
        self.tabs.tabBar().setEnabled(not r)
        for b in (self.btn_add, self.btn_dir, self.btn_clear, self.btn_open_vid, self.btn_scan, self.cmb_cam):
            b.setEnabled(not r)
        self.lst.setEnabled(not r)
        self.btn_run_all.setEnabled(not r and ready and len(self.image_paths) > 0)
        self.btn_stop_img.setEnabled(r and self.mode == "anh")
        self.btn_play.setEnabled(not r and ready and self.video_path is not None)
        self.btn_pause.setEnabled(r and self.mode == "video")
        self.btn_stop_vid.setEnabled(r and self.mode == "video")
        self.btn_cam_on.setEnabled(not r and ready)
        self.btn_cam_off.setEnabled(r and self.mode == "camera")
        self.btn_snap.setEnabled(self.last_frame is not None)
        self.btn_save_cur.setEnabled(self.last_frame is not None)

    # ---------------- Ảnh ----------------
    def _add_paths(self, paths):
        with QSignalBlocker(self.lst):
            for p in paths:
                self.image_paths.append(p)
                self.lst.addItem(f"{len(self.image_paths):03d}  {os.path.basename(p)}")
        self._update_buttons()

    def add_images(self):
        ps, _ = QFileDialog.getOpenFileNames(self, "Chọn ảnh", "", "Ảnh (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)")
        self._add_paths(sorted(ps, key=av.natural_key) if ps else [])

    def add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Chọn thư mục ảnh")
        if d:
            self._add_paths(av.list_images(d))

    def clear_images(self):
        self.image_paths.clear()
        self.lst.clear()
        self._update_buttons()

    def on_pick_image(self, row):
        if row < 0 or self.running or not self.image_paths or not self._need_pose():
            return
        self._launch(ImageWorker(self.pose, self.analyzer, self.cfg, [(row, self.image_paths[row])]), "anh")

    def run_all_images(self):
        if not self._need_pose():
            return
        save_dir = None
        if self.chk_save_all.isChecked():
            save_dir = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu ảnh kết quả", env.OUTPUT_DIR)
            if not save_dir:
                return
        self.prog_img.setValue(0)
        self._launch(ImageWorker(self.pose, self.analyzer, self.cfg, list(enumerate(self.image_paths)), save_dir), "anh")

    def save_current(self):
        if self.last_frame is None:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Lưu ảnh", os.path.join(env.OUTPUT_DIR, "ketqua.png"), "PNG (*.png);;JPG (*.jpg)")
        if p:
            av.imwrite(p, self.last_frame)
            self.statusBar().showMessage(f"Đã lưu {p}")

    # ---------------- Video ----------------
    def open_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "Chọn video", "", "Video (*.mp4 *.avi *.mov *.mkv *.wmv *.flv)")
        if not p:
            return
        try:
            vr = av.VideoReader(p)
            f = vr.first_frame()
            self.lbl_vid.setText(f"{os.path.basename(p)}  ·  {vr.w}x{vr.h}  ·  {vr.fps:.1f} fps  ·  {vr.duration:.1f} s")
            vr.release()
            if f is not None:
                self.view_vid.set_frame(f, fast=False)
            self.video_path = p
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", str(e))
        self._update_buttons()

    def play_video(self):
        if not self._need_pose() or not self.video_path:
            return
        rec = None
        if self.chk_rec_vid.isChecked():
            rec, _ = QFileDialog.getSaveFileName(self, "Lưu video kết quả", os.path.join(env.OUTPUT_DIR, "ketqua.mp4"), "MP4 (*.mp4)")
            if not rec:
                return
        self.prog_vid.setValue(0)
        self._launch(VideoWorker(self.pose, self.analyzer, self.cfg, self.video_path, rec), "video")

    def toggle_pause(self):
        if self.worker and self.running:
            self.worker.paused = not self.worker.paused
            self.btn_pause.setText("▶  Tiếp tục" if self.worker.paused else "⏸  Tạm dừng")

    # ---------------- Camera ----------------
    def scan_cams(self):
        self.statusBar().showMessage("Đang quét camera...")
        QApplication.processEvents()
        cams = list_cameras(5)
        self.cmb_cam.clear()
        self.cmb_cam.addItems([f"Camera {i}" for i in (cams or [0])])
        self.cmb_cam.setItemData(0, cams[0] if cams else 0)
        self._cams = cams or [0]
        self.statusBar().showMessage(f"Tìm thấy {len(cams)} camera: {cams}" if cams else "Không tìm thấy camera nào")

    def start_camera(self):
        if not self._need_pose():
            return
        idx = int(self.cmb_cam.currentText().split()[-1])
        rec = None
        if self.chk_rec_cam.isChecked():
            rec, _ = QFileDialog.getSaveFileName(self, "Lưu video kết quả", os.path.join(env.OUTPUT_DIR, "camera.mp4"), "MP4 (*.mp4)")
            if not rec:
                return
        self._launch(CameraWorker(self.pose, self.analyzer, self.cfg, idx, rec), "camera")

    def snapshot(self):
        if self.last_frame is not None:
            p = os.path.join(env.OUTPUT_DIR, time.strftime("snap_%Y%m%d_%H%M%S.png"))
            av.imwrite(p, self.last_frame)
            self.statusBar().showMessage(f"Đã chụp: {p}")

    # ---------------- Nhận kết quả ----------------
    def on_frame(self, frame, res, t, sample, ms, label):
        try:
            self._on_frame(frame, res, t, sample, ms, label)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage(f"Lỗi hiển thị: {e}")
        finally:
            if self.worker:
                self.worker.ui_busy = False

    def _on_frame(self, frame, res, t, sample, ms, label):
        view = {"anh": self.view_img, "video": self.view_vid, "camera": self.view_cam}[self.mode]
        view.set_frame(frame, fast=self.mode != "anh")
        if self.mode == "anh":
            with QSignalBlocker(self.lst):
                self.lst.setCurrentRow(int(t) - 1)
        self.last_frame = frame
        self.ms_avg = ms if self.ms_avg == 0 else 0.8 * self.ms_avg + 0.2 * ms
        self.cards["ms"].set(self.ms_avg)
        self.cards["fps"].set(1000.0 / max(self.ms_avg, 1e-3))
        vals = res.values() if res else {}
        for k in ("angle", "length", "resistance"):
            self.cards[k].set(vals.get(k))
        self.lbl_det.setText(f"Phát hiện · conf {res.conf:.2f}" if res else "Không phát hiện ngón tay")
        if self.mode == "video":
            self.lbl_time.setText(f"{t:.1f} s")
        if sample:
            self._record(t, label, res, vals)
        if self.worker:
            self.worker.ui_busy = False
        self.btn_snap.setEnabled(True)
        self.btn_save_cur.setEnabled(True)

    def _record(self, t, label, res, vals):
        if self.mode == "camera" and res is None:
            pass                                             # camera: không phát hiện thì bỏ qua dòng CSV
        else:
            self.counter += 1
            key = int(t) if self.mode == "anh" else self.counter
            row = {"nguon": self.mode, "ten_anh": label, "thoi_gian_s": "" if self.mode == "anh" else round(t, 3)}
            for src, dst in (("angle", "goc_tam_giac"), ("length", "do_dai_mm"),
                             ("resistance", "dien_tro_kohm")):
                v = vals.get(src)
                row[dst] = "" if v is None or v != v else round(v, 5)
            row["conf"] = round(res.conf, 3) if res else ""
            for i in range(3):
                row[f"x{i}"] = round(float(res.kpts[i, 0]), 1) if res else ""
                row[f"y{i}"] = round(float(res.kpts[i, 1]), 1) if res else ""
            self.logger.upsert(key, row)
            self.lbl_rows.setText(f"Dữ liệu: {len(self.logger)} dòng")
        if self.chart_on:
            self.chart.add(float(t), vals)

    # ---------------- Biểu đồ / CSV ----------------
    def toggle_chart(self, on):
        self.chart_on = on
        self.btn_chart.setText("Biểu đồ: BẬT" if on else "Biểu đồ: TẮT")

    def clear_data(self):
        self.logger.clear()
        self.chart.clear()
        self.counter = 0
        self.lbl_rows.setText("Dữ liệu: 0 dòng")

    def export_csv(self):
        if len(self.logger) == 0:
            QMessageBox.information(self, "Chưa có dữ liệu", "Hãy chạy dự đoán trước khi xuất CSV.")
            return
        p, _ = QFileDialog.getSaveFileName(self, "Xuất CSV", os.path.join(env.OUTPUT_DIR, time.strftime("data_%Y%m%d_%H%M%S.csv")), "CSV (*.csv)")
        if p:
            self.logger.save(p)
            self.statusBar().showMessage(f"Đã xuất {len(self.logger)} dòng -> {p}")

    def closeEvent(self, e):
        self._stop_worker_wait()
        super().closeEvent(e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(ui_style.QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())