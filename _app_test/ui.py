"""Các widget và bố cục giao diện của ứng dụng."""
import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFrame, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QListWidget, QProgressBar, QPushButton, QSlider,
                               QSplitter, QTabWidget, QVBoxLayout, QWidget)

import moitruong as env

pg.setConfigOptions(antialias=True, foreground="#8b90a8")


class ImageView(QLabel):
    def __init__(self, placeholder):
        super().__init__(placeholder)
        self.setObjectName("view")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(360, 270)
        from PySide6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._img = None
        self._fast = True

    def set_frame(self, bgr, fast=True):
        bgr = np.ascontiguousarray(bgr)
        h, w = bgr.shape[:2]
        self._img = QImage(bgr.data, w, h, 3 * w, QImage.Format.Format_BGR888).copy()
        self._fast = fast
        self._render()

    def _render(self):
        if self._img is None:
            return
        mode = Qt.TransformationMode.FastTransformation if self._fast else Qt.TransformationMode.SmoothTransformation
        self.setPixmap(QPixmap.fromImage(self._img).scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, mode))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._render()


class MetricCard(QFrame):
    def __init__(self, title, unit="", fmt="{:.1f}"):
        super().__init__()
        self.setObjectName("card")
        self.unit, self.fmt = unit, fmt
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(0)
        t = QLabel(title)
        t.setObjectName("cardTitle")
        self.v = QLabel("—")
        self.v.setObjectName("cardValue")
        lay.addWidget(t)
        lay.addWidget(self.v)

    def set(self, val):
        bad = val is None or (isinstance(val, float) and math.isnan(val))
        self.v.setText("—" if bad else self.fmt.format(val) + self.unit)


class ChartPanel(QWidget):
    """Biểu đồ 3 vùng dùng chung trục X, thang cố định:
       Góc 0-180° | Độ dài 0-50 mm | Điện trở 0-30 kΩ.
       Chế độ 'Chồng' vẽ cả 3 lên một đồ thị theo % thang đo để so sánh tương quan."""
    SERIES = [("angle", "Góc tam giác", 0), ("length", "Độ dài", 1), ("resistance", "Điện trở", 2)]
    RANGE = {"angle": 180.0, "length": 50.0, "resistance": 30.0}
    TIME_MAX = 300.0                                                   # trục thời gian: 0 - 300 s
    PANELS = [("Góc (°)", 180), ("Độ dài (mm)", 50), ("Điện trở (kΩ)", 30)]

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.curves, self.ov_curves, self.data, self.image_mode = {}, {}, {}, False
        self.scale = {"length": 1.0, "resistance": 1.0}      # hệ số đưa giá trị model về mm / kΩ
        self.overlay = False

        row = QHBoxLayout()
        row.setSpacing(8)
        self.plots = [self._mk_plot(t, hi) for t, hi in self.PANELS]
        self.ov_plot = self._mk_plot("% thang đo", 100)
        self.ov_plot.addLegend(offset=(8, 4), labelTextColor="#c9ccdd")
        self.plots[0].addLegend(offset=(8, 4), labelTextColor="#c9ccdd")
        for p in self.plots[1:] + [self.ov_plot]:
            p.setXLink(self.plots[0])
        for key, label, grp in self.SERIES:
            col = env.CHART_COLORS[key]
            self.curves[key] = self.plots[grp].plot(
                pen=pg.mkPen(col, width=2), name=label if grp == 0 else None, symbol=None,
                symbolBrush=col, symbolSize=7)
            if key in self.RANGE:
                self.ov_curves[key] = self.ov_plot.plot(
                    pen=pg.mkPen(col, width=2), name=f"{label} (0-{self.RANGE[key]:g})", symbol=None,
                    symbolBrush=col, symbolSize=7)
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.toggled.connect(lambda on, k=key: self._set_visible(k, on))
            cb.setStyleSheet(f"QCheckBox {{ color: {col}; }}")
            row.addWidget(cb)
        row.addStretch()
        lay.addLayout(row)
        for p, st in zip(self.plots, (3, 2, 2)):
            lay.addWidget(p, st)
        lay.addWidget(self.ov_plot, 1)
        self.dirty = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._redraw)
        self.timer.start(100)
        self.set_overlay(False)
        self.set_mode(False)

    @staticmethod
    def _mk_plot(ylabel, ymax):
        pw = pg.PlotWidget()
        pw.setBackground("#14151d")
        pw.showGrid(x=True, y=True, alpha=0.18)
        pw.setLabel("left", ylabel)
        pw.getAxis("left").setWidth(62)                       # canh thẳng hàng các vùng
        pw.setMouseEnabled(x=True, y=False)
        pw.setMenuEnabled(False)
        pw.hideButtons()
        pw.setYRange(0, ymax, padding=0.02)
        pw.enableAutoRange(axis="y", enable=False)
        pw.setLimits(yMin=-0.03 * ymax, yMax=1.03 * ymax)
        pw.setMinimumHeight(70)
        return pw

    def _set_visible(self, key, on):
        self.curves[key].setVisible(on)
        if key in self.ov_curves:
            self.ov_curves[key].setVisible(on)

    def set_overlay(self, on):
        self.overlay = on
        for p in self.plots:
            p.setVisible(not on)
        self.ov_plot.setVisible(on)
        self._update_axes()

    def set_scale(self, key, v):
        self.scale[key] = v
        self.dirty = True

    def _update_axes(self):
        # chỉ vùng dưới cùng hiện số trục X cho đỡ rối
        last = self.ov_plot if self.overlay else self.plots[-1]
        for p in self.plots + [self.ov_plot]:
            show = p is last
            p.getAxis("bottom").setStyle(showValues=show)
            p.getAxis("bottom").setHeight(None if show else 4)
            p.setLabel("bottom", ("Thứ tự ảnh" if self.image_mode else "Thời gian (s)") if show else None)

    def set_mode(self, image_mode):
        self.image_mode = image_mode
        for c in list(self.curves.values()) + list(self.ov_curves.values()):
            c.setSymbol("o" if image_mode else None)
        master = self.plots[0]                                # các vùng khác đã link theo vùng này
        if image_mode:                                        # thứ tự ảnh: từ 0, tự co giãn theo số ảnh
            master.setLimits(xMin=0, xMax=None)
            master.enableAutoRange(axis="x", enable=True)
        else:                                                 # thời gian: cố định 0 - 300 s
            master.setLimits(xMin=0, xMax=self.TIME_MAX)
            master.setXRange(0, self.TIME_MAX, padding=0)
            master.enableAutoRange(axis="x", enable=False)
        self._update_axes()

    def add(self, x, vals):
        self.data[x] = vals
        if not self.image_mode and len(self.data) > max(env.CHART_MAX_POINTS, 20000):
            self.data.pop(next(iter(self.data)))
        self.dirty = True

    def clear(self):
        self.data.clear()
        self.dirty = True

    def _redraw(self):
        if not self.dirty:
            return
        self.dirty = False
        xs = sorted(self.data) if self.image_mode else list(self.data)
        for key, c in self.curves.items():
            k = self.scale.get(key, 1.0)
            ys = np.array([self.data[x].get(key, float("nan")) for x in xs], dtype=float) * k
            c.setData(xs, ys, connect="finite")
            if key in self.ov_curves:
                self.ov_curves[key].setData(xs, ys / self.RANGE[key] * 100.0, connect="finite")


class MainWindowUI:
    def _build(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)
        main.addWidget(self._build_header())
        body = QSplitter(Qt.Orientation.Horizontal)
        body.setContentsMargins(14, 12, 14, 8)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_image(), "  Ảnh  ")
        self.tabs.addTab(self._tab_video(), "  Video  ")
        self.tabs.addTab(self._tab_camera(), "  Camera  ")
        body.addWidget(self.tabs)
        body.addWidget(self._build_side())
        body.setStretchFactor(0, 5)
        body.setStretchFactor(1, 4)
        body.setSizes([780, 700])
        main.addWidget(body, 1)
        self.statusBar().showMessage("Sẵn sàng. Nạp model Pose để bắt đầu.")

    def _build_header(self):
        h = QFrame()
        h.setObjectName("header")
        lay = QHBoxLayout(h)
        lay.setContentsMargins(20, 12, 20, 12)
        box = QVBoxLayout()
        box.setSpacing(0)
        t = QLabel("Finger Bend Analyzer")
        t.setObjectName("title")
        s = QLabel("YOLO Pose · man box + cogel 3 keypoint · CNN1D-GRU")
        s.setObjectName("subtitle")
        box.addWidget(t)
        box.addWidget(s)
        lay.addLayout(box)
        lay.addStretch()
        self.chip_pose, self.chip_gel = QLabel(), QLabel()
        b1, b2 = QPushButton("Nạp model Pose"), QPushButton("Nạp model CNN-GRU")
        b1.clicked.connect(self.pick_pose)
        b2.clicked.connect(self.pick_gel)
        for w in (self.chip_pose, b1, self.chip_gel, b2):
            lay.addWidget(w)
            lay.addSpacing(6)
        self._chips()
        return h

    def _chips(self):
        for chip, ok, text in ((self.chip_pose, self.pose is not None, "Pose"),
                               (self.chip_gel, self.gel.loaded, "CNN-GRU")):
            chip.setText(f"{text}: {'đã nạp' if ok else 'chưa nạp'}")
            chip.setObjectName("chipOk" if ok else "chipBad")
            chip.style().unpolish(chip)
            chip.style().polish(chip)

    @staticmethod
    def _btn(text, primary=False, danger=False):
        b = QPushButton(text)
        b.setProperty("primary", primary)
        b.setProperty("danger", danger)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        return b

    def _tab_image(self):
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(12, 12, 12, 12)
        left = QVBoxLayout()
        self.lst = QListWidget()
        self.lst.setFixedWidth(230)
        self.lst.currentRowChanged.connect(self.on_pick_image)
        self.btn_add, self.btn_dir = self._btn("Thêm ảnh"), self._btn("Thêm thư mục")
        self.btn_clear = self._btn("Xóa danh sách", danger=True)
        left.addWidget(QLabel("Danh sách ảnh (theo thứ tự đưa vào)"))
        left.addWidget(self.lst, 1)
        for b in (self.btn_add, self.btn_dir, self.btn_clear):
            left.addWidget(b)
        self.btn_add.clicked.connect(self.add_images)
        self.btn_dir.clicked.connect(self.add_folder)
        self.btn_clear.clicked.connect(self.clear_images)
        right = QVBoxLayout()
        self.view_img = ImageView("Thêm ảnh rồi chọn ảnh trong danh sách để dự đoán")
        right.addWidget(self.view_img, 1)
        row = QHBoxLayout()
        self.btn_run_all = self._btn("▶  Dự đoán tất cả", primary=True)
        self.btn_stop_img = self._btn("■  Dừng", danger=True)
        self.btn_save_cur = self._btn("Lưu ảnh hiện tại")
        self.chk_save_all = QCheckBox("Lưu ảnh kết quả vào thư mục khi chạy tất cả")
        self.btn_run_all.clicked.connect(self.run_all_images)
        self.btn_stop_img.clicked.connect(self.stop_worker)
        self.btn_save_cur.clicked.connect(self.save_current)
        for x in (self.btn_run_all, self.btn_stop_img, self.btn_save_cur):
            row.addWidget(x)
        row.addStretch()
        right.addLayout(row)
        self.prog_img = QProgressBar()
        right.addWidget(self.prog_img)
        lay.addLayout(left)
        lay.addLayout(right, 1)
        return w

    def _tab_video(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 12, 12, 12)
        top = QHBoxLayout()
        self.btn_open_vid = self._btn("Mở video…")
        self.lbl_vid = QLabel("Chưa chọn video")
        self.lbl_vid.setObjectName("muted")
        self.btn_open_vid.clicked.connect(self.open_video)
        top.addWidget(self.btn_open_vid)
        top.addWidget(self.lbl_vid, 1)
        lay.addLayout(top)
        self.view_vid = ImageView("Mở video để bắt đầu")
        lay.addWidget(self.view_vid, 1)
        row = QHBoxLayout()
        self.btn_play = self._btn("▶  Chạy", primary=True)
        self.btn_pause = self._btn("⏸  Tạm dừng")
        self.btn_stop_vid = self._btn("■  Dừng", danger=True)
        self.btn_play.clicked.connect(self.play_video)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_stop_vid.clicked.connect(self.stop_worker)
        for b in (self.btn_play, self.btn_pause, self.btn_stop_vid):
            row.addWidget(b)
        self.sp_vid = QDoubleSpinBox()
        self.sp_vid.setRange(0.1, 60.0)
        self.sp_vid.setDecimals(1)
        self.sp_vid.setSingleStep(0.5)
        self.sp_vid.setValue(self.cfg["video_sample"])
        self.sp_vid.valueChanged.connect(lambda v: self.cfg.update(video_sample=v))
        self.chk_pace = QCheckBox("Phát đúng nhịp thời gian")
        self.chk_pace.setChecked(True)
        self.chk_pace.toggled.connect(lambda v: self.cfg.update(pace=v))
        self.chk_smooth = QCheckBox("Phát mượt (hiện mọi khung hình)")
        self.chk_smooth.setChecked(True)
        self.chk_smooth.toggled.connect(lambda v: self.cfg.update(smooth=v))
        self.chk_rec_vid = QCheckBox("Ghi video kết quả")
        row.addStretch()
        lay.addLayout(row)
        prow = QHBoxLayout()
        self.prog_vid = QProgressBar()
        self.lbl_time = QLabel("0.0 s")
        self.lbl_time.setObjectName("muted")
        prow.addWidget(self.prog_vid, 1)
        prow.addWidget(self.lbl_time)
        lay.addLayout(prow)
        return w

    def _tab_camera(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 12, 12, 12)
        top = QHBoxLayout()
        top.addWidget(QLabel("Camera:"))
        self.cmb_cam = QComboBox()
        self.cmb_cam.addItems([f"Camera {i}" for i in range(3)])
        self.btn_scan = self._btn("Quét camera")
        self.btn_scan.clicked.connect(self.scan_cams)
        top.addWidget(self.cmb_cam)
        top.addWidget(self.btn_scan)
        top.addStretch()
        lay.addLayout(top)
        self.view_cam = ImageView("Bấm Bật camera")
        lay.addWidget(self.view_cam, 1)
        row = QHBoxLayout()
        self.btn_cam_on = self._btn("▶  Bật camera", primary=True)
        self.btn_cam_off = self._btn("■  Tắt", danger=True)
        self.btn_snap = self._btn("Chụp ảnh")
        self.btn_cam_on.clicked.connect(self.start_camera)
        self.btn_cam_off.clicked.connect(self.stop_worker)
        self.btn_snap.clicked.connect(self.snapshot)
        for b in (self.btn_cam_on, self.btn_cam_off, self.btn_snap):
            row.addWidget(b)
        self.sp_cam = QDoubleSpinBox()
        self.sp_cam.setRange(0.1, 60.0)
        self.sp_cam.setDecimals(1)
        self.sp_cam.setSingleStep(0.5)
        self.sp_cam.setValue(self.cfg["cam_sample"])
        self.sp_cam.valueChanged.connect(lambda v: self.cfg.update(cam_sample=v))
        self.chk_mirror = QCheckBox("Lật ngang")
        self.chk_mirror.setChecked(True)
        self.chk_mirror.toggled.connect(lambda v: self.cfg.update(mirror=v))
        self.chk_rec_cam = QCheckBox("Ghi video kết quả")
        row.addStretch()
        lay.addLayout(row)
        return w

    def _slider(self, lo, hi, val, fmt, on_change):
        row = QHBoxLayout()
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        lbl = QLabel(fmt(val))
        lbl.setMinimumWidth(44)
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        s.valueChanged.connect(lambda v: (lbl.setText(fmt(v)), on_change(v)))
        row.addWidget(s, 1)
        row.addWidget(lbl)
        return row

    def _build_settings(self):
        """Hộp thoại Cài đặt: gom ngưỡng, làm mượt, hệ số quy đổi đơn vị."""
        dlg = QDialog(self)
        dlg.setObjectName("dlg")
        dlg.setWindowTitle("Cài đặt")
        dlg.setMinimumWidth(460)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(6)

        gb = QGroupBox("Phát hiện")
        g = QVBoxLayout(gb)
        g.addWidget(QLabel("Ngưỡng confidence"))
        g.addLayout(self._slider(1, 99, int(self.cfg["conf"] * 100), lambda v: f"{v/100:.2f}",
                                 lambda v: self.cfg.update(conf=v / 100)))
        g.addWidget(QLabel("Ngưỡng IoU (NMS)"))
        g.addLayout(self._slider(10, 95, int(self.cfg["iou"] * 100), lambda v: f"{v/100:.2f}",
                                 lambda v: self.cfg.update(iou=v / 100)))
        g.addWidget(QLabel("Làm mượt góc (video/camera)"))
        g.addLayout(self._slider(0, 90, int(env.DEFAULT_SMOOTH * 100), lambda v: f"{v}%",
                                 lambda v: self.analyzer.set_smooth(v / 100)))
        self.chk_fill = QCheckBox("Tô nền đa giác")
        self.chk_fill.setChecked(True)
        self.chk_fill.toggled.connect(lambda v: self.cfg.update(fill=v))
        g.addWidget(self.chk_fill)
        lay.addWidget(gb)

        gb_img = QGroupBox("Ảnh")
        g_img = QVBoxLayout(gb_img)
        g_img.addWidget(self.chk_save_all)
        lay.addWidget(gb_img)

        gb_vid = QGroupBox("Video")
        g_vid = QGridLayout(gb_vid)
        g_vid.addWidget(QLabel("Lấy mẫu (ảnh/giây)"), 0, 0)
        g_vid.addWidget(self.sp_vid, 0, 1)
        g_vid.addWidget(self.chk_smooth, 1, 0, 1, 2)
        g_vid.addWidget(self.chk_pace, 2, 0, 1, 2)
        g_vid.addWidget(self.chk_rec_vid, 3, 0, 1, 2)
        g_vid.setColumnStretch(2, 1)
        lay.addWidget(gb_vid)

        gb_cam = QGroupBox("Camera")
        g_cam = QGridLayout(gb_cam)
        g_cam.addWidget(QLabel("Lấy mẫu dữ liệu (mẫu/giây)"), 0, 0)
        g_cam.addWidget(self.sp_cam, 0, 1)
        g_cam.addWidget(self.chk_mirror, 1, 0, 1, 2)
        g_cam.addWidget(self.chk_rec_cam, 2, 0, 1, 2)
        g_cam.setColumnStretch(2, 1)
        lay.addWidget(gb_cam)

        gb2 = QGroupBox("Thang biểu đồ (hệ số nhân đưa giá trị model về đơn vị hiển thị)")
        g2 = QGridLayout(gb2)
        for r, (key, text, hint) in enumerate((("length", "Độ dài → mm", "trục 0 – 50 mm"),
                                               ("resistance", "Điện trở → kΩ", "trục 0 – 30 kΩ"))):
            sp = QDoubleSpinBox()
            sp.setDecimals(4)
            sp.setRange(0.0001, 100000)
            sp.setValue(1.0)
            sp.valueChanged.connect(lambda v, k=key: self.chart.set_scale(k, v))
            g2.addWidget(QLabel(text), r, 0)
            g2.addWidget(sp, r, 1)
            h = QLabel(hint)
            h.setObjectName("muted")
            g2.addWidget(h, r, 2)
        lay.addWidget(gb2)

        close = self._btn("Đóng", primary=True)
        close.clicked.connect(dlg.accept)
        lay.addWidget(close, 0, Qt.AlignmentFlag.AlignRight)
        return dlg

    def _build_side(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        grid = QGridLayout()
        grid.setSpacing(6)
        self.cards = {
            "angle": MetricCard("Góc tam giác", "°"), "length": MetricCard("Độ dài", " mm", "{:.3f}"),
            "resistance": MetricCard("Điện trở", " kΩ", "{:.3f}"),
            "fps": MetricCard("FPS suy luận", "", "{:.1f}"), "ms": MetricCard("Thời gian / khung", " ms", "{:.0f}"),
        }
        for i, c in enumerate(self.cards.values()):
            grid.addWidget(c, i // 4, i % 4)
        lay.addLayout(grid)
        self.lbl_det = QLabel("")
        self.lbl_det.setObjectName("muted")
        lay.addWidget(self.lbl_det)

        self.chart = ChartPanel()
        self.dlg_settings = self._build_settings()

        row = QHBoxLayout()
        row.setSpacing(6)
        self.btn_settings = self._btn("⚙  Cài đặt")
        self.btn_settings.clicked.connect(self.dlg_settings.show)
        self.btn_chart = self._btn("Biểu đồ: TẮT")
        self.btn_chart.setCheckable(True)
        self.btn_chart.toggled.connect(self.toggle_chart)
        self.btn_clear_chart = self._btn("Xóa biểu đồ")
        self.btn_clear_chart.clicked.connect(lambda: self.chart.clear())
        self.btn_csv = self._btn("Xuất CSV", primary=True)
        self.btn_csv.clicked.connect(self.export_csv)
        self.btn_clear_data = self._btn("Xóa dữ liệu", danger=True)
        self.btn_clear_data.clicked.connect(self.clear_data)
        for b in (self.btn_settings, self.btn_chart, self.btn_clear_chart, self.btn_csv, self.btn_clear_data):
            row.addWidget(b)
        lay.addLayout(row)

        row2 = QHBoxLayout()
        self.lbl_rows = QLabel("Dữ liệu: 0 dòng")
        self.lbl_rows.setObjectName("muted")
        row2.addWidget(self.lbl_rows)
        row2.addStretch()
        self.btn_overlay = self._btn("Xem: Tách")
        self.btn_overlay.setCheckable(True)
        self.btn_overlay.setToolTip("Chồng: vẽ góc ngón, độ dài, điện trở lên cùng một đồ thị (% thang đo) để xem tương quan")
        self.btn_overlay.toggled.connect(self._toggle_overlay)
        row2.addWidget(self.btn_overlay)
        lay.addLayout(row2)

        lay.addWidget(self.chart, 1)
        return w

    def _toggle_overlay(self, on):
        self.btn_overlay.setText("Xem: Chồng" if on else "Xem: Tách")
        self.chart.set_overlay(on)