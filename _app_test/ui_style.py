"""Giao diện ứng dụng: Qt Style Sheet (QSS)."""

QSS = """
* { font-family: 'Segoe UI', 'Inter', 'Roboto', sans-serif; font-size: 13px; color: #e6e8f0; }
QMainWindow, QWidget#root { background: #12131a; }
QWidget { background: transparent; }
QDialog#dlg { background: #12131a; }
QFrame#header { background: #1a1c27; border-bottom: 1px solid #2a2d3f; }
QLabel#title { font-size: 18px; font-weight: 700; color: #ffffff; }
QLabel#subtitle, QLabel#muted { color: #8b90a8; }
QLabel#chipOk { color: #5ee0a0; background: #1d2b27; border-radius: 10px; padding: 3px 10px; }
QLabel#chipBad { color: #ff8a9b; background: #2d1f26; border-radius: 10px; padding: 3px 10px; }
QTabWidget::pane { border: 1px solid #2a2d3f; border-radius: 12px; background: #181a24; top: -1px; }
QTabBar::tab { background: transparent; color: #8b90a8; padding: 10px 26px; margin-right: 4px;
               border-top-left-radius: 10px; border-top-right-radius: 10px; font-weight: 600; }
QTabBar::tab:selected { background: #181a24; color: #ffffff; border: 1px solid #2a2d3f; border-bottom: none; }
QTabBar::tab:hover:!selected { color: #c9ccdd; }
QFrame#card { background: #1d2030; border: 1px solid #2a2d3f; border-radius: 12px; }
QLabel#cardTitle { color: #8b90a8; font-size: 11px; }
QLabel#cardValue { font-size: 19px; font-weight: 700; color: #ffffff; }
QGroupBox { background: #181a24; border: 1px solid #2a2d3f; border-radius: 12px; margin-top: 14px;
            padding: 14px 10px 10px 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 6px; color: #a99bff; }
QPushButton { background: #252839; border: 1px solid #333754; border-radius: 8px; padding: 7px 14px; }
QPushButton:hover { background: #2e3248; border-color: #4a4f78; }
QPushButton:pressed { background: #1f2233; }
QPushButton:disabled { color: #5a5e75; background: #1a1c27; border-color: #232637; }
QPushButton[primary="true"] { background: #7c6cf0; border: none; color: white; font-weight: 600; }
QPushButton[primary="true"]:hover { background: #8f81f5; }
QPushButton[primary="true"]:disabled { background: #3a3566; color: #8a86b0; }
QPushButton[danger="true"] { background: #3a2230; border: 1px solid #6b2f45; color: #ff8a9b; }
QPushButton[danger="true"]:hover { background: #4a2a3b; }
QPushButton:checked { background: #7c6cf0; border: none; color: white; font-weight: 600; }
QListWidget { background: #14151d; border: 1px solid #2a2d3f; border-radius: 10px; padding: 4px; outline: none; }
QListWidget::item { padding: 6px 8px; border-radius: 6px; }
QListWidget::item:selected { background: #7c6cf0; color: white; }
QListWidget::item:hover:!selected { background: #222538; }
QSlider::groove:horizontal { height: 6px; background: #2a2d3f; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #7c6cf0; border-radius: 3px; }
QSlider::handle:horizontal { background: #ffffff; width: 16px; margin: -5px 0; border-radius: 8px; }
QSpinBox, QDoubleSpinBox, QComboBox { background: #14151d; border: 1px solid #2a2d3f; border-radius: 8px; padding: 5px 8px; }
QComboBox QAbstractItemView { background: #1d2030; selection-background-color: #7c6cf0; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 4px; border: 1px solid #4a4f78; background: #14151d; }
QCheckBox::indicator:checked { background: #7c6cf0; border-color: #7c6cf0; }
QProgressBar { background: #14151d; border: 1px solid #2a2d3f; border-radius: 7px; text-align: center; height: 14px; }
QProgressBar::chunk { background: #7c6cf0; border-radius: 6px; }
QStatusBar { background: #1a1c27; color: #8b90a8; }
QLabel#view { background: #0d0e14; border: 1px solid #2a2d3f; border-radius: 12px; color: #5a5e75; font-size: 15px; }
QSplitter::handle { background: transparent; width: 8px; }
"""