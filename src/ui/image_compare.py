"""图片对照控件: 把"本次检测图"和"信息库记录图"并排放, 供人工目视核对.

为什么要单独做成控件
--------------------
1. 「AI识别与比对」页需要它, 以后「一键检测」页也可能复用
2. 两张图要**独立旋转**、独立缩放, 还要能双击看原图 —— 逻辑不少,
   塞进 match_tab 会让那个文件越来越难读

交互约定
--------
· 每侧各有 ⟲ 左转 / ⟳ 右转, **各自独立** —— 两张面单的拍摄方向常常不一样
· 「两侧同转」勾选后一次转两张, 需要整体摆正时更省事
· 双击某侧图 -> 用系统看图程序打开原图 (要看细节时)
· 图按面板大小等比缩放, 面板越大图越大; 不做放大插值, 免得糊
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSplitter,
    QVBoxLayout, QWidget,
)

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _read_image(path: Any):
    """读图 (兼容中文路径). 失败返回 None."""
    if not path:
        return None
    p = Path(str(path))
    if not p.is_file():
        return None
    try:
        import numpy as np
        buf = np.fromfile(str(p), dtype=np.uint8)      # cv2.imread 遇中文路径静默失败
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:                            # noqa: BLE001
        try:
            return cv2.imread(str(p))
        except Exception:                        # noqa: BLE001
            return None


def _rotate(img, angle: int):
    if not angle:
        return img
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


class _ImagePanel(QWidget):
    """单侧图片面板: 标题 + 旋转按钮 + 图 + 路径提示."""

    rotate_requested = pyqtSignal(int)       # +90 / -90
    reset_requested = pyqtSignal()

    def __init__(self, title: str, hint: str = "尚未载入",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._img = None                     # 原图 (BGR ndarray)
        self._rot = 0
        self._path = ""
        self.setMouseTracking(True)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        bar = QHBoxLayout()
        self.lb_title = QLabel(title)
        self.lb_title.setStyleSheet("font-weight:bold; font-size:13px;")
        bar.addWidget(self.lb_title)
        bar.addStretch(1)
        self.btn_l = QPushButton("⟲")
        self.btn_l.setFixedWidth(34)
        self.btn_l.setToolTip("逆时针转 90°")
        self.btn_l.clicked.connect(lambda: self.rotate_requested.emit(-90))
        bar.addWidget(self.btn_l)
        self.btn_r = QPushButton("⟳")
        self.btn_r.setFixedWidth(34)
        self.btn_r.setToolTip("顺时针转 90°")
        self.btn_r.clicked.connect(lambda: self.rotate_requested.emit(90))
        bar.addWidget(self.btn_r)
        self.btn_0 = QPushButton("复位")
        self.btn_0.setFixedWidth(46)
        self.btn_0.clicked.connect(self.reset_requested.emit)
        bar.addWidget(self.btn_0)
        self.lb_rot = QLabel("0°")
        self.lb_rot.setStyleSheet("color:#777;")
        self.lb_rot.setFixedWidth(34)
        bar.addWidget(self.lb_rot)
        lay.addLayout(bar)

        self.lb_img = QLabel(hint)
        self.lb_img.setAlignment(Qt.AlignCenter)
        self.lb_img.setMinimumSize(300, 340)
        self.lb_img.setStyleSheet(
            "background:#f2f2f2; border:1px solid #ccc; color:#888;")
        self.lb_img.setToolTip("双击用系统看图程序打开原图")
        lay.addWidget(self.lb_img, 1)

        self.lb_info = QLabel("")
        self.lb_info.setStyleSheet("color:#666; font-size:11px;")
        self.lb_info.setWordWrap(True)
        lay.addWidget(self.lb_info)

    # ------------------------------------------------------------------
    def set_image(self, path: Any, hint: str = "没有图片") -> None:
        self._path = str(path or "")
        self._img = _read_image(self._path)
        self._rot = 0
        self.lb_rot.setText("0°")
        if self._img is None:
            self.lb_img.setPixmap(QPixmap())
            if not self._path:
                # 没有路径: 用调用方给的文案 (它知道是"没匹配到"还是
                # "匹配到了但没存图")
                self.lb_img.setText(hint)
            elif not Path(self._path).is_file():
                # 有路径但文件不在 —— 这是"找不到图片文件", 不是"没图片"
                self.lb_img.setText(
                    f"找不到图片文件\n{self._path}\n\n"
                    f"文件可能被移动或删除；可在「信息库」页点"
                    f"『修复图片路径』重新指向")
            else:
                self.lb_img.setText(f"图片打不开（格式损坏？）\n{self._path}")
            self.lb_info.setText("")
            return
        h, w = self._img.shape[:2]
        self.lb_info.setText(f"{w} x {h}　{Path(self._path).name}")
        self._render()

    def rotate(self, delta: int) -> None:
        if self._img is None:
            return
        self._rot = (self._rot + delta) % 360
        self.lb_rot.setText(f"{self._rot}°")
        self._render()

    def reset_rotation(self) -> None:
        self._rot = 0
        self.lb_rot.setText("0°")
        self._render()

    def current_rotation(self) -> int:
        return self._rot

    # ------------------------------------------------------------------
    def _render(self) -> None:
        if self._img is None:
            return
        img = _rotate(self._img, self._rot)
        h, w = img.shape[:2]
        mw = max(160, self.lb_img.width() - 6)
        mh = max(160, self.lb_img.height() - 6)
        sc = min(mw / w, mh / h, 1.0)
        if sc < 1.0:
            img = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))),
                             interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hh, ww = rgb.shape[:2]
        qimg = QImage(rgb.data, ww, hh, 3 * ww, QImage.Format_RGB888)
        self.lb_img.setPixmap(QPixmap.fromImage(qimg.copy()))

    def resizeEvent(self, event) -> None:        # noqa: N802 - Qt 命名
        super().resizeEvent(event)
        self._render()

    def mouseDoubleClickEvent(self, event) -> None:   # noqa: N802
        """双击打开原图, 便于放大看细节."""
        if self._path and Path(self._path).is_file():
            try:
                import os
                os.startfile(self._path)         # noqa: S606 - Windows
            except Exception:                    # noqa: BLE001
                pass
        super().mouseDoubleClickEvent(event)


class ImageCompare(QWidget):
    """两图并排对照: 左=本次检测, 右=信息库记录."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        bar = QHBoxLayout()
        title = QLabel("图片对照（本次检测 vs 信息库记录）")
        title.setStyleSheet("font-weight:bold;")
        bar.addWidget(title)
        bar.addStretch(1)
        self.ck_sync = QCheckBox("两侧同转")
        self.ck_sync.setToolTip(
            "勾选后一次旋转两张图。两张面单拍摄方向不同时, 请取消勾选分别转。")
        bar.addWidget(self.ck_sync)
        self.lb_pair = QLabel("")
        self.lb_pair.setStyleSheet("color:#666;")
        bar.addWidget(self.lb_pair)
        lay.addLayout(bar)

        split = QSplitter(Qt.Horizontal)
        self.left = _ImagePanel("本次检测图", "运行 AI 识别后显示")
        self.right = _ImagePanel("信息库记录图", "匹配到记录后显示该记录的面单")
        split.addWidget(self.left)
        split.addWidget(self.right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setSizes([520, 520])
        lay.addWidget(split, 1)

        for panel in (self.left, self.right):
            panel.rotate_requested.connect(
                lambda d, p=panel: self._on_rotate(p, d))
            panel.reset_requested.connect(
                lambda p=panel: self._on_reset(p))

    # ------------------------------------------------------------------
    def _on_rotate(self, panel: _ImagePanel, delta: int) -> None:
        panel.rotate(delta)
        if self.ck_sync.isChecked():
            other = self.right if panel is self.left else self.left
            other.rotate(delta)

    def _on_reset(self, panel: _ImagePanel) -> None:
        panel.reset_rotation()
        if self.ck_sync.isChecked():
            other = self.right if panel is self.left else self.left
            other.reset_rotation()

    # ------------------------------------------------------------------
    def load(self, detected_path: Any, db_path: Any,
             db_label: str = "", db_hint: str = "") -> None:
        """载入两张图.

        ``db_hint`` 用来区分右侧"没有图"的具体原因 —— 调用方知道
        是"没匹配到记录"还是"匹配到了但没存图片", 面板自己判断不出来。
        """
        self.left.set_image(detected_path, "本次没有检测图")
        self.right.set_image(db_path, db_hint or "未匹配到库中记录，无对照图")
        # 标题保持简短: 运单号较长, 塞进标题会把标题挤断(截图已复现)。
        # 单号改放到下方的信息行, 那里本来就是放元信息的。
        title = "信息库记录图"
        if db_label and not db_path:
            title += "（无图）"
        self.right.lb_title.setText(title)
        if db_label:
            info = self.right.lb_info.text()
            self.right.lb_info.setText(
                f"对应记录 {db_label}" + (f"　{info}" if info else ""))
        same = bool(detected_path and db_path and
                    str(detected_path) == str(db_path))
        self.lb_pair.setText("（两侧是同一张图）" if same else "")

    def clear(self) -> None:
        self.left.set_image("", "本次没有检测图")
        self.right.set_image("", "未匹配到库中记录，无对照图")
        self.lb_pair.setText("")
