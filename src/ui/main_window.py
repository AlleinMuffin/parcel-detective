"""主窗口 —— v2.0.0 (只有两个页签).

::

    ┌────────────────────────────────────────────┐
    │  一键检测 │ 信息库                          │
    ├────────────────────────────────────────────┤
    │                                            │
    │         (当前页)                            │
    │                                            │
    └────────────────────────────────────────────┘
    状态栏: 数据库位置 + 记录数 + 版本

v1 的四个页签 (一键检测/条码/信息库/AI识别与比对) 精简为两个:
  · 条码页 —— 条码路线已放弃
  · AI识别与比对页 —— 匹配结果已并入「一键检测」, 不再单独成页

被移除页面的代码完整保留在 ``D:\\parcel_detective\\parcel_detective_V1``。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QLabel, QMainWindow, QMessageBox, QTabWidget, QWidget,
)

# 允许直接 python src/main.py 运行 (把 src 放进 sys.path)
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core import __version__ as APP_VERSION          # noqa: E402
from core import database as db                      # noqa: E402
from ui.db_tab import DatabaseTab                    # noqa: E402
from ui.detect_tab import DetectTab                  # noqa: E402


class MainWindow(QMainWindow):
    """主窗口."""

    def __init__(self, db_path: Optional[object] = None) -> None:
        super().__init__()
        self._db_path = db_path
        try:
            resolved = db.resolve_db_path(db_path)
            db.init_database(resolved)
            self._db_path = resolved
        except Exception:                            # noqa: BLE001
            pass                                     # 数据库不可用也要能开窗

        self.setWindowTitle(f"面单 VLM 识别与匹配  v{APP_VERSION}")
        # 默认给足宽度: 右侧要同时放字段对照、AI 字段、Top-3 三张表。
        # 再设最小尺寸, 防止用户把窗口缩到内容被裁切的程度。
        self.resize(1600, 1000)
        self.setMinimumSize(1280, 860)

        self.tabs = QTabWidget()

        # 页 1: 一键检测
        self.detect_tab = DetectTab(self._db_path)
        self.tabs.addTab(self.detect_tab, "一键检测")

        # 页 2: 信息库
        self.db_tab = DatabaseTab(self._db_path)
        self.tabs.addTab(self.db_tab, "信息库")

        # 切到信息库时刷新一下 (可能刚入库了新记录)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.setCentralWidget(self.tabs)
        self._build_status()

    # ------------------------------------------------------------------
    def _build_status(self) -> None:
        bar = self.statusBar()
        self.lb_db = QLabel("")
        bar.addWidget(self.lb_db)
        bar.addPermanentWidget(QLabel(f"v{APP_VERSION}"))
        self.refresh_status()

    def refresh_status(self) -> None:
        try:
            n = db.count_parcels(self._db_path)
            self.lb_db.setText(f"数据库: {db.resolve_db_path(self._db_path)}"
                               f"　({n} 条记录)")
        except Exception as exc:                     # noqa: BLE001
            self.lb_db.setText(f"数据库不可用: {exc}")

    def _on_tab_changed(self, index: int) -> None:
        if self.tabs.widget(index) is self.db_tab:
            try:
                self.db_tab.refresh()
            except Exception:                        # noqa: BLE001
                pass
        self.refresh_status()

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:             # noqa: N802 - Qt 命名
        """关窗前确认没有正在跑的识别."""
        worker = getattr(self.detect_tab, "_worker", None)
        if worker is not None:
            r = QMessageBox.question(
                self, "仍在识别",
                "还有识别任务在跑，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                event.ignore()
                return
        # 模型自检线程最多 5 秒; 不等它结束就销毁 QThread 会崩。
        check_thread = getattr(self.detect_tab, "_model_thread", None)
        if check_thread is not None and check_thread.isRunning():
            check_thread.wait(6000)
        super().closeEvent(event)


def main(argv: Optional[list[str]] = None) -> int:
    """启动界面."""
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setApplicationName("ParcelDetective")
    win = MainWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
