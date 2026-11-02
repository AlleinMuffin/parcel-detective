"""「一键检测」页 —— v2.0.1 结果区重做.

用户反馈两条, 对应两个改动:

1. **"批量检查时怎么没有下一张/上一张"**
   按钮其实一直在 (也叫得动), 但它混在"停止 / 自动入库"那一行里,
   又小又不显眼。现在提成**独立导航条**, 放在判定区正上方, 加宽加大。

2. **"信息对照 (检测结果与信息库对比) 怎么没了"**
   v2.0.0 "先瘦身"时把 `ImageCompare` / `FieldCompare` 一起删了。
   现在恢复:
     · **图片对照** —— 本次检测图 与 信息库该记录的面单 并排, 可独立/同步旋转
     · **字段对照** —— 逐字段 AI 值 vs 库值, 鼠标悬停整行高亮
   两者都跟着"选中的候选"变 —— 点 Top-3 里哪一行, 就对照哪一条。

布局::

    ┌────────────────────────────────────────────────┐
    │            [ 上传图片并识别 ]                   │
    │  识别模型: [2B Instruct ▾] [检查]  已就绪        │  ← v2.1.0 新增
    │  状态: 正在启动AI视觉识别...                    │
    ├────────────────────────────────────────────────┤
    │ ◀ 上一张 │ 第 1 / 4 张 │ 下一张 ▶   [⇄ 对调]   │  ← 醒目导航
    ├────────────────────────────────────────────────┤
    │        通过 / 待确认 / 人工   (26pt 大字)       │  ← 整行
    │  单号·得分·耗时·原因        建议                │
    ├──────────────────────┬─────────────────────────┤
    │  图片对照             │ 信息对照 | AI 字段      │
    │ ┌──────┬──────┐      │ (悬停高亮)              │
    │ │检测图│库中图│      │                         │
    │ └──────┴──────┘      ├─────────────────────────┤
    │                      │ Top-3 匹配结果          │
    └──────────────────────┴─────────────────────────┘
    │  处理日志                                       │
    └────────────────────────────────────────────────┘
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSplitter, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from core import detect_pipeline as dp
from core import model_registry as mr
from ui.ai_vision_ui import DetectWorker
from ui.field_compare import FieldCompare
from ui.image_compare import ImageCompare

_IMG_EXTS = "图片 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*)"

#: 判定 -> 主色
_COLOR = {
    dp.Verdict.PASS: "#1e8c1e",
    dp.Verdict.REVIEW: "#c87800",
    dp.Verdict.MANUAL: "#b00000",
}

#: 15 个字段的中文标签
FIELD_LABELS: Dict[str, str] = {
    "tracking_number": "运单号 / 单号",
    "order_number": "订单号",
    "recipient_name": "收件人姓名",
    "recipient_phone": "收件人电话",
    "recipient_address": "收件人地址",
    "sender_name": "寄件人姓名",
    "sender_phone": "寄件人电话",
    "sender_address": "寄件人地址",
    "courier_company": "快递公司",
    "sorting_code": "分拣码 / 三段码",
    "item_name": "内件品名",
    "cod_amount": "到付金额",
    "insured_amount": "保价金额",
    "shipping_date": "发货日期",
    "sender_branch": "寄件网点",
}

_CAND_COLUMNS = (
    ("rank", "排名", 46),
    ("total_score", "得分", 62),
    ("tracking_number", "单号", 150),
    ("recipient_name", "收件人", 84),
    ("recipient_phone", "电话", 112),
    ("courier_company", "快递公司", 92),
    ("match_reason", "匹配理由", 240),
)


def _read_image_bgr(path: Any):
    """读图 (兼容中文路径). cv2.imread 遇中文路径会静默返回 None."""
    if not path:
        return None
    p = Path(str(path))
    if not p.is_file():
        return None
    try:
        import numpy as np
        buf = np.fromfile(str(p), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:                            # noqa: BLE001
        try:
            return cv2.imread(str(p))
        except Exception:                        # noqa: BLE001
            return None


class _ModelCheckThread(QThread):
    """后台检查模型可用性.

    为什么要线程: ``ollama.Client.list()`` 在服务半死 (端口通但不回包) 时
    会一直等到超时。放界面线程里会让整个窗口假死 —— 一个「切换模型」的
    动作不该冻住界面。
    """

    done = pyqtSignal(object)          # ModelStatus

    def __init__(self, model: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._model = model

    def run(self) -> None:
        try:
            st = mr.check_model(self._model, timeout=5.0)
        except Exception as exc:                 # noqa: BLE001 - 界面不能崩
            st = mr.ModelStatus(
                model=self._model, service_ok=False, model_present=False,
                error=f"{type(exc).__name__}: {exc}",
                message=f"检查过程异常: {type(exc).__name__}: {exc}")
        self.done.emit(st)


class DetectTab(QWidget):
    """一键检测页."""

    def __init__(self, db_path: Optional[object] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.db_path = db_path
        self._results: List[Dict[str, Any]] = []
        self._index = 0
        #: 最近一次模型检查结果 (ModelStatus); 切换模型后清空重查
        self._model_status: Optional[mr.ModelStatus] = None
        self._model_thread: Optional[_ModelCheckThread] = None
        self._pending_notify = True
        self._build_ui()

    # ==================================================================
    # 界面
    # ==================================================================
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ---------- 顶部: 居中的大按钮 ----------
        self.btn_run = QPushButton("上传图片并识别")
        self.btn_run.setMinimumHeight(54)
        self.btn_run.setMinimumWidth(300)
        f = QFont()
        f.setPointSize(16)
        f.setBold(True)
        self.btn_run.setFont(f)
        self.btn_run.setStyleSheet(
            "QPushButton { background:#1f6feb; color:white; border-radius:9px;"
            " padding:10px 26px; }"
            "QPushButton:hover { background:#2b7fff; }"
            "QPushButton:disabled { background:#9bb8e0; }")
        self.btn_run.setToolTip(
            "选一张或多张面单图片，直接交给 VLM 识别并比对信息库。\n"
            "单张约 5~25 秒（取决于模型与图片复杂度）。\n"
            "多张时用下面的「上一张 / 下一张」逐张核对。")
        self.btn_run.clicked.connect(self.pick_and_run)
        root.addWidget(self.btn_run, 0, Qt.AlignHCenter)

        # ---------- 识别模型 (v2.1.0 新增) ----------
        model_row = QHBoxLayout()
        model_row.setSpacing(8)
        model_row.addStretch(1)

        lb_model = QLabel("识别模型")
        lb_model.setStyleSheet(
            "font-size:12px; color:#333; font-weight:bold;")
        model_row.addWidget(lb_model)

        self.cb_model = QComboBox()
        self.cb_model.setMinimumWidth(300)
        self.cb_model.setMinimumHeight(30)
        tips = []
        for spec in mr.MODELS:
            self.cb_model.addItem(spec.label, spec.name)
            tips.append(f"  {spec.name}：{spec.hint}")
        default_idx = self.cb_model.findData(mr.DEFAULT_MODEL)
        if default_idx >= 0:
            self.cb_model.setCurrentIndex(default_idx)
        self.cb_model.setToolTip(
            "可选的本地 VLM 模型（需先用 ollama pull 安装）。\n"
            + "\n".join(tips)
            + "\n\n切换后会自动检查 Ollama 服务与模型；检查失败只提示，不退出程序。")
        model_row.addWidget(self.cb_model)

        self.btn_model_check = QPushButton("检查")
        self.btn_model_check.setMinimumHeight(30)
        self.btn_model_check.setToolTip(
            "检查 Ollama 是否在运行、所选模型是否已安装。")
        model_row.addWidget(self.btn_model_check)

        self.lb_model = QLabel("尚未检查")
        self.lb_model.setMinimumWidth(150)
        self.lb_model.setStyleSheet("font-size:12px; color:#777;")
        self.lb_model.setToolTip("点击「检查」确认模型可用性。")
        model_row.addWidget(self.lb_model)

        model_row.addStretch(1)
        root.addLayout(model_row)

        self.cb_model.currentIndexChanged.connect(self._on_model_changed)
        self.btn_model_check.clicked.connect(
            lambda: self._check_model_async(notify=True))

        self.lb_stage = QLabel("就绪。点上面的「上传图片并识别」。")
        self.lb_stage.setAlignment(Qt.AlignCenter)
        f2 = QFont()
        f2.setPointSize(11)
        f2.setBold(True)
        self.lb_stage.setFont(f2)
        self.lb_stage.setStyleSheet("color:#1f6feb; padding:2px;")
        root.addWidget(self.lb_stage)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        # ---------- 导航条 (醒目, 独立一行) ----------
        nav = QHBoxLayout()
        nav.setSpacing(10)
        nav.addStretch(1)

        self.btn_nav_prev = QPushButton("◀ 上一张")
        self.btn_nav_prev.setMinimumHeight(38)
        self.btn_nav_prev.setMinimumWidth(130)
        self.btn_nav_prev.setFont(self._nav_font())
        self.btn_nav_prev.setEnabled(False)
        self.btn_nav_prev.clicked.connect(lambda: self._step(-1))
        nav.addWidget(self.btn_nav_prev)

        self.lb_nav = QLabel("尚未运行")
        self.lb_nav.setAlignment(Qt.AlignCenter)
        self.lb_nav.setMinimumWidth(140)
        self.lb_nav.setStyleSheet(
            "font-size:15px; font-weight:bold; color:#333;"
            "background:#f0f4fa; border:1px solid #cfd8e3;"
            "border-radius:6px; padding:6px;")
        nav.addWidget(self.lb_nav)

        self.btn_nav_next = QPushButton("下一张 ▶")
        self.btn_nav_next.setMinimumHeight(38)
        self.btn_nav_next.setMinimumWidth(130)
        self.btn_nav_next.setFont(self._nav_font())
        self.btn_nav_next.setEnabled(False)
        self.btn_nav_next.clicked.connect(lambda: self._step(1))
        nav.addWidget(self.btn_nav_next)

        nav.addSpacing(24)
        self.btn_swap = QPushButton("⇄ 对调收/发件方")
        self.btn_swap.setMinimumHeight(38)
        self.btn_swap.setMinimumWidth(170)
        fs = QFont()
        fs.setPointSize(11)
        fs.setBold(True)
        self.btn_swap.setFont(fs)
        self.btn_swap.setStyleSheet(
            "QPushButton { background:#8a4b00; color:white; border-radius:6px;"
            " padding:6px 14px; }"
            "QPushButton:hover { background:#a55c00; }"
            "QPushButton:disabled { background:#c9b8a6; }")
        self.btn_swap.setToolTip(
            "VLM 有时会把寄件方和收件方搞反。\n"
            "点这里对调后**重新比对并重判**（阈值与流水线同一套）。")
        self.btn_swap.setEnabled(False)
        self.btn_swap.clicked.connect(self._on_swap)
        nav.addWidget(self.btn_swap)

        nav.addStretch(1)
        sub = QHBoxLayout()
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_run)
        sub.addWidget(self.btn_stop)
        self.ck_autosave = QCheckBox("通过时自动入库")
        self.ck_autosave.setToolTip("只有判定为【通过】时才写入信息库。")
        sub.addWidget(self.ck_autosave)
        nav.addLayout(sub)
        root.addLayout(nav)

        # ---------- 判定区 (整行, 大字) ----------
        verdict_box = QGroupBox("判定结论")
        vb = QHBoxLayout(verdict_box)
        vb.setContentsMargins(10, 8, 10, 8)
        vb.setSpacing(12)

        self.lb_verdict = QLabel("尚未识别")
        fv = QFont()
        fv.setPointSize(26)
        fv.setBold(True)
        self.lb_verdict.setFont(fv)
        self.lb_verdict.setAlignment(Qt.AlignCenter)
        self.lb_verdict.setMinimumSize(340, 96)
        self.lb_verdict.setStyleSheet(
            "color:#777; background:#f7f7f7; border:3px solid #ddd;"
            "border-radius:8px;")
        vb.addWidget(self.lb_verdict)

        right_v = QVBoxLayout()
        right_v.setSpacing(4)
        self.lb_number = QLabel("")
        self.lb_number.setWordWrap(True)
        self.lb_number.setStyleSheet("font-size:12px; color:#222;")
        right_v.addWidget(self.lb_number)
        self.txt_advice = QPlainTextEdit()
        self.txt_advice.setReadOnly(True)
        self.txt_advice.setMaximumHeight(70)
        self.txt_advice.setStyleSheet("font-size:12px;")
        right_v.addWidget(self.txt_advice)
        vb.addLayout(right_v, 1)
        root.addWidget(verdict_box)

        # ---------- 主区: 图片对照 | (字段对照/AI字段 + 候选) ----------
        split = QSplitter(Qt.Horizontal)

        self.img_compare = ImageCompare()
        split.addWidget(self.img_compare)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)

        self.tabs = QTabWidget()

        # 页1: 信息对照 (字段级, 悬停高亮)
        self.field_compare = FieldCompare()
        self.tabs.addTab(self.field_compare, "信息对照")

        # 页2: AI 提取的 15 个字段
        fields_w = QWidget()
        fl = QVBoxLayout(fields_w)
        fl.setContentsMargins(4, 4, 4, 4)
        self.tbl_fields = QTableWidget(0, 2)
        self.tbl_fields.setHorizontalHeaderLabels(["字段", "值"])
        self.tbl_fields.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_fields.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_fields.setAlternatingRowColors(True)
        self.tbl_fields.setWordWrap(False)
        self.tbl_fields.setTextElideMode(Qt.ElideRight)
        _fh = self.tbl_fields.horizontalHeader()
        _fh.setSectionResizeMode(QHeaderView.ResizeToContents)
        _fh.setMinimumSectionSize(60)
        _fh.setSectionResizeMode(1, QHeaderView.Stretch)
        fl.addWidget(self.tbl_fields)
        self.tabs.addTab(fields_w, "AI 提取字段")
        rl.addWidget(self.tabs, 3)

        # 候选表
        cand_box = QGroupBox("Top-3 匹配结果（点某行可换成它来对照）")
        cb = QVBoxLayout(cand_box)
        cb.setContentsMargins(8, 6, 8, 6)
        self.tbl_cands = QTableWidget(0, len(_CAND_COLUMNS))
        self.tbl_cands.setHorizontalHeaderLabels([c[1] for c in _CAND_COLUMNS])
        self.tbl_cands.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_cands.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_cands.setAlternatingRowColors(True)
        self.tbl_cands.setWordWrap(False)
        self.tbl_cands.setTextElideMode(Qt.ElideRight)
        # 自适应: 短列按内容定宽, "匹配理由"占满剩余宽度。
        # 写死像素会在窗口偏窄时把右侧列挤出可视区 (截图已复现)。
        _ch = self.tbl_cands.horizontalHeader()
        _ch.setSectionResizeMode(QHeaderView.ResizeToContents)
        _ch.setMinimumSectionSize(44)
        _ch.setSectionResizeMode(len(_CAND_COLUMNS) - 1, QHeaderView.Stretch)
        self.tbl_cands.itemSelectionChanged.connect(self._on_candidate_pick)
        cb.addWidget(self.tbl_cands)
        rl.addWidget(cand_box, 2)

        split.addWidget(right)
        # 图片按面板高度等比缩放, 左栏窄一点也不影响看图;
        # 右栏要放三张表, 多分一些宽度。
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 6)
        split.setSizes([600, 900])
        root.addWidget(split, 1)

        # ---------- 日志 ----------
        log_box = QGroupBox("处理日志")
        lg = QVBoxLayout(log_box)
        lg.setContentsMargins(8, 4, 8, 4)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(100)
        self.console.setStyleSheet(
            "font-family:Consolas,monospace; font-size:12px;")
        lg.addWidget(self.console)
        root.addWidget(log_box)

    @staticmethod
    def _nav_font() -> QFont:
        f = QFont()
        f.setPointSize(11)
        f.setBold(True)
        return f

    # ==================================================================
    # 模型切换与可用性检查 (v2.1.0)
    # ==================================================================
    def current_model(self) -> str:
        """当前下拉选中的模型名 (传给 Ollama 的原始名字)."""
        data = self.cb_model.currentData()
        return str(data or mr.DEFAULT_MODEL)

    def _on_model_changed(self, _index: int = 0) -> None:
        """切换模型 -> 立即重查可用性.

        需求: 切换后先检查服务与模型; 失败只输出异常报告, 不终止程序。
        这里把旧结果清掉, 异步重查, 避免把上一个模型的结论套到新模型上。
        """
        self._model_status = None
        self._check_model_async(notify=True)

    def _check_model_async(self, notify: bool = True) -> None:
        """在后台线程里检查当前模型, 不冻界面."""
        th = self._model_thread
        if th is not None and th.isRunning():
            return
        model = self.current_model()
        self._pending_notify = bool(notify)
        self.lb_model.setText("检查中…")
        self.lb_model.setStyleSheet("font-size:12px; color:#1f6feb;")
        self.cb_model.setEnabled(False)
        self.btn_model_check.setEnabled(False)

        th = _ModelCheckThread(model, self)
        th.done.connect(self._on_model_checked)
        th.finished.connect(th.deleteLater)
        self._model_thread = th
        th.start()

    def _on_model_checked(self, status: mr.ModelStatus) -> None:
        self._model_thread = None
        self.cb_model.setEnabled(True)
        self.btn_model_check.setEnabled(True)
        self._apply_model_status(status, notify=self._pending_notify)

    def _apply_model_status(self, status: mr.ModelStatus,
                            notify: bool = True) -> None:
        """把检查结果落到界面: 状态标签 + 日志; 失败时按需弹异常报告."""
        self._model_status = status
        if status.ok:
            self.lb_model.setText("已就绪")
            self.lb_model.setStyleSheet(
                "font-size:12px; color:#1e8c1e; font-weight:bold;")
            self.lb_model.setToolTip(status.message)
            self._append_log(f"[模型] OK  {status.message}")
            return

        self.lb_model.setText("不可用")
        self.lb_model.setStyleSheet(
            "font-size:12px; color:#b00000; font-weight:bold;")
        report = mr.format_report(status)
        self.lb_model.setToolTip(report)
        for line in report.splitlines():
            self._append_log(line)
        if notify:
            QMessageBox.warning(self, "模型可用性异常", report)

    def _show_model_report(self, status: mr.ModelStatus, title: str) -> None:
        report = mr.format_report(status)
        for line in report.splitlines():
            self._append_log(line)
        QMessageBox.warning(self, title, report)

    def _ensure_model_ready(self) -> bool:
        """跑识别前的模型预检.

        返回 False 表示**取消本次识别** (不是退出程序)。
        5 分钟内的检查结果直接复用, 避免每张图都去问一次 Ollama。
        """
        model = self.current_model()
        st = self._model_status
        # 只缓存"通过"的结论: 失败后在 5 分钟内重新拉取了模型再点识别,
        # 应该重新检查, 而不是拿旧结论把人挡在门外。
        fresh = (st is not None and st.ok and st.model == model
                 and (time.time() - float(st.checked_at or 0)) < 300)
        if not fresh:
            self._append_log(f"[模型] 检查 {model} …")
            st = mr.check_model(model, timeout=3.0)
            self._apply_model_status(st, notify=False)
        if st.ok:
            return True
        self._show_model_report(st, "所选模型不可用，已取消本次识别")
        return False

    # ==================================================================
    # 运行
    # ==================================================================
    def _config(self) -> dp.DetectConfig:
        return dp.DetectConfig(db_path=self.db_path,
                               ai_model=self.current_model(),
                               auto_save=self.ck_autosave.isChecked())

    def pick_and_run(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择面单图片 (可多选)", "", _IMG_EXTS)
        if paths:
            self.start_run(paths)

    def start_run(self, paths: List[str]) -> None:
        if not paths:
            return
        # 模型不可用就别开跑: 否则要白等三次 90 秒超时。
        # 拒绝的只是这一次识别, 程序与已有结果都不受影响。
        if not self._ensure_model_ready():
            return
        self.btn_run.setEnabled(False)
        self.cb_model.setEnabled(False)
        self.btn_model_check.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(paths))
        self.progress.setValue(0)
        self.console.clear()
        self.lb_stage.setText("正在启动AI视觉识别...")
        self.lb_verdict.setText("识别中…")
        self.lb_verdict.setStyleSheet(
            "color:#777; background:#f7f7f7; border:3px solid #ddd;"
            "border-radius:8px;")

        self._thread = QThread()
        self._worker = DetectWorker(paths, self._config())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.stage.connect(self.lb_stage.setText)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def stop_run(self) -> None:
        if getattr(self, "_worker", None):
            self._worker.stop()
            self.lb_stage.setText("正在停止（当前这张跑完就停）…")

    def _finish(self) -> None:
        th = getattr(self, "_thread", None)
        if th:
            th.quit()
            th.wait(10000)
        self._thread = None
        self._worker = None
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.cb_model.setEnabled(True)
        self.btn_model_check.setEnabled(True)
        self.progress.setVisible(False)

    def _append_log(self, msg: str) -> None:
        self.console.appendPlainText(msg)
        sb = self.console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_progress(self, done: int, total: int, name: str) -> None:
        self.progress.setValue(done)
        if total > 1 and name:
            self.lb_stage.setText(f"处理中 {done + 1}/{total}  {name}")

    def _on_failed(self, msg: str) -> None:
        self._finish()
        self.lb_stage.setText("失败")
        QMessageBox.critical(self, "识别失败", msg)

    def _on_done(self, results: List[Dict[str, Any]]) -> None:
        self._finish()
        self._results = list(results or [])
        self._index = 0
        if not self._results:
            self.lb_stage.setText("处理完成（没有结果）")
            self._update_nav()
            return
        n = len(self._results)
        cnt = {v: 0 for v in dp.Verdict.TEXT}
        for r in self._results:
            cnt[r.get("verdict")] = cnt.get(r.get("verdict"), 0) + 1
        avg = sum((r.get("timings") or {}).get("total_ms", 0)
                  for r in self._results) / max(1, n)
        self.lb_stage.setText(
            f"处理完成　共 {n} 张：通过 {cnt[dp.Verdict.PASS]}、"
            f"待确认 {cnt[dp.Verdict.REVIEW]}、"
            f"人工 {cnt[dp.Verdict.MANUAL]}　平均 {avg:.0f} ms/张")
        self._show(0)

    # ==================================================================
    # 导航
    # ==================================================================
    def _update_nav(self) -> None:
        n = len(self._results)
        if n == 0:
            self.lb_nav.setText("尚未运行")
            self.btn_nav_prev.setEnabled(False)
            self.btn_nav_next.setEnabled(False)
            return
        self.lb_nav.setText(f"第 {self._index + 1} / {n} 张")
        self.btn_nav_prev.setEnabled(self._index > 0)
        self.btn_nav_next.setEnabled(self._index < n - 1)

    def _step(self, delta: int) -> None:
        n = len(self._results)
        if n == 0:
            return
        self._index = max(0, min(n - 1, self._index + delta))
        self._show(self._index)

    # ==================================================================
    # 展示
    # ==================================================================
    def _show(self, idx: int) -> None:
        if not (0 <= idx < len(self._results)):
            return
        self._index = idx
        r = self._results[idx]
        v = r.get("verdict") or dp.Verdict.MANUAL
        color = _COLOR.get(v, "#333")

        self.lb_verdict.setText(str(r.get("verdict_text") or ""))
        self.lb_verdict.setStyleSheet(
            f"color:{color}; background:#fbfbfb;"
            f"border:3px solid {color}; border-radius:8px;")

        # 第一行放短数值(单号/得分/耗时), 第二行放原因。
        # 原来每项各占一行, 四行文字在 74px 的框里会被裁掉。
        head = []
        if r.get("tracking_number"):
            head.append(f"单号：<b>{r['tracking_number']}</b>")
        head.append(f"得分：<b>{float(r.get('score') or 0):.3f}</b>")
        t = r.get("timings") or {}
        head.append(f"耗时：{t.get('total_ms', 0):.0f} ms")
        bits = ["　　".join(head)]
        if r.get("reason"):
            bits.append(f"原因：{r['reason']}")
        if r.get("error"):
            bits.append(f"<span style='color:#b00'>错误：{r['error']}</span>")
        self.lb_number.setText("<br>".join(bits))
        self.txt_advice.setPlainText(str(r.get("advice") or ""))

        self.btn_swap.setEnabled(bool(r.get("ai_fields")))

        self._fill_fields(r.get("ai_fields") or {})
        self._fill_candidates(r.get("candidates") or [])

        # 选中第一候选, 然后**显式**刷新对照区。
        # 不能只靠 selectRow 触发 itemSelectionChanged —— 选中行没真正变时
        # Qt 不发射该信号, 切回上一张就会出现"图还是上一张"的 bug。
        if self.tbl_cands.rowCount() > 0:
            self.tbl_cands.selectRow(0)
            self._on_candidate_pick()
        else:
            self._render_comparison(r, {})

        self._update_nav()

    def _fill_fields(self, fields: Dict[str, Any]) -> None:
        def _blank(v: Any) -> bool:
            return v in (None, "", [], {})

        rows = [(label, fields.get(key), _blank(fields.get(key)))
                for key, label in FIELD_LABELS.items()]
        rows.sort(key=lambda x: x[2])          # 有值的排前面

        self.tbl_fields.setRowCount(len(rows))
        bold = QFont()
        bold.setBold(True)
        for i, (label, val, blank) in enumerate(rows):
            it0 = QTableWidgetItem(label)
            it0.setFont(bold)
            self.tbl_fields.setItem(i, 0, it0)
            it1 = QTableWidgetItem("—" if blank else str(val))
            if blank:
                it1.setForeground(QBrush(QColor(160, 160, 160)))
            self.tbl_fields.setItem(i, 1, it1)

    def _fill_candidates(self, cands: List[Dict[str, Any]]) -> None:
        self.tbl_cands.blockSignals(True)
        # 先把选中态清掉: 让每次填表后 "selectRow(0)" 都成为一次真实变化,
        # 行为才可预期 (而不是碰运气看 Qt 会不会发信号)。
        self.tbl_cands.clearSelection()
        self.tbl_cands.setCurrentCell(-1, -1)
        self.tbl_cands.setRowCount(len(cands))
        bold = QFont()
        bold.setBold(True)
        for row, c in enumerate(cands):
            best = (c.get("rank") == 1)
            for col, (k, _h, _w) in enumerate(_CAND_COLUMNS):
                if k == "total_score":
                    text = f"{float(c.get(k) or 0):.3f}"
                else:
                    text = str(c.get(k, "") or "")
                it = QTableWidgetItem(text)
                if k in ("rank", "total_score"):
                    it.setTextAlignment(Qt.AlignCenter)
                if best:
                    it.setBackground(QBrush(QColor(222, 244, 222)))
                    it.setFont(bold)
                self.tbl_cands.setItem(row, col, it)
        self.tbl_cands.blockSignals(False)

    # ==================================================================
    # 信息对照 (图片 + 字段)
    # ==================================================================
    def _on_candidate_pick(self) -> None:
        """点候选某行 -> 换成该候选的库记录来对照."""
        if not (0 <= self._index < len(self._results)):
            return
        r = self._results[self._index]
        cands = r.get("candidates") or []
        row = self.tbl_cands.currentRow()
        cand = cands[row] if 0 <= row < len(cands) else {}
        self._render_comparison(r, cand)

    def _render_comparison(self, r: Dict[str, Any], cand: Dict[str, Any]) -> None:
        """填充图片对照 + 字段对照."""
        detected = r.get("image_path")
        db_img = cand.get("image_path") or ""
        label = cand.get("tracking_number") or ""

        # 右侧提示要分清三种情况, 不能一律说"未匹配到库中记录":
        #   1. 根本没匹配到记录
        #   2. 匹配到了, 但那条记录在信息库里没存图片
        #   3. 有路径但文件已经不在了 (这种情况由面板自己提示)
        if not cand:
            hint = "未匹配到库中记录，无对照图"
        elif not db_img:
            hint = ("该记录在信息库里没有关联图片\n"
                    "（可以在「信息库」页点『修复图片路径』补上）")
        else:
            hint = ""
        self.img_compare.load(detected, db_img, label, db_hint=hint)

        ai_fields = r.get("ai_fields") or {}
        db_rec = cand.get("db_record") or {}
        self.field_compare.load(ai_fields, db_rec,
                                cand.get("field_scores") or {})

    # ==================================================================
    # 对调
    # ==================================================================
    def _on_swap(self) -> None:
        if not (0 <= self._index < len(self._results)):
            return
        old = self._results[self._index]
        new = dp.swap_and_reevaluate(old, self.db_path,
                                     top_n=len(old.get("candidates") or []) or 3)
        if new.get("swap_error"):
            QMessageBox.warning(self, "对调失败", str(new["swap_error"]))
            return
        self._results[self._index] = new
        self._show(self._index)
        self._append_log(
            f"[一键检测] 已人工对调收/发件方 -> {new.get('verdict_text')} "
            f"({new.get('reason')})")
