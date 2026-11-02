"""字段对照控件: 逐字段把"AI 识别值"和"信息库记录值"摆在同一行, 鼠标悬停高亮.

为什么要单独做
--------------
用户原话: "既然要想人工目测到就要使其放到同一行, 方便做对照对比, 并且框选
鼠标悬停在字段上的时候要高亮框选原数据库信息和对应的 AI 识别信息, 这样
对照显著, 更利于观察。"

所以这个控件的核心不是"显示", 而是**对照**:
  · 同一行左中右: 字段名 | AI 识别值 | 信息库值 | 相似度 | 状态
  · 鼠标移到某行 -> 整行高亮, 两个值单元格加粗描边
  · 状态配色: 一致=绿, 部分一致=橙, 不一致=红, 单边=灰

相似度是**现场算的** (直接调 matcher 的各 sim_* 函数), 不用 field_scores ——
后者只有加权后的 5 个组, 拿不到"快递公司""品名"这些单字段的分数。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

#: 对照字段的顺序 = 权重从大到小, 让用户先看到决定得分的项
COMPARE_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("tracking_number", "单号"),
    ("recipient_phone", "收件人电话"),
    ("recipient_address", "收件人地址"),
    ("recipient_name", "收件人姓名"),
    ("order_number", "订单号(与单号等价)"),
    ("courier_company", "快递公司"),
    ("sorting_code", "分拣码"),
    ("item_name", "品名"),
    ("sender_name", "寄件人姓名"),
    ("sender_phone", "寄件人电话"),
    ("sender_address", "寄件人地址"),
    ("cod_amount", "到付金额"),
    ("insured_amount", "保价金额"),
    ("shipping_date", "发货日期"),
    ("sender_branch", "寄件网点"),
)

_COLUMNS = (
    ("field", "字段", 130),
    ("ai", "AI 识别值", 260),
    ("db", "信息库值", 260),
    ("score", "相似度", 70),
    ("status", "状态", 90),
)

#: 状态 -> (文字, 颜色)
_STATUS_COLOR = {
    "一致": ("#1e8c1e", QColor(232, 248, 232)),
    "部分一致": ("#c87800", QColor(255, 246, 226)),
    "不一致": ("#b00000", QColor(255, 232, 232)),
    "仅 AI 有": ("#5a5a5a", QColor(244, 244, 244)),
    "仅信息库有": ("#5a5a5a", QColor(244, 244, 244)),
    "两边都空": ("#999999", QColor(250, 250, 250)),
}

_HL = QColor(190, 220, 255)          # 悬停高亮底色


def _s(v: Any) -> str:
    return "" if v is None else str(v).strip()


def _sim_of(field: str, ai: str, db: str) -> Optional[float]:
    """调 matcher 的相似度函数现场算分。拿不到就返回 None."""
    if not ai or not db:
        return None
    try:
        from core import matcher as m
    except Exception:                            # noqa: BLE001
        try:
            import matcher as m                  # type: ignore[no-redef]
        except Exception:                        # noqa: BLE001
            return None
    fn = {
        "tracking_number": m.sim_tracking,
        "order_number": m.sim_tracking,          # 与单号同一套规则
        "recipient_phone": m.sim_phone,
        "sender_phone": m.sim_phone,
        "recipient_address": m.sim_address,
        "sender_address": m.sim_address,
        "recipient_name": m.sim_name,
        "sender_name": m.sim_name,
    }.get(field, m.sim_generic)
    try:
        score, _reason = fn(ai, db)
    except Exception:                            # noqa: BLE001
        return None
    return None if score is None else float(score)


def _classify(ai: str, db: str, score: Optional[float]) -> str:
    if not ai and not db:
        return "两边都空"
    if not ai:
        return "仅信息库有"
    if not db:
        return "仅 AI 有"
    if score is None:
        return "不一致"
    if score >= 0.90:
        return "一致"
    if score >= 0.60:
        return "部分一致"
    return "不一致"


class FieldCompare(QWidget):
    """逐字段对照表 (带悬停高亮)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._hover_row = -1
        self._rows: List[Dict[str, Any]] = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        bar = QHBoxLayout()
        t = QLabel("字段对照（鼠标移到某行会整行高亮）")
        t.setStyleSheet("font-weight:bold;")
        bar.addWidget(t)
        bar.addStretch(1)
        self.lb_summary = QLabel("")
        self.lb_summary.setStyleSheet("color:#555;")
        bar.addWidget(self.lb_summary)
        lay.addLayout(bar)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels([c[1] for c in _COLUMNS])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.setTextElideMode(Qt.ElideRight)
        # 自适应: 字段名/相似度/状态按内容定宽, 两个"值"列平分剩余宽度。
        # 原来五列写死 810px, 面板不足 810px 时"相似度""状态"会被切掉。
        _h = self.table.horizontalHeader()
        _h.setSectionResizeMode(QHeaderView.ResizeToContents)
        _h.setMinimumSectionSize(52)
        _h.setSectionResizeMode(1, QHeaderView.Stretch)   # AI 识别值
        _h.setSectionResizeMode(2, QHeaderView.Stretch)   # 信息库值
        # itemEntered 要生效必须打开鼠标跟踪 (在 viewport 上)
        self.table.setMouseTracking(True)
        self.table.viewport().setMouseTracking(True)
        self.table.itemEntered.connect(self._on_hover)
        lay.addWidget(self.table, 1)

    # ------------------------------------------------------------------
    def load(self, ai_fields: Dict[str, Any], db_record: Dict[str, Any],
             field_scores: Optional[Dict[str, float]] = None) -> None:
        """填充对照表。任一侧为空都能照常显示 (只是状态不同)."""
        ai_fields = ai_fields or {}
        db_record = db_record or {}
        self._rows = []
        self._hover_row = -1

        for key, label in COMPARE_FIELDS:
            ai_v = _s(ai_fields.get(key))
            db_v = _s(db_record.get(key))
            if not ai_v and not db_v:
                continue                      # 两边都空的字段不占屏幕
            score = _sim_of(key, ai_v, db_v)
            status = _classify(ai_v, db_v, score)
            self._rows.append({
                "key": key, "label": label, "ai": ai_v, "db": db_v,
                "score": score, "status": status,
            })

        self._fill()
        self._update_summary()

    def _fill(self) -> None:
        bold = QFont()
        bold.setBold(True)
        self.table.setRowCount(len(self._rows))
        for row, r in enumerate(self._rows):
            text_color, bg = _STATUS_COLOR.get(
                r["status"], ("#333", QColor(255, 255, 255)))
            vals = (
                r["label"],
                r["ai"] or "—",
                r["db"] or "—",
                "—" if r["score"] is None else f"{r['score']:.2f}",
                r["status"],
            )
            for col, text in enumerate(vals):
                it = QTableWidgetItem(text)
                if col in (3, 4):
                    it.setTextAlignment(Qt.AlignCenter)
                if col == 0:
                    it.setFont(bold)
                if col == 4:
                    it.setForeground(QBrush(QColor(text_color)))
                # 值单元格给淡淡的底色, 一眼看出哪些字段有差异
                if col in (1, 2) and r["status"] in ("不一致", "部分一致"):
                    it.setBackground(QBrush(bg))
                if col in (1, 2):
                    it.setToolTip(text)
                self.table.setItem(row, col, it)

    def _update_summary(self) -> None:
        if not self._rows:
            self.lb_summary.setText("")
            return
        n = {"一致": 0, "部分一致": 0, "不一致": 0}
        solo = 0
        for r in self._rows:
            if r["status"] in n:
                n[r["status"]] += 1
            elif r["status"] in ("仅 AI 有", "仅信息库有"):
                solo += 1
        self.lb_summary.setText(
            f"共 {len(self._rows)} 项：一致 {n['一致']}、"
            f"部分一致 {n['部分一致']}、不一致 {n['不一致']}、单边 {solo}")

    # ------------------------------------------------------------------
    def _on_hover(self, item: QTableWidgetItem) -> None:
        """鼠标进入某行 -> 整行高亮, 并把两个值单元格加粗描边."""
        row = item.row()
        if row == self._hover_row:
            return
        self._clear_hover()
        self._hover_row = row
        bold = QFont()
        bold.setBold(True)
        for col in range(self.table.columnCount()):
            it = self.table.item(row, col)
            if it is None:
                continue
            it.setBackground(QBrush(_HL))
            if col in (1, 2):
                it.setFont(bold)

    def _clear_hover(self) -> None:
        row = self._hover_row
        if row < 0 or row >= len(self._rows):
            self._hover_row = -1
            return
        r = self._rows[row]
        _tc, bg = _STATUS_COLOR.get(r["status"], ("#333", QColor(255, 255, 255)))
        plain = QFont()
        plain.setBold(False)
        bold = QFont()
        bold.setBold(True)
        for col in range(self.table.columnCount()):
            it = self.table.item(row, col)
            if it is None:
                continue
            if col in (1, 2) and r["status"] in ("不一致", "部分一致"):
                it.setBackground(QBrush(bg))
            else:
                it.setBackground(QBrush(QColor(0, 0, 0, 0)))
            if col == 0:
                it.setFont(bold)
            elif col in (1, 2):
                it.setFont(plain)
        self._hover_row = -1

    def leaveEvent(self, event) -> None:          # noqa: N802 - Qt 命名
        self._clear_hover()
        super().leaveEvent(event)

    def clear(self) -> None:
        self._rows = []
        self._hover_row = -1
        self.table.setRowCount(0)
        self.lb_summary.setText("")
