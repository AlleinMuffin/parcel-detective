"""「信息库」标签页 —— 查看 / 搜索 / 编辑 / 导出本地数据库.

设计说明
--------
- 表格直接**可编辑**: 双击单元格改完, 失焦即写入数据库。比"选中行再点编辑"
  少一步操作, 核对数据时快得多。
- 搜索框走 :func:`core.database.search_any`, 一个关键词跨多字段模糊匹配,
  不用先想"我要按哪个字段查"。
- 导出分两种: 导出**当前筛选结果** (核对用) 与导出**全库** (备份用)。
- 所有数据库异常都在这里兜住, 弹提示而不是让界面崩掉。
"""
from __future__ import annotations

from pathlib import Path

import cv2
from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QBrush, QColor
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QFileDialog, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from core import database as db

#: 表格显示的列: (数据库字段, 表头, 建议列宽)
COLUMNS: Tuple[Tuple[str, str, int], ...] = (
    ("id", "ID", 50),
    ("tracking_number", "单号", 150),
    ("recipient_name", "收件人", 80),
    ("recipient_phone", "收件人电话", 110),
    ("recipient_address", "收件人地址", 220),
    ("sender_name", "寄件人", 80),
    ("sender_phone", "寄件人电话", 110),
    ("sender_address", "寄件人地址", 200),
    ("sorting_code", "分拣码", 90),
    ("order_number", "订单号", 120),
    ("item_name", "内件品名", 100),
    ("cod_amount", "代收货款", 80),
    ("insured_amount", "保价金额", 80),
    ("shipping_date", "寄件日期", 100),
    ("courier_company", "快递公司", 90),
    ("sender_branch", "寄件网点", 120),
    ("is_damaged", "破损", 50),
    ("damage_type", "破损类型", 90),
    ("image_path", "图片路径", 240),
    ("created_at", "录入时间", 140),
)

#: 不允许在表格里直接改的列 (主键与自动生成的时间)
_READONLY_FIELDS = ("id", "created_at")


class DatabaseTab(QWidget):
    """信息库页面."""

    #: 暴露给外部 (测试与集成方) 的列定义, 等同模块级 :data:`COLUMNS`
    COLUMNS = COLUMNS

    def __init__(self, db_path: Optional[object] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.db_path = db_path
        self._rows: List[Dict[str, Any]] = []
        self._loading = False          # 刷新表格时抑制 itemChanged, 否则会回写数据库
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------
    # 界面搭建
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ---- 第一行: 搜索 + 筛选 ----
        bar = QHBoxLayout()
        bar.addWidget(QLabel("搜索:"))
        self.ed_search = QLineEdit()
        self.ed_search.setPlaceholderText(
            "输入任意片段即可, 例如单号后缀 75643、电话尾号 6368、地址关键词 武侯区")
        self.ed_search.returnPressed.connect(self.do_search)
        bar.addWidget(self.ed_search, 1)

        btn_search = QPushButton("搜索")
        btn_search.clicked.connect(self.do_search)
        bar.addWidget(btn_search)

        btn_clear = QPushButton("显示全部")
        btn_clear.clicked.connect(self._clear_search)
        bar.addWidget(btn_clear)

        self.ck_damaged = QCheckBox("只看破损")
        self.ck_damaged.stateChanged.connect(lambda _s: self.refresh())
        bar.addWidget(self.ck_damaged)
        root.addLayout(bar)

        # ---- 统计 ----
        self.lb_stats = QLabel("统计: -")
        self.lb_stats.setWordWrap(True)
        self.lb_stats.setStyleSheet("color:#444;")
        root.addWidget(self.lb_stats)

        # ---- 表格 ----
        box = QGroupBox("记录 (双击单元格可直接修改, 改完自动保存)")
        box_lay = QVBoxLayout(box)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([c[1] for c in COLUMNS])
        for i, (_f, _h, w) in enumerate(COLUMNS):
            self.table.setColumnWidth(i, w)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.table.setAlternatingRowColors(True)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        box_lay.addWidget(self.table)

        # 内嵌预览: 选中一行就能看到对应面单, 不用再去点"查看图片"
        self.lb_preview = QLabel("选中一行可预览面单图片")
        self.lb_preview.setAlignment(Qt.AlignCenter)
        self.lb_preview.setMinimumHeight(180)
        self.lb_preview.setMaximumHeight(260)
        self.lb_preview.setStyleSheet(
            "background:#fafafa; border:1px solid #ddd; color:#888;")
        box_lay.addWidget(self.lb_preview)
        self.table.itemSelectionChanged.connect(self._on_pick_preview)

        root.addWidget(box, 1)

        # ---- 底部按钮 ----
        bottom = QHBoxLayout()

        # 两个主操作单独放前面, 与"查看/删除"这类辅助操作区分开
        self.btn_import_json = QPushButton("导入 JSON 到数据库")
        self.btn_import_json.setToolTip(
            "可多选 .json 文件。支持单个对象、对象数组, 以及本程序导出的识别结果。\n"
            "单号已存在的记录会被跳过; 单个文件出错不影响其它文件。")
        self.btn_import_json.clicked.connect(self.import_json)
        bottom.addWidget(self.btn_import_json)

        self.btn_export_csv = QPushButton("导出 CSV")
        self.btn_export_csv.setToolTip(
            "把数据库全部内容导出为 parcels.csv (放在程序目录下)")
        self.btn_export_csv.clicked.connect(self.export_csv_default)
        bottom.addWidget(self.btn_export_csv)

        self.btn_repair_img = QPushButton("修复图片路径")
        self.btn_repair_img.setToolTip(
            "图片被挪过位置后, 库里记录的 image_path 会失效(打不开)。\n"
            "这里按**文件名主干**把全库路径重新指到真实文件:\n"
            "  parcel_0001.jpg <-> parcel_0001.json <-> 数据库那条记录")
        self.btn_repair_img.clicked.connect(self.repair_image_paths)
        bottom.addWidget(self.btn_repair_img)

        bottom.addSpacing(20)
        for text, slot in (
            ("刷新", self.refresh),
            ("导出当前结果 CSV", self.export_current),
            ("导出全库 CSV", self.export_all),
            ("查看图片", self.open_image),
            ("删除选中行", self.delete_selected),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            bottom.addWidget(b)
        bottom.addStretch(1)
        self.lb_status = QLabel("")
        bottom.addWidget(self.lb_status)
        root.addLayout(bottom)

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------
    def _query(self) -> List[Dict[str, Any]]:
        """按当前搜索条件取数据."""
        kw = self.ed_search.text().strip()
        only_damaged = self.ck_damaged.isChecked()
        try:
            if kw:
                rows = db.search_any(kw, path=self.db_path, limit=2000)
            else:
                rows = db.list_parcels(path=self.db_path, limit=2000)
            if only_damaged:
                rows = [r for r in rows if int(r.get("is_damaged") or 0) == 1]
        except Exception as exc:                      # noqa: BLE001 - 界面不能崩
            QMessageBox.warning(self, "读取失败", f"查询数据库出错:\n{exc}")
            return []
        return rows

    def refresh(self) -> None:
        """重新读库并刷新表格与统计."""
        self._rows = self._query()
        self._loading = True
        try:
            self.table.setRowCount(len(self._rows))
            for r, row in enumerate(self._rows):
                for c, (fld, _h, _w) in enumerate(COLUMNS):
                    item = QTableWidgetItem(self._display(row, fld))
                    if fld in _READONLY_FIELDS:
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                        item.setForeground(QBrush(QColor(120, 120, 120)))
                    if fld == "is_damaged" and int(row.get("is_damaged") or 0) == 1:
                        item.setForeground(QBrush(QColor(200, 40, 40)))
                    self.table.setItem(r, c, item)
        finally:
            self._loading = False

        self._update_stats()
        self.lb_status.setText(f"当前显示 {len(self._rows)} 条")

    @staticmethod
    def _display(row: Dict[str, Any], fld: str) -> str:
        """数据库值 -> 表格显示文本."""
        if fld == "is_damaged":
            return "是" if int(row.get("is_damaged") or 0) == 1 else "否"
        return str(row.get(fld, "") or "")

    def _update_stats(self) -> None:
        try:
            s = db.stats(self.db_path)
        except Exception as exc:                      # noqa: BLE001
            self.lb_stats.setText(f"统计不可用: {exc}")
            return
        dist = "、".join(f"{d['name']} {d['count']}" for d in s["by_courier"][:6]) or "无"
        parts = [
            f"总记录 <b>{s['total']}</b>",
            f"破损样本 <b>{s['damaged']}</b>",
            f"无单号 {s['missing_tracking']}",
            f"按快递公司: {dist}",
            f"数据库: {s['db_path']} ({s['db_size_kb']} KB)",
        ]
        self.lb_stats.setText("　|　".join(parts))

    # ------------------------------------------------------------------
    # 搜索
    # ------------------------------------------------------------------
    def do_search(self) -> None:
        self.refresh()

    def _clear_search(self) -> None:
        self.ed_search.clear()
        self.ck_damaged.setChecked(False)
        self.refresh()

    # ------------------------------------------------------------------
    # 就地编辑
    # ------------------------------------------------------------------
    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """单元格被改后写回数据库."""
        if self._loading:
            return
        r, c = item.row(), item.column()
        if r < 0 or r >= len(self._rows):
            return
        fld, header, _w = COLUMNS[c]
        if fld in _READONLY_FIELDS:
            return

        row = self._rows[r]
        pid = int(row["id"])
        raw = item.text().strip()

        if fld == "is_damaged":
            value: Any = 1 if raw in ("是", "1", "true", "True", "y", "Y") else 0
        else:
            value = raw

        if str(value) == str(row.get(fld, "") or ""):
            return                                   # 没变就不写

        try:
            if not db.update_parcel(pid, path=self.db_path, **{fld: value}):
                QMessageBox.warning(self, "未保存", f"未能更新 id={pid} 的 {header}")
                return
        except Exception as exc:                      # noqa: BLE001
            QMessageBox.warning(self, "保存失败", f"更新 {header} 出错:\n{exc}")
            self.refresh()
            return

        row[fld] = value
        if fld == "is_damaged":
            self._loading = True
            try:
                item.setText("是" if value == 1 else "否")
                item.setForeground(QBrush(QColor(200, 40, 40) if value == 1
                                          else QColor(0, 0, 0)))
            finally:
                self._loading = False
            self._update_stats()
        self.lb_status.setText(f"已保存 id={pid} 的 {header}")

    # ------------------------------------------------------------------
    # 导入 JSON / 导出 CSV
    # ------------------------------------------------------------------
    def import_json(self) -> None:
        """选 JSON 文件 (可多选) 并批量导入数据库."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择要导入的 JSON 文件", "",
            "JSON 文件 (*.json);;所有文件 (*)")
        if not paths:
            return

        self.lb_status.setText(f"正在导入 {len(paths)} 个文件……")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            rep = db.import_json_files(paths, path=self.db_path, log=self._log_like)
        except Exception as exc:                      # noqa: BLE001 - 界面不能崩
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "导入失败", f"导入过程中出错:\n{exc}")
            self.lb_status.setText("导入失败")
            return
        finally:
            # 若上面已 restore 过, 这里再调一次也安全 (Qt 会忽略多余的调用)
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()

        self.refresh()
        self.lb_status.setText(rep.summary_text())
        QMessageBox.information(self, "导入完成", rep.detail_text())

    def _log_like(self, msg: str) -> None:
        """把导入过程的日志回显到界面状态栏 (同时写入 stdout 便于排查)."""
        print(f"[信息库] {msg}", flush=True)

    def export_csv_default(self) -> None:
        """把全库导出为程序目录下的 parcels.csv."""
        try:
            dest = db.default_csv_path(self.db_path)
            n = db.export_csv(dest, path=self.db_path)
        except Exception as exc:                      # noqa: BLE001
            QMessageBox.critical(self, "导出失败", str(exc))
            return

        self.refresh()
        self.lb_status.setText(f"已导出 {n} 条到 {dest.name}")
        QMessageBox.information(
            self, "导出完成",
            f"已导出 {n} 条记录到:\n{dest}\n\n"
            "CSV 带 BOM, 用 Excel 直接双击打开不会乱码。")
        # 导出后顺手打开所在目录, 省得用户去找
        try:
            import os
            os.startfile(str(dest.parent))            # noqa: S606 - Windows 专用
        except Exception:                             # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # 导出 / 图片 / 删除
    # ------------------------------------------------------------------
    def export_current(self) -> None:
        if not self._rows:
            QMessageBox.information(self, "没有数据", "当前没有可导出的记录。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出当前结果", "parcels_当前结果.csv", "CSV 文件 (*.csv)")
        if not path:
            return
        try:
            n = db.export_csv(path, path=self.db_path, rows=self._rows)
            QMessageBox.information(self, "导出完成",
                                    f"已导出 {n} 条到:\n{path}\n\n"
                                    "CSV 带 BOM, 用 Excel 直接打开不会乱码。")
        except Exception as exc:                      # noqa: BLE001
            QMessageBox.warning(self, "导出失败", str(exc))

    def export_all(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "导出全库", "parcels_全库.csv", "CSV 文件 (*.csv)")
        if not path:
            return
        try:
            n = db.export_csv(path, path=self.db_path)
            QMessageBox.information(self, "导出完成", f"已导出 {n} 条到:\n{path}")
        except Exception as exc:                      # noqa: BLE001
            QMessageBox.warning(self, "导出失败", str(exc))

    def _on_pick_preview(self) -> None:
        """选中某行时, 在下方内嵌显示它的面单图片."""
        r = self.table.currentRow()
        if r < 0 or r >= len(self._rows):
            return
        p = str(self._rows[r].get("image_path") or "")
        if not p:
            self.lb_preview.setText("这条记录没有关联图片")
            self.lb_preview.setPixmap(_QPixmap())
            return
        # v2.0.0: 条码模块已移除, 这里改用本文件自带的读图函数。
        # (原先借用 barcode_scanner.imread_unicode, 只为兼容中文路径)
        img = _read_image_bgr(p)
        if img is None:
            self.lb_preview.setText(f"图片打不开:\n{p}\n\n可点「修复图片路径」")
            self.lb_preview.setPixmap(_QPixmap())
            return
        h, w = img.shape[:2]
        mw = max(160, self.lb_preview.width() - 8)
        mh = max(120, self.lb_preview.height() - 8)
        sc = min(mw / w, mh / h, 1.0)
        if sc < 1.0:
            img = _cv2resize(img, (int(w * sc), int(h * sc)))
        rgb = _cv2cvt(img, _cv2bgr)
        hh, ww = rgb.shape[:2]
        qimg = _QImage(rgb.data, ww, hh, 3 * ww, _QImgFmt)
        self.lb_preview.setPixmap(_QPixmap.fromImage(qimg.copy()))

    def repair_image_paths(self) -> None:
        """按文件名主干把全库失效的 image_path 重新指到真实文件."""
        d = QFileDialog.getExistingDirectory(
            self, "选择面单图片所在目录 (会递归查找)")
        if not d:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            rep = db.repair_image_paths(d, path=self.db_path)
        except Exception as exc:                     # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "修复失败", f"{type(exc).__name__}: {exc}")
            return
        QApplication.restoreOverrideCursor()
        self.refresh()
        self.lb_status.setText(rep.get("message", ""))
        detail = "\n".join(rep.get("details") or [])[:2000]
        QMessageBox.information(
            self, "修复完成",
            f"{rep.get('message', '')}\n\n{detail}\n\n"
            "(原库已在同目录备份为 .bak_before_repair)" if rep.get("ok")
            else str(rep.get("message", "")))

    def open_image(self) -> None:
        """用系统默认看图程序打开选中记录关联的原图."""
        r = self.table.currentRow()
        if r < 0 or r >= len(self._rows):
            QMessageBox.information(self, "未选中", "请先选中一行。")
            return
        p = str(self._rows[r].get("image_path") or "")
        if not p:
            QMessageBox.information(self, "无图片", "这条记录没有关联图片路径。")
            return
        if not Path(p).is_file():
            QMessageBox.warning(self, "图片不存在", f"文件已不在原位:\n{p}")
            return
        try:
            import os
            os.startfile(p)                           # noqa: S606 - Windows 专用
        except Exception as exc:                      # noqa: BLE001
            QMessageBox.warning(self, "打开失败", str(exc))

    def delete_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows:
            QMessageBox.information(self, "未选中", "请先选中要删除的行。")
            return
        ids = [int(self._rows[r]["id"]) for r in rows if r < len(self._rows)]
        ans = QMessageBox.question(
            self, "确认删除",
            f"确定删除这 {len(ids)} 条记录吗? 此操作不可撤销。\n"
            f"id: {', '.join(str(i) for i in ids[:20])}"
            + (" ..." if len(ids) > 20 else ""),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ans != QMessageBox.Yes:
            return
        ok = 0
        for pid in ids:
            try:
                ok += 1 if db.delete_parcel(pid, path=self.db_path) else 0
            except Exception as exc:                  # noqa: BLE001
                QMessageBox.warning(self, "删除失败", f"id={pid}: {exc}")
                break
        self.refresh()
        self.lb_status.setText(f"已删除 {ok} 条")



def _read_image_bgr(path):
    """读成 BGR ndarray (兼容中文路径).

    Windows 上 ``cv2.imread`` 遇到中文路径会**静默返回 None**, 不报错。
    必须走 ``np.fromfile + cv2.imdecode``。

    v2.0.0 起这个函数放在本文件里 —— 原来借用 barcode_scanner.imread_unicode,
    条码模块移除后不能再依赖它。
    """
    if not path:
        return None
    fp = Path(str(path))
    if not fp.is_file():
        return None
    try:
        import numpy as np
        buf = np.fromfile(str(fp), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:                            # noqa: BLE001
        try:
            return cv2.imread(str(fp))
        except Exception:                        # noqa: BLE001
            return None


# 就近别名, 让上面的预览代码不必写长前缀
_cv2resize = cv2.resize
_cv2cvt = cv2.cvtColor
_cv2bgr = cv2.COLOR_BGR2RGB
_QImage = QImage
_QPixmap = QPixmap
_QImgFmt = QImage.Format_RGB888
