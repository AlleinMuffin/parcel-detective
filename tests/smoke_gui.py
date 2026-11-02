"""v2.0.0 界面冒烟测试: 两个页签 + 一键检测页的展示逻辑.

不连 Ollama —— 直接往页面里注入结果字典, 验证展示与交互。
真实 AI 链路由 ``test_detect_pipeline.py`` 的 ``TestRealWaybill`` 覆盖。
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from PyQt5.QtWidgets import QApplication        # noqa: E402

from core import database as db                 # noqa: E402
from core import detect_pipeline as dp          # noqa: E402
from ui.main_window import MainWindow           # noqa: E402

_APP = None
_COUNTER = {"pass": 0, "fail": 0}
_FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        _COUNTER["pass"] += 1
        print(f"  [OK]   {name}")
    else:
        _COUNTER["fail"] += 1
        _FAILS.append(f"{name} {detail}")
        print(f"  [FAIL] {name} {detail}")


def setUpModule() -> None:
    global _APP
    _APP = QApplication.instance() or QApplication(sys.argv)


def make_image(path: Path, w: int = 400, h: int = 520) -> Path:
    img = np.full((h, w, 3), 235, np.uint8)
    cv2.rectangle(img, (25, 25), (w - 25, h - 25), (255, 255, 255), -1)
    cv2.putText(img, "WAYBILL", (60, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                1.6, (20, 20, 20), 4)
    cv2.imencode(".png", img)[1].tofile(str(path))
    return path


def fake_result(image: str, verdict: str = dp.Verdict.PASS,
                score: float = 0.93, n_fields: int = 15) -> dict:
    """造一条 detect_pipeline 形状的结果."""
    fields = {}
    sample = {
        "tracking_number": "YT0713590525482", "order_number": "W202609228816",
        "recipient_name": "张伟", "recipient_phone": "13812341234",
        "recipient_address": "广东省深圳市南山区科技园",
        "sender_name": "李四", "sender_phone": "13900001111",
        "sender_address": "广东省广州市天河区", "courier_company": "圆通速递",
        "sorting_code": "130-17-101", "item_name": "鞋", "cod_amount": "0",
        "insured_amount": "0", "shipping_date": "2026-10-11",
        "sender_branch": "深圳南山网点",
    }
    for i, (k, v) in enumerate(sample.items()):
        if i >= n_fields:
            break
        fields[k] = v
    cands = [{
        "rank": i + 1, "id": i + 1,
        "tracking_number": f"YT07135905254{i + 82}",
        "recipient_name": "张伟", "total_score": round(score - 0.1 * i, 3),
        "confidence": "high" if i == 0 else "medium",
        "field_scores": {"tracking_number": 1.0 - 0.1 * i},
        "match_reason": "单号完全一致" if i == 0 else "单号后缀匹配",
        "image_path": image, "db_record": dict(fields),
    } for i in range(3)]
    return {
        "image_path": image, "ok": True, "error": "",
        "verdict": verdict, "verdict_text": dp.Verdict.TEXT[verdict],
        "advice": dp.Verdict.ADVICE[verdict],
        "tracking_number": fields.get("tracking_number", ""),
        "score": score, "candidates": cands, "ai_fields": fields,
        "timings": {"ai_ms": 8000.0, "total_ms": 8100.0},
        "ai": {"ai": fields, "match": {"candidates": cands}},
        "reason": "测试数据", "swapped_manually": False,
        "saved": {}, "logs": ["[一键检测] 测试"],
    }


class TestMainWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._d = tempfile.TemporaryDirectory()
        cls.dir = Path(cls._d.name)
        cls.db = cls.dir / "p.db"
        db.init_database(cls.db)
        cls.img = str(make_image(cls.dir / "waybill.png"))
        cls.win = MainWindow(db_path=cls.db)
        cls.win.resize(1400, 950)
        cls.win.show()
        _APP.processEvents()

    @classmethod
    def tearDownClass(cls):
        cls.win.close()
        cls._d.cleanup()

    # ---- 页签 ----
    def test_two_tabs_only(self):
        n = self.win.tabs.count()
        check("页签数量为 2", n == 2, f"count={n}")
        titles = [self.win.tabs.tabText(i) for i in range(n)]
        check("页签是「一键检测」「信息库」",
              titles == ["一键检测", "信息库"], str(titles))

    def test_removed_pages_absent(self):
        """条码页 / AI比对页已删除, 不该还能访问到."""
        for attr in ("barcode_tab", "match_tab"):
            check(f"已移除: {attr}", not hasattr(self.win, attr))
        titles = [self.win.tabs.tabText(i) for i in range(self.win.tabs.count())]
        for bad in ("条码", "AI识别与比对", "识别"):
            check(f"页签里没有「{bad}」", bad not in titles)

    def test_window_title(self):
        # 不写死具体小版本号 —— 改版本时不该再修测试
        import re as _re
        check("标题含 v2.x 版本号",
              bool(_re.search(r"v2\.\d+\.\d+", self.win.windowTitle())),
              self.win.windowTitle())
        check("标题不再叫 ParcelDeskewOCR 的旧描述",
              "字符提取" not in self.win.windowTitle(),
              self.win.windowTitle())

    # ---- 检测页控件 ----
    def test_detect_widgets(self):
        dt = self.win.detect_tab
        for a in ("btn_run", "btn_stop", "btn_swap", "lb_verdict", "lb_number",
                  "img_compare", "field_compare", "tabs", "tbl_fields",
                  "tbl_cands", "txt_advice", "console", "lb_stage",
                  "progress", "ck_autosave",
                  "btn_nav_prev", "btn_nav_next", "lb_nav"):
            check(f"控件存在: {a}", hasattr(dt, a), a)

    def test_upload_button_prominent(self):
        dt = self.win.detect_tab
        check("上传按钮文字正确", dt.btn_run.text() == "上传图片并识别",
              dt.btn_run.text())
        check("上传按钮够大 (高度>=50)", dt.btn_run.minimumHeight() >= 50,
              str(dt.btn_run.minimumHeight()))

    def test_verdict_is_large_font(self):
        dt = self.win.detect_tab
        check("判定文字是大字号 (>=20pt)",
              dt.lb_verdict.font().pointSize() >= 20,
              str(dt.lb_verdict.font().pointSize()))

    def test_swap_button_prominent(self):
        dt = self.win.detect_tab
        check("对调按钮高度>=36", dt.btn_swap.minimumHeight() >= 36,
              str(dt.btn_swap.minimumHeight()))
        # 注意: "初始禁用"这条不能在这里查 —— 本测试用的是共享窗口,
        # 按字母序 test_show_* 已经先跑过并启用了按钮。
        # 初始状态统一在 test_initial_state 里用独立实例验证。
        check("对调按钮有醒目配色", "background:" in dt.btn_swap.styleSheet())

    def test_initial_state(self):
        # ⚠️ 不能复用 self.win.detect_tab —— 同类的展示测试按字母序先跑,
        #    早把状态填满了。这里自建一个干净的实例。
        from ui.detect_tab import DetectTab
        dt = DetectTab(db_path=self.db)
        dt.resize(900, 700)
        dt.show()
        _APP.processEvents()
        check("初始判定文字为未识别", "尚未" in dt.lb_verdict.text(),
              dt.lb_verdict.text())
        check("字段表初始为空", dt.tbl_fields.rowCount() == 0)
        check("候选表初始为空", dt.tbl_cands.rowCount() == 0)
        check("对调按钮初始禁用", not dt.btn_swap.isEnabled())
        check("上一张/下一张初始禁用",
              not dt.btn_nav_prev.isEnabled() and not dt.btn_nav_next.isEnabled())
        check("初始时没有结果", dt._results == [])
        dt.close()

    # ---- 结果展示 ----
    def test_show_pass_result(self):
        dt = self.win.detect_tab
        dt._results = [fake_result(self.img, dp.Verdict.PASS, 0.93)]
        dt._index = 0
        dt._show(0)
        _APP.processEvents()
        check("大字判定=通过", dt.lb_verdict.text() == "通过", dt.lb_verdict.text())
        check("判定框用绿色", "#1e8c1e" in dt.lb_verdict.styleSheet(),
              dt.lb_verdict.styleSheet()[:60])
        check("字段表 15 行", dt.tbl_fields.rowCount() == 15,
              str(dt.tbl_fields.rowCount()))
        check("候选表 3 行 (Top-3)", dt.tbl_cands.rowCount() == 3,
              str(dt.tbl_cands.rowCount()))
        check("单号已显示", "YT0713590525482" in dt.lb_number.text(),
              dt.lb_number.text()[:60])
        check("建议非空", bool(dt.txt_advice.toPlainText().strip()))
        check("对调按钮可用 (有 AI 字段)", dt.btn_swap.isEnabled())
        lp = dt.img_compare.left.lb_img.pixmap()
        check("本次检测图已显示", lp is not None and not lp.isNull())

    def test_show_review_and_manual(self):
        dt = self.win.detect_tab
        for v, text, color in ((dp.Verdict.REVIEW, "待确认", "#c87800"),
                               (dp.Verdict.MANUAL, "人工", "#b00000")):
            dt._results = [fake_result(self.img, v, 0.5)]
            dt._index = 0
            dt._show(0)
            _APP.processEvents()
            check(f"判定文字={text}", dt.lb_verdict.text() == text,
                  dt.lb_verdict.text())
            check(f"判定配色={color}", color in dt.lb_verdict.styleSheet())

    def test_fields_blank_first_or_last(self):
        """空字段也要列出来(灰字), 让人知道缺什么."""
        dt = self.win.detect_tab
        dt._results = [fake_result(self.img, n_fields=3)]
        dt._index = 0
        dt._show(0)
        _APP.processEvents()
        check("字段表仍是 15 行 (空值也列)", dt.tbl_fields.rowCount() == 15,
              str(dt.tbl_fields.rowCount()))
        texts = [dt.tbl_fields.item(i, 1).text()
                 for i in range(dt.tbl_fields.rowCount())]
        check("空字段显示为破折号", "—" in texts)

    def test_image_rotation(self):
        """旋转现在在 ImageCompare 上 (每侧独立 + 可同步)."""
        dt = self.win.detect_tab
        dt._results = [fake_result(self.img)]
        dt._index = 0
        dt._show(0)
        _APP.processEvents()
        w0 = dt.img_compare.left.lb_img.pixmap().width()
        dt.img_compare.left.rotate(90)
        _APP.processEvents()
        check("旋转后角度=90°",
              dt.img_compare.left.current_rotation() == 90,
              str(dt.img_compare.left.current_rotation()))
        check("旋转后宽高变化",
              dt.img_compare.left.lb_img.pixmap().width() != w0)
        dt.img_compare.left.reset_rotation()
        _APP.processEvents()
        check("复位后角度=0°", dt.img_compare.left.current_rotation() == 0)

    # ---- 对调 ----
    def test_swap_reevaluates(self):
        dt = self.win.detect_tab
        dt._results = [fake_result(self.img, dp.Verdict.REVIEW, 0.72)]
        dt._index = 0
        dt._show(0)
        _APP.processEvents()
        before = (dt._results[0].get("ai_fields") or {}).get("recipient_name")
        dt.btn_swap.click()
        _APP.processEvents()
        after = (dt._results[0].get("ai_fields") or {}).get("recipient_name")
        check("对调后收件人变为原寄件人", before != after, f"{before}->{after}")
        check("对调结果标记了 swapped_manually",
              bool(dt._results[0].get("swapped_manually")))
        check("对调写入日志", "对调" in dt.console.toPlainText())

    def test_swap_disabled_without_fields(self):
        dt = self.win.detect_tab
        r = fake_result(self.img)
        r["ai_fields"] = {}
        dt._results = [r]
        dt._index = 0
        dt._show(0)
        _APP.processEvents()
        check("无 AI 字段时对调禁用", not dt.btn_swap.isEnabled())

    # ---- 多张切换 ----
    def test_navigation(self):
        dt = self.win.detect_tab
        dt._results = [fake_result(self.img, dp.Verdict.PASS),
                       fake_result(self.img, dp.Verdict.MANUAL, 0.3)]
        dt._index = 0
        dt._show(0)
        _APP.processEvents()
        check("导航显示 1/2", "1 / 2" in dt.lb_nav.text(), dt.lb_nav.text())
        check("单张时上一张禁用", not dt.btn_nav_prev.isEnabled())
        check("多张时下一张可用", dt.btn_nav_next.isEnabled())
        dt._step(1)
        _APP.processEvents()
        check("切到第 2 张后判定=人工", dt.lb_verdict.text() == "人工",
              dt.lb_verdict.text())
        check("第 2 张时上一张可用", dt.btn_nav_prev.isEnabled())

    # ---- 导航条 (批量时必须显眼) ----
    def test_nav_prominent(self):
        dt = self.win.detect_tab
        check("导航按钮高度>=36 (够点)",
              dt.btn_nav_prev.minimumHeight() >= 36,
              str(dt.btn_nav_prev.minimumHeight()))
        check("导航按钮宽度>=120",
              dt.btn_nav_prev.minimumWidth() >= 120,
              str(dt.btn_nav_prev.minimumWidth()))
        check("导航计数标签加粗",
              "bold" in dt.lb_nav.styleSheet(), dt.lb_nav.styleSheet()[:50])

    def test_nav_works_in_batch(self):
        dt = self.win.detect_tab
        dt._results = [fake_result(self.img, dp.Verdict.PASS),
                       fake_result(self.img, dp.Verdict.REVIEW, 0.7),
                       fake_result(self.img, dp.Verdict.MANUAL, 0.3)]
        dt._index = 0
        dt._show(0)
        _APP.processEvents()
        check("3 张时导航显示 1/3", "1 / 3" in dt.lb_nav.text(), dt.lb_nav.text())
        check("3 张时下一张可用", dt.btn_nav_next.isEnabled())
        check("第 1 张时上一张禁用", not dt.btn_nav_prev.isEnabled())
        dt._step(1)
        _APP.processEvents()
        check("点下一张后到 2/3", "2 / 3" in dt.lb_nav.text(), dt.lb_nav.text())
        check("第 2 张时上一张可用", dt.btn_nav_prev.isEnabled())
        dt._step(1)
        _APP.processEvents()
        check("到 3/3 时下一张禁用", not dt.btn_nav_next.isEnabled())

    # ---- 信息对照 (v2.0.1 恢复) ----
    def test_comparison_widgets_exist(self):
        dt = self.win.detect_tab
        check("图片对照控件存在", hasattr(dt, "img_compare"))
        check("字段对照控件存在", hasattr(dt, "field_compare"))
        titles = [dt.tabs.tabText(i) for i in range(dt.tabs.count())]
        check("右栏有两个页签", len(titles) == 2, str(titles))
        check("含「信息对照」页", "信息对照" in titles, str(titles))
        check("含「AI 提取字段」页", "AI 提取字段" in titles, str(titles))

    def test_comparison_populates(self):
        """有候选时, 两张图和字段对照都要填上."""
        dt = self.win.detect_tab
        dt._results = [fake_result(self.img, dp.Verdict.REVIEW, 0.72)]
        dt._index = 0
        dt._show(0)
        _APP.processEvents()
        lp = dt.img_compare.left.lb_img.pixmap()
        rp = dt.img_compare.right.lb_img.pixmap()
        check("检测图已填", lp is not None and not lp.isNull())
        check("库中图已填", rp is not None and not rp.isNull())
        check("字段对照有行", dt.field_compare.table.rowCount() > 0,
              str(dt.field_compare.table.rowCount()))
        check("字段对照有摘要统计",
              "共" in dt.field_compare.lb_summary.text(),
              dt.field_compare.lb_summary.text())

    def test_field_compare_hover_highlight(self):
        dt = self.win.detect_tab
        dt._results = [fake_result(self.img, dp.Verdict.REVIEW, 0.72)]
        dt._index = 0
        dt._show(0)
        _APP.processEvents()
        fc = dt.field_compare
        if fc.table.rowCount() == 0:
            check("字段对照有行 (悬停前提)", False)
            return
        fc._on_hover(fc.table.item(0, 0))
        _APP.processEvents()
        check("悬停整行高亮",
              fc.table.item(0, 1).background().color().name() == "#bedcff",
              fc.table.item(0, 1).background().color().name())
        check("悬停值单元格加粗", fc.table.item(0, 1).font().bold())

    def test_candidate_pick_switches_comparison(self):
        """点候选第 2 行 -> 对照对象跟着换."""
        dt = self.win.detect_tab
        dt._results = [fake_result(self.img, dp.Verdict.REVIEW, 0.72)]
        dt._index = 0
        dt._show(0)
        _APP.processEvents()
        if dt.tbl_cands.rowCount() < 2:
            check("候选有 2 行 (切换前提)", False)
            return
        dt.tbl_cands.selectRow(1)
        _APP.processEvents()
        check("切换候选后字段对照仍在",
              dt.field_compare.table.rowCount() > 0)

    # ---- 回归: 切图必须刷新对照区 ----
    def test_navigation_refreshes_comparison(self):
        """批量来回切换时, 左侧"本次检测图"必须每次都换.

        这是实测 bug: 从第 2 张切回第 1 张时, 左图仍停在第 2 张。
        断言用面板**实际加载的路径**(_path), 不用 pixmap 尺寸 ——
        尺寸受面板重排影响, 会误判。
        """
        dt = self.win.detect_tab
        img_a = str(make_image(self.dir / "a.png", 400, 520))
        img_b = str(make_image(self.dir / "b.png", 320, 640))
        dt._results = [
            fake_result(img_a, dp.Verdict.PASS, 0.92),
            fake_result(img_b, dp.Verdict.REVIEW, 0.72),
        ]
        dt._index = 0
        dt._show(0)
        _APP.processEvents()

        def loaded_left() -> str:
            return Path(dt.img_compare.left._path).name

        check("第1张左图=a.png", loaded_left() == "a.png", loaded_left())
        dt._step(1)
        _APP.processEvents()
        check("第2张左图=b.png", loaded_left() == "b.png", loaded_left())
        dt._step(-1)
        _APP.processEvents()
        check("回第1张左图必须回到 a.png (原来这里是坏的)",
              loaded_left() == "a.png", loaded_left())
        dt._step(1)
        _APP.processEvents()
        check("再到第2张左图=b.png", loaded_left() == "b.png", loaded_left())

    def test_navigation_refreshes_db_image(self):
        """右侧信息库图也要跟着换."""
        dt = self.win.detect_tab
        img_a = str(make_image(self.dir / "c.png", 400, 500))
        img_b = str(make_image(self.dir / "d.png", 380, 480))
        r1 = fake_result(str(make_image(self.dir / "e.png", 400, 520)))
        r1["candidates"][0]["image_path"] = img_a
        r2 = fake_result(str(make_image(self.dir / "f.png", 400, 520)))
        r2["candidates"][0]["image_path"] = img_b
        dt._results = [r1, r2]
        dt._index = 0
        dt._show(0)
        _APP.processEvents()
        check("第1张右图=c.png",
              Path(dt.img_compare.right._path).name == "c.png",
              Path(dt.img_compare.right._path).name)
        dt._step(1)
        _APP.processEvents()
        check("第2张右图=d.png",
              Path(dt.img_compare.right._path).name == "d.png",
              Path(dt.img_compare.right._path).name)

    # ---- 回归: 信息库没图的三种提示 ----
    def test_hint_when_no_candidate(self):
        dt = self.win.detect_tab
        r = fake_result(self.img)
        r["candidates"] = []
        dt._results = [r]
        dt._index = 0
        dt._show(0)
        _APP.processEvents()
        txt = dt.img_compare.right.lb_img.text()
        check("无候选 -> 提示未匹配到记录", "未匹配到" in txt, txt[:50])

    def test_hint_when_record_has_no_image(self):
        """记录匹配上了但没存图 —— 文案不能再说"未匹配到"."""
        dt = self.win.detect_tab
        r = fake_result(self.img)
        r["candidates"][0]["image_path"] = ""
        dt._results = [r]
        dt._index = 0
        dt._show(0)
        _APP.processEvents()
        R = dt.img_compare.right
        txt = R.lb_img.text()
        check("记录无图 -> 提示没有关联图片", "没有关联图片" in txt, txt[:60])
        check("记录无图 -> 不再说未匹配到", "未匹配到" not in txt, txt[:60])
        check("标题标注（无图）", "无图" in R.lb_title.text(), R.lb_title.text())

    def test_hint_when_image_file_missing(self):
        """有路径但文件不在 -> 明确说"找不到图片文件"."""
        dt = self.win.detect_tab
        r = fake_result(self.img)
        r["candidates"][0]["image_path"] = str(self.dir / "根本不存在的图.jpg")
        dt._results = [r]
        dt._index = 0
        dt._show(0)
        _APP.processEvents()
        txt = dt.img_compare.right.lb_img.text()
        check("文件缺失 -> 提示找不到图片文件", "找不到图片文件" in txt, txt[:60])

    # ---- 模型切换 (v2.1.0) ----
    def test_model_selector(self):
        from core import model_registry as mr
        dt = self.win.detect_tab
        check("模型下拉存在", hasattr(dt, "cb_model"))
        check("模型下拉 3 项", dt.cb_model.count() == 3,
              str(dt.cb_model.count()))
        names = [dt.cb_model.itemData(i)
                 for i in range(dt.cb_model.count())]
        check("模型清单与注册表一致", names == mr.model_names(), str(names))
        check("默认模型为 2b-instruct",
              dt.current_model() == mr.DEFAULT_MODEL, dt.current_model())

    def test_model_selection_reaches_config(self):
        from core import model_registry as mr
        dt = self.win.detect_tab
        dt.cb_model.blockSignals(True)
        try:
            dt.cb_model.setCurrentIndex(1)
            cfg = dt._config()
        finally:
            dt.cb_model.setCurrentIndex(0)
            dt.cb_model.blockSignals(False)
        check("所选模型进入 DetectConfig",
              cfg.ai_model == mr.MODEL_NAMES[1], str(cfg.ai_model))
        check("恢复默认后配置跟随",
              dt._config().ai_model == mr.DEFAULT_MODEL)

    def test_model_status_reports(self):
        """检查失败只报告, 不终止程序: 标签变红 + 控制台报告 + 弹窗."""
        from core import model_registry as mr
        dt = self.win.detect_tab
        good = mr.ModelStatus(model=mr.DEFAULT_MODEL, service_ok=True,
                              model_present=True,
                              message="Ollama 可用, 模型 x 已就绪")
        dt._apply_model_status(good, notify=False)
        check("检查通过 -> 标签「已就绪」",
              dt.lb_model.text() == "已就绪", dt.lb_model.text())

        bad = mr.ModelStatus(model="qwen3-vl:8b-instruct-q4_k_m",
                             service_ok=True, model_present=False,
                             installed=["qwen3-vl:2b-instruct"],
                             message="服务可用, 但没有找到模型")
        dt.console.clear()
        with mock.patch("PyQt5.QtWidgets.QMessageBox.warning") as warn:
            dt._apply_model_status(bad, notify=True)
        text = dt.console.toPlainText()
        check("检查失败 -> 标签「不可用」",
              dt.lb_model.text() == "不可用", dt.lb_model.text())
        check("检查失败 -> 控制台输出异常报告",
              "模型可用性异常报告" in text, text[:60])
        check("检查失败 -> 给出 ollama pull 建议", "ollama pull" in text)
        check("检查失败 -> 弹出异常报告", warn.called)
        check("检查失败 -> 程序未退出（下拉仍可用）", dt.cb_model.isEnabled())

        dt._apply_model_status(good, notify=False)
        check("恢复 -> 标签回到「已就绪」", dt.lb_model.text() == "已就绪")

    def test_ensure_model_ready_blocks_when_missing(self):
        """模型不可用时取消本次识别, 但只弹报告, 不退出程序."""
        from core import model_registry as mr
        dt = self.win.detect_tab
        bad = mr.ModelStatus(model=dt.current_model(), service_ok=True,
                             model_present=False, installed=[],
                             message="服务可用, 但没有找到模型")
        with mock.patch.object(mr, "check_model", return_value=bad), \
                mock.patch("PyQt5.QtWidgets.QMessageBox.warning") as warn:
            ready = dt._ensure_model_ready()
        check("模型不可用 -> 拒绝开跑", ready is False)
        check("模型不可用 -> 弹出报告", warn.called)
        dt._model_status = None
        dt.lb_model.setText("尚未检查")

    # ---- 信息库页 ----
    def test_db_tab_widgets(self):
        db_tab = self.win.db_tab
        for a in ("table", "lb_preview", "btn_import_json", "btn_repair_img",
                  "btn_export_csv"):
            check(f"信息库控件存在: {a}", hasattr(db_tab, a), a)

    def test_status_bar(self):
        check("状态栏显示数据库", "数据库" in self.win.lb_db.text(),
              self.win.lb_db.text()[:60])


class TestNoRemovedModuleImport(unittest.TestCase):
    """src 里不该再 import 已删除的模块."""

    def test_no_removed_imports(self):
        import ast
        # 注意: image_compare / field_compare 在 v2.0.1 已**特意恢复**,
        # 不再属于"已删除模块"。
        banned = {"barcode_scanner", "pipeline", "ocr_engine", "deskew",
                  "fields", "imageops", "transforms", "barcode_tab",
                  "match_tab"}

        def _guarded_nodes(tree):
            """收集所有被 try/except 包住的节点 (可选导入)."""
            out = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    for child in ast.walk(node):
                        out.add(id(child))
            return out

        found = []
        for f in (ROOT / "src").rglob("*.py"):
            tree = ast.parse(f.read_text(encoding="utf-8"))
            guarded = _guarded_nodes(tree)
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [(node.module or "")] + [a.name for a in node.names]
                for n in names:
                    leaf = n.split(".")[-1]
                    if leaf in banned:
                        # database.py 里的 ``try: from . import fields`` 是
                        # **可选**导入: 失败就 _fields=None, 且 _fields 从未被
                        # 使用。database.py 属于"禁止修改的核心资产", 故放过。
                        if id(node) in guarded:
                            continue
                        found.append(f"{f.name}: {n}")
        check("没有 import 已删除模块 (可选导入除外)", not found, str(found[:5]))


def main() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in (TestMainWindow, TestNoRemovedModuleImport):
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=0)
    runner.run(suite)

    print()
    print("=" * 60)
    total = _COUNTER["pass"] + _COUNTER["fail"]
    print(f"  结果: {_COUNTER['pass']}/{total} 通过")
    if _FAILS:
        print("  失败项:")
        for f in _FAILS:
            print(f"    - {f}")
    print("=" * 60)
    return 0 if _COUNTER["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
