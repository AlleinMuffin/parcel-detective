"""应用界面截图 (真实窗口渲染, 非模拟).

用 QWidget.grab() 截取真实构建出来的窗口 —— 不是拼图, 也是屏幕截图 API 之外
唯一可靠的办法 (不受远程桌面 / 多显示器 / 窗口遮挡影响)。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

V2 = Path(r"D:\parcel_detective\parcel_detective_V2\src")
sys.path.insert(0, str(V2))

from PyQt5.QtWidgets import QApplication      # noqa: E402

from core import database as db               # noqa: E402
from core import detect_pipeline as dp        # noqa: E402
from ui.main_window import MainWindow         # noqa: E402

OUT = Path(r"D:\parcel_detective\parcel_detective_V2\_shots")
OUT.mkdir(exist_ok=True)

PKG = Path(r"D:\parcel_detective\pic_data\面单包裹(1)\面单包裹\完好面单")


def fake_result(detected: str, db_img: str, tn: str, score: float,
                verdict: str = dp.Verdict.REVIEW) -> dict:
    """造一条结果, 让界面填满内容."""
    rec = {"tracking_number": tn, "recipient_name": "曾*",
           "recipient_phone": "*******2535",
           "recipient_address": "广东省茂名市化州市平定镇那宾生活超市",
           "sender_name": "覃**", "sender_phone": "138****5385",
           "courier_company": "中国邮政EMS", "sorting_code": "194-194-Y17-003",
           "order_number": "W202609228816", "item_name": "鞋舌贴",
           "image_path": db_img}
    cands = [{
        "rank": 1, "id": 1, "tracking_number": tn, "recipient_name": "曾*",
        "recipient_phone": "*******2535", "courier_company": "中国邮政EMS",
        "total_score": score, "confidence": "medium",
        "field_scores": {"tracking_number": 1.0, "recipient_phone": 0.0,
                         "recipient_address": 0.36, "recipient_name": 0.0},
        "match_reason": "单号完全一致；地址片段匹配；电话与姓名脱敏无法比对",
        "image_path": db_img, "db_record": rec,
    }]
    return {
        "image_path": detected, "ok": True, "error": "",
        "verdict": verdict, "verdict_text": dp.Verdict.TEXT[verdict],
        "advice": dp.Verdict.ADVICE[verdict],
        "tracking_number": tn, "score": score, "candidates": cands,
        "ai_fields": {"tracking_number": tn, "recipient_name": "曾德明",
                      "recipient_phone": "13812342535",
                      "recipient_address": "广东省茂名市化州市平定镇那宾生活超市",
                      "sender_name": "覃志强", "sender_phone": "13812345385",
                      "courier_company": "中国邮政EMS",
                      "sorting_code": "194-194-Y17-003",
                      "order_number": "W202609228816", "item_name": "鞋舌贴"},
        "timings": {"ai_ms": 4820.0, "total_ms": 4865.0},
        "ai": {}, "reason": "匹配得分 0.63 在 0.6~0.85 之间",
        "swapped_manually": False, "saved": {}, "logs": [],
    }


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)

    d = tempfile.TemporaryDirectory()
    dbp = Path(d.name) / "p.db"
    db.init_database(dbp)

    imgs = sorted(PKG.glob("*.jpg"))[:4]
    if len(imgs) < 2:
        print("  !! 样例图不足")
        return 2

    results = [
        fake_result(str(imgs[0]), str(imgs[2]), "9817883403197", 0.63),
        fake_result(str(imgs[1]), str(imgs[3]), "YT0718248356076", 0.94,
                    dp.Verdict.PASS),
    ]
    # 让状态栏的记录数与被展示的结果一致 (截图是 2 张 → 库里 2 条)
    for r in results:
        rec = dict((r.get("candidates") or [{}])[0].get("db_record") or {})
        rec["image_path"] = str((r.get("candidates") or [{}])[0].get("image_path") or "")
        if rec.get("tracking_number"):
            db.save_record(rec, path=dbp)

    win = MainWindow(db_path=dbp)
    win.resize(1600, 1000)
    win.show()
    app.processEvents()

    # 填充「一键检测」页
    tab = win.detect_tab
    tab._results = results
    tab._index = 0
    tab._show(0)
    tab.lb_stage.setText("处理完成　共 2 张：通过 1、待确认 1、人工 0　"
                         "平均 4865 ms/张")
    tab.tbl_cands.selectRow(0)

    # 模型行显示真实自检结果 (Ollama 在线时约 10 ms; 离线则显示不可用)
    from core import model_registry as mr
    st = mr.check_model()
    tab._apply_model_status(st, notify=False)
    app.processEvents()

    p1 = OUT / "ui-一键检测.png"
    win.grab().save(str(p1))
    print(f"  已保存: {p1}  ({p1.stat().st_size/1024:.0f} KB)")

    # 字段对照页（悬停高亮也一并截出）
    fc = tab.field_compare
    if fc.table.rowCount() > 0:
        fc._on_hover(fc.table.item(0, 0))
    app.processEvents()
    p2 = OUT / "ui-信息对照.png"
    win.grab().save(str(p2))
    print(f"  已保存: {p2}  ({p2.stat().st_size/1024:.0f} KB)")

    # 信息库页
    win.tabs.setCurrentIndex(1)
    app.processEvents()
    if win.db_tab.table.rowCount() > 0:
        win.db_tab.table.selectRow(0)
    app.processEvents()
    p3 = OUT / "ui-信息库.png"
    win.grab().save(str(p3))
    print(f"  已保存: {p3}  ({p3.stat().st_size/1024:.0f} KB)")

    win.close()
    d.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
