"""程序入口 —— v2.0.0.

用法::

    # 图形界面 (默认)
    ParcelDetective.exe

    # 命令行 (无界面)
    ParcelDetective.exe --ai 面单.jpg [面单2.jpg ...] [--db x.db] [--model 模型名]
    ParcelDetective.exe --ai-check          # 只检查 Ollama 服务与默认模型
    ParcelDetective.exe --ai-check --model qwen3-vl:4b-instruct-q4_k_m
    ParcelDetective.exe --models            # 列出可选模型与安装状态
    ParcelDetective.exe --stats             # 打印信息库统计

v2.0.0 只保留与 VLM 相关的入口。v1 的 ``--scan`` (条码) /
``--selftest`` / ``--extract`` (OCR) 已随对应模块移除。
v2.1.0 增加 ``--models`` 与 ``--model``: 模型选择不再只能靠环境变量。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 允许 `python src/main.py` 直接跑
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# 输出被**重定向**（管道 / 写文件）时强制 UTF-8。
# 直接连控制台时保持系统代码页（中文 Windows 是 cp936）—— 否则交互显示会花屏。
# 这样两边都对:
#   · 人在控制台里看              -> 系统代码页, 中文正常
#   · 程序 --ai x.jpg > r.txt     -> UTF-8, 下游按 UTF-8 读不会乱码
if not sys.stdout.isatty():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass


def _arg_value(flag: str) -> str | None:
    """取 `--flag value` 里的 value."""
    argv = sys.argv
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            return argv[i + 1]
    for a in argv:
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def _run_ai(images: list[str], db_path: str | None, top_n: int = 3,
            auto_save: bool = False, out_path: str | None = None,
            model: str | None = None) -> int:
    """命令行跑一键检测.

    退出码: 0 = 全部"通过", 1 = 存在待确认/人工, 2 = 参数或运行错误。
    """
    import json

    from core import detect_pipeline as dp
    from core import model_registry as mr

    if not images:
        print("用法: main.py --ai 图片 [图片...] [--db x.db] [--top 3] "
              "[--model 模型名] [--save] [--out 结果.json]", file=sys.stderr)
        return 2

    model_name = mr.resolve_model(model or os.environ.get("PARCEL_AI_MODEL"))
    cfg = dp.DetectConfig(db_path=db_path, top_n=top_n, auto_save=auto_save,
                          ai_model=model_name)
    print(f"模型: {model_name}")
    results = []
    for p in images:
        r = dp.run_full_pipeline(p, cfg, log=lambda m: print(m, flush=True))
        results.append(r)
        print("")
        print(f"=== {Path(p).name} ===")
        if r.get("error"):
            print(f"  !! {r['error']}")
        print(f"  判定: {r['verdict_text']}　得分 {r['score']:.3f}")
        if r.get("tracking_number"):
            print(f"  单号: {r['tracking_number']}")
        print(f"  原因: {r['reason']}")
        t = r.get("timings") or {}
        print(f"  耗时: AI {t.get('ai_ms', 0):.0f} ms / 总 {t.get('total_ms', 0):.0f} ms")
        print(f"  建议: {r['advice']}")
        for c in (r.get("candidates") or [])[:top_n]:
            print(f"    候选 #{c.get('rank')}  {c.get('total_score', 0):.3f}  "
                  f"{c.get('tracking_number') or '(无单号)'}  "
                  f"{c.get('recipient_name') or '-'}")

    if out_path:
        try:
            Path(out_path).write_text(
                json.dumps(results, ensure_ascii=False, indent=2),
                encoding="utf-8")
            print(f"\n完整结果已写入: {out_path}")
        except Exception as exc:                     # noqa: BLE001
            print(f"写结果文件失败: {exc}", file=sys.stderr)

    n_pass = sum(1 for r in results if r.get("verdict") == "pass")
    print(f"\n汇总: 通过 {n_pass}/{len(results)}")
    return 0 if n_pass == len(results) else 1


def _run_ai_check(model: str | None = None) -> int:
    """只检查 Ollama 服务与模型是否就绪 (可用 --model 指定模型)."""
    from core import model_registry as mr

    st = mr.check_model(model)
    print(("OK  " if st.ok else "FAIL ") + st.message)
    if not st.ok:
        print("")
        print(mr.format_report(st))
    return 0 if st.ok else 1


def _run_models() -> int:
    """列出可选模型与当前安装状态."""
    from core import model_registry as mr

    ok, installed, err = mr.list_installed()
    print(f"Ollama 服务: {mr.DEFAULT_HOST}　" + ("可用" if ok else "不可用"))
    if not ok:
        print(f"  原因: {err}")
        print("  提示: 启动 Ollama 后重试（命令行执行 ollama serve）。")
    print("")
    print("可选模型:")
    for spec in mr.MODELS:
        if ok:
            mark = "已安装" if mr.is_model_installed(spec.name, installed) \
                else "未安装"
        else:
            mark = "未知（服务不可用）"
        default = "  [默认]" if spec.name == mr.DEFAULT_MODEL else ""
        print(f"  - {spec.name}{default}")
        print(f"      {spec.label}；{spec.hint}")
        print(f"      状态: {mark}")
    if ok:
        print("")
        print(f"已安装: {', '.join(installed) if installed else '(无)'}")
    return 0 if ok else 1


def _run_stats(db_path: str | None) -> int:
    """打印信息库统计."""
    from core import database as db

    p = db.resolve_db_path(db_path)
    print(f"数据库: {p}")
    try:
        db.init_database(p)
    except Exception as exc:                         # noqa: BLE001
        print(f"初始化失败: {exc}", file=sys.stderr)
        return 2
    s = db.stats(p)
    for k, v in s.items():
        print(f"  {k}: {v}")
    return 0


def main() -> int:
    # ---- 命令行模式 (有 --xxx 且不是界面) ----
    if "--ai-check" in sys.argv:
        return _run_ai_check(_arg_value("--model"))

    if "--models" in sys.argv:
        return _run_models()

    if "--stats" in sys.argv:
        return _run_stats(_arg_value("--db"))

    if "--ai" in sys.argv:
        i = sys.argv.index("--ai")
        used = {_arg_value("--db"), _arg_value("--out"), _arg_value("--top"),
                _arg_value("--model")}
        imgs = [a for a in sys.argv[i + 1:]
                if not a.startswith("--") and a not in used]
        try:
            top = int(_arg_value("--top") or 3)
        except ValueError:
            top = 3
        return _run_ai(imgs, _arg_value("--db"), top_n=top,
                       auto_save="--save" in sys.argv,
                       out_path=_arg_value("--out"),
                       model=_arg_value("--model"))

    # ---- 图形界面 ----
    from PyQt5.QtWidgets import QApplication
    from ui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("ParcelDetective")
    win = MainWindow(_arg_value("--db"))
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
