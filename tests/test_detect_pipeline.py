"""直连流水线 (v2.0.0) 的测试.

核心验证点:
  1. **判档只有三档**, 阈值就是 0.85 / 0.60
  2. **不再有 barcode 相关字段** (barcode_ms / barcode / boxes /
     barcode_fail_reason 必须全部消失)
  3. **状态回调只有三句**, 顺序固定
  4. 任何异常都不外抛
  5. 对调收/发件方后按同一套阈值重判
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from core import detect_pipeline as dp          # noqa: E402

#: 成功路径的状态回调 (三句, 顺序固定)
STAGES = [
    "正在启动AI视觉识别...",
    "AI识别完成，正在比对数据库...",
    "处理完成",
]

#: 失败路径 —— 第 2 句不能再说"识别完成" (那是谎报)
STAGES_FAILED = [
    "正在启动AI视觉识别...",
    "AI识别失败，未进入比对",
    "处理完成",
]


# ==========================================================================
# 假模块
# ==========================================================================
class _FakeVisionConfig:
    def __init__(self, **kw):
        self.kw = kw
    model = "fake-model"
    release_mode = "strict"


class _FakeAi:
    """假的 ai_vision 模块."""

    VisionConfig = _FakeVisionConfig

    def __init__(self, cands=None, fields=None, error=None):
        self._cands = cands if cands is not None else [_cand(0.92)]
        self._fields = fields if fields is not None else {
            "tracking_number": "YT0713590525482", "recipient_name": "张伟"}
        self._error = error
        self.calls = 0
        self.saved_calls = 0
        self.swapped = None
        self.last_config = None

    def extract_and_match(self, image_path, db_path=None, config=None,
                          top_n=3, log=None):
        self.calls += 1
        self.last_config = config
        ai = dict(self._fields)
        if self._error:
            ai["error"] = self._error
        return {
            "image": str(image_path), "image_name": Path(str(image_path)).name,
            "ai": ai,
            "match": {"candidates": list(self._cands), "match_type": "exact",
                      "confidence": "high", "needs_review": False,
                      "message": ""},
            "ok": True, "elapsed_ms": 8000, "saved": {},
        }

    def save_ai_result(self, result, db_path=None, on_duplicate="skip"):
        self.saved_calls += 1
        return {"status": "inserted", "message": "测试写入"}

    def swap_sender_recipient(self, fields):
        self.swapped = dict(fields)
        out = dict(fields)
        for a, b in (("recipient", "sender"), ("sender", "recipient")):
            for suf in ("name", "phone", "address"):
                ka, kb = f"{a}_{suf}", f"{b}_{suf}"
                out[ka], out[kb] = fields.get(kb), fields.get(ka)
        return out


def _cand(score: float, tracking: str = "YT0713590525482") -> dict:
    return {
        "rank": 1, "id": 1, "tracking_number": tracking,
        "recipient_name": "张伟", "total_score": score,
        "confidence": "high", "field_scores": {"tracking_number": 1.0},
        "match_reason": "单号完全一致",
        "image_path": "x.jpg", "db_record": {"tracking_number": tracking},
    }


def _run(fake_ai, config=None, progress=None, service_ok=True):
    """跑一次流水线.

    ``service_ok=False`` 时把预检 patch 成失败, 用来验证"服务不可用"分支。
    默认 True —— 单元测试不该依赖本机 Ollama 是否在运行。
    """
    probe = ((True, "ok") if service_ok
             else (False, "连不上 Ollama 服务（localhost:11434，TimeoutError）"))
    with mock.patch.object(dp, "_get_ai_module", return_value=fake_ai), \
            mock.patch.object(dp, "_get_matcher_module", return_value=mock.MagicMock(
                match_from_ai=lambda f, db, top_n=3: {
                    "candidates": [_cand(0.5)], "message": ""})), \
            mock.patch.object(dp, "probe_service", return_value=probe):
        return dp.run_full_pipeline("fake.jpg", config, progress=progress)


# ==========================================================================
# 1. 判定常量
# ==========================================================================
class TestVerdict(unittest.TestCase):
    def test_three_verdicts_only(self):
        self.assertEqual(dp.Verdict.PASS, "pass")
        self.assertEqual(dp.Verdict.REVIEW, "review")
        self.assertEqual(dp.Verdict.MANUAL, "manual")
        self.assertEqual(len(dp.Verdict.TEXT), 3)

    def test_chinese_text(self):
        self.assertEqual(dp.Verdict.TEXT[dp.Verdict.PASS], "通过")
        self.assertEqual(dp.Verdict.TEXT[dp.Verdict.REVIEW], "待确认")
        self.assertEqual(dp.Verdict.TEXT[dp.Verdict.MANUAL], "人工")

    def test_every_verdict_has_advice(self):
        for v in (dp.Verdict.PASS, dp.Verdict.REVIEW, dp.Verdict.MANUAL):
            self.assertTrue(dp.Verdict.ADVICE[v])

    def test_thresholds(self):
        self.assertEqual(dp.PASS_SCORE, 0.85)
        self.assertEqual(dp.REVIEW_SCORE, 0.60)


class TestDecide(unittest.TestCase):
    def _d(self, score, fields=True, cands=True):
        return dp._decide(score, fields, cands)[0]

    def test_boundaries(self):
        self.assertEqual(self._d(1.00), dp.Verdict.PASS)
        self.assertEqual(self._d(0.85), dp.Verdict.PASS)      # 含边界
        self.assertEqual(self._d(0.849), dp.Verdict.REVIEW)
        self.assertEqual(self._d(0.60), dp.Verdict.REVIEW)    # 含边界
        self.assertEqual(self._d(0.599), dp.Verdict.MANUAL)
        self.assertEqual(self._d(0.0), dp.Verdict.MANUAL)

    def test_no_fields_always_manual(self):
        """没提取到有效字段时, 哪怕有候选也不能算待确认."""
        self.assertEqual(self._d(0.99, fields=False), dp.Verdict.MANUAL)

    def test_no_candidates_always_manual(self):
        self.assertEqual(self._d(0.99, cands=False), dp.Verdict.MANUAL)


# ==========================================================================
# 2. 主流程
# ==========================================================================
class TestPipeline(unittest.TestCase):
    def test_pass(self):
        r = _run(_FakeAi(cands=[_cand(0.92)]))
        self.assertEqual(r["verdict"], dp.Verdict.PASS)
        self.assertEqual(r["verdict_text"], "通过")
        self.assertAlmostEqual(r["score"], 0.92)
        self.assertEqual(r["tracking_number"], "YT0713590525482")
        self.assertTrue(r["ok"])

    def test_review(self):
        r = _run(_FakeAi(cands=[_cand(0.72)]))
        self.assertEqual(r["verdict"], dp.Verdict.REVIEW)
        self.assertEqual(r["verdict_text"], "待确认")

    def test_manual_low_score(self):
        r = _run(_FakeAi(cands=[_cand(0.30)]))
        self.assertEqual(r["verdict"], dp.Verdict.MANUAL)
        self.assertEqual(r["verdict_text"], "人工")

    def test_manual_no_fields(self):
        r = _run(_FakeAi(fields={}))
        self.assertEqual(r["verdict"], dp.Verdict.MANUAL)
        self.assertIn("未提取", r["reason"])

    def test_manual_vlm_error(self):
        r = _run(_FakeAi(fields={}, error="模型没返回内容"))
        self.assertEqual(r["verdict"], dp.Verdict.MANUAL)
        self.assertIn("模型没返回内容", r["reason"])

    def test_no_candidates(self):
        r = _run(_FakeAi(cands=[]))
        self.assertEqual(r["verdict"], dp.Verdict.MANUAL)
        self.assertIn("没有相似记录", r["reason"])


class TestNoBarcodeResidue(unittest.TestCase):
    """v2.0.0 必须彻底不出现条码相关字段/耗时."""

    def test_result_has_no_barcode_keys(self):
        r = _run(_FakeAi())
        for k in ("barcode", "boxes", "barcode_fail_reason", "level"):
            self.assertNotIn(k, r, f"结果里不该再有 {k}")

    def test_timings_only_ai_and_total(self):
        r = _run(_FakeAi())
        self.assertEqual(sorted(r["timings"]), ["ai_ms", "total_ms"])

    def test_source_has_no_barcode_import(self):
        """用 AST 查真实代码, 而不是查文本 ——

        文档字符串里会提到 ``barcode_ms`` (讲 v1/v2 差异的对照表),
        那是说明文字, 不是残留代码。查文本会误报。
        """
        import ast
        src = (ROOT / "src" / "core" / "detect_pipeline.py").read_text(
            encoding="utf-8")
        tree = ast.parse(src)
        # 1) 不能 import barcode_scanner 之类的模块
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    self.assertNotIn("barcode", a.name.lower(),
                                     f"不该 import {a.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = (node.module or "").lower()
                self.assertNotIn("barcode", mod, f"不该 from {node.module} import")
        # 2) 代码里不能有 barcode_ms 这样的**字符串字面量用作字典键**
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                self.assertNotEqual(node.value, "barcode_ms",
                                    "不该再写 barcode_ms 键")


# ==========================================================================
# 3. 状态回调
# ==========================================================================
class TestProgress(unittest.TestCase):
    def test_exactly_three_stages_in_order(self):
        got = []
        _run(_FakeAi(), progress=got.append)
        self.assertEqual(got, STAGES)

    def test_three_stages_on_failure_too(self):
        """出错时也要把三句走完, 否则界面会卡在中间状态."""
        got = []
        _run(_FakeAi(fields={}, error="炸了"), progress=got.append)
        self.assertEqual(got, STAGES_FAILED)

    def test_failure_does_not_claim_success(self):
        """AI 失败时绝不能显示"AI识别完成" —— 那是谎报."""
        got = []
        _run(_FakeAi(fields={}, error="连不上"), progress=got.append)
        self.assertNotIn("AI识别完成，正在比对数据库...", got)

    def test_success_uses_success_stages(self):
        got = []
        _run(_FakeAi(), progress=got.append)
        self.assertEqual(got, STAGES)

    def test_callback_exception_swallowed(self):
        def boom(_m):
            raise RuntimeError("界面炸了")
        r = _run(_FakeAi(), progress=boom)
        self.assertEqual(r["verdict"], dp.Verdict.PASS)


# ==========================================================================
# 4. 健壮性
# ==========================================================================
class TestServiceUnavailable(unittest.TestCase):
    """服务没起时的行为 —— 必须快速失败且给出可操作提示.

    实测过的痛点: 修复前服务没起会烧 13 秒/张 (三次重试各等超时),
    且最后只说"VLM 未提取出有效字段", 用户看不出真正原因。
    """

    def test_fails_with_actionable_advice(self):
        r = _run(_FakeAi(), service_ok=False)
        self.assertEqual(r["verdict"], dp.Verdict.MANUAL)
        self.assertIn("服务未运行", r["reason"])
        self.assertIn("Ollama", r["advice"])
        self.assertIn("ollama serve", r["advice"])

    def test_does_not_call_ai_when_service_down(self):
        fake = _FakeAi()
        _run(fake, service_ok=False)
        self.assertEqual(fake.calls, 0,
                         "预检失败就不该再调 AI (那才是 13 秒的来源)")

    def test_reports_service_error_not_generic(self):
        """原因里要出现真实细节, 不能只有"未提取出有效字段"."""
        r = _run(_FakeAi(), service_ok=False)
        self.assertNotEqual(r["reason"], "VLM 未提取出有效字段")
        self.assertTrue(r["error"])

    def test_stage_says_failed(self):
        got = []
        _run(_FakeAi(), service_ok=False, progress=got.append)
        self.assertEqual(got, STAGES_FAILED)

    def test_probe_service_returns_tuple(self):
        ok, msg = dp.probe_service()
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(msg, str)
        self.assertTrue(msg)

    def test_probe_is_fast(self):
        """预检必须快 —— 端口过滤时也不能超过 2 秒."""
        import time
        t0 = time.perf_counter()
        dp.probe_service()
        dt = time.perf_counter() - t0
        self.assertLess(dt, 2.0, f"预检太慢: {dt:.2f}s")

    def test_split_host(self):
        self.assertEqual(dp._split_host("http://localhost:11434"),
                         ("localhost", 11434))
        self.assertEqual(dp._split_host("127.0.0.1:8080"),
                         ("127.0.0.1", 8080))
        self.assertEqual(dp._split_host("localhost"), ("localhost", 11434))
        self.assertEqual(dp._split_host(""), ("localhost", 11434))


class TestBatchPreflight(unittest.TestCase):
    """批量时服务不可用要一次性跳过, 不能每张各烧十几秒."""

    def test_batch_skips_all_when_service_down(self):
        with mock.patch.object(dp, "probe_service",
                               return_value=(False, "连不上")):
            rs = dp.run_full_pipeline_batch(["a.jpg", "b.jpg", "c.jpg"])
        self.assertEqual(len(rs), 3)
        for r in rs:
            self.assertEqual(r["verdict"], dp.Verdict.MANUAL)
            self.assertEqual(r["timings"]["ai_ms"], 0.0,
                             "跳过时不该有任何 AI 耗时")
            self.assertIn("服务未运行", r["reason"])

    def test_batch_skips_report_progress(self):
        seen = []
        with mock.patch.object(dp, "probe_service",
                               return_value=(False, "连不上")):
            dp.run_full_pipeline_batch(
                ["a.jpg", "b.jpg"],
                progress=lambda i, n, s: seen.append((i, n)))
        self.assertEqual(seen[0], (0, 2))
        self.assertEqual(seen[-1], (2, 2))


class TestRobustness(unittest.TestCase):
    def test_module_missing(self):
        with mock.patch.object(dp, "_get_ai_module", return_value=None):
            r = dp.run_full_pipeline("fake.jpg")
        self.assertFalse(r["ok"])
        self.assertEqual(r["verdict"], dp.Verdict.MANUAL)
        self.assertIn("ai_vision", r["error"])

    def test_extract_raises(self):
        class Boom(_FakeAi):
            def extract_and_match(self, *a, **k):
                raise RuntimeError("AI 炸了")
        r = _run(Boom())
        self.assertFalse(r["ok"])
        self.assertEqual(r["verdict"], dp.Verdict.MANUAL)
        self.assertIn("AI 炸了", r["error"])

    def test_never_raises(self):
        with mock.patch.object(dp, "_get_ai_module",
                               side_effect=RuntimeError("模块级爆炸")):
            r = dp.run_full_pipeline("fake.jpg")
        self.assertIsInstance(r, dict)
        self.assertIn("verdict", r)

    def test_none_path_safe(self):
        r = _run(_FakeAi())
        self.assertIn("verdict", r)


# ==========================================================================
# 5. 契约与耗时
# ==========================================================================
class TestContract(unittest.TestCase):
    def test_required_keys(self):
        r = _run(_FakeAi())
        for k in ("image_path", "ok", "error", "verdict", "verdict_text",
                  "advice", "tracking_number", "score", "candidates",
                  "ai_fields", "timings", "reason", "logs"):
            self.assertIn(k, r, f"缺少 {k}")

    def test_timings_non_negative(self):
        r = _run(_FakeAi())
        for k in ("ai_ms", "total_ms"):
            self.assertGreaterEqual(r["timings"][k], 0.0)

    def test_json_serialisable(self):
        json.dumps(_run(_FakeAi()), ensure_ascii=False)

    def test_logs_present(self):
        r = _run(_FakeAi())
        self.assertTrue(r["logs"])
        self.assertTrue(any("一键检测" in x for x in r["logs"]))

    def test_log_callback_fires(self):
        got = []
        fake = _FakeAi()
        with mock.patch.object(dp, "_get_ai_module", return_value=fake), \
                mock.patch.object(dp, "_get_matcher_module",
                                  return_value=mock.MagicMock()):
            r = dp.run_full_pipeline("fake.jpg", log=got.append)
        self.assertTrue(got, "log 回调必须真的收到内容")
        self.assertEqual(len(got), len(r["logs"]))


# ==========================================================================
# 6. 自动入库
# ==========================================================================
class TestAutoSave(unittest.TestCase):
    def test_saves_only_on_pass(self):
        fake = _FakeAi(cands=[_cand(0.92)])
        _run(fake, dp.DetectConfig(auto_save=True))
        self.assertEqual(fake.saved_calls, 1, "判定通过时应当入库")

    def test_does_not_save_on_review(self):
        fake = _FakeAi(cands=[_cand(0.72)])
        _run(fake, dp.DetectConfig(auto_save=True))
        self.assertEqual(fake.saved_calls, 0, "待确认不该自动入库")

    def test_no_save_when_disabled(self):
        fake = _FakeAi(cands=[_cand(0.95)])
        _run(fake, dp.DetectConfig(auto_save=False))
        self.assertEqual(fake.saved_calls, 0)


# ==========================================================================
# 7. 对调重判
# ==========================================================================
class TestSwap(unittest.TestCase):
    def _base(self):
        return {
            "image_path": "fake.jpg", "ok": True, "error": "",
            "verdict": dp.Verdict.REVIEW, "verdict_text": "待确认",
            "advice": "x", "tracking_number": "YT1", "score": 0.7,
            "candidates": [_cand(0.7)], "timings": {},
            "ai": {"ai": {"recipient_name": "李四", "sender_name": "张三",
                          "tracking_number": "YT1"}, "match": {}},
            "ai_fields": {"recipient_name": "李四", "sender_name": "张三"},
            "reason": "", "logs": [],
        }

    def test_swaps_fields(self):
        r = dp.swap_and_reevaluate(self._base(), None)
        self.assertEqual(r["ai_fields"]["recipient_name"], "张三")
        self.assertEqual(r["ai_fields"]["sender_name"], "李四")
        self.assertTrue(r["swapped_manually"])

    def test_does_not_mutate_input(self):
        src = self._base()
        before = src["ai_fields"]["recipient_name"]
        dp.swap_and_reevaluate(src, None)
        self.assertEqual(src["ai_fields"]["recipient_name"], before)

    def test_reevaluates_verdict(self):
        r = dp.swap_and_reevaluate(self._base(), None)
        self.assertIn(r["verdict"], (dp.Verdict.PASS, dp.Verdict.REVIEW,
                                     dp.Verdict.MANUAL))
        self.assertIn("对调", r["reason"])

    def test_no_fields_is_safe(self):
        src = self._base()
        src["ai"] = {}
        src["ai_fields"] = {}
        r = dp.swap_and_reevaluate(src, None)
        self.assertIn("swap_error", r)

    def test_none_safe(self):
        r = dp.swap_and_reevaluate(None, None)
        self.assertIn("swap_error", r)


# ==========================================================================
# 8. 批量
# ==========================================================================
class TestBatch(unittest.TestCase):
    def test_batch_empty(self):
        self.assertEqual(dp.run_full_pipeline_batch([]), [])

    def test_batch_progress(self):
        seen = []
        with mock.patch.object(dp, "run_full_pipeline", return_value={"ok": True}):
            dp.run_full_pipeline_batch(
                ["a.jpg", "b.jpg"],
                progress=lambda i, n, s: seen.append((i, n)))
        self.assertEqual(seen[0], (0, 2))
        self.assertEqual(seen[-1], (2, 2))


# ==========================================================================
# 9. 真实面单 (端到端, 需要 Ollama)
# ==========================================================================
REAL_DIR = Path(r"D:\新建文件夹\project\package")


@unittest.skipUnless(REAL_DIR.is_dir(), "没有真实面单目录, 跳过")
class TestRealWaybill(unittest.TestCase):
    def test_real_image_runs(self):
        f = next(REAL_DIR.glob("parcel_*.jpg"), None)
        if f is None:
            self.skipTest("目录里没有图片")
        from core import ai_vision
        ok, msg = ai_vision.check_service()
        if not ok:
            self.skipTest(f"Ollama 不可用: {msg}")
        r = dp.run_full_pipeline(str(f), dp.DetectConfig(), progress=lambda m: None)
        if not r["ok"] and "服务未运行" in str(r.get("reason", "")):
            self.skipTest("运行中 Ollama 掉线了")
        self.assertTrue(r["ok"], r.get("error"))
        self.assertIn(r["verdict"], (dp.Verdict.PASS, dp.Verdict.REVIEW,
                                     dp.Verdict.MANUAL))
        self.assertGreater(r["timings"]["total_ms"], 0)
        self.assertNotIn("barcode_ms", r["timings"])


# ==========================================================================
# 10. 模型选择 (v2.1.0)
# ==========================================================================
class TestModelSelection(unittest.TestCase):
    """``DetectConfig.ai_model`` 必须真正传到 ``VisionConfig.model``."""

    def test_ai_model_forwarded(self):
        fake = _FakeAi()
        cfg = dp.DetectConfig(ai_model="qwen3-vl:4b-instruct-q4_k_m")
        _run(fake, cfg)
        self.assertIsNotNone(fake.last_config, "没有调用 extract_and_match")
        self.assertEqual(fake.last_config.model,
                         "qwen3-vl:4b-instruct-q4_k_m")

    def test_empty_ai_model_keeps_default(self):
        fake = _FakeAi()
        _run(fake, dp.DetectConfig())
        self.assertIsNotNone(fake.last_config)
        self.assertEqual(fake.last_config.model, "fake-model",
                         "ai_model 为空时不该覆盖 VisionConfig 的默认模型")

    def test_release_mode_still_forwarded(self):
        fake = _FakeAi()
        _run(fake, dp.DetectConfig(ai_model="qwen3-vl:2b-instruct",
                                   release_mode="lenient"))
        self.assertEqual(fake.last_config.release_mode, "lenient")


if __name__ == "__main__":
    unittest.main(verbosity=2)
