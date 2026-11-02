"""模型清单与可用性检查的单元测试 (v2.1.0).

不依赖本机 Ollama, 也不发真实网络请求:
    · ``list_installed`` 用假的 ``ollama`` 模块测试;
    · ``check_model`` 直接替换 ``list_installed`` 的返回值;
    · ``ai_vision.check_service`` 的模型名匹配用假客户端回归,
      专门盯住 v2.0.3 之前的"前缀匹配误报已就绪"。

重点不是覆盖率, 而是把两条需求钉死:
    1. 三个模型、默认 2b-instruct, 顺序不能乱;
    2. 切换后检查失败只报告, 不抛异常、不退出。
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core import model_registry as mr              # noqa: E402


def _fake_ollama(models, raise_on_list=None):
    """造一个最小可用的假 ollama 模块."""

    class _Client:
        def __init__(self, *a, **kw):
            self.kw = kw

        def list(self):
            if raise_on_list is not None:
                raise raise_on_list
            return {"models": [{"model": m} for m in models]}

    return types.SimpleNamespace(Client=_Client)


class TestModelCatalogue(unittest.TestCase):
    def test_exactly_three_models_in_order(self):
        self.assertEqual(mr.model_names(), [
            "qwen3-vl:2b-instruct",
            "qwen3-vl:4b-instruct-q4_k_m",
            "qwen3-vl:8b-instruct-q4_k_m",
        ])

    def test_default_is_2b_instruct(self):
        self.assertEqual(mr.DEFAULT_MODEL, "qwen3-vl:2b-instruct")

    def test_default_matches_ai_vision(self):
        """默认模型只能有一处定义 —— 防止注册表与 ai_vision 漂移."""
        from core import ai_vision
        self.assertEqual(mr.DEFAULT_MODEL, ai_vision.MODEL_NAME)

    def test_labels_are_unique_and_non_empty(self):
        labels = [m.label for m in mr.MODELS]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertTrue(all(lbl.strip() for lbl in labels))


class TestNameHandling(unittest.TestCase):
    def test_resolve_none_is_default(self):
        self.assertEqual(mr.resolve_model(None), mr.DEFAULT_MODEL)
        self.assertEqual(mr.resolve_model("   "), mr.DEFAULT_MODEL)

    def test_resolve_keeps_unknown_name(self):
        self.assertEqual(mr.resolve_model(" some-model:tag "), "some-model:tag")

    def test_normalize_appends_latest(self):
        self.assertEqual(mr.normalize_model_name("qwen3-vl:2b"), "qwen3-vl:2b")
        self.assertEqual(mr.normalize_model_name("abc"), "abc:latest")

    def test_normalize_lowercases_and_strips(self):
        self.assertEqual(mr.normalize_model_name("  Qwen3-VL:2B-INSTRUCT "),
                         "qwen3-vl:2b-instruct")

    def test_untagged_matches_latest(self):
        self.assertTrue(mr.is_model_installed("abc", ["abc:latest"]))

    def test_exact_match(self):
        self.assertTrue(mr.is_model_installed(
            "qwen3-vl:2b-instruct", ["qwen3-vl:2b-instruct"]))

    def test_prefix_is_not_a_match(self):
        """回归: 装了 2b 不能说明 4b 已就绪 (v2.0.3 的 check_service 就是这样误报的)."""
        installed = ["qwen3-vl:2b-instruct", "qwen3-vl:2b"]
        self.assertFalse(mr.is_model_installed(
            "qwen3-vl:4b-instruct-q4_k_m", installed))
        self.assertFalse(mr.is_model_installed(
            "qwen3-vl:8b-instruct-q4_k_m", installed))

    def test_empty_never_matches(self):
        self.assertFalse(mr.is_model_installed("", ["x:latest"]))
        self.assertFalse(mr.is_model_installed("x", []))


class TestListInstalled(unittest.TestCase):
    def test_reads_dict_response(self):
        fake = _fake_ollama(["qwen3-vl:2b-instruct", "qwen3-vl:2b"])
        with mock.patch.dict(sys.modules, {"ollama": fake}):
            ok, names, err = mr.list_installed()
        self.assertTrue(ok)
        self.assertEqual(err, "")
        self.assertEqual(names, ["qwen3-vl:2b-instruct", "qwen3-vl:2b"])

    def test_reads_object_response(self):
        class _M:
            def __init__(self, n):
                self.model = n

        class _Resp:
            models = [_M("a:1"), _M("b:2")]

        class _Client:
            def __init__(self, *a, **kw):
                pass

            def list(self):
                return _Resp()

        fake = types.SimpleNamespace(Client=_Client)
        with mock.patch.dict(sys.modules, {"ollama": fake}):
            ok, names, _ = mr.list_installed()
        self.assertTrue(ok)
        self.assertEqual(names, ["a:1", "b:2"])

    def test_connection_error_is_reported_not_raised(self):
        fake = _fake_ollama([], raise_on_list=ConnectionError("boom"))
        with mock.patch.dict(sys.modules, {"ollama": fake}):
            ok, names, err = mr.list_installed()
        self.assertFalse(ok)
        self.assertEqual(names, [])
        self.assertIn("连不上 Ollama", err)

    def test_missing_ollama_package_is_reported(self):
        with mock.patch.dict(sys.modules, {"ollama": None}):
            ok, names, err = mr.list_installed()
        self.assertFalse(ok)
        self.assertIn("未安装 ollama", err)


class TestCheckModel(unittest.TestCase):
    def test_service_down(self):
        with mock.patch.object(mr, "list_installed",
                               return_value=(False, [], "连不上 Ollama")):
            st = mr.check_model("qwen3-vl:2b-instruct")
        self.assertFalse(st.ok)
        self.assertFalse(st.service_ok)
        self.assertFalse(st.model_present)
        self.assertIn("连不上", st.message)

    def test_model_missing(self):
        with mock.patch.object(
                mr, "list_installed",
                return_value=(True, ["qwen3-vl:2b-instruct"], "")):
            st = mr.check_model("qwen3-vl:4b-instruct-q4_k_m")
        self.assertFalse(st.ok)
        self.assertTrue(st.service_ok)
        self.assertFalse(st.model_present)
        self.assertIn("ollama pull", st.message)
        self.assertEqual(st.installed, ["qwen3-vl:2b-instruct"])
        self.assertEqual(st.model, "qwen3-vl:4b-instruct-q4_k_m")

    def test_model_present(self):
        with mock.patch.object(
                mr, "list_installed",
                return_value=(True, ["qwen3-vl:2b-instruct"], "")):
            st = mr.check_model()
        self.assertTrue(st.ok)
        self.assertIn("已就绪", st.message)

    def test_default_model_when_none(self):
        with mock.patch.object(mr, "list_installed",
                               return_value=(True, [mr.DEFAULT_MODEL], "")):
            st = mr.check_model(None)
        self.assertEqual(st.model, mr.DEFAULT_MODEL)

    def test_availability_dict(self):
        with mock.patch.object(
                mr, "list_installed",
                return_value=(True, ["qwen3-vl:2b-instruct"], "")):
            d = mr.check_model_availability()
        self.assertTrue(d["ok"])
        self.assertEqual(d["model"], mr.DEFAULT_MODEL)
        self.assertTrue(d["service_ok"])
        self.assertTrue(d["model_present"])
        self.assertIn("installed", d)
        self.assertIn("message", d)


class TestReport(unittest.TestCase):
    def test_success_report_is_one_line(self):
        st = mr.ModelStatus(model="m:1", service_ok=True, model_present=True,
                            message="Ollama 可用, 模型 m:1 已就绪")
        text = mr.format_report(st)
        self.assertTrue(text.startswith("OK  "))
        self.assertIn("m:1", text)

    def test_failure_report_has_sections_and_pull_command(self):
        st = mr.ModelStatus(
            model="qwen3-vl:4b-instruct-q4_k_m", service_ok=True,
            model_present=False, installed=["qwen3-vl:2b-instruct"],
            message="服务可用, 但没有找到模型")
        text = mr.format_report(st)
        self.assertIn("模型可用性异常报告", text)
        self.assertIn("qwen3-vl:4b-instruct-q4_k_m", text)
        self.assertIn("ollama pull qwen3-vl:4b-instruct-q4_k_m", text)
        self.assertIn("不会退出程序", text)
        self.assertIn("qwen3-vl:2b-instruct", text)

    def test_service_down_report_has_start_advice(self):
        st = mr.ModelStatus(model="m:1", service_ok=False, model_present=False,
                            error="ConnectionError: refused",
                            message="连不上 Ollama")
        text = mr.format_report(st)
        self.assertIn("Ollama 服务: 未连接", text)
        self.assertIn("ollama serve", text)


class TestAiVisionCheckRegression(unittest.TestCase):
    """``ai_vision.check_service`` 的模型名匹配回归 (v2.1.0 修复)."""

    def test_4b_not_reported_ready_when_only_2b(self):
        from core import ai_vision
        fake = _fake_ollama(["qwen3-vl:2b-instruct", "qwen3-vl:2b"])
        with mock.patch.dict(sys.modules, {"ollama": fake}):
            ok, msg = ai_vision.check_service(
                ai_vision.VisionConfig(
                    model="qwen3-vl:4b-instruct-q4_k_m"))
        self.assertFalse(ok, f"4b 没装却报 OK: {msg}")
        self.assertIn("没有找到模型", msg)

    def test_2b_reported_ready(self):
        from core import ai_vision
        fake = _fake_ollama(["qwen3-vl:2b-instruct"])
        with mock.patch.dict(sys.modules, {"ollama": fake}):
            ok, msg = ai_vision.check_service(
                ai_vision.VisionConfig(model="qwen3-vl:2b-instruct"))
        self.assertTrue(ok, msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
