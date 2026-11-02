"""AI 视觉识别 (ai_vision) 的测试.

分两类:
  · **纯函数测试** —— 清洗、规范化、错误处理。不碰网络, 秒级完成。
  · **联调测试**   —— 真调 Ollama + qwen3-vl:2b。Ollama 不在或模型缺失时
    自动 skip, 不会让整个测试套件失败。

运行::

    python tests/test_ai_vision.py            # 全部
    python tests/test_ai_vision.py -k clean   # 只跑名字含 clean 的
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from core import ai_vision as av          # noqa: E402
from core import database as db          # noqa: E402
from core import matcher as matcher      # noqa: E402


# ==========================================================================
# 1. 字段契约
# ==========================================================================
class TestSchema(unittest.TestCase):
    """需求要求字段名固定, 不可增删、不可改名."""

    EXPECTED = (
        "tracking_number", "recipient_name", "recipient_phone",
        "recipient_address", "sender_name", "sender_phone", "sender_address",
        "sorting_code", "order_number", "item_name", "cod_amount",
        "insured_amount", "shipping_date", "courier_company", "sender_branch",
    )

    def test_field_list_exact(self):
        self.assertEqual(tuple(av.FIELDS), self.EXPECTED)
        self.assertEqual(len(av.FIELDS), 15)

    def test_schema_matches_fields(self):
        self.assertEqual(set(av.SCHEMA["properties"]), set(self.EXPECTED))
        self.assertEqual(set(av.SCHEMA["required"]), set(self.EXPECTED))

    def test_schema_allows_null(self):
        """需求: 无法辨认时值为 null, 所以每个字段都必须允许 null."""
        for name, spec in av.SCHEMA["properties"].items():
            self.assertIn("null", spec["type"], f"{name} 不允许 null")

    def test_model_name(self):
        """默认模型是真实面单横评后选定的 instruct 版本.

        16 张真实面单实测: qwen3-vl:2b 成功 14/16、均 10.1 字段、均 48.5 秒;
        qwen3-vl:2b-instruct 成功 16/16、均 14.5 字段、均 8.8 秒。
        """
        self.assertEqual(av.MODEL_NAME, "qwen3-vl:2b-instruct")
        self.assertEqual(av.VisionConfig().model, av.MODEL_NAME)

    def test_model_can_be_overridden_by_env(self):
        """不重新打包也能换模型."""
        import os
        with mock.patch.dict(os.environ, {av.MODEL_ENV_VAR: "某个别的模型"}):
            self.assertEqual(av.VisionConfig().model, "某个别的模型")
        self.assertEqual(av.VisionConfig().model, av.MODEL_NAME)

    def test_prompt_contains_all_fields(self):
        """提示词里必须列全 15 个字段."""
        for f in self.EXPECTED:
            self.assertIn(f, av.PROMPT, f"提示词缺少字段 {f}")

    def test_prompt_has_key_rules(self):
        for kw in ("只提取图片中实际可见的文字", "值设为 null", "不要思考",
                   "直接输出纯 JSON"):
            self.assertIn(kw, av.PROMPT, f"提示词缺少关键要求: {kw}")


# ==========================================================================
# 2. 输出清洗
# ==========================================================================
class TestCleanJsonText(unittest.TestCase):
    """模型返回的形态五花八门, 清洗必须扛得住."""

    def test_plain(self):
        s = '{"a": 1}'
        self.assertEqual(av.clean_json_text(s), s)

    def test_code_fence_json(self):
        s = '```json\n{"a": 1}\n```'
        self.assertEqual(av.clean_json_text(s), '{"a": 1}')

    def test_code_fence_bare(self):
        s = '```\n{"a": 1}\n```'
        self.assertEqual(av.clean_json_text(s), '{"a": 1}')

    def test_uppercase_fence(self):
        s = '```JSON\n{"a": 1}\n```'
        self.assertEqual(av.clean_json_text(s), '{"a": 1}')

    def test_leading_prose(self):
        s = '好的, 结果如下:\n{"a": 1}'
        self.assertEqual(av.clean_json_text(s), '{"a": 1}')

    def test_trailing_prose(self):
        s = '{"a": 1}\n以上就是提取结果。'
        self.assertEqual(av.clean_json_text(s), '{"a": 1}')

    def test_array(self):
        s = '```json\n[{"a": 1}, {"a": 2}]\n```'
        self.assertEqual(av.clean_json_text(s), '[{"a": 1}, {"a": 2}]')

    def test_bom(self):
        self.assertEqual(av.clean_json_text('\ufeff{"a": 1}'), '{"a": 1}')

    def test_fullwidth_quotes(self):
        """模型偶尔输出全角引号, 直接 json.loads 会失败."""
        s = '{“a”: 1}'
        self.assertEqual(av.clean_json_text(s), '{"a": 1}')

    def test_empty(self):
        self.assertEqual(av.clean_json_text(""), "")
        self.assertEqual(av.clean_json_text(None), "")

    def test_double_fence(self):
        s = '```\n```json\n{"a": 1}\n```\n```'
        self.assertIn('{"a": 1}', av.clean_json_text(s))


class TestParseAiJson(unittest.TestCase):
    def test_ok(self):
        data, err = av.parse_ai_json('```json\n{"tracking_number": "X"}\n```')
        self.assertEqual(err, "")
        self.assertEqual(data["tracking_number"], "X")

    def test_array(self):
        data, err = av.parse_ai_json('[{"a":1},{"a":2}]')
        self.assertEqual(err, "")
        self.assertEqual(len(data), 2)

    def test_empty_content(self):
        data, err = av.parse_ai_json("")
        self.assertIsNone(data)
        self.assertIn("没有返回任何内容", err)

    def test_broken_json(self):
        data, err = av.parse_ai_json('{"tracking_number": ')
        self.assertIsNone(data)
        self.assertIn("JSON 解析失败", err)
        self.assertIn("返回内容前", err, "错误信息应带上原文片段, 便于排查")


# ==========================================================================
# 3. 记录规范化
# ==========================================================================
class TestNormalizeRecord(unittest.TestCase):
    def test_exactly_15_fields(self):
        rec = av.normalize_record({"tracking_number": "X", "无关字段": "y"})
        self.assertEqual(set(rec), set(av.FIELDS), "必须是且仅是 15 个字段")

    def test_missing_becomes_none(self):
        rec = av.normalize_record({})
        self.assertTrue(all(v is None for v in rec.values()))

    def test_empty_string_becomes_none(self):
        """需求: 无法辨认时值为 null, 空串不算有效值."""
        self.assertIsNone(av.normalize_record({"recipient_name": ""})["recipient_name"])
        self.assertIsNone(av.normalize_record({"recipient_name": "  "})["recipient_name"])

    def test_string_null_becomes_none(self):
        """模型有时把 null 输出成字符串 "null" 或 "N/A"."""
        for s in ("null", "None", "N/A", "-", "NULL"):
            self.assertIsNone(
                av.normalize_record({"cod_amount": s})["cod_amount"], s)

    def test_strips_whitespace(self):
        self.assertEqual(
            av.normalize_record({"tracking_number": "  X1  "})["tracking_number"], "X1")

    def test_number_to_string(self):
        self.assertEqual(av.normalize_record({"cod_amount": 120})["cod_amount"], "120")

    def test_list_to_string(self):
        rec = av.normalize_record({"item_name": ["衣服", "2件"]})
        self.assertEqual(rec["item_name"], "衣服 2件")

    def test_non_dict_safe(self):
        for bad in (None, "字符串", 123, []):
            rec = av.normalize_record(bad)
            self.assertEqual(set(rec), set(av.FIELDS))
            self.assertTrue(all(v is None for v in rec.values()))


class TestSenderRecipientSwap(unittest.TestCase):
    """回归: AI 会把发货方/收件方的联系人信息搞反位置.

    用户实测反馈的问题。面单上「收」「寄」是两个相邻方框, 小模型容易串位,
    靠提示词治不好 —— 但数据库里存着正确的收发信息, 拿它当裁判即可纠正。
    """

    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.db = Path(self._d.name) / "p.db"
        db.init_database(self.db)
        db.save_record({
            "tracking_number": "YT0713590525482",
            "recipient_name": "曹*", "recipient_phone": "13812341234",
            "recipient_address": "广东省佛山市南海区狮山镇大学城",
            "sender_name": "刘*发", "sender_phone": "17779746667",
            "sender_address": "广东省惠州市惠城区",
        }, path=self.db)

    def tearDown(self):
        self._d.cleanup()

    def _swapped_input(self):
        """模拟模型把收/寄搞反: 收件位填了寄件人的信息, 反之亦然."""
        return av.normalize_record({
            "tracking_number": "YT0713590525482",
            "recipient_name": "刘*发", "recipient_phone": "17779746667",
            "recipient_address": "广东省惠州市惠城区",
            "sender_name": "曹*", "sender_phone": "13812341234",
            "sender_address": "广东省佛山市南海区狮山镇大学城",
        })

    def _correct_input(self):
        return av.normalize_record({
            "tracking_number": "YT0713590525482",
            "recipient_name": "曹*", "recipient_phone": "13812341234",
            "recipient_address": "广东省佛山市南海区狮山镇大学城",
            "sender_name": "刘*发", "sender_phone": "17779746667",
            "sender_address": "广东省惠州市惠城区",
        })

    def test_detects_and_fixes_swap(self):
        fixed, info = av.resolve_sender_recipient(self._swapped_input(), self.db)
        self.assertTrue(info["swapped"], info["reason"])
        self.assertEqual(fixed["recipient_name"], "曹*")
        self.assertEqual(fixed["sender_name"], "刘*发")
        self.assertEqual(fixed["recipient_phone"], "13812341234")
        self.assertGreater(info["gain"], 0)

    def test_leaves_correct_input_alone(self):
        """本来就对的时候绝不能乱动 —— 否则会把正确结果改坏."""
        src = self._correct_input()
        fixed, info = av.resolve_sender_recipient(src, self.db)
        self.assertFalse(info["swapped"], info["reason"])
        self.assertEqual(fixed, src)

    def test_does_not_swap_when_both_assignments_are_bad(self):
        """两种情况都匹配不上时不许换 —— 换了只是把错误换个方向.

        这是关键的安全边界: 库里根本没有这张单, 那就没有裁判依据。
        """
        junk = av.normalize_record({
            "recipient_name": "完全不相干的人", "recipient_phone": "19900000000",
            "recipient_address": "黑龙江省哈尔滨市道里区",
            "sender_name": "另一个不相干的人", "sender_phone": "19800000000",
            "sender_address": "新疆乌鲁木齐市天山区",
        })
        fixed, info = av.resolve_sender_recipient(junk, self.db)
        self.assertFalse(info["swapped"], info["reason"])
        self.assertEqual(fixed, junk)

    def test_empty_fields_untouched(self):
        empty = av.normalize_record({})
        fixed, info = av.resolve_sender_recipient(empty, self.db)
        self.assertFalse(info["swapped"])
        self.assertEqual(fixed, empty)

    def test_swap_helper_is_pure(self):
        """对调函数不能改动传入的字典."""
        src = self._correct_input()
        snapshot = dict(src)
        out = av.swap_sender_recipient(src)
        self.assertEqual(src, snapshot, "原字典被改坏了")
        self.assertEqual(out["recipient_name"], "刘*发")
        self.assertEqual(out["sender_name"], "曹*")

    def test_swap_helper_keeps_other_fields(self):
        src = self._correct_input()
        out = av.swap_sender_recipient(src)
        self.assertEqual(out["tracking_number"], src["tracking_number"])
        self.assertEqual(set(out), set(av.FIELDS))

    def test_extract_and_match_records_the_fix(self):
        """extract_and_match 的结果里要能看到纠正痕迹, 便于事后核对."""
        import core.ai_vision as mod
        swap = self._swapped_input()
        with mock.patch.object(mod, "extract_parcel_info",
                               return_value=dict(swap)):
            r = av.extract_and_match("假装.jpg", self.db)
        self.assertTrue(r["ok"])
        self.assertTrue(r["sender_recipient_fix"]["swapped"])
        self.assertEqual(r["ai"]["recipient_name"], "曹*")
        self.assertIn("ai_before_swap", r, "应保留对调前的原始值以便追溯")
        self.assertEqual(r["ai_before_swap"]["recipient_name"], "刘*发")

    # ---- 判据是"字段相似度", 不是 matcher 总分 ----
    def test_rule_uses_field_similarity_not_total_score(self):
        """回归: 曾经拿 matcher 的**总分**判方向, 完全判不出来.

        原因: 单号精确命中时总分只有 1.0 / 0.60 两档, 对调前后一个样。
        所以判据必须直接比字段相似度。这里断言"两个方向的分数都拿得到",
        就说明走的是字段判据而不是总分。
        """
        _, info = av.resolve_sender_recipient(self._swapped_input(), self.db)
        self.assertIsNotNone(info["score_as_is"])
        self.assertIsNotNone(info["score_swapped"])
        self.assertGreater(info["evidence"], 0, "应记录参与比较的字段权重")

    def test_orientation_pairs_are_not_crossed(self):
        """回归: 曾把两对前缀配错 (recipient 配到 sender), 判据整个反过来.

        表现是: 搞反的说"不用动"、本来就对的反说"要换"。
        这里用一组对称的输入把方向钉死。
        """
        swapped, info_sw = av.resolve_sender_recipient(self._swapped_input(),
                                                       self.db)
        correct, info_ok = av.resolve_sender_recipient(self._correct_input(),
                                                       self.db)
        self.assertTrue(info_sw["swapped"], "搞反了必须判出来")
        self.assertFalse(info_ok["swapped"], "本来就对绝不能判成搞反")
        # 搞反时: 本方相似度低、对方相似度高
        self.assertLess(info_sw["score_as_is"], info_sw["score_swapped"])
        # 本来就对时: 反过来
        self.assertGreater(info_ok["score_as_is"], info_ok["score_swapped"])

    def test_no_basis_when_db_lacks_sender(self):
        """库里那条没有寄件方信息时, 没有裁判依据, 不许乱换."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "nosender.db"
            db.init_database(p)
            db.save_record({"tracking_number": "YT0713590525482",
                            "recipient_name": "曹*"}, path=p)
            fixed, info = av.resolve_sender_recipient(self._swapped_input(), p)
            self.assertFalse(info["swapped"])
            self.assertIn("寄件方", info["reason"])

    def test_fetch_db_row_handles_list_return(self):
        """回归: find_by_tracking 返回的是**列表**, 且元素是 sqlite3.Row.

        直接 dict(row) 会抛
        "dictionary update sequence element #0 has length N; 2 is required"。
        """
        row = av._fetch_db_row("YT0713590525482", self.db)
        self.assertIsNotNone(row)
        self.assertEqual(row["recipient_name"], "曹*")
        self.assertEqual(row["sender_name"], "刘*发", "必须能取到寄件方字段")

    def test_fetch_db_row_missing_returns_none(self):
        self.assertIsNone(av._fetch_db_row("根本不存在的单号", self.db))
        self.assertIsNone(av._fetch_db_row("", self.db))
        self.assertIsNone(av._fetch_db_row(None, self.db))

    # ---- 方案二: 匹配到库中记录时以库为准 ----
    def test_prefer_db_contacts_default_off(self):
        """默认必须关 —— 否则会遮蔽 AI 的读数, 想核对 AI 时就看不见差异了."""
        self.assertFalse(av.VisionConfig().prefer_db_contacts)

    def test_prefer_db_contacts_overrides_and_keeps_original(self):
        """方案二: 方向已经对了, 但个别字段值与库不符 -> 以库为准并留底.

        注意: 不能拿"方向搞反"的输入来测方案二 —— 那种情况方案一就会先对调,
        轮不到方案二动手 (第一版测试就是这么写错的)。
        """
        import core.ai_vision as mod
        # 方向正确, 但姓名读错、寄件电话没读出来
        partly_wrong = av.normalize_record({
            "tracking_number": "YT0713590525482",
            "recipient_name": "曹某",              # 库里是 曹*
            "recipient_phone": "13812341234",
            "recipient_address": "广东省佛山市南海区狮山镇大学城",
            "sender_name": "刘*发",
            # sender_phone 故意留空, 应该被库里的值补上
            "sender_address": "广东省惠州市惠城区",
        })
        cfg = av.VisionConfig(prefer_db_contacts=True)
        with mock.patch.object(mod, "extract_parcel_info",
                               return_value=dict(partly_wrong)):
            r = av.extract_and_match("假装.jpg", self.db, config=cfg)
        self.assertTrue(r.get("db_contacts_applied"))
        self.assertEqual(r["ai"]["sender_phone"], "17779746667", "空字段应被补上")
        self.assertIn("ai_from_model", r, "覆盖前的 AI 读数要留底")
        self.assertEqual(r["ai_from_model"]["sender_phone"], None)

    def test_prefer_db_contacts_off_keeps_model_values(self):
        """关闭时: 方案1 仍然会把搞反的纠正过来, 但不会去补库里别的字段."""
        import core.ai_vision as mod
        swap = self._swapped_input()
        with mock.patch.object(mod, "extract_parcel_info", return_value=dict(swap)):
            r = av.extract_and_match("假装.jpg", self.db)
        self.assertEqual(r["ai"]["recipient_name"], "曹*", "方案1 应对调")
        self.assertNotIn("db_contacts_applied", r)


class TestSortingCodeRelocation(unittest.TestCase):
    """回归: 真实面单上模型常把**大字分拣码**当成单号.

    面单上字体最大的往往不是单号而是目的地分拣码。18 张真实面单实测,
    16 张返回了"单号", 其中 5~6 张填的其实是分拣码 —— 而单号是后续精确匹配的
    唯一依据, 填错会直接导致匹配到错误的包裹。
    """

    REAL_TRACKING = ("YT0713590525482", "9819989866891", "SF3263531736719",
                     "JD0261189133871", "76985198456634", "321350360388470",
                     "79145190061768", "JT3180192749755")

    REAL_SORTING = ("197-197-A45-130", "620A N089-00 101", "155- K8 201 [D10]",
                    "130- A7 101 [A01]", "610 L31 101", "626-M08-00-601")

    def test_real_tracking_numbers_kept(self):
        for v in self.REAL_TRACKING:
            with self.subTest(v=v):
                r = av.normalize_record({"tracking_number": v})
                self.assertEqual(r["tracking_number"], v, "真单号不能被挪走")
                self.assertIsNone(r["sorting_code"])

    def test_sorting_codes_moved_out_of_tracking(self):
        for v in self.REAL_SORTING:
            with self.subTest(v=v):
                r = av.normalize_record({"tracking_number": v})
                self.assertIsNone(r["tracking_number"],
                                  f"{v} 是分拣码, 不该留在单号位")
                self.assertEqual(r["sorting_code"], v)

    def test_existing_sorting_code_not_overwritten(self):
        r = av.normalize_record({"tracking_number": "610 L31 101",
                                 "sorting_code": "已有的分拣码"})
        self.assertEqual(r["sorting_code"], "已有的分拣码")
        self.assertIsNone(r["tracking_number"], "单号位仍须清空")

    def test_privacy_number_is_left_alone(self):
        """隐私号码形态保守处理 —— 不含分隔符的不动, 避免误伤真单号."""
        r = av.normalize_record({"tracking_number": "15783238185转3355"})
        self.assertEqual(r["tracking_number"], "15783238185转3355")

    def test_moved_sorting_code_no_longer_exact_matches(self):
        """清空单号位的实际意义: 不能让分拣码去精确命中错误记录."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.db"
            db.init_database(p)
            db.save_record({"tracking_number": "620A N089-00 101",
                            "recipient_name": "不该被命中"}, path=p)
            rec = av.normalize_record({"tracking_number": "620A N089-00 101"})
            r = matcher.match_from_ai(rec, p)
            self.assertIsNone(rec["tracking_number"])
            self.assertNotEqual(r.get("match_type"), "exact")

    def test_detector_edge_cases(self):
        self.assertFalse(av._looks_like_sorting_code(None))
        self.assertFalse(av._looks_like_sorting_code(""))
        self.assertFalse(av._looks_like_sorting_code("ABC"))       # 太短
        self.assertFalse(av._looks_like_sorting_code("12345678"))  # 长连续段


class TestErrorRecord(unittest.TestCase):
    def test_has_all_fields_and_error(self):
        rec = av._error_record("炸了")
        self.assertEqual(rec["error"], "炸了")
        for f in av.FIELDS:
            self.assertIn(f, rec)
            self.assertIsNone(rec[f])

    def test_error_record_does_not_trigger_bogus_match(self):
        """回归: 失败记录里的 error 文本不能被 matcher 当成有效输入.

        否则会拿着错误信息去和数据库瞎匹配, 产出一堆无意义的候选。
        """
        rec = av._error_record("调用失败: connection refused")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.db"
            db.init_database(p)
            db.save_record({"tracking_number": "ABC123456789"}, path=p)
            r = matcher.match_from_ai(rec, p)
            self.assertEqual(r["candidates"], [], "不应产生候选")
            self.assertTrue(r["needs_review"])
            self.assertIn("AI 识别失败", r["message"])


# ==========================================================================
# 4. 服务检查与异常处理 (不打真实网络)
# ==========================================================================
class TestServiceCheck(unittest.TestCase):
    def test_returns_tuple(self):
        ok, msg = av.check_service(av.VisionConfig(host="http://127.0.0.1:1"))
        self.assertIsInstance(ok, bool)
        self.assertTrue(msg)

    def test_unreachable_host_is_friendly(self):
        ok, msg = av.check_service(
            av.VisionConfig(host="http://127.0.0.1:9", model="不存在的模型"))
        self.assertFalse(ok)
        self.assertTrue(any(k in msg for k in ("连不上", "没有找到")), msg)


class TestExtractErrors(unittest.TestCase):
    """各种失败路径都不能抛异常, 必须返回带 error 的字典."""

    def test_missing_file(self):
        rec = av.extract_parcel_info("/绝对不存在的路径/x.jpg")
        self.assertIn("error", rec)
        self.assertIn("不存在", rec["error"])
        for f in av.FIELDS:
            self.assertIsNone(rec[f])

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "empty.jpg"
            p.write_bytes(b"")
            rec = av.extract_parcel_info(p)
            self.assertIn("error", rec)
            self.assertIn("空文件", rec["error"])

    def test_model_failure_is_caught(self):
        """模拟 ollama 抛异常 -> 应返回 error 字典而不是抛出."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.jpg"
            p.write_bytes(b"\xff\xd8\xff\xd9")      # 最小 jpg 头
            fake = mock.MagicMock()
            fake.chat.side_effect = RuntimeError("connection refused")
            with mock.patch("ollama.Client", return_value=fake):
                rec = av.extract_parcel_info(p)
            self.assertIn("error", rec)
            self.assertIn("调用失败", rec["error"])

    def test_broken_json_response_is_caught(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.jpg"
            p.write_bytes(b"\xff\xd8\xff\xd9")
            fake = mock.MagicMock()
            fake.chat.return_value = {"message": {"content": "这不是 JSON"}}
            with mock.patch("ollama.Client", return_value=fake):
                rec = av.extract_parcel_info(p, config=av.VisionConfig(retries=0))
            self.assertIn("error", rec)
            self.assertIn("JSON 解析失败", rec["error"])

    def test_successful_mock_call(self):
        """用假客户端验证成功路径的字段规范化与日志."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.jpg"
            p.write_bytes(b"\xff\xd8\xff\xd9")
            fake = mock.MagicMock()
            fake.chat.return_value = {"message": {"content": json.dumps({
                "tracking_number": "YT0713590525482",
                "recipient_name": "张**",
                "recipient_phone": "138****1234",
                "sender_name": None,
                "无关": "忽略我",
            }, ensure_ascii=False)}}
            logs: list[str] = []
            with mock.patch("ollama.Client", return_value=fake):
                rec = av.extract_parcel_info(p, log=logs.append)
            self.assertNotIn("error", rec)
            self.assertEqual(rec["tracking_number"], "YT0713590525482")
            self.assertEqual(rec["recipient_name"], "张**")
            self.assertIsNone(rec["sender_name"])
            self.assertEqual(set(rec), set(av.FIELDS))
            self.assertTrue(any("成功" in m for m in logs), "应记录日志")
            self.assertTrue(any("耗时" in m for m in logs), "日志应含耗时")

    def test_array_response_takes_first(self):
        """提示词允许多图输出数组; 单图接口应取第一个元素."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.jpg"
            p.write_bytes(b"\xff\xd8\xff\xd9")
            fake = mock.MagicMock()
            fake.chat.return_value = {"message": {"content": json.dumps(
                [{"tracking_number": "A1"}, {"tracking_number": "A2"}])}}
            with mock.patch("ollama.Client", return_value=fake):
                rec = av.extract_parcel_info(p)
            self.assertEqual(rec["tracking_number"], "A1")


# ==========================================================================
# 5. 批量 (用 mock, 不调真实模型)
# ==========================================================================
class TestBatchMocked(unittest.TestCase):
    def test_isolates_failures(self):
        with tempfile.TemporaryDirectory() as d:
            good = Path(d) / "good.jpg"
            good.write_bytes(b"\xff\xd8\xff\xd9")
            fake = mock.MagicMock()
            # 第一次调用成功, 之后一律返回垃圾。
            # (注意: 不能靠文件名判断 —— images 传的是 bytes, messages 里只有提示词,
            #  文件名根本不出现在请求里。)
            fake.chat.side_effect = [
                {"message": {"content": '{"tracking_number":"G1"}'}},
                {"message": {"content": "垃圾"}},
                {"message": {"content": "垃圾"}},
                {"message": {"content": "垃圾"}},
            ]

            with mock.patch("ollama.Client", return_value=fake):
                out = av.extract_parcel_info_batch(
                    [good, Path(d) / "缺失.jpg"],
                    config=av.VisionConfig(retries=0))
            self.assertEqual(len(out), 2)
            self.assertEqual(out[0].get("tracking_number"), "G1", out[0])
            self.assertIn("error", out[1], "坏的那张应带 error 而不是中断整批")

    def test_progress_callback(self):
        seen: list[tuple] = []
        with mock.patch.object(av, "extract_parcel_info",
                               return_value=av.normalize_record({})):
            av.extract_parcel_info_batch(["a.jpg", "b.jpg"],
                                         progress=lambda i, n, s: seen.append((i, n)))
        self.assertEqual(seen[0], (0, 2))
        self.assertEqual(seen[-1], (2, 2), "结束时应回调满进度")


# ==========================================================================
# 6. 写库 (save_ai_result)
# ==========================================================================
class TestSaveAiResult(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.db = Path(self._d.name) / "p.db"
        db.init_database(self.db)

    def tearDown(self):
        self._d.cleanup()

    def _result(self, **kw):
        ai = av.normalize_record(kw)
        return {"image": "/tmp/a.jpg", "ai": ai, "ok": True}

    def test_save_ok(self):
        r = av.save_ai_result(self._result(tracking_number="SAVE1",
                                           recipient_name="张三"), self.db)
        self.assertEqual(r["status"], "inserted")
        row = db.list_parcels(self.db)[0]
        self.assertEqual(row["tracking_number"], "SAVE1")
        self.assertEqual(row["image_path"], "/tmp/a.jpg", "应记录图片路径")

    def test_save_skips_on_error(self):
        res = {"image": "/tmp/a.jpg", "ai": av._error_record("识别失败")}
        r = av.save_ai_result(res, self.db)
        self.assertEqual(r["status"], "skipped")
        self.assertEqual(db.count_parcels(self.db), 0)

    def test_duplicate_default_skip(self):
        av.save_ai_result(self._result(tracking_number="DUP1"), self.db)
        r = av.save_ai_result(self._result(tracking_number="DUP1"), self.db)
        self.assertEqual(r["status"], "skipped")
        self.assertEqual(db.count_parcels(self.db), 1)

    def test_only_known_fields_written(self):
        """AI 结果里的非表字段不能被写进库.

        注意: 这里刻意用一个**普通的多余键**, 而不是 ``error`` ——
        ``error`` 是保留键, 它的存在会触发"有错就跳过"的分支 (见上一个用例),
        那样就测不到"多余键被过滤"这件事了。
        """
        res = self._result(tracking_number="OK1")
        res["ai"]["some_extra_key"] = "不该被写进库"
        r = av.save_ai_result(res, self.db)
        self.assertEqual(r["status"], "inserted", r.get("message"))
        row = db.get_parcel(r["id"], self.db)
        self.assertNotIn("some_extra_key", row)

    def test_error_key_triggers_skip(self):
        """error 是保留键: 带上它就一律不写库."""
        res = self._result(tracking_number="OK2")
        res["ai"]["error"] = "残留的错误信息"
        r = av.save_ai_result(res, self.db)
        self.assertEqual(r["status"], "skipped")
        self.assertEqual(db.count_parcels(self.db), 0)


# ==========================================================================
# 7. 真实联调 (Ollama 不可用时自动跳过)
# ==========================================================================
def _ollama_ready() -> tuple[bool, str]:
    try:
        return av.check_service()
    except Exception as exc:                     # noqa: BLE001
        return False, str(exc)


_OK, _WHY = _ollama_ready()


@unittest.skipUnless(_OK, f"Ollama 不可用, 跳过联调测试: {_WHY}")
class TestLiveOllama(unittest.TestCase):
    """真实调用 qwen3-vl:2b。较慢 (约 7~25 秒/张), 但验证的是真东西。"""

    @classmethod
    def setUpClass(cls):
        import cv2

        sys.path.insert(0, str(ROOT / "tests"))
        from _samples import make_waybill

        cls.img, cls.gt = make_waybill(seed=5)
        cls.dir = tempfile.TemporaryDirectory(prefix="aiv_live_")
        cls.img_path = Path(cls.dir.name) / "wb.png"
        ok, buf = cv2.imencode(".png", cls.img)
        cls.img_path.write_bytes(buf.tobytes())

        # 用真值建库, 便于验证比对能命中
        cls.db = Path(cls.dir.name) / "p.db"
        db.init_database(cls.db)
        db.save_record({"tracking_number": cls.gt.tracking_no,
                        "recipient_name": "杨小明",
                        "recipient_phone": "17396046368",
                        "recipient_address": "成都市武侯区人民南路3段",
                        "courier_company": "京东物流"}, path=cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_live_extract_returns_schema(self):
        rec = av.extract_parcel_info(self.img_path)
        self.assertNotIn("error", rec, rec.get("error", ""))
        self.assertEqual(set(rec), set(av.FIELDS))

    def test_live_reads_tracking_number(self):
        """最关键的一项: 单号必须读对, 它是精确匹配的依据."""
        rec = av.extract_parcel_info(self.img_path)
        got = str(rec.get("tracking_number") or "").replace(" ", "").upper()
        self.assertIn(self.gt.tracking_no.upper(), got,
                      f"AI 读出 {got!r}, 真值 {self.gt.tracking_no}")

    def test_live_end_to_end_match(self):
        r = av.extract_and_match(self.img_path, self.db)
        self.assertTrue(r["ok"], r.get("message"))
        self.assertEqual(r["match"]["match_type"], "exact")
        self.assertEqual(r["match"]["candidates"][0]["tracking_number"],
                         self.gt.tracking_no)
        self.assertGreaterEqual(r["match"]["candidates"][0]["total_score"], 0.85)

    def test_live_logs_timing(self):
        logs: list[str] = []
        av.extract_and_match(self.img_path, self.db, log=logs.append)
        joined = "\n".join(logs)
        self.assertIn("耗时", joined, "应记录耗时")
        self.assertIn("最高分", joined, "应记录匹配得分")

    def test_live_batch(self):
        out = av.extract_and_match_batch([self.img_path, self.img_path], self.db)
        self.assertEqual(len(out), 2)
        self.assertTrue(all(r["ok"] for r in out))


if __name__ == "__main__":
    unittest.main(verbosity=2)
