"""多字段比对 (matcher) 的测试.

运行::

    python tests/test_matcher.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core import database as db          # noqa: E402
from core import matcher as M            # noqa: E402


# ==========================================================================
# 1. 地址结构解析
# ==========================================================================
class TestParseAddress(unittest.TestCase):
    def test_full_address(self):
        p = M.parse_address("广东省佛山市南海区桂城街道人民路3号")
        self.assertEqual(p["province"], "广东省")
        self.assertEqual(p["city"], "佛山市")
        self.assertEqual(p["district"], "南海区")
        self.assertEqual(p["street"], "桂城街道")
        self.assertIn("人民路3号", p["detail"])

    def test_municipality_without_province(self):
        """直辖市常省略"省", 也要能拆出市级."""
        p = M.parse_address("北京市朝阳区建国路88号")
        self.assertEqual(p["city"], "北京市")
        self.assertEqual(p["district"], "朝阳区")
        self.assertEqual(p["street"], "建国路")

    def test_fragment_only_district(self):
        """AI 只读出一个区名时, 应落进 district 而不是别的段."""
        p = M.parse_address("南海区")
        self.assertEqual(p["district"], "南海区")
        self.assertEqual(p["province"], "")
        self.assertEqual(p["city"], "")

    def test_empty(self):
        p = M.parse_address("")
        self.assertEqual(p["detail"], "")
        self.assertEqual(p["province"], "")

    def test_unparseable_goes_to_detail(self):
        p = M.parse_address("ABCDEFG 12345")
        self.assertEqual(p["province"], "")
        self.assertIn("ABCDEFG", p["detail"])


# ==========================================================================
# 2. 各字段相似度
# ==========================================================================
class TestSimTracking(unittest.TestCase):
    def test_exact(self):
        s, r = M.sim_tracking("YT0713590525482", "yt0713590525482")
        self.assertEqual(s, 1.0)

    def test_ignores_separators(self):
        s, _ = M.sim_tracking("YT 0713-5905-25482", "YT0713590525482")
        self.assertEqual(s, 1.0)

    def test_suffix_fragment(self):
        """需求要求支持后缀匹配 (如只读到 25482), 且理由要写明是后缀."""
        s, r = M.sim_tracking("25482", "YT0713590525482")
        self.assertGreater(s, 0.85)
        self.assertIn("后缀", r)
        self.assertIn("25482", r)

    def test_prefix_fragment(self):
        s, r = M.sim_tracking("YT0713", "YT0713590525482")
        self.assertGreater(s, 0.85)
        self.assertIn("前缀", r)

    def test_prefix_n_digits(self):
        s, r = M.sim_tracking("YT07xxxx", "YT0713590525482")
        self.assertGreater(s, 0.5)

    def test_missing_input(self):
        s, _ = M.sim_tracking("", "YT0713590525482")
        self.assertIsNone(s, "输入为空应返回 None, 表示该字段不参与加权")

    def test_unrelated(self):
        s, _ = M.sim_tracking("SF743505166311", "YT0713590525482")
        self.assertLess(s, 0.5)


class TestSimPhone(unittest.TestCase):
    def test_exact(self):
        s, _ = M.sim_phone("13812341234", "138-1234-1234")
        self.assertEqual(s, 1.0)

    def test_tail_only(self):
        s, r = M.sim_phone("1234", "13812341234")
        self.assertGreater(s, 0.5)
        self.assertLess(s, 0.9, "只凭 4 位尾号不应给出过高分")

    def test_masked_middle(self):
        """脱敏 138****1234 —— 可见的号段+尾号都对上."""
        s, r = M.sim_phone("138****1234", "13812341234")
        self.assertGreater(s, 0.7)
        self.assertIn("片段", r)

    def test_masked_tail_only_no_misalignment(self):
        """回归: 脱敏值比原值短, 绝不能逐位 zip 比较.

        最初用 zip 逐位比, ``****1234``(8 字符) 与 ``13812341234``(11 位)
        会把 1234 拿去和 2341 比, 把一个正确匹配打成分。
        """
        s, _ = M.sim_phone("****1234", "13812341234")
        self.assertGreater(s, 0.5, "掩码尾号匹配不应被错位成低分")

    def test_masked_full_head_and_tail(self):
        s, _ = M.sim_phone("138****1234", "13812341234")
        s2, _ = M.sim_phone("138****9999", "13812341234")
        self.assertGreater(s, s2, "可见位一致时应高于不一致")

    def test_masked_mismatch(self):
        s, _ = M.sim_phone("138****9999", "13812341234")
        self.assertLess(s, 0.5)

    def test_wrong_number_same_tail(self):
        """11 位都齐全但只尾号相同 -> 不能给高分 (容易撞号)."""
        s, _ = M.sim_phone("13900001234", "13812341234")
        self.assertLess(s, 0.7)

    def test_missing(self):
        self.assertIsNone(M.sim_phone("", "13812341234")[0])
        self.assertIsNone(M.sim_phone("13812341234", "")[0])


class TestSimAddress(unittest.TestCase):
    def test_exact(self):
        s, _ = M.sim_address("广东省佛山市南海区桂城街道",
                             "广东省佛山市南海区桂城街道")
        self.assertEqual(s, 1.0)

    def test_fragment_district(self):
        """只读出一个区名 -> 是该段一致, 但**必须被完整度惩罚**.

        一个区下面可能有几千个包裹, "区名对上"远没有"省市区街道全对上"可信。
        最初这里给满分 1.0, 导致 `{姓名:"张", 地址:"南海区"}` 总分越过 0.85
        自动放行 —— 分拣线上这是危险的。
        """
        s, r = M.sim_address("南海区", "广东省佛山市南海区桂城街道")
        self.assertGreater(s, 0.6, "区名一致应给出有意义的分数")
        self.assertLess(s, 0.85, "但不应接近满分 —— 它只覆盖了一个区")
        self.assertIn("区", r)
        self.assertIn("完整", r, "理由里应说明完整度不足")

    def test_full_address_beats_fragment(self):
        full, _ = M.sim_address("广东省佛山市南海区桂城街道",
                                "广东省佛山市南海区桂城街道")
        part, _ = M.sim_address("南海区", "广东省佛山市南海区桂城街道")
        self.assertGreater(full, part)

    def test_city_level_fragment(self):
        """只给一个市名 -> 同样受完整度惩罚 (只覆盖了市级一段)."""
        s, _ = M.sim_address("佛山市", "广东省佛山市南海区桂城街道")
        self.assertGreater(s, 0.6)
        self.assertLess(s, 0.85)

    def test_same_district_different_province(self):
        """区名相同但省市不同 -> 要能区分开."""
        s1, _ = M.sim_address("广东省佛山市南海区", "广东省佛山市南海区桂城街道")
        s2, _ = M.sim_address("广东省佛山市南海区", "湖南省长沙市南海区")
        self.assertGreater(s1, s2)

    def test_unrelated(self):
        s, _ = M.sim_address("北京市朝阳区建国路", "广东省佛山市南海区桂城街道")
        self.assertLess(s, 0.4)

    def test_missing(self):
        self.assertIsNone(M.sim_address("", "广东省佛山市")[0])


class TestSimName(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(M.sim_name("张伟", "张伟")[0], 1.0)

    def test_masked(self):
        """张** vs 张伟."""
        s, _ = M.sim_name("张**", "张伟")
        self.assertGreater(s, 0.6)

    def test_masked_mismatch(self):
        s, _ = M.sim_name("李**", "张伟")
        self.assertLess(s, 0.4)

    def test_surname_only(self):
        s, r = M.sim_name("张", "张伟")
        self.assertGreaterEqual(s, 0.5)
        self.assertIn("姓", r)

    def test_same_surname_partial_given_name(self):
        s, _ = M.sim_name("张伟", "张伟明")
        self.assertGreater(s, 0.6)

    def test_different_person(self):
        s, _ = M.sim_name("李娜", "张伟")
        self.assertLess(s, 0.3)

    def test_missing(self):
        self.assertIsNone(M.sim_name("", "张伟")[0])


# ==========================================================================
# 3. 权重与证据量
# ==========================================================================
class TestWeights(unittest.TestCase):
    def test_full_input_uses_base_weights(self):
        w = M._effective_weights({"tracking_number": "YT0713590525482",
                                  "recipient_phone": "13812341234",
                                  "recipient_address": "广东省佛山市南海区",
                                  "recipient_name": "张伟"})
        self.assertAlmostEqual(w["tracking_number"], 0.40 / 0.90, places=4)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6)

    def test_tracking_missing_redistributes(self):
        """需求: 单号完全缺失时, 0.40 按 0.25:0.25:0.10 分给电话/地址/姓名."""
        w = M._weight_table({"recipient_phone": "13812341234",
                             "recipient_address": "广东省佛山市南海区",
                             "recipient_name": "张伟"})
        self.assertNotIn("tracking_number", w)
        # 电话 = 0.20 + 0.40*(0.25/0.60), 地址同理, 姓名 = 0.10 + 0.40*(0.10/0.60)
        self.assertAlmostEqual(w["recipient_phone"], 0.20 + 0.40 * 0.25 / 0.60, places=4)
        self.assertAlmostEqual(w["recipient_address"], 0.20 + 0.40 * 0.25 / 0.60, places=4)
        self.assertAlmostEqual(w["recipient_name"], 0.10 + 0.40 * 0.10 / 0.60, places=4)

    def test_tracking_fragment_keeps_weight(self):
        """回归: 单号**残缺**不等于缺失, 应保留 0.40 权重并参与模糊匹配.

        最初写成"不够长就剔除", 结果把 ``...25482`` 这种残片整条丢掉了。
        """
        w = M._weight_table({"tracking_number": "...25482",
                             "recipient_address": "广东省佛山市南海区"})
        self.assertIn("tracking_number", w, "单号残片必须保留权重")
        self.assertAlmostEqual(w["tracking_number"], 0.40, places=4)

    def test_only_present_fields_are_normalised(self):
        w = M._effective_weights({"recipient_phone": "13812341234"})
        self.assertEqual(list(w.keys()), ["recipient_phone"])
        self.assertAlmostEqual(w["recipient_phone"], 1.0, places=6)

    def test_empty_input_gives_no_weights(self):
        self.assertEqual(M._effective_weights({}), {})

    def test_others_group_counts_once(self):
        """其它字段合成一个 0.10 权重的组, 不应各自计权."""
        w = M._effective_weights({"courier_company": "圆通速递",
                                  "item_name": "衣服"})
        self.assertIn("others", w)
        self.assertNotIn("courier_company", w)

    def test_evidence_full_input(self):
        ev = M._evidence_weight({"tracking_number": "YT0713590525482",
                                 "recipient_phone": "13812341234",
                                 "recipient_address": "广东省佛山市南海区",
                                 "recipient_name": "张伟",
                                 "courier_company": "圆通速递"})
        self.assertAlmostEqual(ev, 1.0, places=6)

    def test_evidence_only_weak_field(self):
        ev = M._evidence_weight({"courier_company": "圆通速递"})
        self.assertAlmostEqual(ev, 0.10, places=6)

    def test_evidence_phone_only(self):
        ev = M._evidence_weight({"recipient_phone": "13812341234"})
        self.assertAlmostEqual(ev, 0.20 + 0.40 * 0.25 / 0.60, places=4)


# ==========================================================================
# 4. 置信分级与守卫
# ==========================================================================
class TestConfidence(unittest.TestCase):
    cfg = M.DEFAULT_CONFIG

    def test_thresholds_match_requirement(self):
        self.assertEqual(self.cfg.auto_pass, 0.85)
        self.assertEqual(self.cfg.review, 0.60)

    def test_high_score_enough_evidence(self):
        self.assertEqual(M._confidence_of(0.90, 0.80, self.cfg), "high")

    def test_high_score_insufficient_evidence_is_downgraded(self):
        """核心守卫: 分数满分但证据不足时不得自动通过."""
        self.assertEqual(M._confidence_of(1.00, 0.10, self.cfg), "medium")

    def test_medium_band(self):
        self.assertEqual(M._confidence_of(0.70, 1.0, self.cfg), "medium")

    def test_low_band(self):
        self.assertEqual(M._confidence_of(0.40, 1.0, self.cfg), "low")

    def test_ambiguity_downgrades(self):
        """歧义守卫: 前两名分值接近时降级."""
        self.assertEqual(M._confidence_of(0.95, 1.0, self.cfg, runner_up=0.94),
                         "medium")
        self.assertEqual(M._confidence_of(0.95, 1.0, self.cfg, runner_up=0.50),
                         "high")


# ==========================================================================
# 5. compare_fields (端到端, 用临时库)
# ==========================================================================
class _WithDb(unittest.TestCase):
    """建一个带 3 条记录的临时库."""

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory(prefix="match_test_")
        self.db = Path(self._dir.name) / "parcels.db"
        db.init_database(self.db)
        self.records = [
            {"tracking_number": "YT0713590525482", "recipient_name": "张伟",
             "recipient_phone": "13812341234",
             "recipient_address": "广东省佛山市南海区桂城街道",
             "courier_company": "圆通速递", "sorting_code": "626-M08-00-601",
             "item_name": "1KF-FM7【云岩白】"},
            {"tracking_number": "SF743505166311", "recipient_name": "李娜",
             "recipient_phone": "13900001111",
             "recipient_address": "北京市朝阳区建国路88号",
             "courier_company": "顺丰速运"},
            {"tracking_number": "JD681146975643", "recipient_name": "王强",
             "recipient_phone": "17396046368",
             "recipient_address": "四川省成都市武侯区人民南路3段",
             "courier_company": "京东物流"},
        ]
        for r in self.records:
            db.save_record(r, path=self.db)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def match(self, fields, **kw):
        return M.match_from_ai(fields, self.db, **kw)


class TestReleaseModes(_WithDb):
    """两档放行模式 (只有两个)。

    需求:
      · **宽松**: 单号匹配 **且库中唯一** -> 直接放行 (不参与加权打分)
      · **严格**: 即使单号命中, 也要「单号 + 其它信息」的**加权总分**
        达到阈值才放行
    """

    OK_FIELDS = {"tracking_number": "YT0713590525482", "recipient_name": "张伟",
                 "recipient_phone": "13812341234",
                 "recipient_address": "广东省佛山市南海区桂城街道"}
    BAD_FIELDS = {"tracking_number": "YT0713590525482", "recipient_name": "毫不相干",
                  "recipient_phone": "19900000000",
                  "recipient_address": "黑龙江省哈尔滨市道里区"}

    def _run(self, fields, mode, db=None):
        return M.compare_fields(fields, db or self.db,
                                config=M.MatcherConfig(release_mode=mode))

    # ---- 基本契约 ----
    def test_there_are_exactly_two_modes(self):
        """需求明确: 模式只有两个。"""
        self.assertEqual(M._normalise_mode("lenient"), "lenient")
        self.assertEqual(M._normalise_mode("strict"), "strict")
        # 任何其它写法都归到严格 (保守的一档)
        for junk in ("normal", "", None, "宽松", "STRICT ", "随便"):
            self.assertEqual(M._normalise_mode(junk), "strict", junk)

    def test_default_mode_is_strict(self):
        self.assertEqual(M.MatcherConfig().release_mode, "strict")

    def test_mode_reported_in_output(self):
        for mode in ("lenient", "strict"):
            r = self._run(self.OK_FIELDS, mode)
            self.assertEqual(r.get("release_mode"), mode)

    # ---- 宽松 ----
    def test_lenient_releases_when_tracking_unique(self):
        r = self._run(self.OK_FIELDS, "lenient")
        self.assertFalse(r["needs_review"])
        self.assertTrue(r.get("tracking_unique"))
        self.assertIn("宽松模式", r["message"])

    def test_lenient_releases_even_when_all_fields_wrong(self):
        """宽松模式的要点: 只要单号匹配且唯一, 其它字段一概不看."""
        r = self._run(self.BAD_FIELDS, "lenient")
        self.assertFalse(r["needs_review"], "单号唯一就该放行")
        self.assertEqual((r["candidates"] or [{}])[0]["total_score"], 1.0)

    def test_lenient_does_not_release_when_tracking_not_unique(self):
        """库里同号有多条时, 宽松模式的快速通道不成立, 转严格判定.

        回归: 曾漏掉这种组合, 代码掉进"无冲突就放行"的兜底分支,
        把不该放行的放行了。
        """
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "dup.db"
            db.init_database(p)
            db.save_record(dict(self.OK_FIELDS), path=p)
            # 绕开 save_record 的去重, 直接插第二条同号记录
            with db.session(p) as conn:
                conn.execute(
                    "INSERT INTO parcels (tracking_number, recipient_name, "
                    "recipient_phone, recipient_address) VALUES (?,?,?,?)",
                    ("YT0713590525482", "另一个人", "13812341234",
                     "广东省佛山市南海区桂城街道"))
                conn.commit()
            r = self._run(self.OK_FIELDS, "lenient", db=p)
            self.assertFalse(r.get("tracking_unique"))
            self.assertIn("同号", r["message"])

    # ---- 严格: 关键在"加权得分" ----
    def test_strict_uses_weighted_score(self):
        """严格模式必须真的算加权分, 而不是只看"有没有冲突".

        回归: 旧实现只要没冲突就直接给 1.0, 相当于没打分。
        这里断言候选里带出了**逐字段得分**, 说明走的是加权路径。
        """
        r = self._run(self.OK_FIELDS, "strict")
        fs = (r["candidates"] or [{}])[0].get("field_scores") or {}
        self.assertIn("tracking_number", fs)
        self.assertGreater(len(fs), 1, "应有多于单号的字段参与打分")

    def test_strict_releases_when_score_high(self):
        r = self._run(self.OK_FIELDS, "strict")
        self.assertFalse(r["needs_review"])
        self.assertGreaterEqual(
            (r["candidates"] or [{}])[0]["total_score"], M.DEFAULT_CONFIG.auto_pass)

    def test_strict_requires_review_when_score_low(self):
        """单号对但其它字段全错 -> 加权分低 -> 必须转人工 (宽松模式则放行)."""
        strict = self._run(self.BAD_FIELDS, "strict")
        lenient = self._run(self.BAD_FIELDS, "lenient")
        score = (strict["candidates"] or [{}])[0]["total_score"]
        self.assertLess(score, M.DEFAULT_CONFIG.auto_pass)
        self.assertTrue(strict["needs_review"])
        self.assertFalse(lenient["needs_review"], "同一输入宽松模式应放行")
        self.assertIn("加权总分", strict["message"])

    def test_strict_needs_info_beyond_tracking(self):
        """严格模式要求**确实有单号以外的信息**参与打分.

        否则会出现自相矛盾: 输入里只有单号 -> 权重只剩单号 -> 加权分必然
        1.00 -> 严格模式照样放行, 等于没严格。
        """
        only_tn = {"tracking_number": "YT0713590525482"}
        strict = self._run(only_tn, "strict")
        lenient = self._run(only_tn, "lenient")
        self.assertTrue(strict["needs_review"], "只有单号不该在严格模式下放行")
        self.assertIn("单号以外", strict["message"])
        self.assertFalse(lenient["needs_review"], "宽松模式只看单号, 可以放行")

    def test_strict_partial_match_score_is_continuous(self):
        """加权分是连续量: 对得越少分越低 (不是"有冲突就一刀切 0.6")."""
        scores = []
        for name in ("张伟", "张", "毫不相干"):
            f = dict(self.OK_FIELDS, recipient_name=name)
            r = self._run(f, "strict")
            scores.append((r["candidates"] or [{}])[0]["total_score"])
        self.assertGreaterEqual(scores[0], scores[1], f"全对不该低于部分对: {scores}")
        self.assertGreater(scores[1], scores[2], f"部分对不该低于全错: {scores}")

    # ---- 模糊分支 ----
    def test_both_modes_use_weighted_score_on_fuzzy_path(self):
        """单号没精确命中时走模糊分支, 两种模式行为一致 (都看加权分)。"""
        no_tn = {"recipient_phone": "13812341234",
                 "recipient_address": "广东省佛山市南海区桂城街道"}
        for mode in ("lenient", "strict"):
            r = self._run(no_tn, mode)
            self.assertEqual(r["match_type"], "fuzzy")
            self.assertEqual(r.get("release_mode"), mode)




class TestCompareFields(_WithDb):
    def test_returns_required_keys(self):
        """输出格式必须含需求列出的键."""
        r = self.match(self.records[0])
        for k in ("input_fields", "match_type", "candidates",
                  "total_candidates", "message"):
            self.assertIn(k, r, f"缺少必需字段 {k}")
        c = r["candidates"][0]
        for k in ("rank", "tracking_number", "recipient_name", "recipient_phone",
                  "recipient_address", "courier_company", "total_score",
                  "field_scores", "match_reason"):
            self.assertIn(k, c, f"候选缺少必需字段 {k}")

    def test_match_type_is_exact_or_fuzzy(self):
        self.assertIn(self.match(self.records[0])["match_type"], ("exact", "fuzzy"))

    def test_exact_match_full_score(self):
        r = self.match(self.records[0])
        self.assertEqual(r["match_type"], "exact")
        self.assertEqual(r["candidates"][0]["total_score"], 1.0)
        self.assertEqual(r["confidence"], "high")
        self.assertFalse(r["needs_review"])

    def test_exact_match_is_case_and_separator_insensitive(self):
        r = self.match({"tracking_number": "yt 0713-5905-25482"})
        self.assertEqual(r["match_type"], "exact")
        self.assertEqual(r["candidates"][0]["total_score"], 1.0)

    def test_exact_with_conflicting_fields_downgrades(self):
        """单号命中但电话/地址严重冲突 -> 加权分被拉低, 要求人工复核.

        注意: 默认是**严格模式**, 走加权打分, 所以理由文字说的是
        "加权总分不够", 而不是旧版的"字段冲突"。
        """
        r = self.match({"tracking_number": "YT0713590525482",
                        "recipient_phone": "13900001111",
                        "recipient_address": "北京市朝阳区建国路88号"})
        self.assertEqual(r["match_type"], "exact")
        self.assertLess(r["candidates"][0]["total_score"], 0.85)
        self.assertTrue(r["needs_review"])
        self.assertIn("加权总分", r["message"])

    def test_fuzzy_when_tracking_missing(self):
        r = self.match({"recipient_phone": "13812341234",
                        "recipient_address": "广东省佛山市南海区"})
        self.assertEqual(r["match_type"], "fuzzy")
        self.assertEqual(r["candidates"][0]["tracking_number"], "YT0713590525482")
        self.assertGreater(r["candidates"][0]["total_score"], 0.85)

    def test_fuzzy_finds_right_parcel_from_fragments(self):
        """核心场景: 条码破损, 只剩残片, 仍要找到正确的包裹."""
        r = self.match({"recipient_phone": "****1234",
                        "recipient_address": "南海区"})
        self.assertEqual(r["match_type"], "fuzzy")
        self.assertEqual(r["candidates"][0]["tracking_number"], "YT0713590525482")

    def test_candidates_are_ranked(self):
        r = self.match({"recipient_address": "区"})
        scores = [c["total_score"] for c in r["candidates"]]
        self.assertEqual(scores, sorted(scores, reverse=True), "候选必须按分数降序")
        self.assertEqual([c["rank"] for c in r["candidates"]], [1, 2, 3])

    def test_top_n_respected(self):
        r = self.match({"recipient_address": "区"}, top_n=1)
        self.assertEqual(len(r["candidates"]), 1)

    def test_field_scores_present_for_used_fields(self):
        r = self.match({"recipient_phone": "13812341234",
                        "recipient_address": "南海区"})
        fs = r["candidates"][0]["field_scores"]
        self.assertIn("recipient_phone", fs)
        self.assertIn("recipient_address", fs)

    def test_match_reason_is_human_readable(self):
        r = self.match({"recipient_phone": "13812341234",
                        "recipient_address": "广东省佛山市南海区"})
        reason = r["candidates"][0]["match_reason"]
        self.assertTrue(reason)
        self.assertTrue(any(w in reason for w in ("一致", "匹配", "相似")))

    # ---- 守卫 ----
    def test_evidence_guard_blocks_weak_full_score(self):
        """只读到快递公司+品名 (都不是身份字段), 分数可能满分但必须拦下."""
        r = self.match({"courier_company": "圆通速递",
                        "item_name": "1KF-FM7【云岩白】"})
        self.assertLess(r["evidence"], M.DEFAULT_CONFIG.min_evidence)
        self.assertEqual(r["confidence"], "medium")
        self.assertTrue(r["needs_review"])
        self.assertIn("信息过少", r["message"])

    def test_guard_can_be_disabled(self):
        cfg = M.MatcherConfig(min_evidence=0.0)
        r = self.match({"courier_company": "圆通速递",
                        "item_name": "1KF-FM7【云岩白】"}, config=cfg)
        self.assertEqual(r["confidence"], "high")

    def test_ambiguity_guard_blocks_when_two_identical(self):
        db.save_record({"tracking_number": "YT9999999999999",
                        "recipient_name": "张伟",
                        "recipient_phone": "13812341234",
                        "recipient_address": "广东省佛山市南海区桂城街道"},
                       path=self.db)
        r = self.match({"recipient_phone": "13812341234",
                        "recipient_address": "广东省佛山市南海区桂城街道"})
        self.assertTrue(r.get("ambiguous"))
        self.assertTrue(r["needs_review"])
        self.assertIn("多个相似候选", r["message"])

    # ---- 边界 ----
    def test_empty_input(self):
        r = self.match({})
        self.assertEqual(r["candidates"], [])
        self.assertTrue(r["needs_review"], "空输入绝不能被视为可放行")

    def test_all_none_input(self):
        r = self.match({"tracking_number": None, "recipient_phone": None})
        self.assertEqual(r["candidates"], [])
        self.assertTrue(r["needs_review"])

    def test_unrelated_input_gives_low_confidence(self):
        r = self.match({"recipient_name": "完全不认识的人",
                        "recipient_address": "黑龙江省哈尔滨市道里区"})
        self.assertEqual(r["confidence"], "low")
        self.assertTrue(r["needs_review"])

    def test_empty_database(self):
        empty = Path(self._dir.name) / "empty.db"
        db.init_database(empty)
        r = M.compare_fields({"tracking_number": "YT0713590525482"}, empty)
        self.assertEqual(r["candidates"], [])
        self.assertIn("空", r["message"])

    def test_missing_database_file(self):
        r = M.compare_fields({"tracking_number": "X"}, 
                             Path(self._dir.name) / "没有这个.db")
        self.assertEqual(r["candidates"], [])
        self.assertTrue(r["needs_review"])

    def test_candidates_can_be_supplied_directly(self):
        """不查库、直接给候选 (离线评测用)."""
        r = M.compare_fields({"recipient_phone": "13812341234"},
                             candidates=self.records)
        self.assertEqual(r["candidates"][0]["tracking_number"], "YT0713590525482")


# ==========================================================================
# 6. AI 接口
# ==========================================================================
class TestAiInterface(_WithDb):
    def test_dict_input(self):
        r = M.match_from_ai(self.records[0], self.db)
        self.assertEqual(r["match_type"], "exact")

    def test_wrapped_input(self):
        """AI 输出有时会包一层 fields/result/data."""
        for key in ("fields", "result", "data", "parcel", "record"):
            r = M.match_from_ai({key: dict(self.records[0])}, self.db)
            self.assertEqual(r["match_type"], "exact", f"包装键 {key} 未被识别")

    def test_extra_metadata_ignored(self):
        payload = dict(self.records[0])
        payload.update({"image": "/a/b.jpg", "confidence": 0.9})
        r = M.match_from_ai(payload, self.db)
        self.assertEqual(r["match_type"], "exact")

    def test_none_input(self):
        r = M.match_from_ai(None, self.db)
        self.assertEqual(r["candidates"], [])
        self.assertTrue(r["needs_review"])

    def test_wrong_type_input(self):
        r = M.match_from_ai("这应该是个字典", self.db)
        self.assertEqual(r["candidates"], [])
        self.assertIn("格式", r["message"])

    def test_list_passed_to_single_interface(self):
        """误把数组传给单张接口 -> 取第一条并给出提示."""
        r = M.match_from_ai([self.records[0], self.records[1]], self.db)
        self.assertIn("数组", r["message"])
        self.assertEqual(r["match_type"], "exact")

    def test_backend_reported(self):
        self.assertIn(M.similarity_backend(), ("fuzzywuzzy", "difflib"))
        r = M.match_from_ai(self.records[0], self.db)
        self.assertIn("backend", r)


class TestBatchInterface(_WithDb):
    def test_batch_list(self):
        out = M.match_from_ai_batch([self.records[0], self.records[1]], self.db)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["candidates"][0]["tracking_number"],
                         "YT0713590525482")
        self.assertEqual(out[1]["candidates"][0]["tracking_number"],
                         "SF743505166311")

    def test_batch_single_dict_still_returns_list(self):
        out = M.match_from_ai_batch(dict(self.records[0]), self.db)
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 1)

    def test_batch_wrapped_list(self):
        out = M.match_from_ai_batch({"results": [self.records[0]]}, self.db)
        self.assertEqual(len(out), 1)

    def test_batch_isolates_bad_items(self):
        """一条坏数据不影响整批."""
        out = M.match_from_ai_batch([self.records[0], None,
                                     "坏数据", self.records[1]], self.db)
        self.assertEqual(len(out), 4)
        self.assertEqual(out[0]["match_type"], "exact")
        self.assertEqual(out[1]["candidates"], [])
        self.assertEqual(out[2]["candidates"], [])
        self.assertEqual(out[3]["match_type"], "exact", "坏数据不应影响后续条目")

    def test_batch_empty(self):
        self.assertEqual(M.match_from_ai_batch([], self.db), [])
        out = M.match_from_ai_batch(None, self.db)
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0]["needs_review"])

    def test_batch_wrong_type(self):
        out = M.match_from_ai_batch("字符串", self.db)
        self.assertEqual(len(out), 1)


# ==========================================================================
# 7. 混合残缺程度 (需求强调"完整"与"极度残缺"都要能处理)
# ==========================================================================
class TestDegradationLevels(_WithDb):
    """从完整信息一路退化到只剩一个残片, 检查行为是否都合理."""

    CASES = [
        # (名称, 输入, 期望 match_type, 期望找到的单号)
        ("完整", {"tracking_number": "YT0713590525482", "recipient_name": "张伟",
                  "recipient_phone": "13812341234",
                  "recipient_address": "广东省佛山市南海区桂城街道"},
         "exact", "YT0713590525482"),
        ("单号+电话", {"tracking_number": "YT0713590525482",
                       "recipient_phone": "13812341234"},
         "exact", "YT0713590525482"),
        ("只剩单号", {"tracking_number": "YT0713590525482"},
         "exact", "YT0713590525482"),
        ("单号残缺+电话", {"tracking_number": "25482",
                           "recipient_phone": "13812341234"},
         "fuzzy", "YT0713590525482"),
        ("电话+区名", {"recipient_phone": "****1234",
                       "recipient_address": "南海区"},
         "fuzzy", "YT0713590525482"),
        ("只有完整电话", {"recipient_phone": "13812341234"},
         "fuzzy", "YT0713590525482"),
        ("只有姓+区名", {"recipient_name": "张", "recipient_address": "南海区"},
         "fuzzy", "YT0713590525482"),
    ]

    def test_all_levels_find_correct_parcel(self):
        for name, fields, want_type, want_tn in self.CASES:
            with self.subTest(case=name):
                r = self.match(fields)
                self.assertEqual(r["match_type"], want_type,
                                 f"{name}: match_type 不符 ({r['message']})")
                self.assertTrue(r["candidates"], f"{name}: 应至少有一个候选")
                self.assertEqual(r["candidates"][0]["tracking_number"], want_tn,
                                 f"{name}: 找到的是 "
                                 f"{r['candidates'][0]['tracking_number']}")

    def test_only_strong_cases_auto_pass(self):
        """自动放行的必须是证据充分的场景 —— 残片不能自动放行."""
        must_not_auto = {"电话+区名", "只有姓+区名"}
        for name, fields, _t, _tn in self.CASES:
            if name not in must_not_auto:
                continue
            with self.subTest(case=name):
                r = self.match(fields)
                self.assertTrue(r["needs_review"], f"{name} 不该自动放行")


# ==========================================================================
# 8. 输出可直接 JSON 序列化 (前端要能直接吃)
# ==========================================================================
class TestSerialisable(_WithDb):
    def test_result_is_json_serialisable(self):
        r = self.match(self.records[0])
        text = json.dumps(r, ensure_ascii=False)
        back = json.loads(text)
        self.assertEqual(back["match_type"], "exact")

    def test_batch_is_json_serialisable(self):
        out = M.match_from_ai_batch([self.records[0], {"recipient_phone": "1234"}],
                                    self.db)
        json.dumps(out, ensure_ascii=False)

    def test_no_none_in_candidate_values(self):
        """候选里的展示字段不应是 None (前端直接渲染)."""
        r = self.match({"recipient_phone": "13812341234"})
        c = r["candidates"][0]
        for k in ("tracking_number", "recipient_name", "recipient_phone",
                  "recipient_address", "courier_company"):
            self.assertIsNotNone(c[k], f"{k} 不应为 None")


if __name__ == "__main__":
    unittest.main(verbosity=2)
