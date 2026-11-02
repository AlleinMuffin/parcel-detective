"""信息库 (SQLite) 与字段解析的测试.

运行::

    python tests/test_database.py
"""
from __future__ import annotations

import csv
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core import database as db          # noqa: E402


class _TempDb(unittest.TestCase):
    """每个用例用独立的临时数据库, 互不干扰."""

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory(prefix="pdb_test_")
        self.db = Path(self._dir.name) / "parcels.db"
        db.init_db(self.db)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def add(self, **kw):
        return db.save_to_database(kw, path=self.db)


# ==========================================================================
# 1. 建表
# ==========================================================================
class TestInit(_TempDb):
    def test_creates_file_and_table(self):
        self.assertTrue(self.db.is_file(), "init_db 应创建数据库文件")
        with db.connect(self.db) as conn:
            cols = [r["name"] for r in conn.execute("PRAGMA table_info(parcels)")]
        expected = ["id", "tracking_number", "recipient_name", "recipient_phone",
                    "recipient_address", "sender_name", "sender_phone",
                    "sender_address", "sorting_code", "order_number", "item_name",
                    "cod_amount", "insured_amount", "shipping_date",
                    "courier_company", "sender_branch", "image_path",
                    "is_damaged", "damage_type", "created_at"]
        self.assertEqual(cols, expected, "表结构必须与题目给定的一致")

    def test_idempotent(self):
        db.init_db(self.db)
        db.init_db(self.db)
        self.add(tracking_number="A1")
        self.assertEqual(db.count_parcels(self.db), 1, "重复初始化不应丢数据")

    def test_created_at_defaults(self):
        self.add(tracking_number="A1")
        row = db.list_parcels(self.db)[0]
        self.assertTrue(row["created_at"], "created_at 应有默认值")

    def test_migration_adds_missing_column(self):
        """老库缺列时应自动补上, 不用删库重建."""
        with db.connect(self.db) as conn:
            conn.execute("ALTER TABLE parcels DROP COLUMN damage_type")
            conn.commit()
        db.init_db(self.db)
        with db.connect(self.db) as conn:
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(parcels)")}
        self.assertIn("damage_type", cols)


# ==========================================================================
# 2. 录入
# ==========================================================================
class TestSave(_TempDb):
    def test_insert(self):
        out = self.add(tracking_number="JD681146975643", recipient_name="杨小明",
                       recipient_phone="17396046368")
        self.assertEqual(out.status, "inserted")
        self.assertIsNotNone(out.id)
        self.assertEqual(db.count_parcels(self.db), 1)

    def test_chinese_roundtrip(self):
        """中文必须原样存取, 不能乱码."""
        self.add(tracking_number="SF1", recipient_name="王大力",
                 recipient_address="成都市武侯区人民南路3段", courier_company="顺丰速运")
        row = db.list_parcels(self.db)[0]
        self.assertEqual(row["recipient_name"], "王大力")
        self.assertEqual(row["recipient_address"], "成都市武侯区人民南路3段")
        self.assertEqual(row["courier_company"], "顺丰速运")

    def test_missing_fields_become_empty(self):
        self.add(tracking_number="X1")
        row = db.list_parcels(self.db)[0]
        for f in db.PARCEL_FIELDS:
            self.assertIsNotNone(row[f], f"{f} 不应为 None")

    def test_is_damaged_normalised(self):
        self.add(tracking_number="D1", is_damaged=True)
        self.add(tracking_number="D2", is_damaged="是")
        self.add(tracking_number="D3", is_damaged=0)
        rows = {r["tracking_number"]: r["is_damaged"] for r in db.list_parcels(self.db)}
        self.assertEqual(rows, {"D1": 1, "D2": 1, "D3": 0})

    def test_extra_keys_ignored(self):
        """传了表里没有的键不应报错, 也不应写进去."""
        out = db.save_to_database({"tracking_number": "Z1", "不存在的字段": "x"},
                                  path=self.db)
        self.assertEqual(out.status, "inserted")

    # ---- 查重 ----
    def test_duplicate_skip_is_default(self):
        self.add(tracking_number="JD1234567890", recipient_name="甲")
        out = self.add(tracking_number="JD1234567890", recipient_name="乙")
        self.assertEqual(out.status, "skipped")
        self.assertEqual(len(out.existing), 1)
        self.assertEqual(db.count_parcels(self.db), 1)
        self.assertEqual(db.list_parcels(self.db)[0]["recipient_name"], "甲",
                         "skip 策略下原有数据不能被改动")

    def test_duplicate_overwrite(self):
        first = self.add(tracking_number="JD1234567890", recipient_name="甲")
        out = self.add(tracking_number="JD1234567890", recipient_name="乙")
        # 上面那次是 skip, 再显式指定 overwrite
        out = db.save_to_database({"tracking_number": "JD1234567890",
                                   "recipient_name": "乙"},
                                  path=self.db, on_duplicate="overwrite")
        self.assertEqual(out.status, "overwritten")
        self.assertEqual(out.id, first.id, "overwrite 不应新建行")
        self.assertEqual(db.count_parcels(self.db), 1)
        self.assertEqual(db.list_parcels(self.db)[0]["recipient_name"], "乙")

    def test_duplicate_new(self):
        self.add(tracking_number="JD1234567890")
        out = db.save_to_database({"tracking_number": "JD1234567890"},
                                  path=self.db, on_duplicate="new")
        self.assertEqual(out.status, "inserted")
        self.assertEqual(db.count_parcels(self.db), 2)

    def test_duplicate_detection_tolerates_spacing(self):
        """OCR 常把单号读成带空格的, 查重不能因此漏判."""
        self.add(tracking_number="JD681146975643")
        out = self.add(tracking_number="JD 6811 4697 5643")
        self.assertEqual(out.status, "skipped", "规范化后应判定为重复")

    def test_empty_tracking_never_deduped(self):
        """没有单号时无从查重, 应允许重复录入."""
        self.add(recipient_name="甲")
        out = self.add(recipient_name="乙")
        self.assertEqual(out.status, "inserted")
        self.assertEqual(db.count_parcels(self.db), 2)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            db.save_to_database({"tracking_number": "A"}, path=self.db,
                                on_duplicate="乱写")


# ==========================================================================
# 3. 查询
# ==========================================================================
class TestSearch(_TempDb):
    def setUp(self):
        super().setUp()
        self.add(tracking_number="JD681146975643", recipient_name="杨小明",
                 recipient_phone="17396046368",
                 recipient_address="成都市武侯区人民南路3段",
                 courier_company="京东物流", sorting_code="SORT-376", is_damaged=0)
        self.add(tracking_number="SF743505166311", recipient_name="王大力",
                 recipient_phone="13900001111",
                 recipient_address="深圳市南山区科技园南路5号",
                 courier_company="顺丰速运", sorting_code="SORT-848", is_damaged=1,
                 damage_type="tear")
        self.add(tracking_number="STO580548948698", recipient_name="李大壮",
                 recipient_phone="13800002222",
                 recipient_address="上海市浦东新区世纪大道100号",
                 courier_company="申通快递", is_damaged=1, damage_type="stain")

    def test_search_by_tracking_suffix(self):
        """题目要求: 按单号后缀检索."""
        rows = db.search_parcels(tracking_number="25482", path=self.db)
        # 上面三条里没有以 25482 结尾的, 换个真实存在的后缀
        rows = db.search_parcels(tracking_number="75643", path=self.db)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["recipient_name"], "杨小明")

    def test_search_by_phone_tail(self):
        """题目要求: 按电话尾号检索."""
        rows = db.search_parcels(recipient_phone="6368", path=self.db)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["recipient_name"], "杨小明")

    def test_search_by_address_keyword(self):
        """题目要求: 按地址关键词检索."""
        rows = db.search_parcels(recipient_address="武侯区", path=self.db)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["recipient_name"], "杨小明")

    def test_search_case_insensitive_tracking(self):
        rows = db.search_parcels(tracking_number="jd6811", path=self.db)
        self.assertEqual(len(rows), 1)

    def test_search_phone_with_spaces(self):
        """用户输入带空格/横杠也应命中."""
        rows = db.search_parcels(recipient_phone="139 0000 1111", path=self.db)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["recipient_name"], "王大力")

    def test_search_multiple_fields_is_and(self):
        rows = db.search_parcels(courier_company="顺丰", is_damaged=1, path=self.db)
        self.assertEqual(len(rows), 1)
        rows = db.search_parcels(courier_company="顺丰", is_damaged=0, path=self.db)
        self.assertEqual(rows, [], "不同字段之间应是 AND 关系")

    def test_search_is_damaged_filter(self):
        self.assertEqual(len(db.search_parcels(is_damaged=1, path=self.db)), 2)
        self.assertEqual(len(db.search_parcels(is_damaged=0, path=self.db)), 1)

    def test_empty_filter_ignored(self):
        self.assertEqual(len(db.search_parcels(tracking_number="", path=self.db)), 3)

    def test_percent_is_escaped(self):
        """用户输入 % 不应变成"匹配全部"."""
        rows = db.search_parcels(recipient_name="%", path=self.db)
        self.assertEqual(rows, [], "% 必须被转义, 否则会匹配到所有记录")
        rows = db.search_parcels(recipient_name="_", path=self.db)
        self.assertEqual(rows, [])

    def test_unknown_field_raises(self):
        with self.assertRaises(KeyError):
            db.search_parcels(不存在的字段="x", path=self.db)

    def test_search_any_across_fields(self):
        """一个关键词跨字段搜索 (界面搜索框用)."""
        self.assertEqual(len(db.search_any("武侯", path=self.db)), 1)
        self.assertEqual(len(db.search_any("SORT-848", path=self.db)), 1)
        self.assertEqual(len(db.search_any("王大力", path=self.db)), 1)
        self.assertEqual(len(db.search_any("139", path=self.db)), 1)

    def test_search_any_empty_returns_all(self):
        self.assertEqual(len(db.search_any("", path=self.db)), 3)

    def test_search_any_no_match(self):
        self.assertEqual(db.search_any("这个关键词不存在", path=self.db), [])

    def test_find_by_tracking(self):
        rows = db.find_by_tracking("SF743505166311", path=self.db)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["courier_company"], "顺丰速运")
        self.assertEqual(db.find_by_tracking("", path=self.db), [])

    def test_get_and_list(self):
        first = db.list_parcels(self.db, limit=1)[0]
        got = db.get_parcel(first["id"], path=self.db)
        self.assertEqual(got["id"], first["id"])
        self.assertIsNone(db.get_parcel(999999, path=self.db))

    def test_limit(self):
        self.assertEqual(len(db.list_parcels(self.db, limit=2)), 2)


# ==========================================================================
# 4. 修改与删除
# ==========================================================================
class TestEdit(_TempDb):
    def test_update(self):
        pid = self.add(tracking_number="A1", recipient_name="错的名字").id
        self.assertTrue(db.update_parcel(pid, path=self.db, recipient_name="正确的名字"))
        self.assertEqual(db.get_parcel(pid, self.db)["recipient_name"], "正确的名字")

    def test_update_ignores_unknown_fields(self):
        pid = self.add(tracking_number="A1").id
        self.assertFalse(db.update_parcel(pid, path=self.db, 未知="x"))

    def test_update_can_mark_damaged(self):
        pid = self.add(tracking_number="A1").id
        db.update_parcel(pid, path=self.db, is_damaged=1, damage_type="tear")
        row = db.get_parcel(pid, self.db)
        self.assertEqual(row["is_damaged"], 1)
        self.assertEqual(row["damage_type"], "tear")

    def test_delete(self):
        pid = self.add(tracking_number="A1").id
        self.assertTrue(db.delete_parcel(pid, path=self.db))
        self.assertEqual(db.count_parcels(self.db), 0)
        self.assertFalse(db.delete_parcel(pid, path=self.db))


# ==========================================================================
# 5. 统计
# ==========================================================================
class TestStats(_TempDb):
    def test_empty_db(self):
        s = db.stats(self.db)
        self.assertEqual(s["total"], 0)
        self.assertEqual(s["damaged"], 0)
        self.assertTrue(s["db_path"].endswith("parcels.db"))

    def test_counts_and_distribution(self):
        self.add(tracking_number="A", courier_company="京东物流", is_damaged=0)
        self.add(tracking_number="B", courier_company="京东物流", is_damaged=1,
                 damage_type="tear")
        self.add(tracking_number="C", courier_company="顺丰速运", is_damaged=1,
                 damage_type="tear")
        self.add(recipient_name="没有单号")          # tracking 为空

        s = db.stats(self.db)
        self.assertEqual(s["total"], 4)
        self.assertEqual(s["damaged"], 2)
        self.assertEqual(s["missing_tracking"], 1)
        dist = {d["name"]: d["count"] for d in s["by_courier"]}
        self.assertEqual(dist.get("京东物流"), 2)
        self.assertEqual(dist.get("顺丰速运"), 1)
        dmg = {d["name"]: d["count"] for d in s["by_damage_type"]}
        self.assertEqual(dmg.get("tear"), 2)
        self.assertTrue(s["earliest"] and s["latest"])

    def test_missing_db_file_is_safe(self):
        s = db.stats(Path(self._dir.name) / "还没建的.db")
        self.assertEqual(s["total"], 0)


# ==========================================================================
# 6. 导出
# ==========================================================================
class TestExport(_TempDb):
    def test_export_all(self):
        self.add(tracking_number="A", recipient_name="甲")
        self.add(tracking_number="B", recipient_name="乙")
        out = Path(self._dir.name) / "out.csv"
        n = db.export_csv(out, path=self.db)
        self.assertEqual(n, 2)
        with open(out, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 2)
        self.assertEqual(list(rows[0].keys()), list(db.ALL_COLUMNS))
        # 导出按 id 倒序 (最新在前, 与界面一致), 因此这里比集合而不是顺序
        self.assertEqual({r["recipient_name"] for r in rows}, {"甲", "乙"})

    def test_export_subset(self):
        self.add(tracking_number="A", recipient_name="甲")
        self.add(tracking_number="B", recipient_name="乙")
        rows = db.search_parcels(tracking_number="A", path=self.db)
        out = Path(self._dir.name) / "sub.csv"
        self.assertEqual(db.export_csv(out, path=self.db, rows=rows), 1)

    def test_export_has_bom_for_excel(self):
        """带 BOM 才能让 Excel 正确识别 UTF-8 中文."""
        self.add(tracking_number="A", recipient_name="甲")
        out = Path(self._dir.name) / "bom.csv"
        db.export_csv(out, path=self.db)
        self.assertTrue(out.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_export_empty(self):
        out = Path(self._dir.name) / "empty.csv"
        self.assertEqual(db.export_csv(out, path=self.db), 0)
        self.assertTrue(out.is_file())


# ==========================================================================
# 7. 路径解析
# ==========================================================================
class TestPath(unittest.TestCase):
    def test_program_root_is_version_dir(self):
        self.assertEqual(db.program_root(), ROOT)

    def test_resolve_default_location(self):
        p = db.resolve_db_path()
        self.assertEqual(p.name, "parcels.db")
        self.assertTrue(str(p).endswith("parcels.db"))

    def test_resolve_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            p = db.resolve_db_path(Path(d) / "custom.db")
            self.assertEqual(p.name, "custom.db")


# ==========================================================================
# 8. 字段解析
# ==========================================================================
class TestBackup(_TempDb):
    def setUp(self) -> None:
        super().setUp()
        # 每个用例用独立备份目录, 避免互相干扰 (默认目录是全局共享的)
        self.bdir = Path(self._dir.name) / "backup"

    def test_backup_created_outside_program_dir(self):
        """备份必须落在程序目录之外 —— 同目录的备份会被升级一起覆盖掉."""
        self.add(tracking_number="B1", recipient_name="备份测试")
        # 这里刻意用默认目录, 才能验证"默认位置在程序目录之外"
        dest = db.auto_backup(self.db)
        self.assertIsNotNone(dest, "有数据时应生成备份")
        self.assertTrue(Path(dest).is_file())

        prog = db.program_root().resolve()
        try:
            Path(dest).resolve().relative_to(prog)
            self.fail(f"备份落在程序目录内 ({dest}), 升级时会被一起覆盖")
        except ValueError:
            pass                                    # 不在程序目录内 = 正确

    def test_backup_is_readable_and_complete(self):
        self.add(tracking_number="B1")
        self.add(tracking_number="B2")
        dest = db.auto_backup(self.db, dest_dir=self.bdir)
        self.assertIsNotNone(dest)
        self.assertEqual(db.count_parcels(dest), 2, "备份内容应与原库一致")

    def test_backup_once_per_day(self):
        """同一天重复调用不应反复写盘."""
        self.add(tracking_number="B1")
        first = db.auto_backup(self.db, dest_dir=self.bdir)
        self.add(tracking_number="B2")
        second = db.auto_backup(self.db, dest_dir=self.bdir)
        self.assertEqual(str(first), str(second), "当天应复用同一份备份")
        self.assertEqual(db.count_parcels(first), 1, "当天备份内容不会被改写")

    def test_backup_skips_missing_db(self):
        self.assertIsNone(db.auto_backup(Path(self._dir.name) / "没有.db",
                                         dest_dir=self.bdir))

    def test_backup_keeps_only_recent(self):
        """旧备份应被清理, 只留最近 keep 份."""
        import shutil

        self.add(tracking_number="B1")
        db.auto_backup(self.db, dest_dir=self.bdir)          # 生成今天那份
        today = db.auto_backup(self.db, dest_dir=self.bdir)
        # 再伪造三份"旧备份"
        for d in ("20200101", "20200102", "20200103"):
            shutil.copy2(today, self.bdir / f"parcels_{d}.db")
        self.assertEqual(len(list(self.bdir.glob("parcels_*.db"))), 4)

        db.auto_backup(self.db, dest_dir=self.bdir, keep=2)
        left = sorted(p.name for p in self.bdir.glob("parcels_*.db"))
        self.assertEqual(len(left), 2, f"应只留 2 份, 实际 {left}")
        self.assertEqual(left[-1], today.name, "最新的一份必须保留")

    def test_list_backups(self):
        self.add(tracking_number="B1")
        db.auto_backup(self.db, dest_dir=self.bdir)
        items = db.list_backups(dest_dir=self.bdir)
        self.assertTrue(any(i["records"] >= 1 for i in items))
        self.assertTrue(all("path" in i and "time" in i for i in items))
        self.assertEqual(db.list_backups(dest_dir=Path(self._dir.name) / "无"),
                         [])


# ==========================================================================
# 10. JSON 导入
# ==========================================================================
class TestJsonImport(_TempDb):
    """导入 JSON 到数据库.

    覆盖题目要求的全部行为: 单对象/数组、解析失败跳过、单号查重、
    null 转空串、image_path 由文件名反推、计数与提示措辞。
    """

    def setUp(self) -> None:
        super().setUp()
        self.files = Path(self._dir.name) / "json"
        self.files.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, obj=None, *, raw: Optional[str] = None,
              encoding: str = "utf-8") -> Path:
        p = self.files / name
        text = raw if raw is not None else json.dumps(obj, ensure_ascii=False, indent=2)
        p.write_text(text, encoding=encoding)
        return p

    def sample_object(self, tracking: str = "YT0713590525482") -> dict:
        """题目给出的单对象 JSON 格式."""
        return {
            "tracking_number": tracking,
            "recipient_name": "张**",
            "recipient_phone": "138****1234",
            "recipient_address": "广东省佛山市南海区**",
            "sender_name": None, "sender_phone": None, "sender_address": None,
            "sorting_code": "626-M08-00-601", "order_number": None,
            "item_name": "1KF-FM7【云岩白】", "cod_amount": None,
            "insured_amount": None,
            "shipping_date": "2026-01-12 12:29",
            "courier_company": "圆通速递", "sender_branch": None,
        }

    def test_init_database_alias(self):
        """题目要求的函数名 init_database() 应可用."""
        p = Path(self._dir.name) / "别名.db"
        out = db.init_database(p)
        self.assertEqual(out, p)
        self.assertTrue(p.is_file())
        self.assertEqual(db.count_parcels(p), 0)

    def test_single_object(self):
        f = self.write("001.json", self.sample_object())
        rep = db.import_json_files([f], path=self.db)
        self.assertEqual((rep.inserted, rep.skipped, rep.failed), (1, 0, 0))
        self.assertEqual(rep.summary_text(), "成功 1 条，跳过 0 条（重复），失败 0 条")

        row = db.list_parcels(self.db)[0]
        self.assertEqual(row["tracking_number"], "YT0713590525482")
        self.assertEqual(row["recipient_name"], "张**")
        self.assertEqual(row["courier_company"], "圆通速递")
        self.assertEqual(row["sorting_code"], "626-M08-00-601")
        self.assertEqual(row["item_name"], "1KF-FM7【云岩白】")

    def test_null_becomes_empty_string(self):
        """题目要求: JSON 里的 null 写入数据库时存为空字符串."""
        f = self.write("001.json", self.sample_object())
        db.import_json_files([f], path=self.db)
        row = db.list_parcels(self.db)[0]
        for k in ("sender_name", "sender_phone", "sender_address",
                  "order_number", "cod_amount", "insured_amount", "sender_branch"):
            self.assertEqual(row[k], "", f"{k} 应为空字符串而不是 None")
            self.assertIsNotNone(row[k], f"{k} 不应为 None")

    def test_array_of_objects(self):
        f = self.write("002.json", [
            {"tracking_number": "SF743505166311", "recipient_name": "李四"},
            {"tracking_number": "JD681146975643", "recipient_name": "王五"},
        ])
        rep = db.import_json_files([f], path=self.db)
        self.assertEqual(rep.inserted, 2)
        self.assertEqual(db.count_parcels(self.db), 2)
        self.assertEqual(rep.ok_files, 1)

    def test_wrapped_array(self):
        """有些导出工具会把数组包一层, 应能识别."""
        for key in ("records", "parcels", "data", "items"):
            sub = Path(self._dir.name) / f"db_{key}.db"
            f = self.write(f"w_{key}.json", {key: [{"tracking_number": f"T{key}"}]})
            rep = db.import_json_files([f], path=sub)
            self.assertEqual(rep.inserted, 1, f"包装键 {key!r} 未被识别")

    def test_wrapped_single_record(self):
        f = self.write("w.json", {"fields": {"tracking_number": "WRAP001"}})
        rep = db.import_json_files([f], path=self.db)
        self.assertEqual(rep.inserted, 1)

    @unittest.skip("v2.0.0 移除了 fields.py：不再支持从 OCR 行数据（lines）反推结构化字段。\n        v2 的 JSON 导入仍支持结构化的 15 字段格式，只是少了这条 OCR 兜底。如将来恢复 OCR，把 fields.py 放回来即可。")
    def test_our_own_pipeline_export_roundtrip(self):
        """本程序导出的识别结果 (只有 lines, 没有结构化字段) 也要能导回来.

        没有这条兜底, 自己导出的 JSON 就导不回去 —— 那这个功能等于半残。
        """
        f = self.write("识别结果.json", {
            "source_file": "D:/某张面单.jpg",
            "app_version": "1.2.0",
            "lines": [
                {"text": "单号", "confidence": 0.99,
                 "bbox": {"x1": 0, "y1": 0, "x2": 60, "y2": 20}},
                {"text": "JD681146975643", "confidence": 0.99,
                 "bbox": {"x1": 70, "y1": 0, "x2": 260, "y2": 20}},
                {"text": "收件人", "confidence": 0.99,
                 "bbox": {"x1": 0, "y1": 40, "x2": 70, "y2": 60}},
                {"text": "杨小明 17396046368", "confidence": 0.99,
                 "bbox": {"x1": 80, "y1": 40, "x2": 300, "y2": 60}},
            ],
        })
        rep = db.import_json_files([f], path=self.db)
        self.assertEqual(rep.inserted, 1, rep.detail_text())
        row = db.list_parcels(self.db)[0]
        self.assertEqual(row["tracking_number"], "JD681146975643")
        self.assertEqual(row["recipient_name"], "杨小明")
        self.assertEqual(row["recipient_phone"], "17396046368")

    # ---- 查重 ----
    def test_duplicate_is_skipped(self):
        f1 = self.write("a.json", self.sample_object())
        f2 = self.write("b.json", self.sample_object())      # 同一个单号
        rep = db.import_json_files([f1, f2], path=self.db)
        self.assertEqual((rep.inserted, rep.skipped), (1, 1))
        self.assertEqual(db.count_parcels(self.db), 1)
        self.assertEqual(rep.summary_text(), "成功 1 条，跳过 1 条（重复），失败 0 条")

    def test_duplicate_tolerates_spacing(self):
        """OCR 常把单号读成带空格的, 查重不能因此漏判."""
        db.save_to_database({"tracking_number": "YT0713590525482"}, path=self.db)
        f = self.write("sp.json", {"tracking_number": "YT 0713 5905 25482"})
        rep = db.import_json_files([f], path=self.db)
        self.assertEqual(rep.skipped, 1)

    def test_duplicate_inside_same_batch(self):
        f = self.write("dup.json", [{"tracking_number": "SAME1"},
                                    {"tracking_number": "SAME1"}])
        rep = db.import_json_files([f], path=self.db)
        self.assertEqual((rep.inserted, rep.skipped), (1, 1))

    def test_check_duplicate_function(self):
        db.save_to_database({"tracking_number": "YT0713590525482"}, path=self.db)
        self.assertTrue(db.check_duplicate("YT0713590525482", self.db))
        self.assertTrue(db.check_duplicate("yt0713590525482", self.db))
        self.assertTrue(db.check_duplicate("YT 0713 5905 25482", self.db))
        self.assertFalse(db.check_duplicate("不存在的单号", self.db))
        self.assertFalse(db.check_duplicate("", self.db))
        self.assertFalse(db.check_duplicate(None, self.db))

    # ---- image_path ----
    def test_image_path_inferred_from_filename(self):
        self.write("001.json", self.sample_object())
        (self.files / "001.jpg").write_bytes(b"fake")        # 图片确实存在
        rep = db.import_json_files([self.files / "001.json"], path=self.db)
        self.assertEqual(rep.inserted, 1)
        self.assertTrue(db.list_parcels(self.db)[0]["image_path"].endswith("001.jpg"))

    def test_image_path_falls_back_to_jpg(self):
        """图片不存在时也按规则给一个 .jpg 路径."""
        f = self.write("no_img.json", {"tracking_number": "NI1"})
        db.import_json_files([f], path=self.db)
        self.assertTrue(db.list_parcels(self.db)[0]["image_path"]
                        .endswith("no_img.jpg"))

    def test_json_image_path_wins(self):
        """JSON 里已写明 image_path 时应尊重它, 而不是用反推的."""
        f = self.write("x.json", {"tracking_number": "X1",
                                  "image_path": "/已有/路径/photo.png"})
        db.import_json_files([f], path=self.db)
        self.assertEqual(db.list_parcels(self.db)[0]["image_path"],
                         "/已有/路径/photo.png")

    def test_infer_image_path_prefers_existing(self):
        js = self.files / "p.json"
        js.write_text("{}", encoding="utf-8")
        (self.files / "p.png").write_bytes(b"x")
        self.assertTrue(db.infer_image_path(js).endswith("p.png"))

    def test_save_record_function(self):
        """题目要求的函数名 save_record(record, image_path) 应可用."""
        out = db.save_record({"tracking_number": "SR1"}, image_path="/a/b.jpg",
                             path=self.db)
        self.assertEqual(out.status, "inserted")
        self.assertEqual(db.get_parcel(out.id, self.db)["image_path"], "/a/b.jpg")

    # ---- 容错 ----
    def test_bad_json_does_not_break_others(self):
        """题目要求: 单个文件解析失败不影响其它文件."""
        good = self.write("good.json", {"tracking_number": "GOOD1"})
        bad = self.write("bad.json", raw="{ 这不是合法 JSON")
        also_good = self.write("good2.json", {"tracking_number": "GOOD2"})

        rep = db.import_json_files([bad, good, also_good], path=self.db)
        self.assertEqual(rep.inserted, 2, "坏文件不应影响其它文件")
        self.assertEqual(rep.failed, 1)
        self.assertEqual(rep.failed_files, 1)
        self.assertEqual(rep.ok_files, 2)
        self.assertEqual(db.count_parcels(self.db), 2)

        bad_entry = [f for f in rep.files if f["file"].endswith("bad.json")][0]
        self.assertTrue(bad_entry["error"], "失败的文件必须记录原因")
        self.assertIn("bad.json", rep.detail_text())

    def test_missing_file(self):
        rep = db.import_json_files([self.files / "不存在.json"], path=self.db)
        self.assertEqual(rep.failed, 1)
        self.assertIn("不存在", rep.files[0]["error"])

    def test_empty_array(self):
        f = self.write("empty.json", [])
        rep = db.import_json_files([f], path=self.db)
        self.assertEqual(rep.failed, 1)
        self.assertTrue(rep.files[0]["error"])

    def test_unsupported_toplevel(self):
        f = self.write("num.json", raw="123")
        rep = db.import_json_files([f], path=self.db)
        self.assertEqual(rep.failed, 1)

    def test_unrecognised_object(self):
        f = self.write("weird.json", {"foo": "bar", "baz": 1})
        rep = db.import_json_files([f], path=self.db)
        self.assertEqual(rep.failed, 1)
        self.assertIn("没有可识别的面单字段", rep.files[0]["error"])

    def test_utf8_bom_file(self):
        """带 BOM 的 JSON (记事本另存为 UTF-8 会产生) 也要能读."""
        f = self.write("bom.json", {"tracking_number": "BOM1"}, encoding="utf-8-sig")
        rep = db.import_json_files([f], path=self.db)
        self.assertEqual(rep.inserted, 1, rep.detail_text())

    def test_record_without_tracking_is_counted_and_warned(self):
        """没有单号时无法查重, 仍入库但要在报告里提醒."""
        f = self.write("notrack.json", {"recipient_name": "无单号"})
        rep = db.import_json_files([f], path=self.db)
        self.assertEqual(rep.inserted, 1)
        self.assertEqual(rep.empty_tracking, 1)
        self.assertIn("没有单号", rep.detail_text())

    def test_empty_file_list(self):
        rep = db.import_json_files([], path=self.db)
        self.assertEqual((rep.inserted, rep.skipped, rep.failed), (0, 0, 0))
        self.assertEqual(rep.summary_text(), "成功 0 条，跳过 0 条（重复），失败 0 条")

    def test_import_creates_database_if_absent(self):
        """题目要求: 启动时不存在则建库建表; 导入也应保证表存在."""
        fresh = Path(self._dir.name) / "全新.db"
        self.assertFalse(fresh.is_file())
        f = self.write("n.json", {"tracking_number": "NEW1"})
        rep = db.import_json_files([f], path=fresh)
        self.assertEqual(rep.inserted, 1)
        self.assertTrue(fresh.is_file())

    def test_chinese_not_mangled(self):
        f = self.write("cn.json", {"tracking_number": "CN1",
                                   "recipient_address": "广东省佛山市南海区桂城街道",
                                   "item_name": "1KF-FM7【云岩白】"})
        db.import_json_files([f], path=self.db)
        row = db.list_parcels(self.db)[0]
        self.assertEqual(row["recipient_address"], "广东省佛山市南海区桂城街道")
        self.assertEqual(row["item_name"], "1KF-FM7【云岩白】")

    def test_many_files_counts(self):
        paths = [self.write(f"m{i}.json", {"tracking_number": f"M{i}"})
                 for i in range(5)]
        rep = db.import_json_files(paths, path=self.db)
        self.assertEqual(rep.inserted, 5)
        self.assertEqual(rep.total, 5)
        self.assertEqual(len(rep.files), 5)


# ==========================================================================
# 11. 导出到默认 parcels.csv
# ==========================================================================
class TestDefaultExport(_TempDb):
    def test_default_csv_path(self):
        p = db.default_csv_path(self.db)
        self.assertEqual(p.name, "parcels.csv")
        self.assertEqual(p.parent, self.db.parent)

    def test_export_csv_without_arguments(self):
        """题目要求 export_csv() 能直接导出为 parcels.csv."""
        self.add(tracking_number="E1", recipient_name="甲")
        self.add(tracking_number="E2", recipient_name="乙")
        n = db.export_csv(None, path=self.db)          # 不传路径 -> 默认位置
        self.assertEqual(n, 2)
        out = db.default_csv_path(self.db)
        self.assertTrue(out.is_file())
        with open(out, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["recipient_name"] for r in rows}, {"甲", "乙"})

    def test_export_csv_default_has_bom(self):
        self.add(tracking_number="E1")
        db.export_csv(None, path=self.db)
        self.assertTrue(db.default_csv_path(self.db).read_bytes()
                        .startswith(b"\xef\xbb\xbf"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
