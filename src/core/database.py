"""本地信息库 (SQLite) —— 建表 / 录入 / 查重 / 查询 / 更新 / 导出 / 统计.

设计要点
--------
1. **只用 Python 内置的 sqlite3**, 不引入任何第三方依赖。
2. **数据库文件放在程序根目录** (``parcels.db``), 便于随程序一起拷走;
   若程序目录不可写 (例如放在 Program Files 下), 自动退回到用户目录,
   并通过 :func:`resolve_db_path` 的返回值告知实际位置。
3. **线程安全**: 每次操作开一个新连接, 并用模块级锁串行化写操作。
   sqlite3 的连接默认不能跨线程使用, "一次操作一个连接" 从根上规避该问题,
   代价是每次多几毫秒, 对"识别一张写一条"的场景完全可以忽略。
4. **时间**: 表里 ``created_at`` 按题目要求用 ``CURRENT_TIMESTAMP``,
   而 SQLite 的这个函数返回的是 **UTC**。为了方便人工核对, 所有读取接口
   返回的 ``created_at`` 都已转换成**本地时间**字符串, 详见 :func:`_row_to_dict`。
5. **中文**: 全程 UTF-8。连接创建后显式设置 ``text_factory=str``;
   导出的 CSV 使用 ``utf-8-sig`` (带 BOM), 这样 Excel 直接双击打开不乱码。
"""
from __future__ import annotations

import csv
import datetime as _dt
import json as _json
import sqlite3
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# 可选依赖: 只在"JSON 里没有结构化字段、但有 OCR 行数据"时, 用它现场反推字段。
# 用 try 包住是刻意的 —— 本模块要能单独拷贝到别的程序里使用, 那时没有 fields
# 模块也不该导致 import 失败, 只是少一个兜底能力而已。
try:                                                    # pragma: no cover
    from . import fields as _fields
except Exception:                                       # noqa: BLE001
    try:
        import fields as _fields                    # type: ignore[no-redef]
    except Exception:                                   # noqa: BLE001
        _fields = None                                  # type: ignore[assignment]

__all__ = [
    "DEFAULT_DB_NAME",
    "DEFAULT_CSV_NAME",
    "PARCEL_FIELDS",
    "ALL_COLUMNS",
    "SEARCHABLE_FIELDS",
    "DUPLICATE_MODES",
    "SaveOutcome",
    "ImportReport",
    "program_root",
    "resolve_db_path",
    "connect",
    "session",
    "init_db",
    "init_database",
    "save_to_database",
    "save_record",
    "check_duplicate",
    "import_json_files",
    "infer_image_path",
    "repair_image_paths",
    "default_csv_path",
    "find_by_tracking",
    "get_parcel",
    "list_parcels",
    "search_parcels",
    "search_any",
    "update_parcel",
    "delete_parcel",
    "count_parcels",
    "auto_backup",
    "backup_dir",
    "list_backups",
    "stats",
    "export_csv",
    "make_record",
]

#: 数据库文件名 (题目指定)
DEFAULT_DB_NAME = "parcels.db"

#: 默认导出的 CSV 文件名 (题目指定)
DEFAULT_CSV_NAME = "parcels.csv"

#: 除 ``id`` 与 ``created_at`` 外, 由程序写入的业务字段 (顺序与表定义一致)
PARCEL_FIELDS: Tuple[str, ...] = (
    "tracking_number",
    "recipient_name",
    "recipient_phone",
    "recipient_address",
    "sender_name",
    "sender_phone",
    "sender_address",
    "sorting_code",
    "order_number",
    "item_name",
    "cod_amount",
    "insured_amount",
    "shipping_date",
    "courier_company",
    "sender_branch",
    "image_path",
    "is_damaged",
    "damage_type",
)

#: 表里的全部列 (含自增主键与创建时间)
ALL_COLUMNS: Tuple[str, ...] = ("id",) + PARCEL_FIELDS + ("created_at",)

#: 适合做"部分匹配"检索的文本字段 (供界面搜索框使用)
SEARCHABLE_FIELDS: Tuple[str, ...] = (
    "tracking_number",
    "recipient_name",
    "recipient_phone",
    "recipient_address",
    "sender_name",
    "sender_phone",
    "sender_address",
    "sorting_code",
    "order_number",
    "item_name",
    "courier_company",
    "sender_branch",
    "damage_type",
)

#: 需要"去掉空格/横杠再比较"的字段及其规范化方式。
#: 目的: 用户输入 "173 9604 6368" 也能匹配到库里存的 "17396046368"。
_ALNUM_FIELDS = ("tracking_number", "sorting_code", "order_number")
_DIGIT_FIELDS = ("recipient_phone", "sender_phone")

#: 单号重复时的处理策略
#:   skip      - 已有同单号记录则跳过, 不写入 (默认)
#:   overwrite - 用新数据覆盖已有记录
#:   new       - 仍然新增一条 (允许重复, 用于同一单号多次扫描的场景)
DUPLICATE_MODES = ("skip", "overwrite", "new")

# 写操作用它串行化; 读操作不加锁 (SQLite 本身支持多读)
_WRITE_LOCK = threading.RLock()


# ==========================================================================
# 路径
# ==========================================================================
def program_root() -> Path:
    """程序根目录.

    - 打包成 exe 运行时: exe 所在目录
    - 源码运行时:       版本目录 (即 ``src`` 的上一级)

    这里刻意不去 import ``ocr_engine``: 那个模块会连带加载 paddle,
    而信息库功能应当能独立使用 (例如在没有 OCR 环境的机器上查数据)。
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def _is_writable_dir(p: Path) -> bool:
    """目录是否存在且可写 (实际写一个临时文件来确认, 比看权限位可靠)."""
    try:
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".dsh_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:
        return False


def stable_data_dir() -> Path:
    """与程序目录**无关**的固定数据目录 (``%LOCALAPPDATA%\\ParcelDeskewOCR``).

    为什么需要它
    ------------
    用户实测反馈："每次更新完成后都要我往数据库重新输入数据"。

    原因是数据库原先放在**程序根目录**（exe 同级）。这带来两个丢数据的场景：

    1. 重新打包时 PyInstaller 会整个删掉输出目录（构建脚本虽有抢救逻辑，
       但只要中途出错或用户手工清理就会丢）
    2. **更常见**：用户每版都从压缩包解压到**新文件夹**再运行 ——
       上一版目录里的 ``parcels.db`` 自然不会被带过来

    因此把默认位置改到用户数据目录：只要不主动删，重建/换目录都不影响。
    仍然保留"exe 同级有库就用它"的便携模式（见 :func:`resolve_db_path`）。
    """
    import os

    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    d = Path(base) / "ParcelDeskewOCR"
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_db_path(path: Optional[object] = None) -> Path:
    """确定数据库文件位置.

    优先级：

    1. 显式传入 ``path`` —— 直接用（传目录则在其下用默认文件名）
    2. **exe 同级已经有 ``parcels.db``** —— 用它（便携模式，尊重用户放在一起的库）
    3. ``%LOCALAPPDATA%\\ParcelDeskewOCR\\parcels.db`` —— 默认（**重建程序不会碰它**）
    4. 源码运行时用版本目录（开发方便）
    5. 以上都不可写时退回用户数据目录

    注意第 2 条：如果用户把库放在 exe 旁边，仍然优先用它，
    这样"整个文件夹拷走"的用法依然成立。但**默认**不会再往程序目录写。
    """
    if path is not None:
        p = Path(path)
        return p / DEFAULT_DB_NAME if p.is_dir() else p

    root = program_root()

    # 便携模式: exe (或源码目录) 旁边已经有库 -> 尊重它
    beside = root / DEFAULT_DB_NAME
    if beside.is_file():
        return beside

    # 默认: 稳定数据目录
    stable = stable_data_dir() / DEFAULT_DB_NAME
    if _is_writable_dir(stable.parent):
        return stable

    # 兜底: 程序目录可写就用程序目录 (老的题面要求)
    if _is_writable_dir(root):
        return root / DEFAULT_DB_NAME

    return stable


# ==========================================================================
# 连接与建表
# ==========================================================================
_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS parcels (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    tracking_number    TEXT,
    recipient_name     TEXT,
    recipient_phone    TEXT,
    recipient_address  TEXT,
    sender_name        TEXT,
    sender_phone       TEXT,
    sender_address     TEXT,
    sorting_code       TEXT,
    order_number       TEXT,
    item_name          TEXT,
    cod_amount         TEXT,
    insured_amount     TEXT,
    shipping_date      TEXT,
    courier_company    TEXT,
    sender_branch      TEXT,
    image_path         TEXT,
    is_damaged         INTEGER DEFAULT 0,
    damage_type        TEXT,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

#: 常用检索列建索引。刻意**不建 UNIQUE 索引**: 题目要求单号重复时"提示用户
#: 覆盖或跳过", 而不是硬性拒绝; 另外单号可能为空, 空值也不该互相冲突。
_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_parcels_tracking ON parcels(tracking_number)",
    "CREATE INDEX IF NOT EXISTS idx_parcels_created  ON parcels(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_parcels_phone    ON parcels(recipient_phone)",
)


class _ClosingConnection(sqlite3.Connection):
    """退出 ``with`` 块时**真正关闭**的连接.

    sqlite3 自带的 ``Connection.__exit__`` 只提交/回滚事务, **不关闭连接**。
    不关连接就会一直占住数据库文件 —— 在 Windows 上表现为
    ``PermissionError: [WinError 32] 另一个程序正在使用此文件``,
    用户会连"复制/改名/删除 parcels.db"都做不到。

    这里把子类交给 ``sqlite3.connect(factory=...)``, 这样:
      · ``with connect(path) as conn:``  安全 (自动关闭)
      · ``with session(path) as conn:``  也安全
    两种写法都不会再泄漏, 不用指望调用方记住区别。
    """

    def __exit__(self, exc_type, exc, tb):        # type: ignore[override]
        try:
            return super().__exit__(exc_type, exc, tb)
        finally:
            self.close()


def connect(path: Optional[object] = None) -> sqlite3.Connection:
    """打开数据库连接 (不负责建表).

    返回的行可用列名下标访问 (``row["tracking_number"]``)。
    连接支持 ``with`` 且**退出时会自动关闭** (见 :class:`_ClosingConnection`),
    因此 ``with connect(path) as conn:`` 与 ``with session(path) as conn:``
    都是安全的。
    """
    db = resolve_db_path(path)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db), timeout=15.0, factory=_ClosingConnection)
    conn.row_factory = sqlite3.Row
    conn.text_factory = str          # 明确按 UTF-8 文本解码, 避免 bytes
    return conn


@contextmanager
def session(path: Optional[object] = None):
    """连接上下文管理器: 正常退出或抛异常都**保证关闭连接**.

    用法::

        with session(path) as conn:
            rows = conn.execute("SELECT * FROM parcels").fetchall()

    需要写入时在块内显式调用 ``conn.commit()`` (或不提交, 退出时丢弃)。
    """
    conn = connect(path)
    try:
        yield conn
    finally:
        conn.close()


def _ensure_columns(conn: sqlite3.Connection) -> List[str]:
    """补齐缺失的列 (轻量迁移).

    如果以后往 ``PARCEL_FIELDS`` 里加了新字段, 老数据库不需要删库重建,
    这里会自动 ``ALTER TABLE ADD COLUMN``。返回被补上的列名。
    """
    have = {r["name"] for r in conn.execute("PRAGMA table_info(parcels)")}
    added: List[str] = []
    for col in PARCEL_FIELDS:
        if col in have:
            continue
        ctype = "INTEGER DEFAULT 0" if col == "is_damaged" else "TEXT"
        conn.execute(f"ALTER TABLE parcels ADD COLUMN {col} {ctype}")
        added.append(col)
    return added


def init_db(path: Optional[object] = None, *, recreate: bool = False) -> Path:
    """初始化数据库: 文件或表不存在时自动创建.

    参数
    ----
    path
        数据库路径; 省略则用 :func:`resolve_db_path` 的结果。
    recreate
        为 True 时先删除已有表 (慎用, 主要用于测试)。

    返回
    ----
    实际使用的数据库文件路径。
    """
    db = resolve_db_path(path)
    with _WRITE_LOCK:
        conn = connect(db)
        try:
            if recreate:
                conn.execute("DROP TABLE IF EXISTS parcels")
            conn.execute(_CREATE_SQL)
            added = _ensure_columns(conn)
            for sql in _INDEX_SQL:
                conn.execute(sql)
            conn.commit()
            if added:
                # 迁移信息回传给调用方不便, 这里只在标准错误里提示一句
                print(f"[database] 已为 parcels 表补充缺失列: {', '.join(added)}",
                      file=sys.stderr)
        finally:
            conn.close()
    return db


# ==========================================================================
# 行 → 字典
# ==========================================================================
def _utc_to_local_str(ts: Optional[str]) -> str:
    """把 SQLite 的 UTC 时间字符串转成本地时间字符串.

    ``CURRENT_TIMESTAMP`` 写入的是 UTC ("YYYY-MM-DD HH:MM:SS")。
    直接展示会与中国时间差 8 小时, 人工核对时容易误判, 因此读取时统一转换。
    """
    if not ts:
        return ""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            dt = _dt.datetime.strptime(str(ts), fmt).replace(tzinfo=_dt.timezone.utc)
            return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    return str(ts)          # 无法解析就原样返回, 不丢数据


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """sqlite3.Row -> dict, 并把 created_at 转成本地时间."""
    d = {k: row[k] for k in row.keys()}
    if "created_at" in d:
        d["created_at"] = _utc_to_local_str(d.get("created_at"))
    if "is_damaged" in d:
        d["is_damaged"] = int(d.get("is_damaged") or 0)
    # 统一把 None 变成空串, 界面与 CSV 处理起来更省事
    for k, v in d.items():
        if v is None and k != "id":
            d[k] = ""
    return d


# ==========================================================================
# 录入
# ==========================================================================
@dataclass
class SaveOutcome:
    """一次录入的结果."""

    status: str                       # inserted / overwritten / skipped
    id: Optional[int] = None          # 本次写入的行 id (skipped 时为原记录 id)
    existing: List[Dict[str, Any]] = field(default_factory=list)
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("inserted", "overwritten")


def make_record(**kwargs: Any) -> Dict[str, Any]:
    """按表结构规范化一条待录入数据.

    只保留 :data:`PARCEL_FIELDS` 里认识的字段, 缺失的补空串,
    ``is_damaged`` 统一成 0/1。这样调用方不必关心字段完整性。
    """
    rec: Dict[str, Any] = {}
    for f in PARCEL_FIELDS:
        v = kwargs.get(f, "")
        if f == "is_damaged":
            if isinstance(v, str):
                rec[f] = 1 if v.strip().lower() in ("1", "true", "yes", "y", "是", "破损") else 0
            else:
                rec[f] = 1 if v else 0
        else:
            rec[f] = "" if v is None else str(v)
    return rec


def find_by_tracking(tracking_number: str,
                     path: Optional[object] = None) -> List[Dict[str, Any]]:
    """按单号精确查找 (忽略大小写与空格/横杠差异).

    单号常被 OCR 读出多余空格, 或用户输入时带横杠, 因此比较前先规范化。
    """
    key = _norm_alnum(tracking_number)
    if not key:
        return []
    with session(path) as conn:
        rows = conn.execute(
            f"SELECT * FROM parcels WHERE {_alnum_expr('tracking_number')} = ? "
            f"ORDER BY id DESC", (key,)).fetchall()
    return [_row_to_dict(r) for r in rows]


def save_to_database(record: Dict[str, Any],
                     path: Optional[object] = None,
                     on_duplicate: str = "skip") -> SaveOutcome:
    """把一条识别结果写入数据库.

    参数
    ----
    record
        字段字典, 会被 :func:`make_record` 规范化。
    path
        数据库路径, 省略用默认位置。
    on_duplicate
        单号已存在时的策略, 取值见 :data:`DUPLICATE_MODES`:

        - ``"skip"``      跳过不写, 返回 ``status="skipped"`` 并带上已有记录
        - ``"overwrite"`` 用新数据覆盖第一条已有记录 (id 不变)
        - ``"new"``       无视重复, 直接新增

    说明
    ----
    界面上的"询问用户"不在这里做 —— 数据库模块不该弹窗。界面先调
    :func:`find_by_tracking` 判断, 得到用户选择后再用对应策略调用本函数。

    单号为空时不做查重 (没有可比的键), 直接新增。
    """
    if on_duplicate not in DUPLICATE_MODES:
        raise ValueError(f"on_duplicate 只能是 {DUPLICATE_MODES}, 收到 {on_duplicate!r}")

    rec = make_record(**record)
    tracking = rec.get("tracking_number", "")

    with _WRITE_LOCK:
        existing = find_by_tracking(tracking, path) if tracking else []

        if existing and on_duplicate == "skip":
            return SaveOutcome(
                status="skipped", id=int(existing[0]["id"]), existing=existing,
                message=f"单号 {tracking} 已存在 (记录 id={existing[0]['id']}), 未写入")

        conn = connect(path)
        try:
            conn.execute(_CREATE_SQL)
            _ensure_columns(conn)

            cols = list(PARCEL_FIELDS)
            if existing and on_duplicate == "overwrite":
                target_id = int(existing[0]["id"])
                sets = ", ".join(f"{c}=?" for c in cols)
                conn.execute(f"UPDATE parcels SET {sets} WHERE id=?",
                             [rec[c] for c in cols] + [target_id])
                conn.commit()
                return SaveOutcome(
                    status="overwritten", id=target_id, existing=existing,
                    message=f"已覆盖单号 {tracking} 的原有记录 (id={target_id})")

            placeholders = ", ".join("?" for _ in cols)
            cur = conn.execute(
                f"INSERT INTO parcels ({', '.join(cols)}) VALUES ({placeholders})",
                [rec[c] for c in cols])
            conn.commit()
            new_id = int(cur.lastrowid) if cur.lastrowid is not None else None
            return SaveOutcome(status="inserted", id=new_id,
                               message=f"已录入 (id={new_id})")
        finally:
            conn.close()


# ==========================================================================
# 查询
# ==========================================================================
def _norm_alnum(s: Any) -> str:
    """去掉空格/横杠等分隔符并转大写 —— 与 SQL 侧的 ``_alnum_expr`` 保持一致."""
    return "".join(ch for ch in str(s or "") if ch.isalnum()).upper()


def _norm_digits(s: Any) -> str:
    """只保留数字 —— 与 SQL 侧的 ``_digit_expr`` 保持一致."""
    return "".join(ch for ch in str(s or "") if ch.isdigit())


def _alnum_expr(col: str) -> str:
    """SQL 表达式: 去分隔符 + 转大写, 用于单号类字段的比较."""
    e = f"UPPER({col})"
    for ch in (" ", "-", "_", ".", ":", "/", "\\"):
        e = f"REPLACE({e}, '{ch}', '')"
    return e


def _digit_expr(col: str) -> str:
    """SQL 表达式: 只保留数字, 用于电话号码的比较."""
    e = col
    for ch in (" ", "-", "(", ")", "+", ".", "/"):
        e = f"REPLACE({e}, '{ch}', '')"
    return e


def _escape_like(s: str) -> str:
    """转义 LIKE 的通配符, 否则用户输入 ``%`` 会匹配到所有记录."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _build_like(col: str, value: Any) -> Tuple[str, List[Any]]:
    """为单个字段生成"部分匹配"条件.

    - 单号/分拣码/订单号: 先规范化再比较 (忽略空格横杠大小写)
    - 电话:               只留数字比较 (输入 "173 9604" 也能命中)
    - 其余文本:           普通 LIKE, 大小写不敏感
    """
    raw = str(value or "").strip()
    if not raw:
        return "", []

    if col in _ALNUM_FIELDS:
        key = _norm_alnum(raw)
        if not key:
            return "", []
        return f"{_alnum_expr(col)} LIKE ? ESCAPE '\\'", [f"%{_escape_like(key)}%"]

    if col in _DIGIT_FIELDS:
        key = _norm_digits(raw)
        if not key:
            return "", []
        return f"{_digit_expr(col)} LIKE ? ESCAPE '\\'", [f"%{_escape_like(key)}%"]

    return f"{col} LIKE ? ESCAPE '\\'", [f"%{_escape_like(raw)}%"]


def search_parcels(path: Optional[object] = None,
                   limit: int = 200,
                   order_by: str = "id DESC",
                   **filters: Any) -> List[Dict[str, Any]]:
    """按任意字段的部分值检索 (字段之间是 **AND** 关系).

    例::

        # 单号后缀
        search_parcels(tracking_number="25482")
        # 电话尾号
        search_parcels(recipient_phone="1234")
        # 地址关键词 + 只看破损
        search_parcels(recipient_address="武侯区", is_damaged=1)
        # 多字段组合
        search_parcels(courier_company="顺丰", recipient_phone="0000")

    认得的键: :data:`SEARCHABLE_FIELDS` 里的任意字段, 以及
    ``is_damaged`` (精确 0/1)、``damage_type``;
    值为空字符串/None 的条件会被忽略。
    """
    where: List[str] = []
    params: List[Any] = []

    for key, value in filters.items():
        if key == "is_damaged":
            if value in ("", None):
                continue
            where.append("is_damaged = ?")
            params.append(1 if str(value) in ("1", "True", "true", "是") else 0)
            continue
        if key not in PARCEL_FIELDS:
            raise KeyError(f"未知字段 {key!r}; 可用字段见 PARCEL_FIELDS")
        cond, ps = _build_like(key, value)
        if cond:
            where.append(cond)
            params.extend(ps)

    sql = "SELECT * FROM parcels"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" ORDER BY {order_by} LIMIT ?"
    params.append(int(limit))

    with session(path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(r) for r in rows]


def search_any(keyword: str,
               path: Optional[object] = None,
               fields: Sequence[str] = SEARCHABLE_FIELDS,
               limit: int = 200,
               order_by: str = "id DESC") -> List[Dict[str, Any]]:
    """一个关键词跨多个字段模糊搜索 (字段之间是 **OR** 关系).

    这是界面搜索框用的接口: 用户只记得"25482"或"武侯区", 不用指定是哪个字段。
    """
    kw = str(keyword or "").strip()
    if not kw:
        return list_parcels(path=path, limit=limit, order_by=order_by)

    ors: List[str] = []
    params: List[Any] = []
    for col in fields:
        cond, ps = _build_like(col, kw)
        if cond:
            ors.append(cond)
            params.extend(ps)

    if not ors:
        return []

    sql = (f"SELECT * FROM parcels WHERE ({' OR '.join(ors)}) "
           f"ORDER BY {order_by} LIMIT ?")
    params.append(int(limit))
    with session(path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_parcel(pid: int, path: Optional[object] = None) -> Optional[Dict[str, Any]]:
    """按主键取一条记录."""
    with session(path) as conn:
        row = conn.execute("SELECT * FROM parcels WHERE id=?", (int(pid),)).fetchone()
    return _row_to_dict(row) if row else None


def list_parcels(path: Optional[object] = None,
                 limit: int = 500,
                 offset: int = 0,
                 order_by: str = "id DESC") -> List[Dict[str, Any]]:
    """按顺序取一页记录 (界面表格用)."""
    sql = f"SELECT * FROM parcels ORDER BY {order_by} LIMIT ? OFFSET ?"
    with session(path) as conn:
        rows = conn.execute(sql, (int(limit), int(offset))).fetchall()
    return [_row_to_dict(r) for r in rows]


def count_parcels(path: Optional[object] = None) -> int:
    """总记录数."""
    with session(path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM parcels").fetchone()[0])


# ==========================================================================
# 自动备份
# ==========================================================================
def backup_dir() -> Path:
    """自动备份的默认存放目录 (程序目录之外).

    可用环境变量 ``PARCEL_DESKEW_BACKUP_DIR`` 覆盖, 便于把备份直接放进
    自己的网盘 / 移动硬盘同步目录。
    """
    import os

    env = os.environ.get("PARCEL_DESKEW_BACKUP_DIR")
    if env:
        return Path(env)
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "ParcelDeskewOCR" / "backup"


def auto_backup(path: Optional[object] = None, keep: int = 7,
                log: Optional[Any] = None,
                dest_dir: Optional[object] = None) -> Optional[Path]:
    """把数据库按天备份到程序目录**之外**, 返回备份文件路径 (未备份则 None).

    为什么必须备份到程序目录之外
    ---------------------------
    题目要求 ``parcels.db`` 放在程序根目录, 这带来一个真实的丢数据场景:
    **升级时用户用新版覆盖/替换整个程序文件夹, 数据库就一起没了。**
    同目录下再放一份 ``.bak`` 也没用 —— 它同样会被覆盖掉。

    因此这里每天把数据库复制一份到 ``%LOCALAPPDATA%\\ParcelDeskewOCR\\backup``,
    按日期命名, 只保留最近 ``keep`` 份。该目录在程序文件夹之外, 升级换版本
    都不会动到它。

    一天只备份一次 (同名文件已存在就跳过), 避免每次启动都写盘。

    参数
    ----
    dest_dir
        自定义备份目录; 省略则用 :func:`backup_dir`。
        (测试靠它隔离, 用户靠它把备份放进网盘同步目录。)
    """
    def _say(msg: str) -> None:
        if log:
            try:
                log(msg)
            except Exception:
                pass

    src = resolve_db_path(path)
    if not src.is_file() or src.stat().st_size == 0:
        return None

    try:
        bdir = Path(dest_dir) if dest_dir is not None else backup_dir()
        bdir.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d")
        dest = bdir / f"parcels_{stamp}.db"
        if not dest.is_file():
            # 用 sqlite 官方的 backup API 而不是 shutil.copy2:
            # 直接拷文件可能拿到"正在写入中"的不一致状态。
            # 目标库用 session() 打开, 保证退出即关闭 —— 用原生 sqlite3.connect
            # 加 with 是不关连接的, 会让备份文件一直被占用 (踩过一次)。
            with session(src) as sconn, session(dest) as dconn:
                sconn.backup(dconn)
            _say(f"信息库已自动备份: {dest}")

        # 清理旧备份, 只留最近 keep 份
        olds = sorted(bdir.glob("parcels_*.db"))
        for old in (olds[:-keep] if len(olds) > keep else []):
            try:
                old.unlink()
            except Exception:
                pass
        return dest
    except Exception as exc:                          # noqa: BLE001
        _say(f"自动备份失败 (不影响正常使用): {type(exc).__name__}: {exc}")
        return None


def list_backups(dest_dir: Optional[object] = None) -> List[Dict[str, Any]]:
    """列出已有的自动备份 (按时间倒序), 供界面"从备份恢复"使用."""
    bdir = Path(dest_dir) if dest_dir is not None else backup_dir()
    if not bdir.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for f in sorted(bdir.glob("parcels_*.db"), reverse=True):
        try:
            st = f.stat()
            with session(f) as c:
                n = int(c.execute("SELECT COUNT(*) FROM parcels").fetchone()[0])
            out.append({"path": str(f), "name": f.name,
                        "size_kb": round(st.st_size / 1024.0, 1),
                        "records": n,
                        "time": _dt.datetime.fromtimestamp(st.st_mtime)
                                .strftime("%Y-%m-%d %H:%M:%S")})
        except Exception:
            continue
    return out


# ==========================================================================
# 修改与删除
# ==========================================================================
def update_parcel(pid: int, path: Optional[object] = None, **changes: Any) -> bool:
    """修改一条记录 (人工核对后纠正字段用).

    只接受 :data:`PARCEL_FIELDS` 里的字段; 没有任何有效字段时返回 False。
    """
    valid = {k: v for k, v in changes.items() if k in PARCEL_FIELDS}
    if not valid:
        return False
    if "is_damaged" in valid:
        valid["is_damaged"] = 1 if valid["is_damaged"] in (1, "1", True, "true", "是") else 0

    sets = ", ".join(f"{k}=?" for k in valid)
    with _WRITE_LOCK:
        conn = connect(path)
        try:
            cur = conn.execute(f"UPDATE parcels SET {sets} WHERE id=?",
                               list(valid.values()) + [int(pid)])
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()


def delete_parcel(pid: int, path: Optional[object] = None) -> bool:
    """删除一条记录."""
    with _WRITE_LOCK:
        conn = connect(path)
        try:
            cur = conn.execute("DELETE FROM parcels WHERE id=?", (int(pid),))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()


# ==========================================================================
# 统计
# ==========================================================================
def stats(path: Optional[object] = None) -> Dict[str, Any]:
    """基本信息统计.

    返回字典含: ``total`` 总数, ``damaged`` 破损样本数,
    ``by_courier`` 按快递公司分布, ``by_damage_type`` 按破损类型分布,
    ``missing_tracking`` 无单号条数, ``db_path`` / ``db_size_kb`` 数据库自身信息。
    """
    db = resolve_db_path(path)
    out: Dict[str, Any] = {
        "total": 0, "damaged": 0, "missing_tracking": 0,
        "by_courier": [], "by_damage_type": [],
        "db_path": str(db), "db_size_kb": 0.0,
        "earliest": "", "latest": "",
    }

    # 文件可能还没建 (没识别过任何图), 此时返回全 0 而不是报错
    if not db.is_file():
        return out
    out["db_size_kb"] = round(db.stat().st_size / 1024.0, 1)

    with session(path) as conn:
        try:
            out["total"] = int(conn.execute("SELECT COUNT(*) FROM parcels").fetchone()[0])
            out["damaged"] = int(conn.execute(
                "SELECT COUNT(*) FROM parcels WHERE is_damaged=1").fetchone()[0])
            out["missing_tracking"] = int(conn.execute(
                "SELECT COUNT(*) FROM parcels "
                "WHERE tracking_number IS NULL OR TRIM(tracking_number)=''").fetchone()[0])

            out["by_courier"] = [
                {"name": (r[0] or "(未识别)"), "count": int(r[1])}
                for r in conn.execute(
                    "SELECT courier_company, COUNT(*) c FROM parcels "
                    "GROUP BY courier_company ORDER BY c DESC, courier_company")
            ]
            out["by_damage_type"] = [
                {"name": (r[0] or "(未填写)"), "count": int(r[1])}
                for r in conn.execute(
                    "SELECT damage_type, COUNT(*) c FROM parcels WHERE is_damaged=1 "
                    "GROUP BY damage_type ORDER BY c DESC, damage_type")
            ]
            row = conn.execute(
                "SELECT MIN(created_at), MAX(created_at) FROM parcels").fetchone()
            out["earliest"] = _utc_to_local_str(row[0])
            out["latest"] = _utc_to_local_str(row[1])
        except sqlite3.OperationalError:
            # 表还没建起来 (例如空文件), 保持全 0
            pass
    return out


# ==========================================================================
# 导出
# ==========================================================================
def export_csv(out_path: Optional[object] = None,
               path: Optional[object] = None,
               rows: Optional[Iterable[Dict[str, Any]]] = None,
               include_header: bool = True) -> int:
    """导出为 CSV, 返回写出的记录条数.

    - 不传 ``rows`` 时导出全表; 传了就只导出给的那批 (界面"导出当前搜索结果")。
    - ``out_path`` 省略时写到数据库同级的 ``parcels.csv`` (题目要求的默认行为)。
    - 编码用 ``utf-8-sig``: 带 BOM, Excel 双击打开不会把中文显示成乱码。
    """
    if out_path is None:
        out_path = default_csv_path(path)
    data = list(rows) if rows is not None else list_parcels(path=path, limit=10 ** 9)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ALL_COLUMNS), extrasaction="ignore")
        if include_header:
            writer.writeheader()
        for r in data:
            writer.writerow({k: r.get(k, "") for k in ALL_COLUMNS})
    return len(data)


def default_csv_path(path: Optional[object] = None) -> Path:
    """默认导出位置: 数据库文件同级的 ``parcels.csv``."""
    return resolve_db_path(path).parent / DEFAULT_CSV_NAME


# ==========================================================================
# JSON 导入
# ==========================================================================
#: JSON 里可能是"记录列表"的包装键 (有些导出工具会把数组包一层)
_LIST_KEYS = ("records", "parcels", "data", "items", "results", "list", "rows")
#: JSON 里可能是"单条记录"的包装键
_RECORD_KEYS = ("fields", "extraction", "parsed", "record", "parcel")


@dataclass
class ImportReport:
    """一次 JSON 导入的结果汇总."""

    total: int = 0                  # 解析出的记录总数
    inserted: int = 0               # 成功入库
    skipped: int = 0                # 因单号重复被跳过
    failed: int = 0                 # 失败 (解析失败或写库出错)
    empty_tracking: int = 0         # 入库了但没有单号 (无法查重, 提醒用户注意)
    files: List[Dict[str, Any]] = field(default_factory=list)
    ok_files: int = 0
    failed_files: int = 0

    def summary_text(self) -> str:
        """给弹窗用的一行摘要 (题目要求的措辞)."""
        return (f"成功 {self.inserted} 条，跳过 {self.skipped} 条（重复），"
                f"失败 {self.failed} 条")

    def detail_text(self) -> str:
        """摘要 + 逐文件明细, 便于定位是哪个文件出的问题."""
        lines = [self.summary_text(), ""]
        lines.append(f"共 {len(self.files)} 个文件 "
                     f"(正常 {self.ok_files}, 失败 {self.failed_files}), "
                     f"解析出记录 {self.total} 条")
        if self.empty_tracking:
            lines.append(f"注意: 其中 {self.empty_tracking} 条没有单号 "
                         f"(无法查重, 已照常入库)")
        bad = [f for f in self.files if f.get("error")]
        if bad:
            lines.append("")
            lines.append("出错的文件:")
            for f in bad[:20]:
                lines.append(f"  · {Path(f['file']).name}: {f['error']}")
            if len(bad) > 20:
                lines.append(f"  …… 还有 {len(bad) - 20} 个")
        return "\n".join(lines)


def init_database(path: Optional[object] = None, *, recreate: bool = False) -> Path:
    """初始化数据库 (题目指定的函数名).

    :func:`init_db` 的别名。程序启动时调用一次: 文件或表不存在就自动创建。
    """
    return init_db(path, recreate=recreate)


def check_duplicate(tracking_number: Any, path: Optional[object] = None) -> bool:
    """单号是否已存在于数据库中.

    比较前会规范化 (忽略空格、横杠与大小写), 因此 OCR 读出的
    ``"YT 0713 5905 25482"`` 与库里的 ``"YT0713590525482"`` 视为同一个。
    单号为空时一律返回 False (没有可比对的键)。
    """
    if not _norm_alnum(tracking_number):
        return False
    return len(find_by_tracking(str(tracking_number), path)) > 0


def infer_image_path(json_path: object) -> str:
    """由 JSON 文件名反推对应的图片路径 (如 ``001.json`` → ``001.jpg``).

    会先在同目录下依次找 ``.jpg/.jpeg/.png/.bmp/.webp``; 都找不到时,
    仍按规则返回 ``.jpg`` 路径 (题目要求"反推", 不要求文件必须存在)。
    """
    p = Path(json_path)
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    # 搜索范围**不能只有 JSON 同目录**。实际数据里 json 和图片常分开放,
    # 例如 json\parcel_0001.json 配 package\parcel_0001.jpg ——
    # 只找同目录就会反推出一个不存在的路径 (库里 15 条全是这种情况)。
    search_dirs = [p.parent]
    for sibling in ("package", "pictures", "images", "photos", "img"):
        search_dirs.append(p.parent / sibling)
        search_dirs.append(p.parent.parent / sibling)
    search_dirs.append(p.parent.parent)

    for d in search_dirs:
        for ext in exts:
            cand = d / (p.stem + ext)
            if cand.is_file():
                return str(cand)

    # 都没有: 仍按规则返回同目录的 .jpg (题目要求"反推", 不要求文件存在)
    return str(p.with_suffix(".jpg"))


def repair_image_paths(picture_root: object,
                       path: Optional[object] = None,
                       *,
                       db_path: Optional[object] = None
                       ) -> Dict[str, Any]:
    """按**文件名主干**把库里失效的 image_path 重新指到真实文件.

    用法场景
    --------
    数据库里的 ``image_path`` 写的是导入当时的路径, 后来图片被挪了位置
    (例如从 ``json\\`` 挪到 ``picture_json\\package\\``), 记录就全"打不开"了。
    这个函数扫一遍 ``picture_root``, 按主干 (``parcel_0001``) 建索引,
    再把每条记录重新指过去。

    为什么按**主干**而不是按单号: 图片文件名和 JSON 同名
    (``parcel_0001.jpg`` ↔ ``parcel_0001.json``), 而 JSON 是数据库记录的来源,
    所以主干是三者之间唯一可靠的纽带。

    返回 ``{"ok","total","fixed","already","missing","details","message"}``。
    """
    target = resolve_db_path(db_path or path)
    root = Path(picture_root)
    if not root.is_dir():
        return {"ok": False, "total": 0, "fixed": 0, "already": 0,
                "missing": 0, "details": [],
                "message": f"图片目录不存在: {root}"}

    # 建索引: 主干 -> 文件 (跳过 AI 缩放缓存 _1600 之类)
    index: Dict[str, Path] = {}
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for f in root.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in exts:
            continue
        stem = f.stem
        if stem.endswith("_1600") or "_ai_resized" in str(f).lower():
            continue
        index.setdefault(stem.lower(), f)

    if not index:
        return {"ok": False, "total": 0, "fixed": 0, "already": 0,
                "missing": 0, "details": [],
                "message": f"目录下没找到图片: {root}"}

    fixed = already = missing = total = 0
    details: List[str] = []
    try:
        with session(target) as conn:
            rows = conn.execute("SELECT id, tracking_number, image_path "
                                "FROM parcels").fetchall()
            for row in rows:
                total += 1
                rid = row["id"] if not isinstance(row, dict) else row["id"]
                cur = (row["image_path"] if not isinstance(row, dict)
                       else row["image_path"]) or ""
                stem = Path(str(cur)).stem.lower() if cur else ""
                hit = index.get(stem)
                if hit is not None and Path(str(hit)) == Path(str(cur)):
                    already += 1
                    continue
                if hit is None:
                    missing += 1
                    details.append(f"  ✗ {Path(str(cur)).name or '(空)'} "
                                   f"在图片目录里没找到对应文件")
                    continue
                conn.execute("UPDATE parcels SET image_path=? WHERE id=?",
                             (str(hit), rid))
                fixed += 1
                details.append(f"  ✓ {Path(str(cur)).name or '(空)'} -> {hit}")
            # ⚠️ session() 不自动提交 —— 文档写明"需要写入时在块内显式 commit"。
            # 原先漏了这句, UPDATE 全被丢弃, 却仍报"修复 15 条"。
            conn.commit()
    except Exception as exc:                     # noqa: BLE001
        return {"ok": False, "total": total, "fixed": fixed,
                "already": already, "missing": missing, "details": details,
                "message": f"修复失败: {type(exc).__name__}: {exc}"}

    msg = (f"共 {total} 条: 修复 {fixed}, 本来就对 {already}, 找不到 {missing}")
    return {"ok": True, "total": total, "fixed": fixed, "already": already,
            "missing": missing, "details": details, "message": msg}


def save_record(record: Dict[str, Any],
                image_path: Optional[object] = None,
                path: Optional[object] = None,
                on_duplicate: str = "skip") -> SaveOutcome:
    """保存一条记录 (题目指定的函数名).

    与 :func:`save_to_database` 的区别只有一点: ``image_path`` 单独作为参数传入,
    符合"由调用方决定图片来源"的用法。

    优先级: 记录里自带的非空 ``image_path`` > 传入的 ``image_path``。
    (JSON 里已经写明了图片路径时应当尊重它; 没写才用反推出来的。)
    """
    rec = dict(record or {})
    if not str(rec.get("image_path") or "").strip() and image_path:
        rec["image_path"] = str(image_path)
    return save_to_database(rec, path=path, on_duplicate=on_duplicate)


def _extract_records(data: Any) -> Tuple[List[Dict[str, Any]], str]:
    """从解析好的 JSON 里取出记录列表.

    返回 ``(记录列表, 错误说明)``; 错误说明非空表示这个文件没法用。

    兼容以下几种实际会遇到的形态::

        1. 单个对象          {"tracking_number": "...", ...}
        2. 对象数组          [{...}, {...}]
        3. 包一层的数组      {"records": [{...}]}   (也认 parcels/data/items/...)
        4. 包一层的单条      {"fields": {...}}
        5. 本程序导出的格式  {"lines": [...], "deskew": {...}, ...}
           —— 没有结构化字段, 但有 OCR 行数据, 现场用 fields 模块反推

    第 5 条很重要: 没有它, 本程序自己导出的 JSON 就导不回来。
    """
    # ---- 情况 2: 数组 ----
    if isinstance(data, list):
        recs = [x for x in data if isinstance(x, dict)]
        if not recs:
            return [], "JSON 数组里没有对象"
        return recs, ""

    if not isinstance(data, dict):
        return [], f"不支持的 JSON 顶层类型: {type(data).__name__}"

    # ---- 情况 1: 本身就是一条记录 ----
    if any(k in data for k in PARCEL_FIELDS):
        return [data], ""

    # ---- 情况 3: 包装的数组 ----
    for key in _LIST_KEYS:
        v = data.get(key)
        if isinstance(v, list):
            recs = [x for x in v if isinstance(x, dict)]
            if recs:
                return recs, ""
            return [], f"字段 {key!r} 是空数组或其中没有对象"

    # ---- 情况 4: 包装的单条记录 ----
    for key in _RECORD_KEYS:
        v = data.get(key)
        if isinstance(v, dict):
            return [v], ""

    # ---- 情况 5: 本程序导出的流水线结果, 从 lines 反推字段 ----
    if isinstance(data.get("lines"), list):
        if _fields is None:
            return [], ("这是本程序导出的识别结果格式, 但当前环境缺少 fields "
                        "模块, 无法从 lines 反推字段")
        try:
            ext = _fields.extract_fields(data["lines"],
                                         image_path=str(data.get("source_file") or ""))
            return [ext.as_record()], ""
        except Exception as exc:                        # noqa: BLE001
            return [], f"从 lines 反推字段失败: {type(exc).__name__}: {exc}"

    keys = ", ".join(list(data.keys())[:8])
    return [], f"JSON 对象里没有可识别的面单字段 (顶层键: {keys})"


def _read_json_file(fp: object) -> Tuple[Any, str]:
    """读并解析一个 JSON 文件; 返回 ``(数据, 错误说明)``.

    编码按顺序尝试, 覆盖实际会遇到的几种情况:

    * ``utf-8-sig`` —— **首选**。它同时兼容"有 BOM"和"没有 BOM"的 UTF-8,
      因为 Python 的 ``json.loads`` 不接受开头的 ``\\ufeff``,
      而带 BOM 的文件用普通 ``utf-8`` 读**不会报错**、只会把 BOM 留在字符串里,
      随后 JSON 解析就失败, 且报错完全看不出是 BOM 引起的。
      (用记事本"另存为 UTF-8"、或 Excel 导出都会产生 BOM。)
    * ``utf-8``      —— 兜底。
    * ``gb18030``    —— 手工编辑过的文件在中文 Windows 上可能是 ANSI/GBK 编码。
    """
    p = Path(fp)
    text: Optional[str] = None
    last_err = ""
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            text = p.read_text(encoding=enc)
            break
        except UnicodeDecodeError as exc:
            last_err = str(exc)
            continue
        except FileNotFoundError:
            return None, "文件不存在"
        except PermissionError:
            return None, "文件被占用或没有读取权限"
        except IsADirectoryError:
            return None, "这是一个目录, 不是文件"
        except Exception as exc:                        # noqa: BLE001
            return None, f"读取失败: {type(exc).__name__}: {exc}"

    if text is None:
        return None, f"文件编码无法识别 (试过 UTF-8/UTF-8-BOM/GB18030): {last_err}"

    # 双保险: 万一还有残留 BOM, 去掉再解析
    text = text.lstrip("\ufeff")
    try:
        return _json.loads(text), ""
    except Exception as exc:                            # noqa: BLE001
        return None, f"JSON 解析失败: {exc}"


def import_json_files(file_paths: Iterable[object],
                      path: Optional[object] = None,
                      on_duplicate: str = "skip",
                      log: Optional[Any] = None) -> ImportReport:
    """批量导入 JSON 文件到数据库 (题目指定的核心函数).

    行为
    ----
    - **单个文件失败不影响其它文件**: 每个文件单独 try, 出错只记进报告。
    - 单号已存在 -> 计入"跳过"; 不存在 -> 插入, 计入"成功"。
    - JSON 里的 ``null`` 写入数据库时统一存成空字符串 ``""``。
    - ``image_path`` 用 JSON 文件名反推 (``001.json`` -> ``001.jpg``);
      若 JSON 里已经写了非空的 ``image_path``, 则尊重 JSON 里的值。

    参数
    ----
    file_paths
        JSON 文件路径的可迭代对象 (支持多选得到的列表)。
    path
        数据库路径; 省略则用 :func:`resolve_db_path` 的结果。
    on_duplicate
        单号重复时的策略, 见 :data:`DUPLICATE_MODES`。
        **默认 ``"skip"``**, 即题目要求的"已存在则跳过"。

    返回
    ----
    :class:`ImportReport`, 含成功/跳过/失败计数与逐文件明细。
    """
    def _say(msg: str) -> None:
        if log:
            try:
                log(msg)
            except Exception:
                pass

    # 先确保库和表存在, 否则下面第一条插入就报 no such table
    target = init_db(path)
    rep = ImportReport()

    for fp in file_paths:
        entry: Dict[str, Any] = {"file": str(fp), "inserted": 0, "skipped": 0,
                                 "failed": 0, "records": 0, "error": ""}
        _say(f"导入: {Path(fp).name}")

        data, err = _read_json_file(fp)
        if err:
            entry["error"] = err
            entry["failed"] = 1
            rep.failed += 1
            rep.failed_files += 1
            rep.files.append(entry)
            _say(f"  !! {err}")
            continue

        records, err = _extract_records(data)
        if err:
            entry["error"] = err
            entry["failed"] = 1
            rep.failed += 1
            rep.failed_files += 1
            rep.files.append(entry)
            _say(f"  !! {err}")
            continue

        entry["records"] = len(records)
        rep.total += len(records)
        img = infer_image_path(fp)

        for rec in records:
            try:
                out = save_record(rec, image_path=img, path=target,
                                  on_duplicate=on_duplicate)
            except Exception as exc:                    # noqa: BLE001
                entry["failed"] += 1
                rep.failed += 1
                entry["error"] = entry["error"] or f"写库失败: {type(exc).__name__}: {exc}"
                continue

            if out.status == "skipped":
                entry["skipped"] += 1
                rep.skipped += 1
            else:
                entry["inserted"] += 1
                rep.inserted += 1
                if not _norm_alnum(rec.get("tracking_number")):
                    rep.empty_tracking += 1

        if entry["failed"] == 0:
            rep.ok_files += 1
        else:
            rep.failed_files += 1
        rep.files.append(entry)
        _say(f"  -> 成功 {entry['inserted']}, 跳过 {entry['skipped']}, "
             f"失败 {entry['failed']}")

    _say(f"导入结束: {rep.summary_text()}")
    return rep
