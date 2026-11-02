"""多字段比对 —— 在数据库里找出与识别结果最匹配的包裹.

业务位置
--------
本模块服务于快递分拣线::

    条码扫描 ──成功──> 查库命中 ──> 放行          (不进入本模块)
        │
        └──失败(破损/遮挡)──> AI 识别文字 ──> **本模块** ──> 候选包裹

也就是说: **只有当条码读不出来时才会走到这里**。此时手上的信息可能
很完整(文字都清晰), 也可能极度残缺(只剩几个残片)。两种都必须能处理。

两层比对
--------
第一层 **精确匹配**: 单号完整有效 -> 直接按单号查库。
        查到且电话/地址没有严重冲突 -> 满分返回;
        查到但冲突严重 -> 降分并提示人工复核。

第二层 **模糊加权匹配**: 单号残缺或缺失 -> 多字段加权打分,
        按总分排序返回 Top-N。

⛔ 一个必须知道的取舍: 证据量守卫
--------------------------------
需求给的规则是"总分 = Σ(字段得分 × 归一化权重)"。但**归一化**本身有风险:

    如果 AI 只读出一个电话尾号 "1234", 该字段归一化权重就是 1.0,
    一旦匹配上总分就是 1.0 -> 按 ≥0.85 的规则**自动放行**。

在分拣线上这等于"凭 4 位残片自动分拣一个包裹", 错分代价很高。

因此本实现额外计算 **证据量** ``evidence`` (= 输入里实际有值的字段基础权重之和),
并对置信等级设下限: 证据量不足时,**即使满分也不允许自动通过**, 一律降为人工复核。
阈值见 :attr:`MatcherConfig.min_evidence`。这是对需求的一处主动加固,
不是偏离 —— 需求本身也写了"Ai 识别结果可能极度残缺"。

相似度后端
----------
按需求使用 ``fuzzywuzzy`` + ``python-Levenshtein``。但在打包/部署环境里这两个
包可能缺失, 因此做了**自动降级**: 找不到时退回标准库 ``difflib``, 并可通过
:func:`similarity_backend` 查询当前用的是哪个。降级不影响接口与调用方式。
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "MatcherConfig",
    "DEFAULT_CONFIG",
    "FIELD_LABELS",
    "similarity_backend",
    "acceleration_info",
    "compare_fields",
    "match_from_ai",
    "match_from_ai_batch",
    "parse_address",
]

# --------------------------------------------------------------------------
# 相似度后端 (优先 fuzzywuzzy, 缺失则退回 difflib)
# --------------------------------------------------------------------------
try:                                                    # pragma: no cover
    from fuzzywuzzy import fuzz as _fuzz

    _BACKEND = "fuzzywuzzy"
except Exception:                                       # noqa: BLE001
    import difflib as _difflib

    _BACKEND = "difflib"

    class _FuzzShim:                                    # 类型占位, 见下面的函数
        pass

    _fuzz = None                                        # type: ignore[assignment]

#: fuzzywuzzy 是否真的用上了 C 加速的 python-Levenshtein。
#:
#: 这件事**必须单独检测**: fuzzywuzzy 内部是
#: ``try: from Levenshtein import ratio / except ImportError: 纯 Python 兜底``,
#: 也就是**装了 fuzzywuzzy 并不代表有加速**。漏掉 C 扩展时程序不会报错,
#: 只是慢一截 —— 打包时很容易悄悄丢掉, 所以显式暴露出来以便验证。
try:                                                    # pragma: no cover
    from Levenshtein import ratio as _lev_ratio     # noqa: F401

    _ACCEL = "python-Levenshtein"
except Exception:                                       # noqa: BLE001
    _ACCEL = ""


def similarity_backend() -> str:
    """当前使用的相似度后端名 (``"fuzzywuzzy"`` 或 ``"difflib"``)."""
    return _BACKEND


def acceleration_info() -> str:
    """C 加速后端是否可用.

    用于确认打包产物没有静默降级 —— 例如::

        >>> matcher.similarity_backend(), matcher.acceleration_info()
        ('fuzzywuzzy', 'python-Levenshtein')
    """
    return _ACCEL or "无(纯 Python, 较慢)"


def _ratio(a: str, b: str) -> float:
    """整体相似度 0~1."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if _fuzz is not None:
        return float(_fuzz.ratio(a, b)) / 100.0
    return _difflib.SequenceMatcher(None, a, b).ratio()


def _partial_ratio(a: str, b: str) -> float:
    """子串相似度 0~1 (a 是 b 的一部分时得分高)."""
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    if _fuzz is not None:
        return float(_fuzz.partial_ratio(a, b)) / 100.0
    sm = _difflib.SequenceMatcher(None, a, b)
    return sm.find_longest_match(0, len(a), 0, len(b)).size / max(len(a), len(b))


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
#: 字段中文名, 用于生成"匹配理由"
FIELD_LABELS: Dict[str, str] = {
    "tracking_number": "单号",
    "recipient_phone": "收件人电话",
    "recipient_address": "收件人地址",
    "recipient_name": "收件人姓名",
    "others": "其它字段",
    "courier_company": "快递公司",
    "sorting_code": "分拣码",
    "order_number": "订单号",
    "item_name": "内件品名",
    "sender_name": "寄件人",
    "sender_phone": "寄件人电话",
    "sender_address": "寄件人地址",
    "cod_amount": "代收货款",
    "insured_amount": "保价金额",
    "shipping_date": "寄件日期",
    "sender_branch": "寄件网点",
}

#: 主字段的基础权重 (合计 1.00)
_BASE_WEIGHTS: Dict[str, float] = {
    "tracking_number": 0.40,
    "recipient_phone": 0.20,
    "recipient_address": 0.20,
    "recipient_name": 0.10,
    "others": 0.10,
}

#: "其它字段" 组里参与的字段 (组合成一个 0.10 权重的合成字段)
_OTHER_FIELDS: Tuple[str, ...] = (
    # 注意: order_number 已**移出**本组 —— 它和 tracking_number 视为同一个
    # 概念 (用户明确要求), 由 _tracking_values / sim_tracking 统一处理,
    # 留在这里会重复计分。
    "courier_company", "sorting_code", "item_name",
    "sender_name", "sender_phone", "sender_address",
    "cod_amount", "insured_amount", "shipping_date", "sender_branch",
)

#: 单号缺失时, 其 0.40 权重按下面的比例分给这三个字段
#: (需求写的是"按比例分配给电话(0.25)、地址(0.25)、姓名(0.10)",
#:  这三个数字是**比例**而非绝对权重, 合计 0.60, 因此按 5:5:2 拆分 0.40)
_REDISTRIBUTE_RATIO: Dict[str, float] = {
    "recipient_phone": 0.25,
    "recipient_address": 0.25,
    "recipient_name": 0.10,
}

#: 单号"有效"的判定: 至少这么长且含数字
_MIN_TRACKING_LEN = 8


@dataclass
class MatcherConfig:
    """比对参数. 所有阈值都可调, 便于按现场数据标定."""

    #: 自动通过阈值
    auto_pass: float = 0.85
    #: 人工复核阈值 (低于它建议重新录入)
    review: float = 0.60
    #: 返回的候选数量
    top_n: int = 3
    #: **证据量下限**. 输入里实际有值的字段基础权重之和低于它时,
    #: 即使满分也不允许"自动通过", 强制降为人工复核。
    #: 设为 0 可关闭该守卫 (即完全按需求的纯归一化规则)。
    min_evidence: float = 0.30
    #: 精确匹配时, 电话/地址低于多少算"严重冲突"
    conflict_threshold: float = 0.35
    #: 单次比对最多从数据库取多少行做模糊打分 (性能保护)
    max_scan: int = 20000
    #: **歧义裕度**. 第一名与第二名分差小于它时, 说明库里有多个同样像的包裹,
    #: 此时不允许"自动通过" —— 分拣线上选错等于错分。
    ambiguity_margin: float = 0.05
    #: 精确匹配时是否也返回候选列表
    exact_returns_candidates: bool = True
    #: **放行模式**, 只有两个取值。
    #:
    #: ``"lenient"`` **宽松**
    #:     单号匹配 **且该单号在库中唯一** -> 直接放行, 不参与后续加权打分。
    #:     这是给分拣线开的一条快速通道。
    #:
    #: ``"strict"`` **严格** (默认)
    #:     **即使单号命中, 也要看"单号 + 其它信息"的加权得分** ——
    #:     加权总分达到 :attr:`auto_pass` (默认 0.85) 才放行。
    #:
    #:     与旧行为的区别很关键: 以前单号一命中就只做"有没有字段冲突"的
    #:     二值判断 (没冲突直接给 1.0); 现在改成算真正的加权分,
    #:     **其它字段对得越少分越低**, 是连续量而不是一刀切。
    release_mode: str = "strict"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "auto_pass": self.auto_pass, "review": self.review,
            "top_n": self.top_n, "min_evidence": self.min_evidence,
            "conflict_threshold": self.conflict_threshold,
            "max_scan": self.max_scan, "ambiguity_margin": self.ambiguity_margin,
            "release_mode": self.release_mode,
        }


DEFAULT_CONFIG = MatcherConfig()


# --------------------------------------------------------------------------
# 文本工具
# --------------------------------------------------------------------------
_MASK_CHARS = set("*×xX#?○●□■")
_CODE_STRIP = re.compile(r"[\s\-_.:/\\|]+")


def _s(v: Any) -> str:
    """统一成字符串 (None -> "")."""
    return "" if v is None else str(v)


def _norm_code(v: Any) -> str:
    """单号类字段规范化: 去分隔符、转大写."""
    return _CODE_STRIP.sub("", _s(v)).upper()


def _digits(v: Any) -> str:
    """只保留数字."""
    return re.sub(r"\D", "", _s(v))


def _masked_compare(a: str, b: str) -> Tuple[int, int]:
    """掩码感知的**逐位**比较; 返回 ``(相同位数, 可比位数)``.

    仅适用于等长的脱敏值。长度不同时必须用 :func:`_masked_similarity`,
    否则会因 zip 截断而错位 —— 实测 ``****1234``(8 字符) 与
    ``13812341234``(11 位) 逐位比较会把 ``1234`` 拿去和 ``2341`` 比,
    把一个正确的匹配打成低分。
    """
    same = total = 0
    for ca, cb in zip(a, b):
        if ca in _MASK_CHARS or cb in _MASK_CHARS:
            continue
        total += 1
        if ca == cb:
            same += 1
    return same, total


def _visible_segments(s: str, digits_only: bool = True) -> List[str]:
    """把脱敏字符串切成若干**可见片段** (掩码作为分隔).

    ``"138****1234"`` -> ``["138", "1234"]``
    ``"****1234"``    -> ``["1234"]``
    ``"138-1234"``    -> ``["1381234"]``  (连字符等分隔符不切断片段)
    """
    segs: List[str] = []
    cur: List[str] = []
    for ch in s:
        if ch in _MASK_CHARS:
            if cur:
                segs.append("".join(cur))
                cur = []
        elif digits_only and not ch.isdigit():
            continue                     # 非数字字符忽略, 不当作分隔
        else:
            cur.append(ch)
    if cur:
        segs.append("".join(cur))
    return [x for x in segs if x]


def _masked_similarity(inp: Any, dbv: Any, *,
                       digits_only: bool = True) -> Optional[Tuple[float, str]]:
    """脱敏值的相似度: 按可见片段对齐比对.

    返回 ``(得分, 理由)``; 输入里没有可见片段时返回 ``None``。

    打分方式::

        得分 = 片段命中率 × (0.45 + 0.55 × 覆盖率)

    乘上"覆盖率"是关键 —— 否则只看见 4 位尾号也能拿到 1.0。
    覆盖率 = 可见字符数 / 库中值的长度, 用于体现"这个字段看到了多少"。
    """
    raw = _s(inp)
    segs = _visible_segments(raw, digits_only)
    if not segs:
        return None

    target = _digits(dbv) if digits_only else _s(dbv).strip()
    if not target:
        return None

    total = sum(len(x) for x in segs)
    if total <= 0:
        return None

    matched = 0
    hits: List[str] = []
    for i, seg in enumerate(segs):
        is_first, is_last = (i == 0), (i == len(segs) - 1)
        if is_first and target.startswith(seg):
            matched += len(seg)
            hits.append(seg)
        elif is_last and target.endswith(seg):
            matched += len(seg)
            hits.append(seg)
        elif seg in target:
            matched += len(seg)
            hits.append(seg)

    hit_ratio = matched / float(total)
    coverage = min(1.0, total / float(len(target)))
    score = hit_ratio * (0.45 + 0.55 * coverage)
    if hit_ratio > 0:
        reason = (f"可见片段匹配({'/'.join(hits)}), "
                  f"覆盖 {coverage:.0%}") if hits else ""
    else:
        reason = ""
    return max(0.0, min(1.0, score)), reason


# --------------------------------------------------------------------------
# 地址结构解析
# --------------------------------------------------------------------------
_PROVINCE_RE = re.compile(r"^(.{2,8}?(?:省|自治区|特别行政区))")
_MUNICIPALITY = ("北京", "上海", "天津", "重庆")
_CITY_RE = re.compile(r"^(.{2,8}?(?:市|自治州|地区|盟))")
_DISTRICT_RE = re.compile(r"^(.{2,8}?(?:区|县|市|旗|新区))")
_STREET_RE = re.compile(r"^(.{2,10}?(?:街道|镇|乡|路|街|大道|开发区))")


def parse_address(addr: Any) -> Dict[str, str]:
    """把中文地址拆成 省/市/区/街道/详细 五段.

    只做**轻量规则切分**, 不查行政区划表 —— 目的是让"省市区是否一致"
    这类结构化判断成为可能, 而不是做精确的地理编码。
    拆不出来也没关系, 剩下的部分会落进 ``detail``。
    """
    s = _s(addr).strip()
    out = {"province": "", "city": "", "district": "", "street": "", "detail": s}
    if not s:
        return out

    rest = s
    m = _PROVINCE_RE.match(rest)
    if m:
        out["province"] = m.group(1)
        rest = rest[m.end():]
    else:
        # 直辖市 (北京/上海/天津/重庆) 常省略"省", 直接以市开头
        for city in _MUNICIPALITY:
            if rest.startswith(city):
                out["province"] = city + "市"
                break

    m = _CITY_RE.match(rest)
    if m:
        out["city"] = m.group(1)
        rest = rest[m.end():]

    m = _DISTRICT_RE.match(rest)
    if m:
        out["district"] = m.group(1)
        rest = rest[m.end():]

    m = _STREET_RE.match(rest)
    if m:
        out["street"] = m.group(1)
        rest = rest[m.end():]

    out["detail"] = rest
    return out


# --------------------------------------------------------------------------
# 各字段相似度 (返回 (得分|None, 理由))
#   得分为 None 表示"输入没有这个字段", 不参与加权
# --------------------------------------------------------------------------
def sim_tracking(inp: Any, dbv: Any) -> Tuple[Optional[float], str]:
    """单号相似度: 精确 / 前后缀 / 片段 / 字符相似.

    需求明确要求"精确匹配**或前缀/后缀匹配**", 因此片段匹配是主路径之一,
    并且理由要写清是前缀还是后缀 —— 这决定了人工复核时该信多少。
    """
    x, y = _norm_code(inp), _norm_code(dbv)
    if not x or not y:
        return None, ""
    if x == y:
        return 1.0, "单号完全一致"

    short, long_ = (x, y) if len(x) <= len(y) else (y, x)

    # 片段关系: 优先判"完整的前缀/后缀", 它比"后 4 位一致"信息量大得多
    if len(short) >= 3:
        r = len(short) / float(len(long_))
        if long_.endswith(short):
            return 0.86 + 0.12 * r, f"单号后缀匹配({short}, 占 {r:.0%})"
        if long_.startswith(short):
            return 0.86 + 0.12 * r, f"单号前缀匹配({short}, 占 {r:.0%})"
        if short in long_:
            return 0.85 + 0.12 * r, f"单号片段匹配({short}, 占 {r:.0%})"

    # 无包含关系时, 再比固定位数
    for n in (8, 6, 4):
        if len(x) >= n and len(y) >= n:
            if x[:n] == y[:n]:
                return 0.88, f"单号前 {n} 位一致"
            if x[-n:] == y[-n:]:
                return 0.86, f"单号后 {n} 位一致"

    base = _ratio(x, y)
    if base >= 0.75:
        return base * 0.9, f"单号相似度 {base:.0%}"
    return base * 0.4, ""


def sim_phone(inp: Any, dbv: Any) -> Tuple[Optional[float], str]:
    """电话相似度: 掩码感知 + 尾号/号段 + 编辑距离."""
    ra, rb = _s(inp).strip(), _s(dbv).strip()
    if not ra or not rb:
        return None, ""

    da, db_ = _digits(ra), _digits(rb)
    if not da or not db_:
        return None, ""

    if da == db_:
        return 1.0, "电话完全一致"

    # 任一方带掩码 -> 按可见片段比对
    # (绝不能逐位 zip: 脱敏值比原值短, zip 会截断错位)
    if any(c in _MASK_CHARS for c in ra) or any(c in _MASK_CHARS for c in rb):
        got = _masked_similarity(ra, rb)
        if got is not None and got[0] > 0:
            return got
        # 输入的数字被掩码全遮住了 -> 换另一方当基准再试
        got = _masked_similarity(rb, ra)
        if got is not None and got[0] > 0:
            return got[0] * 0.95, got[1]
        return 0.0, ""

    # 片段包含: 输入可能是残缺的
    if da in db_ or db_ in da:
        short = da if len(da) <= len(db_) else db_
        n = len(short)
        if n >= 7:
            return 0.95, f"电话片段匹配({n} 位)"
        if n >= 4:
            return 0.72, f"电话尾号匹配({short})"
        return 0.40, f"电话片段过短({short})"

    # 等长比较: 尾号 4 位 + 号段 3 位
    if len(da) == len(db_) >= 7:
        tail = da[-4:] == db_[-4:]
        head = da[:3] == db_[:3]
        if tail and head:
            return 0.90, "电话尾号与号段均一致"
        if tail:
            return 0.55, "仅电话尾号一致"
        if head:
            return 0.40, "仅电话号段一致"

    base = _ratio(da, db_)
    return max(0.0, base * 0.5), ""


#: 地址各段的权重。合计 1.00, 同时用作"输入地址有多完整"的分母。
_ADDR_SEG_WEIGHTS: Tuple[Tuple[str, float], ...] = (
    ("province", 0.15), ("city", 0.25), ("district", 0.30),
    ("street", 0.20), ("detail", 0.10),
)


def sim_address(inp: Any, dbv: Any) -> Tuple[Optional[float], str]:
    """地址相似度: 结构化分段 (省/市/区/街道/详细) + **完整度惩罚**.

    ⚠️ 为什么要乘完整度
    -------------------
    如果只按"输入里有的段"计分, 那么输入只有一个"南海区"时, 该段一致就是
    满分 1.0 —— 和完整地址同分。可一个区下面可能有几千个包裹,
    "区名对上"远没有"省市区街道全对上"可信。

    实测后果: 输入 ``{姓名:"张", 地址:"南海区"}`` 总分 0.859, 越过 0.85 线
    **自动放行**。这在分拣线上是危险的。

    因此最终得分再乘一个完整度系数::

        得分 = 分段命中率 × (0.55 + 0.45 × 完整度)

    完整度 = 输入里出现过的段权重之和 / 全部段权重之和。
    """
    sa, sb = _s(inp).strip(), _s(dbv).strip()
    if not sa or not sb:
        return None, ""
    if sa == sb:
        return 1.0, "地址完全一致"

    pa, pb = parse_address(sa), parse_address(sb)

    got = 0.0
    total = 0.0
    hits: List[str] = []
    for seg, w in _ADDR_SEG_WEIGHTS:
        if seg == "detail":
            continue                       # 详细地址单独处理
        va, vb = pa.get(seg, ""), pb.get(seg, "")
        if not va:
            continue
        total += w
        if not vb:
            continue
        if va == vb or va in vb or vb in va:
            got += w
            hits.append({"province": "省", "city": "市",
                         "district": "区", "street": "街道"}[seg])
        else:
            got += w * _ratio(va, vb)

    if pa.get("detail"):
        total += 0.10
        got += 0.10 * _partial_ratio(pa["detail"], sb)

    if total <= 0:
        # 没解析出结构 (纯英文/数字地址等) -> 退化为整串相似度
        r = _partial_ratio(sa, sb)
        return r, (f"地址相似度 {r:.0%}" if r >= 0.6 else "")

    raw = min(1.0, got / total)

    # 完整度: 输入地址覆盖了多少个段
    full_w = sum(w for _s2, w in _ADDR_SEG_WEIGHTS)
    coverage = min(1.0, total / full_w) if full_w > 0 else 1.0
    score = raw * (0.55 + 0.45 * coverage)

    if hits:
        seen, uniq = set(), []
        for h in hits:
            if h not in seen:
                seen.add(h)
                uniq.append(h)
        reason = "地址" + "/".join(uniq) + "匹配"
        if coverage < 0.6:
            reason += f"(仅 {coverage:.0%} 完整)"
    else:
        reason = f"地址相似度 {score:.0%}" if score >= 0.6 else ""
    return max(0.0, min(1.0, score)), reason


def sim_name(inp: Any, dbv: Any) -> Tuple[Optional[float], str]:
    """姓名相似度: 掩码感知 + 姓氏/模糊匹配."""
    x, y = _s(inp).strip(), _s(dbv).strip()
    if not x or not y:
        return None, ""
    if x == y:
        return 1.0, "姓名完全一致"

    # 脱敏姓名 (张** / 张*伟 / *伟) -> 按可见片段比对
    if any(c in _MASK_CHARS for c in x) or any(c in _MASK_CHARS for c in y):
        got = _masked_similarity(x, y, digits_only=False)
        if got is None:
            got = _masked_similarity(y, x, digits_only=False)
        if got is not None:
            # 姓名只有 2~3 个字, 覆盖率惩罚会过重, 这里把分数往上兜一点
            raw = got[0]
            score = min(1.0, 0.35 + 0.65 * raw) if raw > 0 else 0.0
            if score >= 0.5:
                return score, (got[1] or "姓名可见字匹配")
            return score, ""

    # 输入只是姓 (AI 只读出一个字)
    if len(x) == 1 and y.startswith(x):
        return 0.55, f"姓氏一致({x})"
    if len(y) == 1 and x.startswith(y):
        return 0.55, f"姓氏一致({y})"

    # 姓相同 + 后续模糊
    if len(x) >= 2 and len(y) >= 2 and x[0] == y[0]:
        r = _ratio(x[1:], y[1:])
        return 0.5 + 0.4 * r, (f"姓氏一致且名字相似 {r:.0%}" if r >= 0.5 else "")

    base = _ratio(x, y)
    return base, (f"姓名相似度 {base:.0%}" if base >= 0.6 else "")


def sim_generic(inp: Any, dbv: Any) -> Tuple[Optional[float], str]:
    """通用字段相似度 (快递公司/分拣码/订单号等)."""
    x, y = _s(inp).strip(), _s(dbv).strip()
    if not x or not y:
        return None, ""
    if x == y:
        return 1.0, ""
    # 单号类字段先按规范化比较
    if _norm_code(x) == _norm_code(y):
        return 1.0, ""
    base = _ratio(x.upper(), y.upper())
    if base >= 0.8:
        return base, ""
    # 一方包含另一方 (如 "顺丰" 与 "顺丰速运")
    if x in y or y in x:
        short, long_ = (x, y) if len(x) <= len(y) else (y, x)
        return 0.60 + 0.35 * len(short) / float(len(long_)), ""
    return base * 0.6, ""


# --------------------------------------------------------------------------
# 权重计算
# --------------------------------------------------------------------------
def _tracking_present(input_fields: Dict[str, Any]) -> bool:
    """单号是否有**可用内容** —— 哪怕只是残片.

    注意区分两件事 (需求里是分开说的):
      · **完全缺失** -> 把 0.40 权重按比例分给电话/地址/姓名
      · **残缺** (如 ``...25482``) -> 保留 0.40 权重, 用前缀/后缀模糊匹配打分

    最初写成"不够长就剔除", 结果把 ``...25482`` 这种残片整条丢掉了 ——
    那正是分拣线上最需要利用的信息。
    """
    # 单号/订单号视为同一概念: 两列**任一**有内容就算在场。
    # 否则 AI 把号码填到 order_number 时, 这里会误判成"完全缺失",
    # 把 0.40 权重错分给电话/地址/姓名, 单号那一项等于没比。
    return any(len(_norm_code(v)) >= 2 for v in _tracking_values(input_fields))


def _weight_table(input_fields: Dict[str, Any]) -> Dict[str, float]:
    """基础权重表 (按单号是否**完全缺失**决定要不要重新分配)."""
    w = dict(_BASE_WEIGHTS)
    if not _tracking_present(input_fields):
        rest = w.pop("tracking_number", 0.0)
        ratio_sum = sum(_REDISTRIBUTE_RATIO.values())
        for k, r in _REDISTRIBUTE_RATIO.items():
            w[k] = w.get(k, 0.0) + rest * (r / ratio_sum)
    return w


#: 与"单号"等价的字段名。AI 经常把同一个号码填到其中任意一个。
_TRACKING_ALIASES: Tuple[str, ...] = ("tracking_number", "order_number")


def _tracking_values(fields: Optional[Dict[str, Any]]) -> List[str]:
    """取出输入里所有**非空**的"单号/订单号"写法.

    AI 有时填 tracking_number, 有时填 order_number, 有时两个都填。
    这里把它们都收进来, 后面分别与库里两列比对、取最高分。
    """
    out: List[str] = []
    for k in _TRACKING_ALIASES:
        v = _s((fields or {}).get(k)).strip()
        if v and v not in out:
            out.append(v)
    return out


def _has_tracking(fields: Optional[Dict[str, Any]]) -> bool:
    """单号"在场"的判定: 两列任一有值即算在场."""
    return bool(_tracking_values(fields))


def _present_fields(input_fields: Dict[str, Any],
                    w: Dict[str, float]) -> Dict[str, float]:
    """筛出输入里真的有值的字段及其基础权重."""
    present: Dict[str, float] = {}
    for k, v in w.items():
        if k == "others":
            if any(_s(input_fields.get(f)).strip() for f in _OTHER_FIELDS):
                present[k] = v
        elif k == "tracking_number":
            # 单号: tracking_number 或 order_number 任一有值都算在场
            if _has_tracking(input_fields):
                present[k] = v
        elif _s(input_fields.get(k)).strip():
            present[k] = v
    return present


def _effective_weights(input_fields: Dict[str, Any]) -> Dict[str, float]:
    """按输入里实际有哪些字段, 算出归一化权重.

    - 单号**完全缺失**时: 把 0.40 按 5:5:2 分给 电话/地址/姓名
      (对应需求里的 0.25 / 0.25 / 0.10 比例)
    - 单号**残缺**时: 保留 0.40, 由 :func:`sim_tracking` 做前后缀匹配
    - 最后只保留输入里真有值的字段, 再归一化到合计 1.0
    """
    present = _present_fields(input_fields, _weight_table(input_fields))
    total = sum(present.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in present.items()}


def _evidence_weight(input_fields: Dict[str, Any]) -> float:
    """输入实际携带的"证据量" = 有值字段的**基础权重**之和 (未归一化, 0~1).

    用于 :attr:`MatcherConfig.min_evidence` 守卫: 残片越少, 证据量越低。
    """
    return min(1.0, sum(_present_fields(input_fields,
                                        _weight_table(input_fields)).values()))


# --------------------------------------------------------------------------
# 候选检索
# --------------------------------------------------------------------------
def _like(v: str) -> str:
    return "%" + v.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _alnum_sql(col: str) -> str:
    e = f"UPPER({col})"
    for ch in (" ", "-", "_", ".", ":", "/", "\\"):
        e = f"REPLACE({e}, '{ch}', '')"
    return e


def _digits_sql(col: str) -> str:
    e = col
    for ch in (" ", "-", "(", ")", "+", ".", "/", "*", "×"):
        e = f"REPLACE({e}, '{ch}', '')"
    return e


def _fetch_candidates(conn: sqlite3.Connection, input_fields: Dict[str, Any],
                      cfg: MatcherConfig) -> List[Dict[str, Any]]:
    """从数据库取候选行 (统一转成普通 dict).

    先用 SQL 粗筛 (单号片段 / 电话片段 / 地址关键词任一命中即可),
    避免对整张表做模糊打分 —— 分拣线上库可能有几万条, 每次都全表模糊匹配太慢。
    粗筛结果太少时再退回有上限的全表扫描, 保证召回。

    注意: 返回的**必须是普通 dict**。sqlite3.Row 没有 ``.get()``,
    后续打分代码大量使用 ``record.get(...)``, 直接用 Row 会抛
    ``AttributeError: 'sqlite3.Row' object has no attribute 'get'``。
    """
    conds: List[str] = []
    params: List[Any] = []

    tn = _norm_code(input_fields.get("tracking_number"))
    if len(tn) >= 4:
        conds.append(f"{_alnum_sql('tracking_number')} LIKE ? ESCAPE '\\'")
        params.append(_like(tn))

    ph = _digits(input_fields.get("recipient_phone"))
    if len(ph) >= 4:
        conds.append(f"{_digits_sql('recipient_phone')} LIKE ? ESCAPE '\\'")
        params.append(_like(ph[-6:] if len(ph) >= 6 else ph))

    addr = _s(input_fields.get("recipient_address")).strip()
    if addr:
        pa = parse_address(addr)
        # 取最有区分度的那一段当关键词 (区 > 街道 > 市 > 省)
        for seg in ("district", "street", "city", "province"):
            kw = pa.get(seg, "")
            if len(kw) >= 2:
                conds.append("recipient_address LIKE ? ESCAPE '\\'")
                params.append(_like(kw))
                break

    name = _s(input_fields.get("recipient_name")).strip().strip("*×")
    if len(name) >= 2:
        conds.append("recipient_name LIKE ? ESCAPE '\\'")
        params.append(_like(name))

    rows: List[Dict[str, Any]] = []
    if conds:
        sql = f"SELECT * FROM parcels WHERE ({' OR '.join(conds)}) LIMIT ?"
        rows = [_row_to_plain(r) for r in conn.execute(sql, params + [cfg.max_scan])]

    # 粗筛不足 -> 补一次有上限的全表扫描
    if len(rows) < max(cfg.top_n, 5):
        seen = {r.get("id") for r in rows}
        for r in conn.execute("SELECT * FROM parcels ORDER BY id DESC LIMIT ?",
                              (cfg.max_scan,)):
            plain = _row_to_plain(r)
            if plain.get("id") not in seen:
                rows.append(plain)
    return rows


def _row_to_plain(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: ("" if row[k] is None else row[k]) for k in row.keys()}


# --------------------------------------------------------------------------
# 打分核心
# --------------------------------------------------------------------------
@dataclass
class Candidate:
    """一个候选包裹的打分结果."""

    record: Dict[str, Any]
    total_score: float
    field_scores: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    evidence: float = 0.0

    def to_dict(self, rank: int, cfg: MatcherConfig) -> Dict[str, Any]:
        return {
            "rank": rank,
            "id": self.record.get("id"),
            "tracking_number": self.record.get("tracking_number", ""),
            "recipient_name": self.record.get("recipient_name", ""),
            "recipient_phone": self.record.get("recipient_phone", ""),
            "recipient_address": self.record.get("recipient_address", ""),
            "courier_company": self.record.get("courier_company", ""),
            "sorting_code": self.record.get("sorting_code", ""),
            "total_score": round(self.total_score, 4),
            "confidence": _confidence_of(self.total_score, self.evidence, cfg),
            "field_scores": {k: round(v, 4) for k, v in self.field_scores.items()},
            "match_reason": "，".join(self.reasons) if self.reasons else "无明确匹配点",
            # 下面两项是给"字段对照 + 图片对照"界面用的:
            #   · image_path —— 信息库那条记录自己的面单照片, 要和本次检测图并排比对
            #   · db_record  —— 完整记录, 供逐字段对照(上面只暴露了 7 个字段)
            # 不放进来的话界面拿不到库里的图片和大部分字段, 对照就无从谈起。
            "image_path": self.record.get("image_path", "") or "",
            "db_record": {k: v for k, v in self.record.items()
                          if k not in ("id",)},
        }


def _confidence_of(score: float, evidence: float, cfg: MatcherConfig,
                   runner_up: Optional[float] = None) -> str:
    """把总分 + 证据量 + 歧义度映射成置信等级.

    **这是本模块偏离需求字面规则的地方**, 有两道守卫:

    1. **证据量守卫**. 需求只按总分分级, 但归一化会让"只剩一个弱字段"的输入
       拿到很高的分: AI 只读出一个电话尾号 "1234", 该字段归一化权重就是 1.0,
       匹配上总分即 1.0 -> 按 ≥0.85 就该"自动放行"。分拣线上这等于凭 4 位
       残片自动分拣一个包裹, 错分代价很高。因此证据量不足时强制降为人工复核。

    2. **歧义守卫**. 如果库里有多个包裹同样像 (第一、二名分值接近),
       高分并不代表"选对了", 而只代表"这几个都像"。此时也降为人工复核。
    """
    if runner_up is not None and (score - runner_up) < cfg.ambiguity_margin:
        return "medium" if score >= cfg.review else "low"
    if score >= cfg.auto_pass:
        if evidence >= cfg.min_evidence:
            return "high"          # 自动通过
        return "medium"            # 证据不足, 降级为人工复核
    if score >= cfg.review:
        return "medium"
    return "low"


def _score_one(input_fields: Dict[str, Any], weights: Dict[str, float],
               record: Dict[str, Any], evidence: float) -> Candidate:
    """对单个候选打分."""
    scores: Dict[str, float] = {}
    reasons: List[str] = []

    for key, w in weights.items():
        if key == "others":
            sub: List[Tuple[float, str]] = []
            for f in _OTHER_FIELDS:
                s, r = sim_generic(input_fields.get(f), record.get(f))
                if s is not None:
                    sub.append((s, r))
            if not sub:
                continue
            s = sum(x[0] for x in sub) / len(sub)
            scores["others"] = s
            for _, r in sub:
                if r:
                    reasons.append(r)
        elif key == "tracking_number":
            # 单号/订单号视为同一个概念: 输入的每个号码分别与
            # **库里 tracking_number 和 order_number 两列**比, 取最高分。
            # 这样 AI 把号码填到哪一列都不影响判定。
            best: Optional[float] = None
            best_reason = ""
            for iv in _tracking_values(input_fields):
                for dk in _TRACKING_ALIASES:
                    s, r = sim_tracking(iv, record.get(dk))
                    if s is None:
                        continue
                    if best is None or s > best:
                        best, best_reason = s, r
            if best is None:
                continue
            scores[key] = best
            if best_reason:
                reasons.append(best_reason)
        else:
            fn = {"recipient_phone": sim_phone,
                  "recipient_address": sim_address,
                  "recipient_name": sim_name}.get(key, sim_generic)
            s, r = fn(input_fields.get(key), record.get(key))
            if s is None:
                continue
            scores[key] = s
            if r:
                reasons.append(r)

    total = sum(scores.get(k, 0.0) * w for k, w in weights.items())
    return Candidate(record=record, total_score=total, field_scores=scores,
                     reasons=reasons, evidence=evidence)


# --------------------------------------------------------------------------
# 对外主接口
# --------------------------------------------------------------------------
def _empty_result(input_fields: Dict[str, Any], match_type: str,
                  message: str) -> Dict[str, Any]:
    """空/失败结果.

    ⚠️ 必须显式给出 ``needs_review: True``。否则调用方读到的 ``needs_review``
    是 ``None`` (假值), 会把这个"什么都没比对上"的结果误当成"可以放行"。
    """
    return {"input_fields": dict(input_fields or {}), "match_type": match_type,
            "candidates": [], "total_candidates": 0, "message": message,
            "confidence": "low", "needs_review": True,
            "backend": _BACKEND}


def _tracking_is_valid(tn: Any) -> bool:
    """单号是否"完整且有效" (用于判定走精确匹配还是模糊匹配)."""
    t = _norm_code(tn)
    if len(t) < _MIN_TRACKING_LEN:
        return False
    if not any(c.isdigit() for c in t):
        return False
    # 掩码值 (JD****5643) 不算完整
    if any(c in _MASK_CHARS for c in _s(tn)):
        return False
    return True


#: 比对时**认得的输入字段**. 用来判断"输入里到底有没有实质内容"。
#:
#: ⚠️ 为什么要单独列一份: AI 视觉模块识别失败时会返回
#: ``{15 个字段全 None, "error": "原因"}``。若按"任意键非空"来判断,
#: 那句 error 文本会被当成有输入, 于是拿着错误信息去和数据库瞎匹配,
#: 产出一堆毫无意义的候选。只看认识的面单字段就能避免。
_KNOWN_INPUT_FIELDS: Tuple[str, ...] = (
    "tracking_number", "recipient_name", "recipient_phone", "recipient_address",
    "sender_name", "sender_phone", "sender_address", "sorting_code",
    "order_number", "item_name", "cod_amount", "insured_amount",
    "shipping_date", "courier_company", "sender_branch",
)


def _has_any_input(fields: Dict[str, Any]) -> bool:
    """输入里是否有**实质内容** (只算认识的面单字段).

    ⚠️ 单号要按 :func:_has_tracking 判 —— `order_number` 可能就在
    `_KNOWN_INPUT_FIELDS` 里, 但两者视为同一概念, 用统一判定更稳。
    """
    if _has_tracking(fields):
        return True
    return any(_s(fields.get(f)).strip() for f in _KNOWN_INPUT_FIELDS)


def compare_fields(input_fields: Optional[Dict[str, Any]],
                   db_path: Optional[object] = None,
                   *,
                   config: Optional[MatcherConfig] = None,
                   top_n: Optional[int] = None,
                   candidates: Optional[Iterable[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """多字段比对核心.

    参数
    ----
    input_fields
        残缺或完整的面单字段字典 (AI 输出)。
    db_path
        ``parcels.db`` 路径; 省略则用 :func:`core.database.resolve_db_path`。
    config / top_n
        覆盖默认参数。
    candidates
        直接给一批候选记录 (不查库)。**测试与离线评测用**,
        传了它就不会碰数据库。

    返回
    ----
    见模块文档与 README 的输出示例。
    """
    cfg = config or DEFAULT_CONFIG
    n = int(top_n or cfg.top_n)
    fields = dict(input_fields or {}) if isinstance(input_fields, dict) else {}

    if not fields or not _has_any_input(fields):
        # AI 失败时会给 {15 字段全 None, "error": "..."}。这里刻意给出更明确的
        # 提示, 免得用户看到"输入为空"却明明有错误信息、一头雾水。
        hint = _s(fields.get("error")).strip()
        msg = (f"AI 识别失败, 未做比对: {hint}" if hint
               else "输入为空, 没有可比对的字段")
        return _empty_result(fields, "fuzzy", msg)

    tn_valid = _tracking_is_valid(fields.get("tracking_number"))

    # ---- 准备候选 ----
    try:
        if candidates is not None:
            rows = [_row_to_plain_like(r) for r in candidates]
        else:
            _db = _get_db_module()
            if _db is None:
                return _empty_result(fields, "exact" if tn_valid else "fuzzy",
                                     "找不到 database 模块, 无法读取数据库")
            dbp = _db.resolve_db_path(db_path)
            if not Path(dbp).is_file():
                return _empty_result(fields, "exact" if tn_valid else "fuzzy",
                                     f"数据库不存在: {dbp}")
            with _db.session(dbp) as conn:
                _ensure_table(conn)
                if conn.execute("SELECT COUNT(*) FROM parcels").fetchone()[0] == 0:
                    return _empty_result(fields, "exact" if tn_valid else "fuzzy",
                                         "数据库为空, 没有可比对的包裹")
                rows = _fetch_candidates(conn, fields, cfg)
    except Exception as exc:                        # noqa: BLE001 - 对外要友好
        return _empty_result(fields, "exact" if tn_valid else "fuzzy",
                             f"读取数据库失败: {type(exc).__name__}: {exc}")

    if not rows:
        return _empty_result(fields, "exact" if tn_valid else "fuzzy",
                             "没有检索到任何候选包裹")

    # ---- 第一层: 精确匹配 ----
    if tn_valid:
        want = _norm_code(fields["tracking_number"])
        hits = [r for r in rows if _norm_code(r.get("tracking_number")) == want]
        if hits:
            # 库中该单号有几条 —— 宽松模式靠它判断"唯一"
            return _exact_result(fields, hits[0], cfg, same_tracking=len(hits))

    # ---- 第二层: 模糊加权 ----
    return _fuzzy_result(fields, rows, cfg, n)


def _row_to_plain_like(r: Any) -> Dict[str, Any]:
    """把 dict / sqlite3.Row 统一成普通 dict."""
    if isinstance(r, dict):
        return {k: ("" if v is None else v) for k, v in r.items()}
    try:
        return _row_to_plain(r)
    except Exception:                               # noqa: BLE001
        return {}


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS parcels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tracking_number TEXT, recipient_name TEXT, recipient_phone TEXT,
            recipient_address TEXT, sender_name TEXT, sender_phone TEXT,
            sender_address TEXT, sorting_code TEXT, order_number TEXT,
            item_name TEXT, cod_amount TEXT, insured_amount TEXT,
            shipping_date TEXT, courier_company TEXT, sender_branch TEXT,
            image_path TEXT, is_damaged INTEGER DEFAULT 0, damage_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")


def _normalise_mode(mode: Any) -> str:
    """放行模式只有两个: lenient / strict。写错一律按 strict (保守的一档)。"""
    m = ("" if mode is None else str(mode)).strip().lower()
    return "lenient" if m == "lenient" else "strict"


def _release_note(mode: str, unique: bool) -> str:
    """给结果里附一句"为什么这么判", 便于现场理解。"""
    if mode == "lenient":
        return ("宽松模式: 单号匹配且库中唯一, 直接放行 (不参与加权打分)"
                if unique else
                "宽松模式: 单号匹配但库中有多条同号记录, 转严格模式判定")
    return "严格模式: 需加权总分达标才放行"


def _exact_result(fields: Dict[str, Any], record: Dict[str, Any],
                  cfg: MatcherConfig, same_tracking: int = 1) -> Dict[str, Any]:
    """单号命中分支.

    **宽松模式**: 单号匹配且库中唯一 -> 直接放行, 不算加权分。

    **严格模式** (含宽松但库中不唯一): 走**加权打分** ——
    "单号 + 电话 + 地址 + 姓名 + 其它" 加权后的总分达到
    :attr:`MatcherConfig.auto_pass` 才放行, 否则转人工。

    两种模式对同一条记录的差别, 举个直观例子::

        单号对(0.40) + 电话对(0.20) + 地址对(0.20) + 姓名不符(0) + 其它(0)
            -> 加权总分约 0.80 -> 严格模式下不达标, 转人工
        宽松模式下 (库中唯一) -> 直接放行
    """
    mode = _normalise_mode(getattr(cfg, "release_mode", "strict"))
    unique = same_tracking <= 1
    evidence = _evidence_weight(fields)

    # 先算一遍字段相似度, 用于展示与说明 (两种模式都用得上)
    checked: Dict[str, float] = {"tracking_number": 1.0}
    reasons = ["单号精确命中数据库"]
    if same_tracking > 1:
        reasons.append(f"库中有 {same_tracking} 条同号记录")
    for key, fn in (("recipient_phone", sim_phone),
                    ("recipient_address", sim_address),
                    ("recipient_name", sim_name)):
        s, _r = fn(fields.get(key), record.get(key))
        if s is None:
            continue
        checked[key] = s
        label = FIELD_LABELS.get(key, key)
        if s < cfg.conflict_threshold:
            reasons.append(f"{label}与库中记录严重不符(相似度 {s:.0%})")
        elif s < 0.85:
            reasons.append(f"{label}基本一致({s:.0%})")
        else:
            reasons.append(f"{label}一致")

    note = _release_note(mode, unique)

    # ---------- 宽松模式的快速通道 ----------
    if mode == "lenient" and unique:
        cand = Candidate(record=record, total_score=1.0, field_scores=checked,
                         reasons=reasons, evidence=evidence)
        return {
            "input_fields": fields,
            "match_type": "exact",
            "candidates": [cand.to_dict(1, cfg)],
            "total_candidates": 1,
            "message": "精确匹配成功, 可直接放行 (宽松模式: 单号匹配且库中唯一)",
            "confidence": "high",
            "evidence": round(evidence, 4),
            "needs_review": False,
            "release_mode": mode,
            "tracking_unique": unique,
            "release_note": note,
            "backend": _BACKEND,
        }

    # ---------- 严格模式: 加权打分 ----------
    weights = _effective_weights(fields)
    if not weights:
        # 输入里只有单号, 没有别的可加权字段 —— 退化成"单号就是全部证据"
        weights = {"tracking_number": 1.0}
    cand = _score_one(fields, weights, record, evidence)
    # 单号既然已经精确命中, 就把这一项的分数钉成 1.0
    # (_score_one 走的是通用相似度, 规范化差异可能让它略低于 1)
    cand.field_scores["tracking_number"] = 1.0
    cand.reasons = reasons + [r for r in cand.reasons if "单号" not in r]

    score = cand.total_score
    ok_score = score >= cfg.auto_pass
    ok_evidence = evidence >= cfg.min_evidence
    # 严格模式还要求"确实有单号以外的信息参与打分"。
    # 否则会出现这种自相矛盾: 输入里只有单号 -> 权重只剩单号 -> 加权分必然
    # 1.00 -> 严格模式照样放行, 等于没严格。需求说的是"单号**和其他信息**的
    # 加权得分", 所以至少要有一项其它字段参与。
    other_scored = [k for k in cand.field_scores
                    if k != "tracking_number" and cand.field_scores.get(k) is not None]
    ok_other = bool(other_scored)

    if ok_score and ok_evidence and ok_other:
        return {
            "input_fields": fields,
            "match_type": "exact",
            "candidates": [cand.to_dict(1, cfg)],
            "total_candidates": 1,
            "message": (f"精确匹配成功, 加权总分 {score:.2f} 达标, 可放行"
                        f" [{note}]"),
            "confidence": "high",
            "evidence": round(evidence, 4),
            "needs_review": False,
            "release_mode": mode,
            "tracking_unique": unique,
            "release_note": note,
            "backend": _BACKEND,
        }

    if not ok_other:
        msg = (f"单号命中, 但输入里**没有单号以外的信息**可参与加权打分, "
               f"严格模式下不足以放行, 建议人工复核")
    elif not ok_score:
        msg = (f"单号命中, 但加权总分 {score:.2f} < {cfg.auto_pass}, "
               f"建议人工复核")
    else:
        msg = (f"加权总分 {score:.2f} 虽达标, 但输入信息过少"
               f"(证据量 {evidence:.2f} < {cfg.min_evidence}), 建议人工复核")
    if mode == "lenient" and not unique:
        msg = (f"单号命中, 但库中有 {same_tracking} 条同号记录, "
               f"无法确定是哪一条; 加权总分 {score:.2f}, 建议人工复核")
    msg += f" [{note}]"

    return {
        "input_fields": fields,
        "match_type": "exact",
        "candidates": [cand.to_dict(1, cfg)],
        "total_candidates": 1,
        "message": msg,
        "confidence": "medium" if score >= cfg.review else "low",
        "evidence": round(evidence, 4),
        "needs_review": True,
        "release_mode": mode,
        "tracking_unique": unique,
        "release_note": note,
        "backend": _BACKEND,
    }


def _fuzzy_result(fields: Dict[str, Any], rows: Sequence[Dict[str, Any]],
                  cfg: MatcherConfig, n: int) -> Dict[str, Any]:
    """模糊加权匹配分支."""
    weights = _effective_weights(fields)
    if not weights:
        return _empty_result(fields, "fuzzy", "输入里没有可用于比对的字段")

    evidence = _evidence_weight(fields)
    scored = [_score_one(fields, weights, r, evidence) for r in rows]
    scored.sort(key=lambda c: (-c.total_score,
                               _s(c.record.get("tracking_number"))))
    top = scored[:max(1, n)]

    best = top[0]
    runner_up = top[1].total_score if len(top) > 1 else None
    level = _confidence_of(best.total_score, evidence, cfg, runner_up)

    ambiguous = (runner_up is not None
                 and (best.total_score - runner_up) < cfg.ambiguity_margin)
    if level == "high":
        msg = "匹配成功, 置信度高, 可自动通过"
    elif ambiguous:
        msg = (f"存在多个相似候选 (第1名 {best.total_score:.2f} / "
               f"第2名 {runner_up:.2f}, 差距小于 {cfg.ambiguity_margin}), "
               f"不可自动通过, 请人工复核")
    elif best.total_score >= cfg.auto_pass and evidence < cfg.min_evidence:
        msg = (f"分数虽高({best.total_score:.2f})但输入信息过少"
               f"(证据量 {evidence:.2f} < {cfg.min_evidence}), "
               f"不可自动通过, 请人工复核")
    elif level == "medium":
        msg = "匹配到中等置信度候选, 建议人工复核"
    else:
        msg = "未找到足够相似的包裹, 建议人工重新录入"

    # 放行模式: 模糊分支 (单号没精确命中) 两种模式行为一致 ——
    # 本来就走加权打分, 达到 auto_pass 才放行。宽松模式的"快速通道"
    # 只针对"单号匹配且唯一"这一种情况。
    mode = _normalise_mode(getattr(cfg, "release_mode", "strict"))
    needs_review = level != "high"
    if mode == "lenient" and not needs_review:
        msg += " [宽松模式: 单号未精确命中, 按加权得分放行]"

    return {
        "input_fields": fields,
        "match_type": "fuzzy",
        "candidates": [c.to_dict(i, cfg) for i, c in enumerate(top, 1)],
        "total_candidates": len(scored),
        "message": msg,
        "confidence": level,
        "evidence": round(evidence, 4),
        "ambiguous": ambiguous,
        "weights_used": {k: round(v, 4) for k, v in weights.items()},
        "needs_review": needs_review,
        "release_mode": mode,
        "backend": _BACKEND,
    }


# --------------------------------------------------------------------------
# AI 接口
# --------------------------------------------------------------------------
#: 可能包着记录的键 (AI 输出有时会包一层)
_WRAPPER_KEYS = ("fields", "result", "data", "parcel", "record", "ai_result",
                 "prediction", "output")


def _get_db_module():
    """取得 database 模块.

    本模块既可作为 ``core.matcher`` 被导入, 也允许**单独拷贝出去**当脚本用,
    因此两种导入方式都要兼容。
    """
    try:
        from . import database as mod            # 包内导入
        return mod
    except Exception:                            # noqa: BLE001
        pass
    try:
        import database as mod                   # 独立脚本/同目录导入
        return mod
    except Exception:                            # noqa: BLE001
        return None


def _unwrap_ai(ai_result: Any) -> Dict[str, Any]:
    """把 AI 输出规整成一个字段字典.

    容忍三种常见形态:
      1. 直接就是字段字典            ``{"tracking_number": ...}``
      2. 包一层                      ``{"fields": {...}}``
      3. 带元信息的                   ``{"image": "...", "fields": {...}}``
    """
    if not isinstance(ai_result, dict):
        return {}
    for k in _WRAPPER_KEYS:
        v = ai_result.get(k)
        if isinstance(v, dict):
            return dict(v)
    return dict(ai_result)


def match_from_ai(ai_result: Any,
                  db_path: Optional[object] = None,
                  *,
                  config: Optional[MatcherConfig] = None,
                  top_n: Optional[int] = None) -> Dict[str, Any]:
    """处理**单张**面单的 AI 输出.

    自动判断走精确匹配还是模糊匹配, 其余交给 :func:`compare_fields`。
    """
    try:
        if ai_result is None:
            return _empty_result({}, "fuzzy", "AI 输出为空")
        if isinstance(ai_result, (list, tuple)):
            # 传进来是数组却调了单张接口 -> 取第一条并提示
            if not ai_result:
                return _empty_result({}, "fuzzy", "AI 输出是空数组")
            res = match_from_ai(ai_result[0], db_path, config=config, top_n=top_n)
            res["message"] = ("注意: 传入的是数组, 只处理了第一条。"
                              "批量请用 match_from_ai_batch()。" + res.get("message", ""))
            return res
        if not isinstance(ai_result, dict):
            return _empty_result({}, "fuzzy",
                                 f"AI 输出格式不认识: {type(ai_result).__name__}")

        fields = _unwrap_ai(ai_result)
        return compare_fields(fields, db_path, config=config, top_n=top_n)
    except Exception as exc:                        # noqa: BLE001 - 对外要友好
        return _empty_result({}, "fuzzy",
                             f"比对过程出错: {type(exc).__name__}: {exc}")


def match_from_ai_batch(ai_results: Any,
                        db_path: Optional[object] = None,
                        *,
                        config: Optional[MatcherConfig] = None,
                        top_n: Optional[int] = None) -> List[Dict[str, Any]]:
    """处理**多张**面单的 AI 输出 (JSON 数组).

    单张输入也会被接受 (返回只含一个元素的列表), 便于上游统一调用。
    单个元素出错只影响它自己, 不会中断整批。
    """
    if ai_results is None:
        return [_empty_result({}, "fuzzy", "批量输入为空")]
    if isinstance(ai_results, dict):
        # 兼容 {"results": [...]} 这类包装
        for k in ("results", "items", "data", "parcels", "records", "list"):
            if isinstance(ai_results.get(k), list):
                ai_results = ai_results[k]
                break
        else:
            ai_results = [ai_results]
    if not isinstance(ai_results, (list, tuple)):
        return [_empty_result({}, "fuzzy",
                              f"批量输入格式不认识: {type(ai_results).__name__}")]

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(ai_results):
        try:
            out.append(match_from_ai(item, db_path, config=config, top_n=top_n))
        except Exception as exc:                    # noqa: BLE001
            out.append(_empty_result({}, "fuzzy",
                                     f"第 {i + 1} 条处理失败: "
                                     f"{type(exc).__name__}: {exc}"))
    return out


# --------------------------------------------------------------------------
# 便于直接运行: 造一个演示库并跑两种输入
# --------------------------------------------------------------------------
def _demo() -> int:                                 # pragma: no cover
    import json
    import tempfile

    _db = _get_db_module()
    if _db is None:
        print("找不到 database 模块, 无法运行演示。请从 src 目录或包内运行。")
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="matcher_demo_"))
    dbp = tmp / "parcels.db"
    _db.init_database(dbp)

    samples = [
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
    for s in samples:
        _db.save_record(s, path=dbp)

    print(f"演示库: {dbp}  ({_db.count_parcels(dbp)} 条记录)")
    print(f"相似度后端: {similarity_backend()}")
    print("=" * 78)

    print("\n【场景一】AI 读全了 -> 应走精确匹配")
    ai_full = dict(samples[0])
    print(json.dumps(match_from_ai(ai_full, dbp), ensure_ascii=False, indent=2))

    print("\n【场景二】条码破损 + 文字残缺 -> 只能模糊匹配")
    ai_partial = {"recipient_phone": "****1234",
                  "recipient_address": "南海区",
                  "tracking_number": None}
    print(json.dumps(match_from_ai(ai_partial, dbp), ensure_ascii=False, indent=2))

    print("\n【场景三】只读出一个电话尾号 -> 分数可能很高但证据不足")
    ai_tiny = {"recipient_phone": "1234"}
    print(json.dumps(match_from_ai(ai_tiny, dbp), ensure_ascii=False, indent=2))

    print("\n【场景四】批量输入")
    batch = match_from_ai_batch([ai_full, ai_partial], dbp)
    for i, r in enumerate(batch, 1):
        c = (r.get("candidates") or [{}])[0]
        print(f"  {i}. match_type={r['match_type']} score={c.get('total_score')} "
              f"tn={c.get('tracking_number')}  {r.get('message')}")
    return 0


if __name__ == "__main__":                          # pragma: no cover
    import sys

    # 允许直接 `python src/core/matcher.py` 运行: 把 src 目录加进搜索路径,
    # 这样 _get_db_module() 里的 `import database` 能生效。
    _src = Path(__file__).resolve().parents[1]
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    import os

    _core = str(Path(__file__).resolve().parent)
    if _core not in sys.path:
        sys.path.insert(0, _core)
    raise SystemExit(_demo())
