"""AI 视觉识别 —— 用本地 Ollama 的视觉模型从面单图片直接提取结构化字段.

**默认模型**: ``qwen3-vl:2b-instruct``

    这个默认值是**用真实面单横评后**定的。最初用的是 ``qwen3-vl:2b``,
    16 张真实面单实测下来明显更差 (成功 14/16、均 10.1 字段、均 48.5 秒),
    换成 ``-instruct`` 后是 16/16、均 14.5 字段、均 8.8 秒。
    详见 :data:`MODEL_NAME` 处的完整对比数据。

    换模型不必改代码: 设环境变量 ``PARCEL_AI_MODEL`` 即可。

在整个系统里的位置
------------------
::

    面单图片 → ai_vision.extract_parcel_info() → 15 字段 JSON
             → matcher.match_from_ai()          → Top-N 候选
             → 界面展示 + 人工确认
             → (可选) 写入数据库

与原有的 PaddleOCR 链路是**并列的两条路**::

    条码扫描 ─成功→ 放行
        └─失败→ ┬─ PaddleOCR (本项目原有) ─→ fields.py 版面解析 ─→ matcher
                └─ AI 视觉   (本模块)     ─→ 直接出结构化字段 ─→ matcher

本模块**不依赖** PaddleOCR / OpenCV 的识别链路, 可以单独使用。

几个实现上的关键决定 (都有实测依据)
----------------------------------
1. **用 JSON Schema 约束输出** (``format=<schema>``)。
   需求要求"字段名固定, 不可增删、不可改名"。实测把 schema 传给 Ollama 后,
   返回的键数正好是 15, 无多余无缺失 —— 比在 Python 侧清洗更硬。
   万一所用 Ollama 版本不支持 schema, 会自动退回 ``format="json"``,
   再由 :func:`normalize_record` 在 Python 侧兜底补齐。

2. **``think=False`` 会传, 但不要指望它彻底关掉思考**。
   实测 (在 qwen3-vl:2b 上) 这个 Ollama 版本的 think=False 并不生效, 仍然会产生思考内容,
   但 Ollama 把思考放在响应的独立 ``thinking`` 字段里, **``content`` 是干净 JSON**,
   所以解析不受影响。需求里"避免输出冗长思考过程"的实际目的已经达到。

3. **图片先降采样**。手机照片动辄 4000×3000, 对 2B 模型既慢又无益。
   默认长边压到 1600 (可配)。

4. **失败不抛异常**, 返回带 ``error`` 键的字典 (需求明确要求)。

耗时参考 (本机 RTX 5060, 实测)
    · 首次调用 (含模型加载): 约 23 秒
    · 模型常驻后:            约 8~10 秒/张
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "FIELDS",
    "MODEL_NAME",
    "PROMPT",
    "VisionConfig",
    "DEFAULT_CONFIG",
    "extract_parcel_info",
    "extract_parcel_info_batch",
    "extract_and_match",
    "extract_and_match_batch",
    "resolve_sender_recipient",
    "swap_sender_recipient",
    "check_service",
    "clean_json_text",
    "normalize_record",
]

#: 默认模型名。
#:
#: ⚠️ 这是**用真实面单横评后**改的: 由 ``qwen3-vl:2b`` 换成 ``qwen3-vl:2b-instruct``。
#: 同一批 16 张真实面单、同一套配置, 两个模型的实测对比:
#:
#: ====================  ==========  ==========  ==========  ==========
#: 模型                   成功        均字段      均耗时      中位耗时
#: ====================  ==========  ==========  ==========  ==========
#: qwen3-vl:2b           14/16       10.1        48.5s       21.6s
#: qwen3-vl:2b-instruct  **16/16**   **14.5**    **8.8s**    **7.0s**
#: ====================  ==========  ==========  ==========  ==========
#:
#: 决定性的一题 (中国邮政那张, 单号 9819989866891, 面单上有个很显眼的
#: 大字分拣码 197-197-A45-130):
#:
#:   · qwen3-vl:2b          194.7 秒 -> 单号读出"空"(把大字看成了别的)
#:   · qwen3-vl:2b-instruct   7.8 秒 -> 单号 **9819989866891 正确**,
#:                                       且正确区分了单号与分拣码
#:
#: 两个模型都读出单号的 10 张里, **10 张完全一致** —— 互相印证。
#:
#: 想换回去或试别的模型, 不用改代码: 设环境变量 ``PARCEL_AI_MODEL`` 即可
#: (见 :func:`_default_model`), 或直接传 ``VisionConfig(model=...)``。
MODEL_NAME = "qwen3-vl:2b-instruct"

#: 可用环境变量覆盖模型名, 方便不重新打包就切换模型。
MODEL_ENV_VAR = "PARCEL_AI_MODEL"


def _default_model() -> str:
    """取默认模型名: 环境变量 ``PARCEL_AI_MODEL`` 优先。"""
    return (os.environ.get(MODEL_ENV_VAR) or "").strip() or MODEL_NAME


#: 上下文长度可用环境变量覆盖 —— 换回"会思考"的模型时才需要调大。
NUM_CTX_ENV_VAR = "PARCEL_AI_NUM_CTX"


def _default_num_ctx() -> int:
    """取默认上下文长度: 环境变量 ``PARCEL_AI_NUM_CTX`` 优先, 否则 8192."""
    raw = (os.environ.get(NUM_CTX_ENV_VAR) or "").strip()
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return 8192
    return val if val >= 2048 else 8192


def cache_dir() -> Path:
    """AI 模块自己的缓存目录 (在系统临时目录下).

    **不往用户的图片目录里写任何东西。** 缩放后的中间图放这里。
    """
    import tempfile

    d = Path(tempfile.gettempdir()) / "ParcelDeskewOCR_ai"
    d.mkdir(parents=True, exist_ok=True)
    return d

#: Ollama 服务默认地址
DEFAULT_HOST = "http://localhost:11434"

#: 严格固定的 15 个字段 —— 不可增删、不可改名
FIELDS: Tuple[str, ...] = (
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
)

#: 提交给 Ollama 的 JSON Schema (结构化输出用).
#: 全部允许 null —— 需求要求"无法辨认时值为 null"。
SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {f: {"type": ["string", "null"]} for f in FIELDS},
    "required": list(FIELDS),
    "additionalProperties": False,
}

#: 提示词。
#:
#: ⚠️ 相对最初给定的提示词做了**两处必要修改**，都有实测依据：
#:
#: **修改一：去掉举例里的字面值 ``"张**""138****1234"``**
#:
#: 实测发现模型会**逐字照抄提示词里的示例值**当作答案。同一张真实面单，
#: 只改提示词里那两个举例值，输出跟着一起变：
#:
#:     提示词举例 = 张** / 138****1234    ->  输出 张** / 138****1234
#:     提示词举例 = 王小明 / 13900001111  ->  输出 王小明 / 13900001111
#:
#: 也就是说面单上那人的真实姓名电话根本没被读，输出的是提示词里的样板。
#: 改成不含字面值的描述后，这条路被堵住。
#:
#: **修改二：新增第 8 条，明确「收」/「寄」两个栏位的关系**
#:
#: 用户实测反馈"发货方/收件方的联系人信息会搞反位置"。面单上「收」和「寄」
#: 是两个相邻方框，模型容易串位，因此把版面约定直接写进规则。
#: 另加第 9 条：姓名栏里若实际是地址/商品名/网点名，应填 null 而不是硬塞
#: （实测模型会把"武汉余集发货区""米山药养吐司"这类文字填进姓名栏）。
PROMPT = """你是一个快递面单信息提取助手。请仔细查看这张快递面单图片，从中提取指定的字段信息，严格按 JSON 格式输出。

提取规则：
1. 只提取图片中实际可见的文字，不要猜测，不要编造。
2. 如果某个字段不存在、被遮挡、模糊不清或无法辨认，值设为 null。
3. 对于脱敏信息（姓名里保留的星号、电话号码中被遮挡的位），按原样输出，不要补全、不要替换成别的号码。
4. 忽略广告语、客服电话、快递员信息、签收提示等无关内容。
5. 不要输出任何解释文字，不要用 Markdown 代码块包裹，直接输出纯 JSON。
6. 不要思考，直接给出结果。
7. 如果面单上有多个单号，只提取字体最大、最显眼的主单号。
8. 面单上标有「收」字的方框是**收件人**（收方），标有「寄」字的方框是**寄件人**（寄方）。
   recipient_* 字段只填「收」方的内容，sender_* 字段只填「寄」方的内容，不要弄反。
9. 姓名栏只填人名。若该栏实际是地址、商品名、网点名、中转标签或时间，
   说明没有找到人名，该字段一律填 null。

需要提取的字段（固定，不可增删、不可改名）：
tracking_number, recipient_name, recipient_phone, recipient_address,
sender_name, sender_phone, sender_address, sorting_code,
order_number, item_name, cod_amount, insured_amount,
shipping_date, courier_company, sender_branch

单张图片输出一个 JSON 对象。
多张图片输出一个 JSON 数组，每个元素对应一张面单，按上传顺序排列。

直接输出 JSON。"""

#: 日志器 (调用方可自行配置 handler)
_LOGGER = logging.getLogger("ai_vision")

# 静音底层 HTTP 库: ollama 客户端基于 httpx, 它会把每个请求打成 INFO 日志
# (``HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"``)。
# 一旦外面调了 basicConfig(level=INFO) 这些行就会刷屏, 把有用的输出淹掉。
for _noisy in ("httpx", "httpcore", "urllib3", "ollama"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def _log(log: Optional[Callable[[str], None]], msg: str) -> None:
    """同时写标准日志与调用方回调 (界面要把日志显示到控制台)."""
    _LOGGER.info(msg)
    if log:
        try:
            log(msg)
        except Exception:                        # noqa: BLE001
            pass


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
@dataclass
class VisionConfig:
    """AI 视觉识别参数."""

    #: 模型名。默认取 :data:`MODEL_NAME`, 但**可被环境变量
    #: ``PARCEL_AI_MODEL`` 覆盖** —— 这样不重新打包也能换模型。
    model: str = field(default_factory=_default_model)
    #: Ollama 服务地址
    host: str = DEFAULT_HOST
    #: 单次请求超时 (秒)。VL 模型首次调用含加载, 给足余量。
    timeout: float = 300.0
    #: 采样温度, 取 0 让输出尽量稳定可复现
    temperature: float = 0.0
    #: 是否用 JSON Schema 约束输出。True 时若服务端不支持会自动退回 "json"。
    use_schema: bool = True
    #: **上下文长度** —— 直接决定显存占用与推理是否落在 GPU 上。
    #:
    #: 历史: 早期用 qwen3-vl:2b (会思考) 时, 思考内容能写到 3000~31000 字,
    #: 上下文给小了会""来不及输出 JSON""; 当时把默认值定到 32768。
    #:
    #: ⚠️ v2.1.1 实测: 现在的默认模型是 **2b-instruct (不思考)**, 32768 完全没必要,
    #: 而且会把模型撑到 9.8 GB —— 8 GB 显存装不下, Ollama 只能 41%/59% CPU/GPU
    #: 混跑, 单张 58~137 秒。改成 8192 后显存 2.6 GB、**100% GPU**, 单张 2~3 秒。
    #:
    #: 真实面单的提示词约 2300 token + 输出约 250 token, 8192 有充足余量;
    #: 只有换回"会思考"的模型时才需要调大 (可用 PARCEL_AI_NUM_CTX 覆盖)。
    num_ctx: int = field(default_factory=_default_num_ctx)
    #: 是否在提示词前加 Qwen3 的 ``/no_think`` 控制标记。
    #:
    #: 需求第 3 条要求"关闭思考模式", 并说明"可以在 prompt 中明确要求,
    #: 或使用 Ollama 的 nothink 参数"。实测 Ollama 的 ``think=False`` 对这个模型
    #: **不生效** (思考照旧产生, 三种 format 表现完全一样); 而 ``/no_think``
    #: 在真实面单上把平均思考量从 12540 字压到 3822 字 (约 1/3)。
    #: **原始提示词一字未改**, 只是在最前面加了这个控制标记。
    no_think: bool = True
    #: 单次尝试的超时 (秒)。超时就换更省资源的配置重试,
    #: 避免一张特别难的图把整批拖到几分钟。
    attempt_timeout: float = 25.0
    #: 单张**总预算**（秒）。超过就不再尝试下一档，直接返回最后一次的错误。
    #:
    #: 正常 2~4 秒；病态图（模型反复输出 1024 token 的垃圾）在 3 档重试下
    #: 能烧掉 30 秒以上。设总预算后，最坏情况也有上限。
    total_budget_s: float = 20.0
    #: 模型在显存里的保留时间（Ollama ``keep_alive``）。
    #:
    #: 默认 ``5m`` 时，两次识别间隔一长就要重新加载模型（GPU 上约 7 秒）。
    #: 调到 30 分钟，连续核件时模型常驻，单张只剩推理时间。
    keep_alive: str = "30m"
    #: 单次输出 token 上限。面单 15 字段的 JSON 约 250 token，
    #: 给 1024 足够；设 0 表示不限制（Ollama 默认）。
    max_output_tokens: int = 1024
    #: **放行模式**, 透传给 matcher。取值 ``"lenient"`` / ``"normal"`` / ``"strict"``:
    #:
    #: ``lenient`` 宽松
    #:     单号正确且库中唯一 -> 直接放行 (不再看其它字段)
    #: ``normal``  常规 (默认)
    #:     单号命中且无字段冲突才放行
    #: ``strict``  严格
    #:     任何情况都建议人工复核, 包括单号完全正确的情况
    release_mode: str = "strict"
    #: **匹配到库中记录时, 收发信息以库为准** (方案二)。默认关闭。
    #:
    #: 打开后: 单号命中库里某条记录时, 用那条记录的 ``recipient_*`` /
    #: ``sender_*`` 覆盖 AI 的读数。理由是单号一旦对上, 库里那条就是事实,
    #: 不必再纠结模型把收件方/寄件方读成了哪个方向。
    #:
    #: ⚠️ 为什么默认关: 它会**遮蔽 AI 的读数**。如果你用这个功能来核对
    #: 模型读得对不对, 开了就看不见差异了。覆盖前的原值仍保留在结果的
    #: ``ai_from_model`` 里。
    prefer_db_contacts: bool = False
    #: 图片长边超过它就先等比缩小 (省时间; 手机照片动辄 4000px)
    max_image_side: int = 1600
    #: 额外尝试次数 (不含首次)。决定重试阶梯走几档:
    #: 0 = 只试主配置; 1 = 主配置 + 加大上下文; 2 = 再缩图重试 (默认)。
    retries: int = 2
    #: 追加给 Ollama 的其它 options
    extra_options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"model": self.model, "host": self.host,
                "timeout": self.timeout, "max_image_side": self.max_image_side,
                "use_schema": self.use_schema, "retries": self.retries,
                "num_ctx": self.num_ctx, "no_think": self.no_think,
                "attempt_timeout": self.attempt_timeout,
                "total_budget_s": self.total_budget_s,
                "keep_alive": self.keep_alive,
                "max_output_tokens": self.max_output_tokens}


DEFAULT_CONFIG = VisionConfig()


# --------------------------------------------------------------------------
# 输出清洗与规范化
# --------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")


def clean_json_text(text: Any) -> str:
    """清洗模型返回的文本, 得到可以喂给 ``json.loads`` 的字符串.

    要处理的实际形态 (都真实遇到过):

    * ````` ```json ... ``` ````` 代码块包裹
    * 前后夹带解释文字 (``好的, 结果如下: {...}``)
    * 输出被截断或多余尾巴
    * 全角引号 / BOM

    做法: 去掉代码围栏 → 若整段解析不了, 再退回"截取第一个 ``{`` 或 ``[``
    到最后一个配对的 ``}`` / ``]``"。
    """
    t = "" if text is None else str(text)
    t = t.strip().lstrip("\ufeff")
    if not t:
        return ""

    # 反复剥掉代码围栏 (模型偶尔会套两层)
    for _ in range(3):
        new = _FENCE_RE.sub("", t).strip()
        if new == t:
            break
        t = new

    # 全角引号会让 json.loads 直接失败, 顺手换成半角
    t = t.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    # 截取最外层的 JSON。
    # ⚠️ 这一步**不能**因为"文本以 { 开头"就跳过: 模型完全可能先输出正确的
    # JSON, 后面再补一句"以上就是提取结果。" —— 那样 json.loads 会因为尾部
    # 有杂质而整体失败, 白白丢掉一份本来正确的结果。
    starts = [i for i in (t.find("{"), t.find("[")) if i >= 0]
    if not starts:
        return t
    start = min(starts)
    end = max(t.rfind("}"), t.rfind("]"))
    if end > start:
        return t[start:end + 1]
    return t[start:]


def parse_ai_json(text: Any) -> Tuple[Any, str]:
    """解析模型返回的 JSON; 返回 ``(对象, 错误说明)``."""
    cleaned = clean_json_text(text)
    if not cleaned:
        return None, "模型没有返回任何内容"
    try:
        return json.loads(cleaned), ""
    except Exception as exc:                     # noqa: BLE001
        preview = cleaned[:200].replace("\n", " ")
        return None, f"JSON 解析失败: {exc} (返回内容前 200 字: {preview})"


#: 面单上的"分拣码"形态。
#:
#: ⚠️ 真实面单上最要命的失败模式: 面单上**字体最大最显眼**的往往不是单号,
#: 而是目的地分拣码 (实测遇到 ``197-197-A45-130``、``620A N089-00 101``、
#: ``155- K8 201 [D10]``、``610 L31 101``)。2B 模型经常把这个大字当成单号
#: 填进 ``tracking_number``, 而单号恰恰是后续精确匹配的**唯一依据**。
#:
#: 18 张真实面单实测: 16 张返回了"单号", 其中 5~6 张填的其实是分拣码。
#:
#: 判别依据 (只用**高置信度**的那一条, 命中才动, 不猜)::
#:
#:     真单号  YT0713590525482 / 9819989866891 / SF3263531736719
#:             -> 都有一段 >= 8 位的连续字母数字
#:     分拣码  197-197-A45-130 / 620A N089-00 101 / 610 L31 101
#:             -> 被分隔符切成若干短段, 最长连续段很短
_SORTING_SEP = re.compile(r"[\s\[\]【】()（）\-_/、,，]")
_ALNUM_RUN = re.compile(r"[0-9A-Za-z]+")


def _looks_like_sorting_code(value: Any) -> bool:
    """判断这个值是不是"其实是个分拣码"而不是快递单号."""
    s = "" if value is None else str(value).strip()
    if not s:
        return False
    core = re.sub(r"[^0-9A-Za-z]", "", s)
    if not (2 <= len(core) <= 20):
        return False
    # 真单号: 存在一段 >= 8 位的连续字母数字
    longest = max((len(r) for r in _ALNUM_RUN.findall(s)), default=0)
    if longest >= 8:
        return False
    # 没有长连续段, 又带分隔符 -> 分拣码
    return bool(_SORTING_SEP.search(s))


def _s_ai(v: Any) -> str:
    """转成去空白的字符串 (None -> "")."""
    return "" if v is None else str(v).strip()


def normalize_record(data: Any) -> Dict[str, Any]:
    """把任意字典规范成**严格 15 个字段**的记录.

    - 多余的键丢弃
    - 缺失的键补 ``None``
    - 空字符串统一成 ``None`` (需求要求"无法辨认时值为 null")
    - 非字符串的值转成字符串 (模型有时把金额输出成数字)
    - **单号位被填成分拣码时把它挪回分拣码** (见 :data:`_SORTING_CODE_HINT`)
    """
    out: Dict[str, Any] = {}
    src = data if isinstance(data, dict) else {}
    for f in FIELDS:
        v = src.get(f, None)
        if v is None:
            out[f] = None
            continue
        if isinstance(v, str):
            s = v.strip()
            out[f] = None if s == "" or s.lower() in ("null", "none", "n/a", "-") else s
        elif isinstance(v, (int, float)):
            out[f] = str(v)
        elif isinstance(v, (list, tuple)):
            # 模型偶尔把"数量"类字段输出成数组, 拼成可读文本
            parts = [str(x).strip() for x in v if str(x).strip()]
            out[f] = " ".join(parts) if parts else None
        else:
            out[f] = str(v)

    # ---- 修正: 把误填进单号的分拣码挪到分拣码字段 ----
    tn = out.get("tracking_number")
    if tn and _looks_like_sorting_code(tn):
        if not out.get("sorting_code"):
            out["sorting_code"] = tn
        out["tracking_number"] = None
    return out


def _error_record(message: str) -> Dict[str, Any]:
    """失败时的返回值: 15 个字段全 None + ``error``.

    需求既要求"字段固定不可增删", 又要求"失败时返回含 error 字段的字典",
    两者在失败这一种情况下冲突。这里按后者处理, 并把 15 个字段同样补全,
    这样调用方**两种情况下都能安全地按字段名取值**。

    注意: ``error`` 会成为 ``matcher`` 眼里的"非空输入"。因此
    ``matcher`` 侧已改为**只看它认识的面单字段**来判断输入是否为空,
    不会拿错误信息去瞎匹配。
    """
    rec: Dict[str, Any] = {f: None for f in FIELDS}
    rec["error"] = message
    return rec


# --------------------------------------------------------------------------
# 服务可用性
# --------------------------------------------------------------------------
def check_service(config: Optional[VisionConfig] = None) -> Tuple[bool, str]:
    """检查 Ollama 服务与模型是否可用.

    返回 ``(可用, 说明)``。不抛异常 —— 界面拿它做启动自检。
    """
    cfg = config or DEFAULT_CONFIG
    try:
        import ollama
    except Exception as exc:                     # noqa: BLE001
        return False, (f"未安装 ollama Python 库 ({type(exc).__name__})。"
                       f"请执行: pip install ollama")

    try:
        client = ollama.Client(host=cfg.host, timeout=15.0)
        listed = client.list()
    except Exception as exc:                     # noqa: BLE001
        return False, (f"连不上 Ollama 服务 ({cfg.host}): "
                       f"{type(exc).__name__}: {exc}。"
                       f"请确认 Ollama 已启动 (命令行执行 ollama serve)")

    names: List[str] = []
    try:
        models = listed.get("models", []) if isinstance(listed, dict) else listed.models
        for m in models or []:
            if isinstance(m, dict):
                names.append(str(m.get("model") or m.get("name") or ""))
            else:
                names.append(str(getattr(m, "model", "") or getattr(m, "name", "")))
    except Exception:                            # noqa: BLE001
        pass

    # 名称可能带 ":latest" 后缀 —— 归一后**精确**比对, 不做前缀匹配。
    # 早期写法是 ``n.startswith(cfg.model.split(":")[0])``, 后果是
    # qwen3-vl:2b-instruct 会让 qwen3-vl:4b-instruct-q4_k_m 也判成"已就绪"
    # (同族不同规格), 模型没装却报 OK。v2.1.0 修正。
    def _key(x: str) -> str:
        x = str(x or "").strip().lower()
        if x and ":" not in x:
            x += ":latest"
        return x

    target = _key(cfg.model)
    if any(_key(n) == target for n in names):
        return True, f"Ollama 可用, 模型 {cfg.model} 已就绪"
    return False, (f"服务可用, 但没有找到模型 {cfg.model}。"
                   f"当前已安装: {', '.join(n for n in names if n) or '(无)'}。"
                   f"请先执行: ollama pull {cfg.model}")


# --------------------------------------------------------------------------
# 图片准备
# --------------------------------------------------------------------------
def _prepare_image(image_path: Any, cfg: VisionConfig) -> Tuple[Optional[Path], str]:
    """校验图片并按需降采样; 返回 ``(可用路径, 错误说明)``.

    降采样用 OpenCV —— 它在本项目里本来就是必备依赖。如果 OpenCV 不可用
    (本模块要能单独跑), 就直接用原图, 不因此失败。
    """
    p = Path(image_path)
    if not p.is_file():
        return None, f"图片不存在: {p}"
    if p.stat().st_size == 0:
        return None, f"图片是空文件: {p}"

    if cfg.max_image_side <= 0:
        return p, ""

    try:
        import cv2
    except Exception:                            # noqa: BLE001
        return p, ""                             # 没有 OpenCV 也能用原图

    try:
        import numpy as np
        data = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return p, ""                         # 交给模型自己判断, 不阻断
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest <= cfg.max_image_side:
            return p, ""
        scale = cfg.max_image_side / float(longest)
        small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
        # 用 OpenCV 编码 + Python 写文件, 避开 cv2.imwrite 不支持中文路径的问题
        ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return p, ""
        # ⚠️ 缩放后的临时图**必须写到系统临时目录, 不能写到用户图片所在目录**。
        # 之前写成 ``Path(p).parent / "_ai_resized"``, 结果在用户的照片文件夹里
        # 堆了一坨副本 —— 往用户的数据目录里写东西是不能接受的。
        out_dir = cache_dir() / "resized"
        out_dir.mkdir(parents=True, exist_ok=True)
        # 用「原文件绝对路径 + 尺寸」的哈希命名, 避免不同目录下的同名文件互相覆盖
        try:
            key = hashlib.md5(
                f"{Path(p).resolve()}|{cfg.max_image_side}|{p.stat().st_mtime_ns}"
                .encode("utf-8")).hexdigest()[:16]
        except OSError:
            key = hashlib.md5(
                f"{Path(p)}|{cfg.max_image_side}".encode("utf-8")).hexdigest()[:16]
        out = out_dir / f"{p.stem}_{cfg.max_image_side}_{key}.jpg"
        out.write_bytes(buf.tobytes())
        return out, ""
    except Exception:                            # noqa: BLE001
        return p, ""                             # 任何环节出问题都退回原图


def _as_image_arg(path: Path) -> Any:
    """把图片读成 bytes 交给 ollama.

    直接传 ``bytes`` 而不是路径字符串: ollama 内部对 ``str`` 会走
    ``Path(value).read_bytes()``, 对 ``bytes`` 则直接 base64。
    自己读一次可以精确控制错误信息 (文件被占用/权限不足等),
    也避免路径编码在中间环节出问题。
    """
    return path.read_bytes()


# --------------------------------------------------------------------------
# 核心: 单张识别
# --------------------------------------------------------------------------
def _build_prompt(cfg: VisionConfig) -> str:
    """构造实际发给模型的提示词.

    原始提示词一字不改; 需要关思考时只在**最前面**加一个 ``/no_think``
    控制标记 (Qwen3 系列的写法, 需求第 3 条允许这么做)。
    """
    if cfg.no_think and not PROMPT.lstrip().startswith("/no_think"):
        return "/no_think\n" + PROMPT
    return PROMPT


def _call_once(client: Any, cfg: VisionConfig, image_arg: Any,
               fmt: Any, *, num_ctx: Optional[int] = None
               ) -> Tuple[Any, str, int, str]:
    """发起一次调用并解析; 返回 ``(解析结果, 错误说明, thinking 字数, 结束原因)``.

    ``thinking`` 字数会一并返回 —— 排查"content 为空"时它就是最有用的线索
    (空 content + 上千字 thinking = 上下文被思考吃光了)。
    ``finish`` 是 Ollama 的 ``done_reason``: ``stop`` 正常结束, ``length``
    表示撞上 ``num_predict`` 上限。后者说明模型在打转, 同图重试不会变好。
    """
    opts: Dict[str, Any] = {"temperature": cfg.temperature}
    if num_ctx:
        opts["num_ctx"] = int(num_ctx)
    if int(cfg.max_output_tokens or 0) > 0:
        # 不设上限时模型偶尔会继续"自言自语", 生成时间不可控
        opts["num_predict"] = int(cfg.max_output_tokens)
    opts.update(cfg.extra_options)

    resp = client.chat(
        model=cfg.model,
        messages=[{"role": "user", "content": _build_prompt(cfg),
                   "images": [image_arg]}],
        format=fmt,
        think=False,                    # 需求要求关思考; 见 VisionConfig.no_think 说明
        keep_alive=cfg.keep_alive,      # 连续识别时模型常驻, 避免每次重载
        options=opts,
    )
    finish = str((resp.get("done_reason") if isinstance(resp, dict)
                  else getattr(resp, "done_reason", "")) or "")
    msg = resp.get("message", {}) if isinstance(resp, dict) else resp.message
    content = msg.get("content") if isinstance(msg, dict) else msg.content
    thinking = (msg.get("thinking") if isinstance(msg, dict)
                else getattr(msg, "thinking", "")) or ""
    data, err = parse_ai_json(content)
    if not err and data is None:
        err = "模型没有返回任何内容"
    return data, err, len(str(thinking)), finish


def _shrink_image(image_path: Any, max_side: int) -> Optional[bytes]:
    """把图片缩到指定长边并返回 JPEG 字节; 失败返回 None."""
    try:
        import cv2
        import numpy as np
    except Exception:                            # noqa: BLE001
        return None
    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest <= max_side:
            return None                          # 本来就不大, 不用缩
        scale = max_side / float(longest)
        small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return buf.tobytes() if ok else None
    except Exception:                            # noqa: BLE001
        return None


def extract_parcel_info(image_path: Any,
                        config: Optional[VisionConfig] = None,
                        log: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """对**单张**面单图片做 AI 识别, 返回 15 个固定字段的字典.

    参数
    ----
    image_path
        本地图片路径。
    config
        识别参数; 省略用 :data:`DEFAULT_CONFIG`。
    log
        日志回调 (界面用它把过程显示到控制台)。

    返回
    ----
    成功: ``{15 个字段}``, 值可能是字符串或 ``None``。
    失败: ``{15 个字段全 None, "error": "原因"}`` —— **不抛异常**。
    """
    cfg = config or DEFAULT_CONFIG
    t0 = time.perf_counter()

    def _fail(msg: str) -> Dict[str, Any]:
        _log(log, f"  [AI] 失败: {msg}  (耗时 {(time.perf_counter()-t0)*1000:.0f} ms)")
        return _error_record(msg)

    try:
        import ollama
    except Exception as exc:                     # noqa: BLE001
        return _fail(f"未安装 ollama Python 库 ({type(exc).__name__}); "
                     f"请执行 pip install ollama")

    path, err = _prepare_image(image_path, cfg)
    if err:
        return _fail(err)

    try:
        image_arg = _as_image_arg(path)          # type: ignore[arg-type]
    except Exception as exc:                     # noqa: BLE001
        return _fail(f"读取图片失败: {type(exc).__name__}: {exc}")

    try:
        client = ollama.Client(host=cfg.host, timeout=cfg.timeout)
    except Exception as exc:                     # noqa: BLE001
        return _fail(f"创建 Ollama 客户端失败: {type(exc).__name__}: {exc}")

    # ---- 重试阶梯 ----
    # 关键: 真实面单上最常见的失败是"content 为空" —— 思考把上下文吃光了。
    # 因此重试**必须调整真正有用的两个东西**: 上下文长度 与 图片大小。
    # (最初只换 format, 实测三种 format 表现完全一样, 白试。)
    fmt_primary: Any = SCHEMA if cfg.use_schema else "json"
    # 第二档最多翻到 16384: 再高就会把 8 GB 显存撑爆, 退化成 CPU 混跑,
    # 越"重试"越慢 (实测 65536 时 9.8 GB, 41% CPU)。
    ctx_retry = min(int(cfg.num_ctx) * 2, 16384)
    ladder: List[Dict[str, Any]] = [
        {"label": f"schema/ctx{cfg.num_ctx}", "fmt": fmt_primary,
         "num_ctx": cfg.num_ctx, "img": image_arg},
        {"label": f"json/ctx{ctx_retry}", "fmt": "json",
         "num_ctx": ctx_retry, "img": image_arg},
    ]
    # 第三档: 把图缩到 1200 再试 —— 图小则图像 token 少, 给思考留出更多预算。
    shrunk = _shrink_image(path, 1200)
    if shrunk is not None:
        ladder.append({"label": "json/ctx16384/图缩到1200", "fmt": "json",
                       "num_ctx": max(cfg.num_ctx, 16384), "img": shrunk})

    # retries 控制走几档 (0 = 只试主配置)
    ladder = ladder[:max(1, int(cfg.retries) + 1)]

    last_err = ""
    for step in ladder:
        # 总预算: 正常 2~4 秒; 病态图不再一档一档耗下去
        elapsed = time.perf_counter() - t0
        if (cfg.total_budget_s and elapsed >= float(cfg.total_budget_s)):
            last_err = last_err or f"总预算 {cfg.total_budget_s:.0f} 秒已用完"
            _log(log, f"  [AI] 跳过 {step['label']}: 已耗时 {elapsed:.1f}s, "
                      f"超过总预算 {cfg.total_budget_s:.0f}s")
            break
        # 每次尝试单独限时: 避免一张特别难的图把整批拖到几分钟
        try:
            client = ollama.Client(host=cfg.host,
                                   timeout=float(cfg.attempt_timeout))
        except Exception as exc:                 # noqa: BLE001
            return _fail(f"创建 Ollama 客户端失败: {type(exc).__name__}: {exc}")

        try:
            data, perr, think_len, finish = _call_once(
                client, cfg, step["img"], step["fmt"], num_ctx=step["num_ctx"])
        except Exception as exc:                 # noqa: BLE001
            last_err = f"{step['label']} 调用失败: {type(exc).__name__}: {exc}"
            _log(log, f"  [AI] {last_err}")
            continue

        if perr:
            last_err = f"{step['label']}: {perr}"
            hint = (f" (思考 {think_len} 字, 疑似上下文被吃光)"
                    if think_len > 1500 else "")
            _log(log, f"  [AI] {step['label']} 未成功: {perr}{hint}")
            if finish == "length":
                # 撞上 num_predict 上限仍不是完整 JSON: 模型在打转,
                # 同图重试只会再烧一遍 token, 直接停。
                _log(log, f"  [AI] 输出达到 {cfg.max_output_tokens} token 上限"
                          f"仍不完整, 停止重试")
                break
            continue

        # 模型可能返回数组 (提示词里允许多图); 单图取第一个
        if isinstance(data, list):
            data = data[0] if data else {}
        rec = normalize_record(data)
        dt = (time.perf_counter() - t0) * 1000
        filled = sum(1 for v in rec.values() if v)
        _log(log, f"  [AI] 成功: {filled}/15 字段有值, 单号="
                  f"{rec.get('tracking_number') or '(空)'}  耗时 {dt:.0f} ms  "
                  f"[{step['label']}, 思考 {think_len} 字]")
        return rec

    return _fail(last_err or "未知失败")


# --------------------------------------------------------------------------
# 批量识别
# --------------------------------------------------------------------------
def extract_parcel_info_batch(image_paths: Iterable[Any],
                              config: Optional[VisionConfig] = None,
                              log: Optional[Callable[[str], None]] = None,
                              progress: Optional[Callable[[int, int, str], None]] = None
                              ) -> List[Dict[str, Any]]:
    """逐张调用 :func:`extract_parcel_info`.

    单张失败只影响它自己, 不会中断整批。

    ``progress`` 回调签名: ``(已完成数, 总数, 当前文件名)``。
    """
    paths = list(image_paths or [])
    out: List[Dict[str, Any]] = []
    total = len(paths)
    _log(log, f"[AI] 批量识别 {total} 张图片")
    for i, p in enumerate(paths, 1):
        name = Path(p).name
        _log(log, f"[AI] ({i}/{total}) {name}")
        if progress:
            try:
                progress(i - 1, total, name)
            except Exception:                    # noqa: BLE001
                pass
        try:
            out.append(extract_parcel_info(p, config=config, log=log))
        except Exception as exc:                 # noqa: BLE001 - 兜底
            out.append(_error_record(f"未预期异常: {type(exc).__name__}: {exc}"))
    if progress:
        try:
            progress(total, total, "")
        except Exception:                        # noqa: BLE001
            pass
    return out


# --------------------------------------------------------------------------
# 识别 + 比对
# --------------------------------------------------------------------------
def _get_matcher():
    """取得 matcher 模块 (兼顾包内导入与单独拷贝使用)."""
    try:
        from . import matcher as mod
        return mod
    except Exception:                            # noqa: BLE001
        pass
    try:
        import matcher as mod
        return mod
    except Exception:                            # noqa: BLE001
        return None


#: 寄件方字段与收件方字段的一一对应关系 (用于整体对调)
_SENDER_RECIPIENT_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("sender_name", "recipient_name"),
    ("sender_phone", "recipient_phone"),
    ("sender_address", "recipient_address"),
)


def swap_sender_recipient(fields: Dict[str, Any]) -> Dict[str, Any]:
    """把收件方与寄件方的字段整体对调, 返回新字典 (不改原字典)."""
    out = dict(fields)
    for a, b in _SENDER_RECIPIENT_PAIRS:
        out[a], out[b] = fields.get(b), fields.get(a)
    return out


#: 判方向时各类字段的权重。
#:
#: 姓名和电话是"人"的标识, 主导判断; 地址降权, 因为收件人和寄件人
#: 常常在同一个城市 (比如收件人在佛山, 发货仓也在佛山), 地址相似度
#: 两边都高, 区分度差。
_ORIENT_WEIGHTS: Tuple[Tuple[str, float], ...] = (
    ("name", 1.0),
    ("phone", 1.0),
    ("address", 0.5),
)


def _fetch_db_row(tracking: Any,
                  db_path: Optional[object] = None) -> Optional[Dict[str, Any]]:
    """按单号取数据库里的**整行**.

    为什么要单独取: ``matcher`` 返回的候选只带展示用的几个字段
    (单号/收件人/电话/地址/快递公司), **没有 sender_*** —— 而判方向
    恰恰需要库里的寄件方信息。
    """
    tn = _s_ai(tracking)
    if not tn:
        return None
    try:
        from . import database as _db
    except Exception:                            # noqa: BLE001
        try:
            import database as _db               # type: ignore[no-redef]
        except Exception:                        # noqa: BLE001
            return None
    try:
        row = _db.find_by_tracking(tn, db_path)
    except Exception:                            # noqa: BLE001
        return None
    if not row:
        return None
    # ⚠️ find_by_tracking 返回的是**列表** (同一单号可能有多条), 取最新一条。
    if isinstance(row, (list, tuple)):
        row = row[0] if row else None
        if row is None:
            return None
    if isinstance(row, dict):
        return dict(row)
    # sqlite3.Row 是**序列**不是映射, 直接 dict(row) 会抛
    # "dictionary update sequence element #0 has length 20; 2 is required"。
    try:
        return {k: row[k] for k in row.keys()}
    except Exception:                            # noqa: BLE001
        return None


def _pair_similarity(ai_fields: Dict[str, Any], db_row: Dict[str, Any],
                     ai_prefix: str, db_prefix: str
                     ) -> Tuple[Optional[float], float]:
    """AI 的 ``ai_prefix`` 方 与 库里 ``db_prefix`` 方 的加权相似度.

    返回 ``(相似度 | None, 参与比较的权重和)``。
    只统计**两边都有值**的字段 —— 一边为空说明没有可比信息, 不能算作"不像"。
    """
    m = _get_matcher()
    if m is None:
        return None, 0.0
    fns = {"name": m.sim_name, "phone": m.sim_phone, "address": m.sim_address}
    acc = 0.0
    tw = 0.0
    for suffix, w in _ORIENT_WEIGHTS:
        av = ai_fields.get(f"{ai_prefix}_{suffix}")
        dv = db_row.get(f"{db_prefix}_{suffix}")
        if not _s_ai(av) or not _s_ai(dv):
            continue
        try:
            s, _reason = fns[suffix](av, dv)
        except Exception:                        # noqa: BLE001
            continue
        if s is None:
            continue
        acc += float(s) * w
        tw += w
    return ((acc / tw) if tw > 0 else None), tw


def _orientation_score(ai_fields: Dict[str, Any], db_row: Dict[str, Any],
                       swapped: bool) -> Tuple[Optional[float], float]:
    """给定"是否对调", 算 AI 与库中记录的**一致度**.

    两侧都算: 收件方对收件方 (或对寄件方), 寄件方对寄件方 (或对收件方)。
    两边能互相印证时可信度更高。
    """
    # ⚠️ 这里曾把两对前缀配错 (recipient 配到了 sender), 结果判据整个反过来:
    #    搞反的说"不用动"、本来就对的反说"要换"。用显式的成对写法避免再错。
    if swapped:
        # 对调后: AI 的收件方 应当对应 库的寄件方, 反之亦然
        pairs = (("recipient", "sender"), ("sender", "recipient"))
    else:
        # 原样: 各对各
        pairs = (("recipient", "recipient"), ("sender", "sender"))

    parts = [_pair_similarity(ai_fields, db_row, ai_pre, db_pre)
             for ai_pre, db_pre in pairs]
    acc = 0.0
    tw = 0.0
    for score, w in parts:
        if score is None:
            continue
        acc += score * w
        tw += w
    return ((acc / tw) if tw > 0 else None), tw


def resolve_sender_recipient(
    fields: Dict[str, Any],
    db_path: Optional[object] = None,
    *,
    top_n: int = 3,
    min_gain: float = 0.15,
    min_confidence: float = 0.55,
    candidates: Optional[Iterable[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """用**数据库当裁判**, 判断收件方/寄件方是不是被模型搞反了.

    用户实测反馈的问题: AI 会把发货方和收件方的联系人信息搞反位置
    (收件人姓名写成寄件人、反之亦然)。

    ⚠️ 这里换过一次判据, 记下来免得再走回头路
    -------------------------------------------
    最初比的是 ``matcher`` 的**总分** —— 实测**根本行不通**:

        单号精确命中时, matcher 走"精确匹配"分支, 总分只有
        1.0 (无冲突) 或 0.60 (有冲突) 两档, **完全不看字段加权分**。
        于是对调前后都是 0.60, 增益恒为 0, 方向判不出来。

    改成**直接比字段相似度**才对路: 把"方向"当成独立问题 ——
    拿 AI 读到的收件方 (姓名/电话/地址) 分别去和库里的**收件方**、
    库里的**寄件方**比, 哪边像就是哪边。

        实测同一批真实数据:
            旧规则(比总分) 判出 1/9
            新规则(比字段) 判出 2/9
        其中漏掉的那例信号非常干净: AI 的收件方和库里收件方相似度 0.03,
        和库里寄件方 1.00 —— 教科书级的串位, 旧规则完全没发现。

    判定规则 (两道, 缺一不可)
    ------------------------
    1. **对调方向必须明显更像**：``swapped - as_is >= min_gain``
    2. **像的那个方向本身要够像**：``swapped >= min_confidence``

    只满足第 1 条不够 —— 可能两个方向都很差 (AI 读的就是垃圾), 这时
    乱换只是把错误换个方向。**宁可不动。**

    返回 ``(采用的字段, 说明)``。
    """
    info: Dict[str, Any] = {"swapped": False, "gain": 0.0,
                            "score_as_is": None, "score_swapped": None,
                            "evidence": 0.0, "reason": "",
                            "db_tracking": None}
    m = _get_matcher()
    if m is None:
        info["reason"] = "找不到 matcher 模块, 不做收/寄纠正"
        return fields, info

    has_any = any(_s_ai(fields.get(k))
                  for k in ("recipient_name", "recipient_phone",
                            "recipient_address", "sender_name",
                            "sender_phone", "sender_address"))
    if not has_any:
        info["reason"] = "收/寄字段都为空, 无从判断"
        return fields, info

    # ---- 1. 找到库里的对照记录 ----
    row: Optional[Dict[str, Any]] = None
    try:
        r = m.compare_fields(fields, db_path, top_n=max(1, top_n),
                             candidates=candidates)
        cands = r.get("candidates") or []
        if cands:
            info["db_tracking"] = cands[0].get("tracking_number")
            row = _fetch_db_row(cands[0].get("tracking_number"), db_path)
    except Exception as exc:                     # noqa: BLE001
        info["reason"] = f"比对失败, 不做收/寄纠正: {type(exc).__name__}"
        return fields, info

    if row is None:
        info["reason"] = "库里没有可对照的记录, 无裁判依据"
        return fields, info

    # 库里这条必须**有寄件方信息**, 否则无从判断方向
    if not any(_s_ai(row.get(f"sender_{s}"))
               for s in ("name", "phone", "address")):
        info["reason"] = "库里该记录没有寄件方信息, 无裁判依据"
        return fields, info

    # ---- 2. 两个方向各打一个分 ----
    a, wa = _orientation_score(fields, row, swapped=False)
    b, wb = _orientation_score(fields, row, swapped=True)
    info["score_as_is"] = None if a is None else round(a, 4)
    info["score_swapped"] = None if b is None else round(b, 4)
    info["evidence"] = round(max(wa, wb), 2)

    if a is None and b is None:
        info["reason"] = "两边没有可比对的字段, 无从判断"
        return fields, info

    # as_is 无从计算时 (AI 没读出收件方), 只要对调方向够像就换
    if a is None:
        if b is not None and b >= min_confidence:
            info.update(swapped=True, gain=round(b, 4))
            info["reason"] = f"原方向无法判断; 对调后与库一致 {b:.2f}, 已对调"
            return swap_sender_recipient(fields), info
        info["reason"] = "证据不足, 不做收/寄纠正"
        return fields, info

    if b is None:
        info["reason"] = f"对调方向无从判断 (原样 {a:.2f}), 保持原样"
        return fields, info

    gain = b - a
    info["gain"] = round(gain, 4)

    if gain >= min_gain and b >= min_confidence:
        info["swapped"] = True
        info["reason"] = (f"收/寄疑似搞反: 收件方与库中【寄件方】更像 "
                          f"({b:.2f}) 而与本方不像 ({a:.2f}), 已自动对调")
        return swap_sender_recipient(fields), info

    info["reason"] = (f"收/寄位置正常 (本方 {a:.2f} / 对方 {b:.2f})")
    return fields, info
def extract_and_match(image_path: Any,
                      db_path: Optional[object] = None,
                      config: Optional[VisionConfig] = None,
                      top_n: int = 3,
                      log: Optional[Callable[[str], None]] = None,
                      ) -> Dict[str, Any]:
    """识别一张面单并直接比对出候选包裹.

    返回::

        {
          "image": "xxx.jpg",
          "ai":    {15 个字段},              # 可能含 error
          "match": {...matcher 的返回...},   # AI 失败时为 None
          "elapsed_ms": 12345.6,
          "ok": true/false,
          "message": "……"
        }
    """
    t0 = time.perf_counter()
    name = Path(image_path).name if image_path else ""
    ai = extract_parcel_info(image_path, config=config, log=log)

    result: Dict[str, Any] = {
        "image": str(image_path or ""), "image_name": name,
        "ai": ai, "match": None, "ok": False,
        "elapsed_ms": 0.0, "message": "",
    }

    if ai.get("error"):
        result["message"] = f"AI 识别失败: {ai['error']}"
        result["elapsed_ms"] = (time.perf_counter() - t0) * 1000
        _log(log, f"[AI→比对] {name} 跳过比对 ({result['message']})")
        return result

    m = _get_matcher()
    if m is None:
        result["message"] = "找不到 matcher 模块, 无法比对"
        result["elapsed_ms"] = (time.perf_counter() - t0) * 1000
        return result

    # ---- 用数据库当裁判, 纠正收件方/寄件方搞反 ----
    # 放在比对之前: 先把位置理顺, 再拿顺好的字段正式比对。
    cfg = config or DEFAULT_CONFIG
    fixed, swap_info = resolve_sender_recipient(ai, db_path, top_n=top_n)
    result["sender_recipient_fix"] = swap_info
    if swap_info.get("swapped"):
        result["ai_before_swap"] = dict(ai)
        result["ai"] = fixed
        ai = fixed
        _log(log, f"[AI→比对] {name} {swap_info['reason']}")

    try:
        match = m.match_from_ai(
            ai, db_path, top_n=top_n,
            config=m.MatcherConfig(release_mode=cfg.release_mode))
    except Exception as exc:                     # noqa: BLE001
        result["message"] = f"比对失败: {type(exc).__name__}: {exc}"
        result["elapsed_ms"] = (time.perf_counter() - t0) * 1000
        _log(log, f"[AI→比对] {name} {result['message']}")
        return result

    best = (match.get("candidates") or [{}])[0]
    result["match"] = match
    result["ok"] = True
    result["elapsed_ms"] = (time.perf_counter() - t0) * 1000
    result["message"] = match.get("message", "")

    # ---- 方案二: 匹配到库中记录时, 收发信息以库为准 ----
    # 单号一旦命中, 库里那条就是事实, 没必要再纠结 AI 读成什么方向。
    # 默认**关闭**: 打开后会遮蔽 AI 的读数, 想核对 AI 时不该开。
    cfg = config or DEFAULT_CONFIG
    if getattr(cfg, "prefer_db_contacts", False):
        row = _fetch_db_row(best.get("tracking_number"), db_path)
        if row:
            filled = dict(ai)
            changed = []
            for suffix in ("name", "phone", "address"):
                for side in ("recipient", "sender"):
                    key = f"{side}_{suffix}"
                    dbv = _s_ai(row.get(key))
                    if dbv and dbv != _s_ai(filled.get(key)):
                        if _s_ai(filled.get(key)):
                            changed.append(key)
                        filled[key] = row.get(key)
            if changed:
                # 保留 AI 原始读数, 便于核对
                result.setdefault("ai_from_model", dict(ai))
            result["ai"] = filled
            ai = filled
            result["db_contacts_applied"] = True
            _log(log, f"[AI→比对] {name} 收发信息以库中记录为准 "
                      f"(覆盖了 {len(changed)} 个字段: {', '.join(changed) or '仅补空'})")

    _log(log, f"[AI→比对] {name} match_type={match.get('match_type')} "
              f"最高分={best.get('total_score')} "
              f"候选={(best.get('tracking_number') or '无')} "
              f"放行={'是' if not match.get('needs_review') else '否'} "
              f"耗时={result['elapsed_ms']:.0f} ms")
    return result


def extract_and_match_batch(image_paths: Iterable[Any],
                            db_path: Optional[object] = None,
                            config: Optional[VisionConfig] = None,
                            top_n: int = 3,
                            log: Optional[Callable[[str], None]] = None,
                            progress: Optional[Callable[[int, int, str], None]] = None
                            ) -> List[Dict[str, Any]]:
    """逐张识别并比对.

    ``progress`` 回调签名: ``(已完成数, 总数, 当前文件名)``。
    单张出错不影响其它。
    """
    paths = list(image_paths or [])
    out: List[Dict[str, Any]] = []
    total = len(paths)
    _log(log, f"[AI] 批量识别+比对 {total} 张")
    for i, p in enumerate(paths, 1):
        name = Path(p).name
        if progress:
            try:
                progress(i - 1, total, name)
            except Exception:                    # noqa: BLE001
                pass
        try:
            out.append(extract_and_match(p, db_path, config=config,
                                         top_n=top_n, log=log))
        except Exception as exc:                 # noqa: BLE001 - 兜底
            out.append({"image": str(p), "image_name": name,
                        "ai": _error_record(f"未预期异常: {type(exc).__name__}: {exc}"),
                        "match": None, "ok": False, "elapsed_ms": 0.0,
                        "message": "未预期异常"})
    if progress:
        try:
            progress(total, total, "")
        except Exception:                        # noqa: BLE001
            pass
    return out


# --------------------------------------------------------------------------
# 便捷入口: 直接把 AI 结果写进数据库
# --------------------------------------------------------------------------
def save_ai_result(result: Dict[str, Any],
                   db_path: Optional[object] = None,
                   on_duplicate: str = "skip") -> Dict[str, Any]:
    """把一条 :func:`extract_and_match` 的结果写入数据库.

    只有 AI 识别成功、且**匹配得分达到自动放行线**时才写 (调用方负责判断,
    本函数只负责写)。``image_path`` 用 AI 那张原图, 便于事后核对。

    返回 ``{"status": ..., "message": ...}``; 失败不抛异常。
    """
    ai = (result or {}).get("ai") or {}
    if ai.get("error"):
        return {"status": "skipped", "message": "AI 识别失败, 不写入"}

    try:
        from . import database as _db
    except Exception:                            # noqa: BLE001
        try:
            import database as _db               # type: ignore[no-redef]
        except Exception as exc:                 # noqa: BLE001
            return {"status": "failed", "message": f"找不到 database 模块: {exc}"}

    rec = {k: v for k, v in ai.items() if k in _db.PARCEL_FIELDS}
    rec["image_path"] = result.get("image") or ""
    try:
        out = _db.save_to_database(rec, path=db_path, on_duplicate=on_duplicate)
    except Exception as exc:                     # noqa: BLE001
        return {"status": "failed", "message": f"写库失败: {type(exc).__name__}: {exc}"}
    return {"status": out.status, "message": out.message, "id": out.id}


# --------------------------------------------------------------------------
# __main__: 直接跑一张图, 打印识别结果与比对候选
# --------------------------------------------------------------------------
def _main(argv: Sequence[str]) -> int:           # pragma: no cover
    import argparse
    import json as _json
    import sys

    ap = argparse.ArgumentParser(
        description=f"AI 视觉识别 ({MODEL_NAME}) —— 识别面单并比对数据库")
    ap.add_argument("images", nargs="*", help="面单图片路径 (可多张)")
    ap.add_argument("--db", default=None, help="parcels.db 路径")
    ap.add_argument("--top", type=int, default=3, help="返回候选数 (默认 3)")
    ap.add_argument("--model", default=MODEL_NAME, help=f"模型名 (默认 {MODEL_NAME})")
    ap.add_argument("--host", default=DEFAULT_HOST, help="Ollama 地址")
    ap.add_argument("--no-match", action="store_true", help="只识别, 不比对")
    ap.add_argument("--mode", default="strict",
                    choices=("lenient", "strict"),
                    help="放行模式: lenient 宽松 / strict 严格(默认)")
    ap.add_argument("--max-side", type=int, default=1600,
                    help="图片长边上限, 0 表示不缩放")
    ap.add_argument("--check", action="store_true", help="只检查服务与模型是否可用")
    args = ap.parse_args(list(argv))

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = VisionConfig(model=args.model, host=args.host,
                       max_image_side=args.max_side,
                       release_mode=args.mode)

    ok, msg = check_service(cfg)
    print(f"[服务检查] {'OK' if ok else '不可用'}: {msg}")
    if args.check or not ok:
        return 0 if ok else 2
    if not args.images:
        ap.print_help()
        print("\n提示: 也可以只跑 --check 检查环境。")
        return 2

    def _p(prefix: str):
        def _fn(s: str) -> None:
            print(f"{prefix}{s}", flush=True)
        return _fn

    if args.no_match:
        results = extract_parcel_info_batch(args.images, config=cfg, log=_p(""))
        for path, rec in zip(args.images, results):
            print(f"\n=== {Path(path).name} ===")
            print(_json.dumps(rec, ensure_ascii=False, indent=2))
        return 0

    results = extract_and_match_batch(args.images, args.db, config=cfg,
                                      top_n=args.top, log=_p(""))
    print("\n" + "=" * 78)
    for r in results:
        print(f"\n=== {r['image_name']} ===")
        print("AI 提取字段:")
        print(_json.dumps(r["ai"], ensure_ascii=False, indent=2))
        m = r.get("match")
        if not m:
            print(f"未比对: {r['message']}")
            continue
        print(f"\n比对结果: match_type={m.get('match_type')} "
              f"置信度={m.get('confidence')} "
              f"自动放行={'是' if not m.get('needs_review') else '否'}")
        for c in m.get("candidates", []):
            mark = " <== 最高分" if c["rank"] == 1 else ""
            print(f"  #{c['rank']} {c['total_score']:.4f}  "
                  f"{c['tracking_number'] or '(无单号)':<20s} "
                  f"{c['recipient_name'] or '-'} {c['recipient_phone'] or '-'}{mark}")
            print(f"       {c['match_reason']}")
        print(f"  {m.get('message')}")
        print(f"  本张总耗时: {r['elapsed_ms']:.0f} ms")
    return 0


if __name__ == "__main__":                       # pragma: no cover
    import sys as _sys

    _src = Path(__file__).resolve().parents[1]
    if str(_src) not in _sys.path:
        _sys.path.insert(0, str(_src))
    _core = str(Path(__file__).resolve().parent)
    if _core not in _sys.path:
        _sys.path.insert(0, _core)
    raise SystemExit(_main(_sys.argv[1:]))
