"""可选 VLM 模型清单与可用性检查 —— v2.1.0.

为什么单独一个模块
------------------
模型切换有两件事必须在**同一个地方**回答, 否则界面和命令行早晚各写一套:

1. **可选哪些模型** —— 清单只有一份, 界面下拉与 ``--models`` 都从这里取。
2. **这个模型现在能不能用** —— 先看 Ollama 服务在不在, 再看模型装没装。

为什么不用 ``ai_vision.check_service()``
----------------------------------------
那个函数把两件事混在一句话里 (服务 + 模型), 而且它的模型名匹配是
``name.startswith(model.split(':')[0])`` —— 对 ``qwen3-vl:2b-instruct``
与 ``qwen3-vl:4b-instruct-q4_k_m`` 这类同族不同规格的模型会**误判为已就绪**。
本模块按 ``family:tag`` 精确比对 (只忽略大小写与 ``:latest`` 后缀),
不复用那个缺陷。

约定
----
* 任何函数都**不抛异常**。服务没起、模型没装、ollama 库缺失, 一律
  返回结构化的失败结果, 由调用方决定怎么展示。
* ``check_model()`` 的结果可以直接 ``to_dict()`` 给命令行 ``--json`` 用,
  也可以用 :func:`format_report` 变成给人看的「异常报告」。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ModelSpec",
    "ModelStatus",
    "MODELS",
    "MODEL_NAMES",
    "DEFAULT_MODEL",
    "model_names",
    "resolve_model",
    "normalize_model_name",
    "is_model_installed",
    "list_installed",
    "check_model",
    "check_model_availability",
    "format_report",
]

#: Ollama 默认服务地址。与 ``ai_vision.DEFAULT_HOST`` 保持一致。
DEFAULT_HOST = "http://localhost:11434"


# ==========================================================================
# 模型清单
# ==========================================================================
@dataclass(frozen=True)
class ModelSpec:
    """一个可选模型.

    ``label`` 是界面上显示的中文说明; ``name`` 才是传给 Ollama 的模型名,
    必须与 ``ollama list`` 里的一字不差。
    """

    name: str
    label: str
    #: 大致体积/定位, 仅用于界面提示
    hint: str


#: 界面下拉、命令行 ``--models`` 共用这一份清单。
MODELS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        name="qwen3-vl:2b-instruct",
        label="2B Instruct（默认 · 速度优先）",
        hint="约 1.8 GB，单张约 5~25 秒；默认模型",
    ),
    ModelSpec(
        name="qwen3-vl:4b-instruct-q4_k_m",
        label="4B Instruct Q4（平衡）",
        hint="4B 量化版，显存/耗时高于 2B，精度通常更好",
    ),
    ModelSpec(
        name="qwen3-vl:8b-instruct-q4_k_m",
        label="8B Instruct Q4（精度优先）",
        hint="8B 量化版，最慢；适合疑难面单",
    ),
)

#: 默认模型 —— 需求指定为 2B Instruct。
#: 注意与 ``ai_vision.MODEL_NAME`` 必须一致 (有单元测试盯着)。
DEFAULT_MODEL = "qwen3-vl:2b-instruct"

#: 全部模型名, 顺序与 :data:`MODELS` 一致。
MODEL_NAMES: Tuple[str, ...] = tuple(m.name for m in MODELS)


def model_names() -> List[str]:
    """返回全部可选模型名 (副本, 调用方可以随便改)."""
    return list(MODEL_NAMES)


def resolve_model(name: Optional[str] = None) -> str:
    """把 ``None`` / 空串归一成默认模型; 其它值原样返回 (去掉首尾空白).

    这里**不做模糊匹配** —— 模型名是 Ollama 的键, 猜错了只会让人更困惑。
    """
    text = str(name or "").strip()
    return text or DEFAULT_MODEL


def normalize_model_name(name: Any) -> str:
    """模型名归一化, 用于**精确**比对.

    * 去首尾空白、转小写 (Ollama 模型名本身是小写)
    * 没有 ``:tag`` 的补 ``:latest`` —— ``ollama list`` 里 ``x`` 就是 ``x:latest``
    * 有 ``:tag`` 的原样保留, **不**做前缀匹配
    """
    text = str(name or "").strip().lower()
    if not text:
        return ""
    if ":" not in text:
        text += ":latest"
    return text


def _spec_for(name: str) -> Optional[ModelSpec]:
    key = normalize_model_name(name)
    for spec in MODELS:
        if normalize_model_name(spec.name) == key:
            return spec
    return None


# ==========================================================================
# 可用性检查
# ==========================================================================
@dataclass
class ModelStatus:
    """一次模型可用性检查的结果.

    ``ok`` 是一个整体结论: 服务可用 **且** 模型已安装。
    两个中间状态分别放在 ``service_ok`` / ``model_present`` 里,
    界面可以据此给出不同建议 (启动服务 vs 拉取模型)。
    """

    model: str = DEFAULT_MODEL
    service_ok: bool = False
    model_present: bool = False
    installed: List[str] = field(default_factory=list)
    #: 服务连接失败 / 库缺失时的原始错误文本
    error: str = ""
    #: 给用户看的一句话结论
    message: str = ""
    #: 检查时刻 (epoch 秒), 界面用来判断缓存是否过期
    checked_at: float = field(default_factory=lambda: time.time())

    @property
    def ok(self) -> bool:
        return bool(self.service_ok and self.model_present)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "ok": self.ok,
            "service_ok": self.service_ok,
            "model_present": self.model_present,
            "installed": list(self.installed),
            "error": self.error,
            "message": self.message,
            "checked_at": self.checked_at,
        }


def _parse_model_names(listed: Any) -> List[str]:
    """从 ``ollama.Client.list()`` 的返回里抽模型名 (兼容 dict / 对象)."""
    names: List[str] = []
    try:
        models = listed.get("models", []) if isinstance(listed, dict) \
            else getattr(listed, "models", [])
        for m in models or []:
            if isinstance(m, dict):
                names.append(str(m.get("model") or m.get("name") or ""))
            else:
                names.append(str(getattr(m, "model", "")
                                 or getattr(m, "name", "")))
    except Exception:                            # noqa: BLE001
        pass
    return [n for n in names if n]


def list_installed(host: Optional[str] = None,
                   timeout: float = 5.0) -> Tuple[bool, List[str], str]:
    """查 Ollama 服务与已安装模型.

    返回 ``(服务可用, 模型名列表, 错误说明)``。**不抛异常**:

    * 没装 ``ollama`` Python 库  -> ``(False, [], "未安装 ollama ...")``
    * 服务没起 / 连不上          -> ``(False, [], "连不上 Ollama ...")``
    * 服务正常                   -> ``(True, [名字...], "")``
    """
    try:
        import ollama
    except Exception as exc:                     # noqa: BLE001
        return False, [], (f"未安装 ollama Python 库 ({type(exc).__name__})。"
                           f"请执行: pip install ollama")

    try:
        client = ollama.Client(host=str(host or DEFAULT_HOST), timeout=timeout)
        listed = client.list()
    except Exception as exc:                     # noqa: BLE001
        return False, [], (f"连不上 Ollama 服务 ({host or DEFAULT_HOST}): "
                           f"{type(exc).__name__}: {exc}")
    return True, _parse_model_names(listed), ""


def is_model_installed(model: str, installed: Sequence[str]) -> bool:
    """模型是否已安装 —— 按 ``family:tag`` 精确比对, 不做前缀匹配."""
    target = normalize_model_name(model)
    if not target:
        return False
    return any(normalize_model_name(n) == target for n in installed)


def check_model(model: Optional[str] = None,
                host: Optional[str] = None,
                timeout: float = 5.0) -> ModelStatus:
    """检查指定模型是否可用 (服务在跑 + 模型已安装).

    这是给界面「切换后自检」和命令行 ``--ai-check --model`` 用的统一入口。
    无论发生什么, 都返回一个 :class:`ModelStatus`, **不抛异常**。
    """
    name = resolve_model(model)
    service_ok, installed, err = list_installed(host=host, timeout=timeout)
    st = ModelStatus(model=name, service_ok=service_ok, installed=installed,
                     error=err)

    if not service_ok:
        st.model_present = False
        st.message = err or "Ollama 服务不可用"
        return st

    st.model_present = is_model_installed(name, installed)
    if st.model_present:
        st.message = f"Ollama 可用, 模型 {name} 已就绪"
    else:
        have = ", ".join(installed) if installed else "(无)"
        st.message = (f"服务可用, 但没有找到模型 {name}。"
                      f"当前已安装: {have}。请先执行: ollama pull {name}")
    return st


def check_model_availability(model: Optional[str] = None,
                             host: Optional[str] = None,
                             timeout: float = 5.0) -> Dict[str, Any]:
    """对外接口: 检查模型可用性, 返回可序列化的 dict.

    供外部调用方 (命令行 / 其它模块 / 脚本) 使用::

        from core import check_model_availability
        report = check_model_availability("qwen3-vl:4b-instruct-q4_k_m")
        if not report["ok"]:
            print(report["message"])
    """
    return check_model(model=model, host=host, timeout=timeout).to_dict()


def format_report(status: ModelStatus) -> str:
    """把检查结果渲染成给用户看的「模型可用性异常报告」.

    只在失败时才会被界面弹窗使用; 成功时返回简短一行, 便于写日志。
    """
    if status.ok:
        return f"OK  {status.message}"

    lines = [
        "模型可用性异常报告",
        "=" * 40,
        f"模型: {status.model}",
        f"检查时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(status.checked_at))}",
        "",
    ]
    if not status.service_ok:
        lines += [
            "Ollama 服务: 未连接",
            f"原因: {status.error or status.message}",
            "",
            "处理建议:",
            "  1. 确认 Ollama 已启动（开始菜单 → Ollama，或在命令行执行 ollama serve）",
            "  2. 确认系统托盘出现 Ollama 图标",
            f"  3. 确认服务地址可访问（默认 {DEFAULT_HOST}）",
            "  4. 启动后回到本页点「检查」重试",
        ]
    else:
        have = ", ".join(status.installed) if status.installed else "(无)"
        lines += [
            "Ollama 服务: 已连接",
            "模型是否安装: 否",
            f"当前已安装: {have}",
            "",
            "处理建议:",
            f"  1. 在命令行执行: ollama pull {status.model}",
            "  2. 拉取完成后回到本页点「检查」重试",
            "  3. 也可以先把模型切回已安装的 2B Instruct 继续用",
        ]
    lines += [
        "",
        "说明: 本次检查失败不会退出程序，仍可切换其它模型或继续使用已有模型。",
    ]
    return "\n".join(lines)
