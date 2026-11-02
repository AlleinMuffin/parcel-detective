"""parcel_detective 核心包 —— v2.0.0 (VLM 兜底系统).

v2.0.0 是**彻底转型**版本: 放弃软件条码识别与传统 OCR, 只保留
「VLM 大模型视觉识别 + 多字段匹配」这一条主线。

因此本文件**刻意极简**, 且**不在顶层导入任何重依赖**:

    from core import database          # 只依赖标准库, 随时可用
    from core import detect_pipeline   # 调度层
    from core import ai_vision         # 需要 ollama
    from core import matcher           # 需要 fuzzywuzzy (可选)

为什么顶层不 import 子模块
--------------------------
早期版本在 ``__init__.py`` 顶层 ``from .pipeline import ...``, 结果任何
``from core import database`` 都会连带拉起整条 OCR 依赖链 (paddle 几百 MB)。
v2.0.0 已经删掉 OCR, 但这个教训保留: **让 "只想用数据库" 的场景不必
为 AI 依赖付代价**。

v1 的历史版本 (v1.0.0 ~ v1.7.0) 连同条码/OCR 模块完整保留在
``D:\\parcel_detective\\parcel_detective_V1``, 需要时可回溯。
"""

__all__ = [
    "run_full_pipeline",
    "run_full_pipeline_batch",
    "swap_and_reevaluate",
    "DetectConfig",
    "Verdict",
    "DEFAULT_CONFIG",
    "check_model_availability",
    "model_registry",
]

__version__ = "2.1.0"


def __getattr__(name: str):
    """按需导出流水线接口 (PEP 562).

    这样 ``from core import run_full_pipeline`` 仍然可用, 但只有在真正
    取用时才会 import ``detect_pipeline`` 及其依赖。
    """
    _lazy = {"run_full_pipeline", "run_full_pipeline_batch",
             "swap_and_reevaluate", "DetectConfig", "Verdict",
             "DEFAULT_CONFIG"}
    if name in _lazy:
        # 用 importlib 直接加载子模块: ``from . import x`` 在模块尚未
        # 初始化完时会再次触发本 __getattr__, 形成无限递归。
        import importlib
        _dp = importlib.import_module(".detect_pipeline", __name__)
        return getattr(_dp, name)
    if name == "model_registry":
        import importlib
        return importlib.import_module(".model_registry", __name__)
    if name == "check_model_availability":
        import importlib
        _mr = importlib.import_module(".model_registry", __name__)
        return _mr.check_model_availability
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
