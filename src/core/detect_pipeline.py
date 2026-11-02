"""一键检测 —— v2.0.0 **直连模式**.

v1 是「三级流水线」: 条码 -> VLM -> 人工. v2.0.0 砍掉条码那一级,
变成**一条直线**:

::

    图片 ──> ai_vision.extract_and_match() ──> 判定

判档依据只有匹配得分:

    >= 0.85            -> 【通过】    自动打单 / 放行
    0.60 ~ 0.85        -> 【待确认】  人工核验 (给 Top-3 候选)
    <  0.60 或 无字段  -> 【人工】    转人工处理通道

为什么砍掉条码
--------------
条码确实快 (几十毫秒), 但它是**另一条技术路线**, 会让整个系统的职责变成
"两套识别 + 互相兜底", 复杂度高、边界多。v2.0.0 的定位是**纯 VLM 兜底
系统** —— 只做"AI 能不能认出来、认出来像不像库里的某一条"。

条码能力在 v1 里完整保留 (``parcel_detective_V1``), 需要时可回溯。

与 v1 的接口差异
----------------
========================  =====================  =====================
项                        v1 (三级)              v2.0.0 (直连)
========================  =====================  =====================
``timings`` 键            barcode_ms/ai_ms/total ai_ms/total
``level`` (停在第几级)     1/2/3                  **移除**
``barcode`` / ``boxes``   ScanResult + 框坐标    **移除**
``barcode_fail_reason``   有                     **移除**
``progress`` 状态         四句                   三句
========================  =====================  =====================

**不修改** ``ai_vision.py`` / ``matcher.py`` / ``database.py`` —— 本模块
只做调度与判档。
"""
from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "DetectConfig",
    "DEFAULT_CONFIG",
    "Verdict",
    "run_full_pipeline",
    "run_full_pipeline_batch",
    "swap_and_reevaluate",
    "PASS_SCORE",
    "REVIEW_SCORE",
]

_LOGGER = logging.getLogger("detect_pipeline")


class _ServiceDown(RuntimeError):
    """AI 服务不可用（预检失败）。

    单独一个异常类型，是为了和"代码缺陷"区分开：服务没起是**环境问题**,
    应该给用户可操作的建议, 而不是"报给开发"。
    """


def _log(cb: Optional[Callable[[str], None]], msg: str) -> None:
    _LOGGER.info(msg)
    if cb:
        try:
            cb(msg)
        except Exception:                        # noqa: BLE001
            pass


# ==========================================================================
# 判定
# ==========================================================================
class Verdict:
    """三档最终判定."""

    PASS = "pass"          # 通过
    REVIEW = "review"      # 待确认
    MANUAL = "manual"      # 人工

    TEXT = {PASS: "通过", REVIEW: "待确认", MANUAL: "人工"}

    ADVICE = {
        PASS: "自动打单 / 放行。",
        REVIEW: "待确认：请从候选里核对是哪一件，确认后放行。",
        MANUAL: "人工处理：信息不足或匹配度过低，无法自动找回。",
    }


#: 判档阈值 (用户指定)
PASS_SCORE = 0.85
REVIEW_SCORE = 0.60


# ==========================================================================
# 配置
# ==========================================================================
@dataclass
class DetectConfig:
    """一键检测参数."""

    #: 数据库路径 (None = 用 database.resolve_db_path 的默认位置)
    db_path: Optional[object] = None
    #: 返回几个候选
    top_n: int = 3
    #: VLM 模型名 (None = 用 ai_vision 的默认, 可用环境变量覆盖)
    ai_model: Optional[str] = None
    #: 匹配成功后是否入库
    auto_save: bool = False
    #: 放行模式, 透传给 matcher ("strict"/"lenient")
    release_mode: str = "strict"

    def to_dict(self) -> Dict[str, Any]:
        return {"top_n": self.top_n, "ai_model": self.ai_model,
                "auto_save": self.auto_save,
                "release_mode": self.release_mode}


DEFAULT_CONFIG = DetectConfig()


# ==========================================================================
# 结果
# ==========================================================================
@dataclass
class DetectResult:
    """一次一键检测的结果."""

    image_path: str = ""
    ok: bool = False
    error: str = ""
    #: Verdict.*
    verdict: str = ""
    verdict_text: str = ""
    advice: str = ""
    #: 最高分候选的单号
    tracking_number: str = ""
    #: 匹配得分
    score: float = 0.0
    #: Top-N 候选
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    #: AI 提取到的字段 (15 个)
    ai_fields: Dict[str, Any] = field(default_factory=dict)
    #: 耗时: {"ai_ms":..., "total_ms":...}
    timings: Dict[str, float] = field(default_factory=dict)
    #: 第二级原始结果 (extract_and_match 的返回值)
    ai: Optional[Dict[str, Any]] = None
    #: 判定原因说明
    reason: str = ""
    #: 是否人工对调过收发件方
    swapped_manually: bool = False
    #: 是否已入库
    saved: Dict[str, Any] = field(default_factory=dict)
    #: 日志行
    logs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path, "ok": self.ok, "error": self.error,
            "verdict": self.verdict, "verdict_text": self.verdict_text,
            "advice": self.advice,
            "tracking_number": self.tracking_number, "score": self.score,
            "candidates": self.candidates, "ai_fields": self.ai_fields,
            "timings": self.timings, "ai": self.ai, "reason": self.reason,
            "swapped_manually": self.swapped_manually, "saved": self.saved,
            "logs": self.logs,
        }


# ==========================================================================
# 懒加载 (也便于测试时替换)
# ==========================================================================
def _get_ai_module():
    try:
        from . import ai_vision as mod
        return mod
    except Exception:                            # noqa: BLE001
        pass
    try:
        import ai_vision as mod                  # type: ignore[no-redef]
        return mod
    except Exception:                            # noqa: BLE001
        return None


def _get_matcher_module():
    try:
        from . import matcher as mod
        return mod
    except Exception:                            # noqa: BLE001
        pass
    try:
        import matcher as mod                    # type: ignore[no-redef]
        return mod
    except Exception:                            # noqa: BLE001
        return None


# ==========================================================================
# 服务预检 (快速失败)
# ==========================================================================
def _split_host(host: str) -> Tuple[str, int]:
    """把 ``http://localhost:11434`` 拆成 ("localhost", 11434)."""
    h = str(host or "").strip()
    for pre in ("http://", "https://"):
        if h.startswith(pre):
            h = h[len(pre):]
    h = h.rstrip("/")
    if ":" in h:
        name, _, port = h.partition(":")
        try:
            return name or "localhost", int(port)
        except ValueError:
            return name or "localhost", 11434
    return h or "localhost", 11434


def probe_service(cfg: Optional[Any] = None,
                  timeout: float = 0.8) -> Tuple[bool, str]:
    """**快速**探测 Ollama 服务是否在监听.

    为什么不用 ``ai_vision.check_service()``: 它内部把超时写死 15 秒
    (那是受保护的文件, 不改), 服务没起时反应太慢。这里直接做 TCP 连接:
    端口关闭时操作系统立即返回 RST, 毫秒级失败; 主机不可达才等 timeout。

    返回 ``(可用, 说明)``, 不抛异常。
    """
    import socket

    host = "localhost"
    port = 11434
    try:
        from_cfg = getattr(cfg, "host", None) if cfg is not None else None
        if from_cfg:
            host, port = _split_host(from_cfg)
        else:
            mod = _get_ai_module()
            if mod is not None:
                host, port = _split_host(
                    getattr(getattr(mod, "DEFAULT_CONFIG", None), "host", "")
                    or "localhost:11434")
    except Exception:                            # noqa: BLE001
        pass

    # ⚠️ 不要用 socket.create_connection: 它会对 getaddrinfo 返回的**每个**
    # 地址各试一次。localhost 同时有 ::1 与 127.0.0.1, 端口被过滤
    # (TimeoutError 而非 ConnectionRefused) 时每个都耗满超时 —— 实测 2×2s=4s。
    # 这里显式只取 IPv4, 一次就够, 0.8 秒内出结果。
    last_exc: Optional[BaseException] = None
    try:
        infos = socket.getaddrinfo(host, port, socket.AF_INET,
                                  socket.SOCK_STREAM)
    except Exception:                            # noqa: BLE001
        infos = []
    for family, type_, proto, _canon, sockaddr in infos[:1] or []:
        try:
            s = socket.socket(family, type_, proto)
            try:
                s.settimeout(timeout)
                s.connect(sockaddr)
                return True, f"Ollama 服务在监听 {host}:{port}"
            finally:
                s.close()
        except Exception as exc:                 # noqa: BLE001
            last_exc = exc
    if last_exc is None:                         # 解析失败, 退回一次普通连接
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True, f"Ollama 服务在监听 {host}:{port}"
        except Exception as exc:                 # noqa: BLE001
            last_exc = exc
    return False, (f"连不上 Ollama 服务（{host}:{port}，"
                   f"{type(last_exc).__name__}）。\n"
                   f"请先启动 Ollama（开始菜单打开 Ollama，或在命令行执行 "
                   f"ollama serve），确认托盘有 Ollama 图标后重试。")


def _service_failure(host_hint: str = "") -> Tuple[str, str]:
    """服务不可用时的 (原因, 建议)."""
    reason = "AI 服务未运行，无法识别" + (f"（{host_hint}）" if host_hint else "")
    advice = ("请先启动 Ollama 服务后重试：\n"
              "  · 开始菜单 → Ollama，或命令行执行 ollama serve\n"
              "  · 确认系统托盘出现 Ollama 图标\n"
              "  · 再点一次「上传图片并识别」")
    return reason, advice


# ==========================================================================
# 判档
# ==========================================================================
def _decide(score: float, has_fields: bool, has_candidates: bool
            ) -> Tuple[str, str]:
    """按得分判档, 返回 ``(verdict, reason)``.

    注意"没有有效字段"优先于得分 —— 那说明 VLM 根本没读出东西,
    哪怕有个 0 分的候选也不能算"待确认"。
    """
    if not has_fields:
        return Verdict.MANUAL, "VLM 未提取出有效字段"
    if not has_candidates:
        return Verdict.MANUAL, "信息库里没有相似记录"
    if score >= PASS_SCORE:
        return Verdict.PASS, f"匹配得分 {score:.2f} ≥ {PASS_SCORE}"
    if score >= REVIEW_SCORE:
        return (Verdict.REVIEW,
                f"匹配得分 {score:.2f} 在 {REVIEW_SCORE}~{PASS_SCORE} 之间")
    return Verdict.MANUAL, f"匹配得分 {score:.2f} < {REVIEW_SCORE}"


# ==========================================================================
# 主流程
# ==========================================================================
def run_full_pipeline(image_path: Any,
                      config: Optional[DetectConfig] = None,
                      log: Optional[Callable[[str], None]] = None,
                      progress: Optional[Callable[[str], None]] = None
                      ) -> Dict[str, Any]:
    """一键检测 (直连模式): 图片 -> AI 提取 + 匹配 -> 判定.

    参数
    ----
    image_path
        面单图片路径。
    config
        参数; 省略用 :data:`DEFAULT_CONFIG`。
    log
        日志回调 (逐行)。
    progress
        **状态回调**, 只发这三句, 顺序固定::

            正在启动AI视觉识别...
            AI识别完成，正在比对数据库...
            处理完成

    返回
    ----
    dict, 含 ``verdict`` / ``verdict_text`` / ``advice`` / ``score`` /
    ``tracking_number`` / ``candidates`` / ``ai_fields`` / ``timings`` /
    ``logs`` 等。**保证不抛异常**。
    """
    cfg = config or DEFAULT_CONFIG
    logs: List[str] = []
    res = DetectResult(image_path=str(image_path or ""))
    started = time.perf_counter()

    def _sink(msg: str) -> None:
        logs.append(msg)
        _log(log, msg)

    def _say(msg: str) -> None:
        _sink(msg)
        if progress:
            try:
                progress(msg)
            except Exception:                    # noqa: BLE001
                pass

    def _elapsed() -> float:
        return (time.perf_counter() - started) * 1000

    name = Path(str(image_path)).name if image_path else "(空)"
    _sink(f"[一键检测] 开始: {name}")

    try:
        mod = _get_ai_module()
        if mod is None:
            raise RuntimeError("找不到 ai_vision 模块")

        # ---- 启动 ----
        _say("正在启动AI视觉识别...")

        # 快速预检: 服务没起就别去烧三次超时 (实测 13 秒/张)。
        ok_svc, svc_msg = probe_service(getattr(mod, "DEFAULT_CONFIG", None))
        if not ok_svc:
            _sink(f"  [一键] 预检失败: {svc_msg.splitlines()[0]}")
            raise _ServiceDown(svc_msg)

        t_ai = time.perf_counter()
        try:
            vcfg = mod.VisionConfig()
            if cfg.ai_model:
                vcfg.model = cfg.ai_model
            if hasattr(vcfg, "release_mode"):
                vcfg.release_mode = cfg.release_mode
            raw = mod.extract_and_match(
                image_path, cfg.db_path, config=vcfg,
                top_n=cfg.top_n, log=_sink)
        except Exception as exc:                 # noqa: BLE001
            a_ms = (time.perf_counter() - t_ai) * 1000
            res.timings = {"ai_ms": round(a_ms, 1),
                           "total_ms": round(_elapsed(), 1)}
            raise RuntimeError(f"AI 阶段异常: {type(exc).__name__}: {exc}") from exc
        a_ms = (time.perf_counter() - t_ai) * 1000
        res.timings["ai_ms"] = round(a_ms, 1)
        _sink(f"  [一键] AI 阶段耗时 {a_ms:.0f} ms")

        res.ai = raw or {}
        ai_block = (raw or {}).get("ai") or {}
        match = (raw or {}).get("match") or {}
        cands = match.get("candidates") or []

        res.ai_fields = dict(ai_block)
        res.candidates = cands
        res.saved = (raw or {}).get("saved") or {}
        best = cands[0] if cands else {}
        res.score = float(best.get("total_score") or 0.0)
        res.tracking_number = str(best.get("tracking_number") or "")

        # ---- 先判断 AI 这一步到底成没成 ----
        # ai_vision 的失败信息可能在三个地方, 都要读:
        #   · raw["ok"]          False 表示这一步没成
        #   · raw["message"]     "AI 识别失败: ..."
        #   · ai_block["error"]  原始异常文本
        # 早先只读 ai.error 且日志打印的是局部变量, 导致用户只看到
        # "VLM 未提取出有效字段", 完全看不出真正原因是连不上服务。
        ai_msg = str((raw or {}).get("message") or "").strip()
        ai_err = str(ai_block.get("error") or "").strip()
        ai_failed = (not (raw or {}).get("ok", True)) or bool(ai_err)

        if ai_failed:
            _say("AI识别失败，未进入比对")
        else:
            _say("AI识别完成，正在比对数据库...")

        # ---- 有没有提取到有效字段 ----
        # AI 报错, 或 15 个字段全空, 都算"没读出东西"
        meaningful = {k: v for k, v in ai_block.items()
                      if v not in (None, "", [], {}) and k != "error"}
        has_fields = bool(meaningful) and not ai_failed

        # ---- 判档 ----
        verdict, reason = _decide(res.score, has_fields, bool(cands))
        res.verdict = verdict
        res.verdict_text = Verdict.TEXT[verdict]
        res.advice = Verdict.ADVICE[verdict]
        res.reason = reason
        # 把真实原因拼进去 —— 用户必须能看到"连不上服务"这种关键信息
        detail = ai_err or ai_msg
        if ai_failed and detail:
            res.reason = f"{reason}　真实原因: {detail}"
            res.advice = (Verdict.ADVICE[verdict] + "\n\n"
                          f"AI 侧错误: {detail}")
            if "connect" in detail.lower() or "连不上" in detail:
                _r, _a = _service_failure()
                res.advice = _a
        res.ok = True

        # ---- 自动入库 (只在判定"通过"时) ----
        # ⚠️ extract_and_match() **没有** save 参数, 必须单独调 save_ai_result。
        # 早先按印象写成 save=..., 会直接 TypeError。
        if cfg.auto_save and verdict == Verdict.PASS:
            try:
                res.saved = mod.save_ai_result(raw, cfg.db_path) or {}
                _sink(f"  [一键] 已入库: {res.saved.get('status')}")
            except Exception as exc:             # noqa: BLE001
                _sink(f"  [一键] 入库失败(不影响判定): "
                      f"{type(exc).__name__}: {exc}")
                res.saved = {"status": "error", "message": str(exc)}

        res.timings["total_ms"] = round(_elapsed(), 1)
        _say("处理完成")
        # ⚠️ 必须打印 res.reason 而不是局部变量 reason,
        #    否则刚拼进去的错误细节会在日志里丢掉。
        _sink(f"[一键检测] 结果: {res.verdict_text}　{res.reason}")
        _sink(f"[一键检测] 耗时: AI {a_ms:.0f} ms, 总 {_elapsed():.0f} ms")
        res.logs = logs
        return res.to_dict()

    except _ServiceDown as exc:
        # 预检就失败 —— 这不是"异常", 是可预期的环境问题, 给可操作建议。
        res.ok = False
        res.error = str(exc)
        res.verdict = Verdict.MANUAL
        res.verdict_text = Verdict.TEXT[Verdict.MANUAL]
        res.reason, res.advice = _service_failure()
        try:
            _say("AI识别失败，未进入比对")
        except Exception:                        # noqa: BLE001
            pass
        _sink(f"[一键检测] {res.reason}")
        _sink(f"[一键检测] {str(exc)}")
        res.timings["ai_ms"] = round((time.perf_counter() - started) * 1000, 1)
        res.timings["total_ms"] = round(_elapsed(), 1)
        try:
            _say("处理完成")
        except Exception:                        # noqa: BLE001
            pass
        res.logs = logs
        return res.to_dict()

    except Exception as exc:                     # noqa: BLE001 - 绝不外抛
        res.ok = False
        res.error = str(exc)
        res.verdict = Verdict.MANUAL
        res.verdict_text = Verdict.TEXT[Verdict.MANUAL]
        res.advice = ("处理过程出错，已按人工处理。\n"
                      f"错误: {res.error}\n"
                      "建议: 检查 AI 服务是否在运行，或把日志报给开发。")
        res.reason = f"异常: {res.error}"
        res.timings.setdefault("total_ms", round(_elapsed(), 1))
        _sink(f"[一键检测] 未预期异常: {res.error}")
        _sink(traceback.format_exc())
        try:
            _say("处理完成")
        except Exception:                        # noqa: BLE001
            pass
        res.logs = logs
        return res.to_dict()


def run_full_pipeline_batch(image_paths: Sequence[Any],
                            config: Optional[DetectConfig] = None,
                            log: Optional[Callable[[str], None]] = None,
                            progress: Optional[Callable[[int, int, str], None]] = None
                            ) -> List[Dict[str, Any]]:
    """批量一键检测. 单张失败不影响其它."""
    paths = list(image_paths or [])
    out: List[Dict[str, Any]] = []
    total = len(paths)

    # 批量前先探一次服务。不可用就直接把每一张都快速标成人工,
    # 别让 N 张图各烧十几秒 (实测那是 2 分钟白等)。
    if total > 1:
        ok_svc, svc_msg = probe_service(
            getattr(config or DEFAULT_CONFIG, "host", None))
        if not ok_svc:
            reason, advice = _service_failure()
            if log:
                try:
                    log(f"[一键] 预检失败，{total} 张全部跳过: "
                        f"{svc_msg.splitlines()[0]}")
                except Exception:                # noqa: BLE001
                    pass
            for i, pth in enumerate(paths, 1):
                if progress:
                    try:
                        progress(i - 1, total, Path(str(pth)).name)
                    except Exception:            # noqa: BLE001
                        pass
                r = DetectResult(image_path=str(pth), ok=False,
                                 error=svc_msg, verdict=Verdict.MANUAL,
                                 verdict_text=Verdict.TEXT[Verdict.MANUAL],
                                 advice=advice, reason=reason)
                r.timings = {"ai_ms": 0.0, "total_ms": 0.0}
                r.logs = [f"[一键检测] {reason}"]
                out.append(r.to_dict())
            if progress:
                try:
                    progress(total, total, "")
                except Exception:                # noqa: BLE001
                    pass
            return out
    for i, p in enumerate(paths, 1):
        if progress:
            try:
                progress(i - 1, total, Path(str(p)).name)
            except Exception:                    # noqa: BLE001
                pass
        out.append(run_full_pipeline(p, config, log=log))
    if progress:
        try:
            progress(total, total, "")
        except Exception:                        # noqa: BLE001
            pass
    return out


# ==========================================================================
# 人工对调收/发件方后重判
# ==========================================================================
def swap_and_reevaluate(result: Dict[str, Any],
                        db_path: Optional[object] = None,
                        top_n: int = 3) -> Dict[str, Any]:
    """把结果里的收件方/寄件方字段对调, 重新比对并按同样阈值重判.

    为什么需要它: VLM 会把发货方和收件方搞反。界面要提供人工对调的按钮,
    但**判档逻辑不能写在界面里** —— 否则阈值会有两份实现, 早晚不一致。

    ``ai_vision`` / ``matcher`` 的核心逻辑不改, 这里只是调用它们的公开函数。

    返回**新的**结果字典, 不改传入的那个。
    """
    r = dict(result or {})
    ai_block = dict(r.get("ai") or {})
    fields = ai_block.get("ai") or r.get("ai_fields") or {}
    if not fields:
        r["swap_error"] = "没有 AI 提取到的字段, 无法对调"
        return r

    mod_ai = _get_ai_module()
    mod_m = _get_matcher_module()
    if mod_ai is None or mod_m is None:
        r["swap_error"] = "找不到 ai_vision 或 matcher 模块"
        return r

    try:
        swapped = mod_ai.swap_sender_recipient(dict(fields))
        new_match = mod_m.match_from_ai(swapped, db_path, top_n=top_n)
    except Exception as exc:                     # noqa: BLE001
        r["swap_error"] = f"对调失败: {type(exc).__name__}: {exc}"
        return r

    ai_block["ai"] = swapped
    ai_block["match"] = new_match
    ai_block["message"] = new_match.get("message", "")
    r["ai"] = ai_block
    r["ai_fields"] = dict(swapped)

    cands = new_match.get("candidates") or []
    r["candidates"] = cands
    best = cands[0] if cands else {}
    score = float(best.get("total_score") or 0.0)
    r["score"] = score
    r["tracking_number"] = str(best.get("tracking_number") or "")

    meaningful = {k: v for k, v in swapped.items()
                  if v not in (None, "", [], {})}
    verdict, reason = _decide(score, bool(meaningful), bool(cands))
    r["verdict"] = verdict
    r["verdict_text"] = Verdict.TEXT[verdict]
    r["advice"] = Verdict.ADVICE[verdict]
    r["reason"] = f"已人工对调收/发件方：{reason}"
    r["swapped_manually"] = True
    r.pop("swap_error", None)
    return r
