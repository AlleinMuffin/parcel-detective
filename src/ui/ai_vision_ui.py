"""一键检测的界面支撑: 后台工作线程.

为什么需要线程
--------------
VLM 单张 5~25 秒。放界面线程里会假死 —— 连"停止"按钮都点不动。
所以识别放到 QThread 里, 用信号回传日志、状态、进度。

与 v1 的区别
------------
v1 的 ``AiVisionWorker`` 直接调 ``ai_vision.extract_and_match_batch``,
还需要自己拼判定。v2.0.0 把判定收敛到 ``detect_pipeline`` 里, 这里
**只负责跑和回报**, 不含任何业务判断。
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import QObject, pyqtSignal

from core import detect_pipeline as dp


class DetectWorker(QObject):
    """在独立线程里跑一键检测 (直连模式)."""

    log = pyqtSignal(str)                 # 逐行日志
    stage = pyqtSignal(str)               # 状态文字 (三句之一)
    progress = pyqtSignal(int, int, str)  # (已完成, 总数, 当前文件名)
    finished = pyqtSignal(list)           # List[dict]
    failed = pyqtSignal(str)

    def __init__(self, image_paths: List[Any],
                 config: Optional[dp.DetectConfig] = None) -> None:
        super().__init__()
        self.image_paths = list(image_paths or [])
        self.config = config or dp.DEFAULT_CONFIG
        self._stop = False

    def stop(self) -> None:
        """请求停止 (当前这张跑完就停, 不强行打断 VLM 调用)."""
        self._stop = True

    def run(self) -> None:
        try:
            out: List[Dict[str, Any]] = []
            total = len(self.image_paths)
            for i, p in enumerate(self.image_paths, 1):
                if self._stop:
                    self.log.emit("[一键] 用户请求停止")
                    break
                self.progress.emit(i - 1, total, Path(str(p)).name)
                try:
                    r = dp.run_full_pipeline(
                        p, self.config,
                        log=self.log.emit,
                        progress=self.stage.emit)
                except Exception as exc:         # noqa: BLE001 - 界面不能崩
                    r = dp.DetectResult(
                        image_path=str(p),
                        error=f"{type(exc).__name__}: {exc}",
                        verdict=dp.Verdict.MANUAL,
                        verdict_text=dp.Verdict.TEXT[dp.Verdict.MANUAL],
                        advice="处理异常，转人工。").to_dict()
                out.append(r)
            self.progress.emit(total, total, "")
            self.finished.emit(out)
        except Exception as exc:                 # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}\n"
                             f"{traceback.format_exc()}")
