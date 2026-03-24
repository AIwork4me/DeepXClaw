"""DeepX M1 NPU inference wrapper using dx_engine."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np


class DeepXDetector:
    """Wraps dx_engine InferenceEngine for YOLOv26n on DeepX M1 NPU."""

    def __init__(self, model_path: str | Path, dxrt_bin: str | None = None):
        """Initialize NPU detector.

        Args:
            model_path: Path to .dxnn model file.
            dxrt_bin: Path to dx_rt/bin directory (for dxrt.dll).
                      If provided, prepended to PATH before importing dx_engine.
        """
        if dxrt_bin:
            os.environ["PATH"] = str(dxrt_bin) + ";" + os.environ["PATH"]

        from dx_engine import InferenceEngine

        self.model_path = str(model_path)
        self.engine = InferenceEngine(self.model_path)
        self._input_size = (640, 640)  # H, W
        self._frame_count = 0
        self._fps = 0.0
        self._fps_time = time.time()

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_size

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize and format frame for NPU input.

        Args:
            frame: BGR frame from OpenCV, shape (H, W, 3).

        Returns:
            NPU input tensor, shape (1, 640, 640, 3), dtype uint8.
        """
        # Resize with letterbox-style padding (aspect ratio preserved)
        h, w = frame.shape[:2]
        target_h, target_w = self._input_size

        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        import cv2
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )

        # Convert BGR to RGB (dx_engine expects RGB)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        return rgb.reshape(1, target_h, target_w, 3).astype(np.uint8)

    def detect(self, frame: np.ndarray) -> tuple[list[dict], float]:
        """Run detection on a single frame.

        Args:
            frame: BGR frame from OpenCV.

        Returns:
            (detections, latency_ms) where detections is a list of dicts
            with x1,y1,x2,y2,score,class_id,label keys.
        """
        from .postprocess import decode_yolo26n

        input_tensor = self.preprocess(frame)

        t0 = time.perf_counter()
        outputs = self.engine.run(input_tensor)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        detections = decode_yolo26n(outputs[0], input_size=self._input_size)

        # Track FPS
        self._frame_count += 1
        elapsed = time.time() - self._fps_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_time = time.time()

        return detections, latency_ms

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def npu_time_us(self) -> float:
        """Last NPU inference time in microseconds."""
        try:
            return self.engine.get_npu_inference_time()
        except Exception:
            return 0.0

    def dispose(self):
        self.engine.dispose()