"""USB Webcam capture using OpenCV with thread-safe frame queue."""

from __future__ import annotations

import threading
from queue import Queue

import cv2
import numpy as np


class Camera:
    """Threaded USB webcam capture.

    Usage:
        cam = Camera(device_id=0)
        cam.start()
        frame = cam.read()  # Returns None if no frame available
        cam.stop()
    """

    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480):
        self.device_id = device_id
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None
        self._queue: Queue[np.ndarray] = Queue(maxsize=2)
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        """Open camera and start capture thread. Returns True on success."""
        self._cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            # Fallback without DSHOW
            self._cap = cv2.VideoCapture(self.device_id)

        if not self._cap.isOpened():
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def _capture_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                # Drop old frames if queue is full (keep latest)
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except Exception:
                        pass
                self._queue.put(frame)

    def read(self) -> np.ndarray | None:
        """Read latest frame. Returns None if camera not running."""
        if not self._running:
            return None
        try:
            return self._queue.get(timeout=1.0)
        except Exception:
            return None

    def stop(self):
        """Stop capture and release camera."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()