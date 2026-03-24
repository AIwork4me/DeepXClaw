"""Gradio WebUI for YOLOv26 + DeepX M1 real-time object detection.

Architecture:
  - Detection runs in a daemon thread (camera + NPU)
  - Gradio Timer always active, polls latest frame from shared state
  - Start/Stop buttons only toggle a flag, no outputs (no loading state)
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def load_config() -> dict:
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / ".deepxclaw.json"
    if not config_path.exists():
        raise FileNotFoundError(f".deepxclaw.json not found at {config_path}")
    with open(config_path) as f:
        return json.load(f)


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    from .labels import COLORS
    for det in detections:
        x1, y1 = int(det["x1"]), int(det["y1"])
        x2, y2 = int(det["x2"]), int(det["y2"])
        score = det["score"]
        cls_id = det["class_id"]
        label = det["label"]
        color = tuple(int(c) for c in COLORS[cls_id % len(COLORS)])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame


def create_app():
    config = load_config()
    dxrt_bin = config.get("dxrt_bin", "")
    if dxrt_bin:
        os.environ["PATH"] = str(dxrt_bin) + ";" + os.environ["PATH"]

    model_path = config.get(
        "model_path",
        str(Path(__file__).resolve().parent.parent.parent / "models" / "yolo26n-1.dxnn"),
    )

    state = {
        "running": False,
        "latest_rgb": None,
        "status": "Please click Start button...",
        "fps_display": 0.0,
        "frame_count": 0,
        "fps_time": time.time(),
        "last_latency": 0.0,
        "lock": threading.Lock(),
    }

    def _detection_worker():
        """Background thread: open camera -> load NPU -> detection loop."""
        from .camera import Camera
        from .detector import DeepXDetector

        with state["lock"]:
            state["status"] = "Opening camera..."

        cam = Camera(device_id=0, width=640, height=480)
        if not cam.start():
            with state["lock"]:
                state["status"] = "ERROR: Cannot open webcam"
                state["running"] = False
            return

        with state["lock"]:
            state["status"] = "Loading NPU model..."

        try:
            detector = DeepXDetector(model_path=model_path, dxrt_bin=dxrt_bin if dxrt_bin else None)
        except Exception as e:
            cam.stop()
            with state["lock"]:
                state["status"] = f"ERROR: {e}"
                state["running"] = False
            return

        with state["lock"]:
            state["status"] = "Running"
            state["frame_count"] = 0
            state["fps_time"] = time.time()

        while state["running"]:
            frame = cam.read()
            if frame is None:
                time.sleep(0.005)
                continue
            try:
                detections, latency = detector.detect(frame)
                annotated = draw_detections(frame, detections)
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                state["frame_count"] += 1
                state["last_latency"] = latency
                elapsed = time.time() - state["fps_time"]
                if elapsed >= 1.0:
                    state["fps_display"] = state["frame_count"] / elapsed
                    state["frame_count"] = 0
                    state["fps_time"] = time.time()
                with state["lock"]:
                    state["latest_rgb"] = rgb
                    state["status"] = (
                        f"Running | FPS: {state['fps_display']:.1f} | "
                        f"Latency: {latency:.1f}ms | Objects: {len(detections)}"
                    )
            except Exception as e:
                with state["lock"]:
                    state["status"] = f"Error: {e}"
                break

        cam.stop()
        detector.dispose()
        with state["lock"]:
            state["latest_rgb"] = None
            if state["running"]:
                state["status"] = "Stopped"
            state["running"] = False

    def on_start():
        """Toggle start - returns status instantly, heavy work in background."""
        if state["running"]:
            return "Already running"
        state["running"] = True
        threading.Thread(target=_detection_worker, daemon=True).start()
        return "Starting..."

    def on_stop():
        """Toggle stop - returns instantly."""
        state["running"] = False
        return "Stopping..."

    def poll():
        """Timer poll - reads latest frame from shared state."""
        with state["lock"]:
            return state["latest_rgb"], state["status"]

    def screenshot(frame):
        if frame is None:
            return frame, "No frame"
        save_path = f"screenshot_{int(time.time())}.png"
        cv2.imwrite(save_path, cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        return frame, f"Saved to {save_path}"

    with gr.Blocks(title="YOLOv26 + DeepX M1 Demo") as app:
        gr.Markdown("# YOLOv26 + DeepX M1 Demo\nReal-time object detection powered by DeepX M1 NPU")
        video = gr.Image(label="Detection View", interactive=False, height=480)
        with gr.Row():
            btn_start = gr.Button("Start", variant="primary", size="sm")
            btn_stop = gr.Button("Stop", variant="stop", size="sm")
            btn_screenshot = gr.Button("Screenshot", size="sm")
        status_text = gr.Textbox(value="Please click Start button...", label="Status", interactive=False, max_lines=2)

        # Timer ALWAYS active - polls shared state every 100ms
        timer = gr.Timer(value=0.1, active=True)
        timer.tick(fn=poll, outputs=[video, status_text])

        # Buttons update status text only (instant, lightweight)
        btn_start.click(fn=on_start, outputs=[status_text])
        btn_stop.click(fn=on_stop, outputs=[status_text])
        btn_screenshot.click(fn=screenshot, inputs=[video], outputs=[video, status_text])

    return app


def main():
    import gradio.networking as _gn
    _orig_url_ok = getattr(_gn, "url_ok", None)
    if _orig_url_ok:
        _gn.url_ok = lambda url: True
    import httpx as _hx
    _orig_get = _hx.get
    def _safe_get(url, **kw):
        if "startup-events" in str(url):
            return _hx.Response(200, request=_hx.Request("GET", url))
        return _orig_get(url, **kw)
    _hx.get = _safe_get
    try:
        app = create_app()
        app.launch(server_name="0.0.0.0", server_port=7860)
    finally:
        _hx.get = _orig_get
        if _orig_url_ok:
            _gn.url_ok = _orig_url_ok


if __name__ == "__main__":
    main()