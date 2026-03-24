<div align="center">

# DeepXClaw

**YOLOv26 + DeepX M1 NPU | Real-time Object Detection at 25ms**

[![License](https://img.shields.io/github/license/AIwork4me/DeepXClaw?color=blue)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=gold)](https://www.python.org/)
[![Stars](https://img.shields.io/github/stars/AIwork4me/DeepXClaw?style=social)](https://github.com/AIwork4me/DeepXClaw/stargazers)

<br>

<img src="https://img.shields.io/badge/YOLOv26n-Ultralytics-00FFFF?logo=yolo&logoColor=white" alt="YOLOv26">
<img src="https://img.shields.io/badge/NPU-DeepX_M1-FF6F00" alt="DeepX M1">
<img src="https://img.shields.io/badge/Inference-25ms-brightgreen" alt="25ms">
<img src="https://img.shields.io/badge/Throughput-30_FPS-brightgreen" alt="30 FPS">
<img src="https://img.shields.io/badge/Power-15W_TDP-orange" alt="15W">
<img src="https://img.shields.io/badge/UI-Gradio_5-F97316?logo=gradio&logoColor=white" alt="Gradio">

<br>

**English** | [中文](README_CN.md)

<br><br>

<img width="80%" src="screenshots/20260324-191357.png" alt="DeepXClaw Demo — YOLOv26 real-time detection on DeepX M1 NPU">

<br>

*Real-time COCO 80-class object detection — USB webcam to annotated display in ~35ms end-to-end*

</div>

<br>

## ✨ Highlights

- **25ms NPU inference** — YOLOv26n on DeepX M1 (3 cores, LPDDR5 3.92 GiB, PCIe Gen3 x4)
- **~30 FPS end-to-end** — camera capture, NPU inference, NMS, and live display at 15W power
- **One-command launch** — Gradio WebUI at `localhost:7860`, no ML expertise needed
- **Zero-copy pipeline** — threaded camera capture with frame-drop queue keeps latency low
- **80-class detection** — full COCO label set with per-class NMS (IoU 0.45, confidence 0.25)

## 📊 Performance

> Benchmarks: YOLOv26n 640×640, USB webcam 720p input, DeepX M1 NPU, Windows 11.

| Metric | Value |
|:-------|------:|
| NPU inference latency | **~25 ms** |
| End-to-end latency | **~35 ms** |
| Throughput | **~30 FPS** |
| Max detections per frame | 300 |
| Model input | 640×640 RGB uint8 |
| Power consumption | 15W TDP |

## 🚀 Quick Start

<details open>
<summary><b>Prerequisites</b></summary>

| Requirement | Version |
|:------------|:--------|
| OS | Windows 10 / 11 |
| Hardware | DeepX M1 NPU (PCIe) |
| Runtime | DeepX Runtime v3.2.0+ |
| Python | 3.10+ |
| Camera | USB Webcam |

</details>

<details open>
<summary><b>Install & Run</b></summary>

```bash
git clone https://github.com/AIwork4me/DeepXClaw.git
cd DeepXClaw

# Install dependencies
pip install uv
uv sync

# Place your compiled NPU model
cp /path/to/yolo26n-1.dxnn models/

# Configure local paths
cp .deepxclaw.json.example .deepxclaw.json
# Edit .deepxclaw.json — set dxrt_bin and model_path

# Launch
uv run deepxclaw
```

Open **http://localhost:7860** → click **Start** → see real-time detections.

</details>

<details>
<summary><b>Configuration (.deepxclaw.json)</b></summary>

```json
{
  "dxrt_bin": "C:/path/to/dx_rt_windows/m1/v3.2.0/dx_rt/bin",
  "model_path": "models/yolo26n-1.dxnn",
  "sdk_repo": "C:/path/to/dx_rt_windows",
  "fw_repo": "C:/path/to/dx_fw",
  "models_cdn": "https://example.com/models"
}
```

| Field | Description |
|:------|:------------|
| `dxrt_bin` | Path to DeepX Runtime binaries (contains `dxrt.dll`) |
| `model_path` | Path to compiled `.dxnn` model file |
| `sdk_repo` | DeepX SDK repository root |
| `fw_repo` | DeepX firmware repository root |
| `models_cdn` | CDN URL for downloading pre-compiled models |

</details>

## 🏗️ Architecture

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│ USB      │    │ Threaded     │    │ DeepX M1     │    │ Per-class    │    │ Gradio   │
│ Webcam   │───>│ Capture      │───>│ NPU          │───>│ NMS          │───>│ WebUI    │
│          │    │ (camera.py)  │    │ (detector.py)│    │ (postproc.py)│    │ (app.py) │
└──────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘
  720p BGR       Queue(max=2)        640×640 uint8       IoU 0.45           Timer 100ms
                 drop-old-frames     [1,300,6] output    conf ≥ 0.25        poll & render
```

**Threading model**: Camera capture runs in a daemon thread with a 2-frame queue (stale frames dropped). NPU inference runs in a separate detection worker thread. The Gradio UI polls the latest annotated frame every 100ms via `gr.Timer`, ensuring smooth display without blocking the inference pipeline.

## 📁 Project Structure

<details>
<summary>Click to expand</summary>

```
DeepXClaw/
├── src/deepxclaw/
│   ├── __init__.py          # Package version
│   ├── app.py               # Gradio WebUI + detection worker thread
│   ├── camera.py            # Threaded USB webcam capture (OpenCV)
│   ├── detector.py          # DeepX M1 NPU inference (dx_engine)
│   ├── labels.py            # COCO 80-class labels + color palette
│   └── postprocess.py       # YOLOv26n output decode + NMS
├── models/                  # NPU model files (.dxnn) — gitignored
├── screenshots/             # Demo screenshots
├── .deepxclaw.json.example  # Config template
├── pyproject.toml           # Dependencies & CLI entry point
└── README.md
```

</details>

## 🔩 DeepX M1 NPU Specs

| Spec | Detail |
|:-----|:-------|
| Cores | 3 NPU cores |
| Memory | LPDDR5 3.92 GiB |
| Interface | PCIe Gen3 x4 |
| Power | 15W TDP |
| Model format | `.dxnn` (compiled from ONNX) |

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv26 architecture
- [DeepX](https://www.deepx.ai/) — M1 NPU hardware & runtime SDK
- [Gradio](https://gradio.app/) — WebUI framework

---

<div align="center">

**Built by [OpenClaw](https://github.com/AIwork4me)** — AI Agent Platform

If this project helped you, consider giving it a ⭐

</div>
