# DeepXClaw - YOLOv26 + DeepX M1 NPU Real-time Object Detection

Real-time object detection WebUI powered by DeepX M1 NPU (3 cores, 3.92 GiB LPDDR5).

## Features

- **YOLOv26n** model optimized for DeepX M1 NPU
- **~25ms inference latency** on NPU
- **Gradio WebUI** for easy interaction
- **USB Webcam** support with threaded capture

## Requirements

- Windows 10/11
- DeepX M1 NPU hardware
- DeepX Runtime v3.2.0+
- Python 3.10+
- USB Webcam

## Installation

```bash
# Clone the repo
git clone https://github.com/Tinkerclaw/deepxclaw.git
cd deepxclaw

# Install dependencies with uv
pip install uv
uv sync

# Copy model to models/
# yolo26n-1.dxnn should be placed in models/

# Create config
cp .deepxclaw.json.example .deepxclaw.json
# Edit .deepxclaw.json with your local paths

# Run
uv run python -m deepxclaw.app
```

## Project Structure

```
deepxclaw/
├── src/deepxclaw/
│   ├── __init__.py
│   ├── app.py          # Gradio WebUI
│   ├── camera.py       # Threaded webcam capture
│   ├── detector.py     # DeepX NPU inference
│   ├── labels.py       # COCO class labels & colors
│   └── postprocess.py  # YOLO NMS post-processing
├── models/             # NPU models (.dxnn)
├── pyproject.toml
└── README.md
```

## Hardware

- **NPU**: DeepX M1 (3 cores, LPDDR5 3.92 GiB)
- **Interface**: PCIe Gen3 x4
- **Power**: 15W TDP

## Model

- **Architecture**: YOLOv26n
- **Input**: 640x640 RGB uint8
- **Output**: 300 detections max (cx, cy, w, h, conf, class_id)
- **Classes**: COCO 80 classes

## Performance

| Metric | Value |
|--------|-------|
| Inference latency | ~25ms |
| End-to-end latency | ~35ms |
| Throughput | ~30 FPS |

## License

MIT

## Developed by

**OpenClaw** - AI Agent Platform

---

*This project is a demo for DeepX M1 NPU capabilities.*