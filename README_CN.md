<div align="center">

# DeepXClaw

**YOLOv26 + DeepX M1 NPU | 25ms 实时目标检测**

[![License](https://img.shields.io/github/license/AIwork4me/DeepXClaw?color=blue)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=gold)](https://www.python.org/)
[![Stars](https://img.shields.io/github/stars/AIwork4me/DeepXClaw?style=social)](https://github.com/AIwork4me/DeepXClaw/stargazers)

<br>

<img src="https://img.shields.io/badge/YOLOv26n-Ultralytics-00FFFF?logo=yolo&logoColor=white" alt="YOLOv26">
<img src="https://img.shields.io/badge/NPU-DeepX_M1-FF6F00" alt="DeepX M1">
<img src="https://img.shields.io/badge/推理延迟-25ms-brightgreen" alt="25ms">
<img src="https://img.shields.io/badge/吞吐量-30_FPS-brightgreen" alt="30 FPS">
<img src="https://img.shields.io/badge/功耗-15W_TDP-orange" alt="15W">
<img src="https://img.shields.io/badge/界面-Gradio_5-F97316?logo=gradio&logoColor=white" alt="Gradio">

<br>

[English](README.md) | **中文**

<br>

<img width="80%" src="screenshots/20260324-191357.png" alt="DeepXClaw 演示 — YOLOv26 在 DeepX M1 NPU 上的实时检测">

<br>

*COCO 80 类实时目标检测 — USB 摄像头到标注显示端到端仅需 ~35ms*

</div>

<br>

## ✨ 亮点

- **25ms NPU 推理** — YOLOv26n 运行在 DeepX M1（3 核心, LPDDR5 3.92 GiB, PCIe Gen3 x4）
- **~30 FPS 端到端** — 摄像头采集、NPU 推理、NMS 后处理、实时显示，全链路仅需 15W 功耗
- **一键启动** — Gradio WebUI 在 `localhost:7860` 即开即用，无需 ML 经验
- **零拷贝流水线** — 线程化摄像头采集，帧丢弃队列保持低延迟
- **80 类检测** — 完整 COCO 标签集，逐类 NMS（IoU 0.45，置信度阈值 0.25）

## 📊 性能

> 测试条件：YOLOv26n 640×640，USB 摄像头 720p 输入，DeepX M1 NPU，Windows 11。

| 指标 | 数值 |
|:-----|-----:|
| NPU 推理延迟 | **~25 ms** |
| 端到端延迟 | **~35 ms** |
| 吞吐量 | **~30 FPS** |
| 每帧最大检测数 | 300 |
| 模型输入 | 640×640 RGB uint8 |
| 功耗 | 15W TDP |

## 🚀 快速开始

<details open>
<summary><b>环境要求</b></summary>

| 依赖项 | 版本 |
|:-------|:-----|
| 操作系统 | Windows 10 / 11 |
| 硬件 | DeepX M1 NPU（PCIe 接口） |
| 运行时 | DeepX Runtime v3.2.0+ |
| Python | 3.10+ |
| 摄像头 | USB 摄像头 |

</details>

<details open>
<summary><b>安装与运行</b></summary>

```bash
git clone https://github.com/AIwork4me/DeepXClaw.git
cd DeepXClaw

# 安装依赖
pip install uv
uv sync

# 放置编译好的 NPU 模型
cp /path/to/yolo26n-1.dxnn models/

# 配置本地路径
cp .deepxclaw.json.example .deepxclaw.json
# 编辑 .deepxclaw.json — 设置 dxrt_bin 和 model_path

# 启动
uv run deepxclaw
```

打开 **http://localhost:7860** → 点击 **Start** → 即可看到实时检测结果。

</details>

<details>
<summary><b>配置文件 (.deepxclaw.json)</b></summary>

```json
{
  "dxrt_bin": "C:/path/to/dx_rt_windows/m1/v3.2.0/dx_rt/bin",
  "model_path": "models/yolo26n-1.dxnn",
  "sdk_repo": "C:/path/to/dx_rt_windows",
  "fw_repo": "C:/path/to/dx_fw",
  "models_cdn": "https://example.com/models"
}
```

| 字段 | 说明 |
|:-----|:-----|
| `dxrt_bin` | DeepX Runtime 二进制文件路径（包含 `dxrt.dll`） |
| `model_path` | 编译后的 `.dxnn` 模型文件路径 |
| `sdk_repo` | DeepX SDK 仓库根目录 |
| `fw_repo` | DeepX 固件仓库根目录 |
| `models_cdn` | 预编译模型下载 CDN 地址 |

</details>

## 🏗️ 架构

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│ USB      │    │ 线程化       │    │ DeepX M1     │    │ 逐类         │    │ Gradio   │
│ 摄像头   │───>│ 采集         │───>│ NPU          │───>│ NMS          │───>│ WebUI    │
│          │    │ (camera.py)  │    │ (detector.py)│    │ (postproc.py)│    │ (app.py) │
└──────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘
  720p BGR       Queue(max=2)        640×640 uint8       IoU 0.45           Timer 100ms
                 丢弃旧帧            [1,300,6] 输出      置信度 ≥ 0.25       轮询渲染
```

**线程模型**：摄像头采集运行在守护线程中，使用容量为 2 的帧队列（旧帧自动丢弃）。NPU 推理运行在独立的检测工作线程中。Gradio UI 每 100ms 通过 `gr.Timer` 轮询最新的标注帧，确保流畅显示且不阻塞推理流水线。

## 📁 项目结构

<details>
<summary>点击展开</summary>

```
DeepXClaw/
├── src/deepxclaw/
│   ├── __init__.py          # 包版本号
│   ├── app.py               # Gradio WebUI + 检测工作线程
│   ├── camera.py            # 线程化 USB 摄像头采集（OpenCV）
│   ├── detector.py          # DeepX M1 NPU 推理引擎（dx_engine）
│   ├── labels.py            # COCO 80 类标签 + 颜色生成
│   └── postprocess.py       # YOLOv26n 输出解码 + NMS
├── models/                  # NPU 模型文件（.dxnn）— 已 gitignore
├── screenshots/             # 演示截图
├── .deepxclaw.json.example  # 配置模板
├── pyproject.toml           # 依赖与 CLI 入口
└── README.md
```

</details>

## 🔩 DeepX M1 NPU 规格

| 参数 | 详情 |
|:-----|:-----|
| 核心数 | 3 NPU 核心 |
| 内存 | LPDDR5 3.92 GiB |
| 接口 | PCIe Gen3 x4 |
| 功耗 | 15W TDP |
| 模型格式 | `.dxnn`（从 ONNX 编译） |

## 🤝 贡献

欢迎贡献代码！请随时提交 Issue 或 Pull Request。

1. Fork 本仓库
2. 创建功能分支（`git checkout -b feature/amazing-feature`）
3. 提交更改（`git commit -m 'Add amazing feature'`）
4. 推送分支（`git push origin feature/amazing-feature`）
5. 发起 Pull Request

## 📜 许可证

本项目采用 [MIT 许可证](LICENSE)。

## 🙏 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv26 模型架构
- [DeepX](https://www.deepx.ai/) — M1 NPU 硬件与运行时 SDK
- [Gradio](https://gradio.app/) — WebUI 框架

---

<div align="center">

**由 [OpenClaw](https://github.com/AIwork4me) 构建** — AI Agent 平台

如果这个项目对你有帮助，欢迎给个 ⭐

</div>
