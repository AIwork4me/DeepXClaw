<div align="center">

# DeepXClaw

### 零行代码。插上硬件。AI 搞定一切。

<br>

<img src="https://img.shields.io/badge/人工代码-0_行-blueviolet?style=for-the-badge" alt="0 行代码">
<img src="https://img.shields.io/badge/构建者-OpenClaw_Agent-FF6F00?style=for-the-badge" alt="OpenClaw">
<img src="https://img.shields.io/badge/效果-30_FPS_检测-brightgreen?style=for-the-badge" alt="30 FPS">

<br><br>

[![License](https://img.shields.io/github/license/AIwork4me/DeepXClaw?color=blue)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=gold)](https://www.python.org/)
[![Stars](https://img.shields.io/github/stars/AIwork4me/DeepXClaw?style=social)](https://github.com/AIwork4me/DeepXClaw/stargazers)

<br>

[English](README.md) | **中文**

<br>

<img width="80%" src="screenshots/20260324-191357.png" alt="DeepXClaw — 完全由 OpenClaw AI Agent 构建的实时目标检测应用">

<br>

<em>这个应用的每一行代码、每一个配置、每一个架构决策，全部由 <a href="https://github.com/AIwork4me">OpenClaw</a> AI Agent 完成。人类只是插上了硬件。</em>

</div>

<br>

## 🤯 发生了什么？

一个用户把 [DeepX M1 NPU](https://www.deepx.ai/) 插到了电脑上。就这样。这就是他做的全部事情。

**[OpenClaw](https://github.com/AIwork4me)** — 一个 AI Agent — 接管了一切，自主完成了：

| 步骤 | OpenClaw 做了什么 | 人工操作 |
|:-----|:-------------------|:---------|
| 1 | 安装 DeepX NPU 驱动和运行时 SDK | 无 |
| 2 | 更新 NPU 固件到最新版本 | 无 |
| 3 | 为硬件选择最优模型 YOLOv26n | 无 |
| 4 | 将模型编译为 NPU 专用的 `.dxnn` 格式 | 无 |
| 5 | 编写完整推理流水线（摄像头、预处理、NPU 推理、NMS、可视化） | 无 |
| 6 | 构建 Gradio WebUI 实时流式界面 | 无 |
| 7 | 调试、测试、交付应用 | 无 |

**结果**：一个生产就绪的、30 FPS 实时目标检测应用。人工编写代码零行。

## ✨ 为什么这很重要

大多数 AI 硬件的 Demo 需要花几天时间手动搞定：翻 SDK 文档、折腾驱动、写推理样板代码、调试线程问题、搭 UI……

**OpenClaw 消除了这一切。** 它是一个端到端的 AI Agent，填平了"我有一块芯片"到"我有一个可用产品"之间的鸿沟。

```
 你要做的：                      OpenClaw 做的：
 ┌─────────────────┐             ┌─────────────────────────────────────┐
 │                 │             │  安装驱动                           │
 │  把 DeepX M1    │             │  更新固件                           │
 │  NPU 插到电脑上  │────────>   │  选择并编译模型                     │
 │                 │             │  编写应用代码                       │
 │  （完事了。）    │             │  构建 WebUI                         │
 │                 │             │  测试并交付                         │
 └─────────────────┘             └─────────────────────────────────────┘
```

> **这个仓库就是证明。** 你看到的每一个 `.py` 文件都由 OpenClaw 生成。人类贡献了零行应用代码。

## 📊 OpenClaw 构建的成果

一个完整的实时目标检测系统：

| 指标 | 数值 |
|:-----|-----:|
| NPU 推理延迟 | **~25 ms** |
| 端到端延迟 | **~35 ms** |
| 吞吐量 | **~30 FPS** |
| 检测类别数 | 80（COCO） |
| 功耗 | 15W TDP |
| 人工编写代码 | **0 行** |

## 🚀 亲自试试

<details open>
<summary><b>前提条件</b></summary>

| 你提供的 | OpenClaw 搞定的 |
|:---------|:----------------|
| Windows 10/11 电脑 | 驱动安装 |
| DeepX M1 NPU（PCIe） | 固件更新 |
| USB 摄像头 | 其他所有事情 |

</details>

<details open>
<summary><b>运行</b></summary>

```bash
git clone https://github.com/AIwork4me/DeepXClaw.git
cd DeepXClaw

pip install uv
uv sync

cp .deepxclaw.json.example .deepxclaw.json
# 编辑 .deepxclaw.json — 设置 dxrt_bin 和 model_path

uv run deepxclaw
```

打开 **http://localhost:7860** → 点击 **Start** → 实时检测已就绪。

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
| `dxrt_bin` | DeepX Runtime 二进制文件路径 |
| `model_path` | 编译后的 `.dxnn` 模型文件路径 |
| `sdk_repo` | DeepX SDK 仓库根目录 |
| `fw_repo` | DeepX 固件仓库根目录 |
| `models_cdn` | 预编译模型下载 CDN 地址 |

</details>

## 🏗️ 架构（由 OpenClaw 设计）

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│ USB      │    │ 线程化       │    │ DeepX M1     │    │ 逐类         │    │ Gradio   │
│ 摄像头   │───>│ 采集         │───>│ NPU          │───>│ NMS          │───>│ WebUI    │
│          │    │ (camera.py)  │    │ (detector.py)│    │ (postproc.py)│    │ (app.py) │
└──────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘
  720p BGR       Queue(max=2)        640×640 uint8       IoU 0.45           Timer 100ms
                 丢弃旧帧            [1,300,6] 输出      置信度 ≥ 0.25       轮询渲染
```

OpenClaw 选择了多线程架构：摄像头采集在守护线程中运行（帧丢弃队列），NPU 推理在独立工作线程中，Gradio 以 100ms 间隔轮询。这种设计在保持 UI 响应的同时最大化吞吐量——这是 Agent 根据硬件约束自主做出的决策。

<details>
<summary><b>📁 项目结构</b>（全部由 OpenClaw 生成）</summary>

```
DeepXClaw/
├── src/deepxclaw/
│   ├── app.py               # Gradio WebUI + 检测工作线程
│   ├── camera.py            # 线程化 USB 摄像头采集（OpenCV）
│   ├── detector.py          # DeepX M1 NPU 推理引擎（dx_engine）
│   ├── labels.py            # COCO 80 类标签 + 颜色生成
│   └── postprocess.py       # YOLOv26n 输出解码 + NMS
├── models/                  # NPU 模型文件（.dxnn）
├── pyproject.toml           # 依赖与 CLI 入口
└── README.md
```

</details>

## 🔩 硬件：DeepX M1 NPU

| 参数 | 详情 |
|:-----|:-----|
| 核心数 | 3 NPU 核心 |
| 内存 | LPDDR5 3.92 GiB |
| 接口 | PCIe Gen3 x4 |
| 功耗 | 15W TDP |
| 模型格式 | `.dxnn`（从 ONNX 编译） |

## 🤝 贡献

欢迎贡献！请随时提交 Issue 或 Pull Request。

## 📜 许可证

[MIT 许可证](LICENSE)

## 🙏 致谢

- [DeepX](https://www.deepx.ai/) — M1 NPU 硬件与运行时 SDK
- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv26 模型架构
- [Gradio](https://gradio.app/) — WebUI 框架

---

<div align="center">

**完全由 [OpenClaw](https://github.com/AIwork4me) 构建** — 把硬件变成产品的 AI Agent。

*插上芯片。获得应用。无需代码。*

如果这改变了你对 AI 硬件开发的看法，给个 ⭐

</div>
