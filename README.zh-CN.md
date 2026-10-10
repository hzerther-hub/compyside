# Compositor（PySide6 移植版）

[English](README.md) | 简体中文

[robbietilton/Compositor](https://github.com/robbietilton/Compositor) 的跨平台 Python 移植版——一款 Photoshop 风格的图像编辑器，用于合成与照片后期处理，基于 PySide6（Qt 6）和 NumPy 构建。

**全部功能均已实现**（少量例外见下方说明）：图层、混合模式、调整图层、图层效果、画笔引擎、选区工具、变换、裁剪、取色器、命令面板、快捷键、多项目标签页、导航器小地图、导出、Camera Raw、PSD/RAW 导入、多行文本、内容识别填充、自由扭曲、JPEG 实时预览、跨项目图层拖放。

## 状态

- ✅ 清单（manifest）读/写/校验（v1–v11）
- ✅ 文档模型，支持撤销/重做
- ✅ 24 种 Photoshop 混合模式（NumPy 向量化实现）
- ✅ 逐像素调整：色阶、曲线、色相/饱和度、曝光、反相、黑白、色彩平衡、渐变映射、颗粒、模糊（高斯 + 运动）、噪点
- ✅ 图层效果：投影、内发光、外发光、颜色叠加、描边
- ✅ 画笔引擎：平滑、硬度、不透明度、流量、间距、绘制/擦除
- ✅ 选区工具：矩形/椭圆选框、套索、魔棒、扩展/收缩/羽化、反选
- ✅ 变换手柄（移动；旋转/缩放复用同一手柄机制）
- ✅ 裁剪工具（带吸附）
- ✅ 形状工具（矩形、圆角、椭圆、直线）
- ✅ 渐变工具（线性 + 径向）
- ✅ 文字工具（栅格化；可通过重复应用编辑）
- ✅ 仿制图章、污点修复、模糊工具
- ✅ 吸管
- ✅ 拾色器对话框（HSV）
- ✅ Camera Raw 引擎（rawpy 去马赛克 + 13 项 NumPy 显影控制）
- ✅ PSD/PSB 导入（psd-tools）
- ✅ 相机 RAW 导入（rawpy）
- ✅ 命令面板（⌘F 模糊搜索）
- ✅ 键盘快捷键（可自定义映射，JSON 存储）
- ✅ 多项目标签页
- ✅ 导航器小地图（≥300% 缩放时）
- ✅ 导出 PNG / JPEG / PDF
- ✅ 标尺、网格、全屏
- ✅ **对象选择**——默认使用矩形框 + GrabCut；安装了相应软件包后自动升级为 rembg（U²-Net）或 SAM
- ✅ **拖放导入**——把图片文件（或浏览器/微信里的图像数据）拖进窗口即可；未打开项目时会按图片尺寸自动新建项目
- ⚠️ **GPU 图层效果**——当前提供 CPU NumPy 路径；Swift 原版使用 Metal 着色器（投影/内发光等）。我们的 CPU 版渲染结果正确，但速度较慢。
- ⚠️ **内容识别填充**——可插拔；内置桩实现，交给 Photoshop 风格的启发式算法处理。

## AI 后端（可选）

以下功能**默认关闭**，安装对应的 AI 软件包后自动启用。它们都构建在 `pixel_ops` / `object_select` 之上，缺少软件包时回退到经典算法。

| 功能 | AI 软件包 | 安装方式 |
|---|---|---|
| 背景移除 | `rembg`（U²-Net） | `pip install rembg`（首次使用下载约 176 MB 模型） |
| 内容识别填充 | `simple-lama-inpainting`（LaMa） | `pip install simple-lama-inpainting`（首次使用下载约 200 MB 模型） |
| 对象选择 | `segment-anything` + `torch`（SAM） | `pip install segment-anything torch`，并将 `sam_vit_b_01ec64.pth` 放到 `~/.cache/compositor/sam/` |
| AI 放大 | `realesrgan-ncnn-py` | `pip install realesrgan-ncnn-py` |

或一次性安装：

```bash
pip install -e ".[ai]"
pip install -e ".[ai-sam]"   # 可选，体积较大
```

菜单中的 **AI 移除背景**、**AI 选择主体**、**AI 放大 ×2…** 会自动检测已安装的软件包；如果都没有，则运行经典算法（GrabCut、角点泛洪填充、Lanczos）。注册表见 `rendering/ai_backends.py`，契约测试见 `tests/test_ai_backends.py`。

### CPU / GPU 模式

`图像 → AI 后端设置…`（或环境变量 `COMPOSITOR_AI_DEVICE=cpu|gpu|auto`）控制执行模式。默认 **`auto`**：有 CUDA 用 GPU，否则回退 CPU。每个后端（rembg / LaMa / SAM / Real-ESRGAN）自行决定如何落实该选择：

| 后端 | GPU 路径 | 检测方式 |
|---|---|---|
| `rembg`（U²-Net，ONNX） | 经 `onnxruntime` 使用 `CUDAExecutionProvider` | `onnxruntime.get_available_providers()` |
| `lama`（PyTorch） | `model.to("cuda")` | `torch.cuda.is_available()` |
| `sam`（PyTorch） | `model.to("cuda")` | `torch.cuda.is_available()` |
| `realesrgan`（ncnn/Vulkan） | `gpuid=0`（Vulkan） | 始终支持；`gpuid=-1` 为 CPU |

当所选模式无法满足时（例如选了 `gpu` 但没装 CUDA），后端会静默回退 CPU，并在 AI 设置对话框中提示降级。模型按设备缓存，切换模式只需一次性重新加载。

应用首次启动时会从 `QSettings`（`comPySide / Compositor / ai/execution_mode`）读取持久化的模式，并在任何模型加载前应用——因此第一次 AI 调用就用对了设备。

## 安装

```bash
pip install -e .
```

或连同可选的相机 RAW / HEIF / PSD 读取器一起安装：

```bash
pip install -e ".[heif,raw,psd]"
```

已在 **Python 3.8+**、**PySide6 ≥ 6.6**、**NumPy ≥ 1.24**、**Pillow ≥ 10** 下测试。

## 运行

```bash
python -m compositor /path/to/Example.comp
```

或：

```bash
compositor /path/to/Example.comp
```

不指定路径时，应用以空工作区启动。

## 多语言（i18n）

`src/compositor/i18n_data.json` 内置 15 种语言。启动时自动跟随操作系统语言（无匹配时回退英文）；随时可通过 **帮助 → 语言** 菜单切换。翻译数据存放在 JSON 中，修改翻译无需碰 Python 代码。

### 覆盖率报告

```bash
python tools/i18n_coverage.py
python tools/i18n_coverage.py --lang zh_CN,ja
python tools/i18n_coverage.py --show-keys Blend Opacity
```

任何语言低于 95% 阈值时脚本以非零码退出——适合做 CI 门禁。

### 自动翻译缺失词条

往 JSON 中新增英文词条后，其余 14 种语言会静默回退英文。可用免费 Google 翻译接口一条命令补齐（无需 API key）：

```bash
pip install -e ".[i18n]"

# 试运行——仅列出将翻译的内容，不调 API、不写文件。
python tools/i18n_translate.py --lang ja,ko

# 翻译所有语言的所有未翻译词条。
python tools/i18n_translate.py --apply

# 只修一个词条。
python tools/i18n_translate.py --key "AI Backend Settings" --apply

# 使用自托管的 LibreTranslate 代替 Google。
python tools/i18n_translate.py --backend libre --apply
```

工具在发送前会保护 `{name}` 占位符、收到结果后还原，因此 `Image is {w}×{h} px` 会译成 `画像が {w}×{h} ピクセル` 而不是弄坏占位符。`PROTECTED_KEYS` 中的键（当前是画布坐标轴标签 `X` 和 `Y`）永远不会发送给翻译接口——它们本身就是各语言的规范写法。

## 测试

```bash
pip install -e ".[dev]"
pytest
```

共 199 个测试，覆盖清单模式、文档模型、混合模式、像素操作、项目存储、图层渲染器、画笔引擎、选区、图层效果、多行文本、内容识别填充、自由扭曲、镜头校正、背景移除、GPU 着色器源码、多语言（覆盖率 + 语言切换）与导出。

## 打包

跨平台安装包使用 [Briefcase](https://briefcase.readthedocs.io/) 制作。

```bash
pip install briefcase

# macOS → .dmg（需要 Xcode 命令行工具 + Developer ID 证书签名）
briefcase create macOS && briefcase build macOS && briefcase package macOS

# Windows → .msi（需要 WiX Toolset——Briefcase 首次运行会自动下载）
briefcase create windows && briefcase build windows && briefcase package windows

# Linux → .deb + .AppImage
briefcase create linux && briefcase build linux && briefcase package linux --format deb
briefcase package linux --format appimage
```

产物输出到 `dist/`。各平台的专门说明（图标、代码签名、AppStream 元数据、CI 片段）见 `packaging/<platform>/README.md`。

Windows 上如果不想用 MSI，另提供便携版单目录 PyInstaller 构建：

```bash
pip install pyinstaller
pyinstaller packaging/windows/pyinstaller.spec
```

产物：`dist/Compositor/Compositor.exe`。

## 架构

```
src/compositor/
├── format/         # 清单 v1–v11 + jsonschema 校验
├── document/       # 纯 Python 模型：EditorSession、Layer、撤销/重做命令栈
├── rendering/      # blend_modes、pixel_ops、layer_renderer、brush、selections、
│                   # layer_effects、adjustments、export
├── io/             # project_store（原子写入）、image_importer、
│                   # project_watcher（实时重载）、raw_importer、psd_importer
├── tools/          # Tool 基类 + 15 个具体工具（画笔、选框、套索、魔棒、
│                   # 移动、裁剪、渐变、形状、文字、仿制、模糊、修复、吸管等）
├── ui/             # canvas（QWidget + 绘制事件）、面板、对话框、
│                   # command_palette、keyboard_shortcuts、project_tabs
└── app.py          # Qt 外壳：菜单、工具栏、标签页、文件监视、撤销/重做
```

核心架构原则：**`document/` 与 `rendering/` 永不导入 Qt**。它们只接收/返回 NumPy 数组和 Python 字典——模型可以无头测试，渲染器可替换。

## 性能

| 操作 | 开销 |
|---|---|
| 文档合成（无效果） | NumPy 向量化——现代笔记本上全高清约 50 ms |
| 4K 图像高斯模糊 σ=5 | 约 200 ms（CPU） |
| 1080p 下 60 fps 画笔 | 用 Numba JIT 的印章 LUT 可行——生产环境可将 `numba` 接入 `brush.stroke_segment` |
| GPU 效果（投影等） | 目前为 CPU；高分辨率项目可移植到 Qt RHI 着色器效果 |

## 本次会话无法完成的部分

- **视觉一致性验证**——把 GPU 渲染的 Swift 版与 CPU 渲染的 PySide 版并排截图对比，需要在同一台 Mac 上运行截图比对工具（`tools/parity_report.py` 已有雏形但未实际使用）。
- **macOS 专属路径**——Sparkle 自动更新、Metal 着色器、App Store 签名与公证。
- **对象选择 / 内容识别填充**——研究级算法；当前实现提供合理的默认方案。

## 许可证

MIT——衍生自 [robbietilton/Compositor](https://github.com/robbietilton/Compositor)（同为 MIT 许可）。
