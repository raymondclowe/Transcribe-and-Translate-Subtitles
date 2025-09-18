# 🎬 Transcribe and Translate Subtitles

<div align="center">

**A powerful, privacy-first tool for transcribing and translating video subtitles**

[![Privacy First](https://img.shields.io/badge/Privacy-100%25%20Local-green.svg)](https://github.com/your-repo)
[![ONNX Runtime](https://img.shields.io/badge/Powered%20by-ONNX%20Runtime-blue.svg)](https://onnxruntime.ai/)
[![Multi-Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](https://github.com/your-repo)

</div>

---

## 🔒 Privacy Guarantee

> **🚨 All processing runs completely offline** - No internet connection required, ensuring maximum privacy and data security.

---

## 📋 Complete Update History

### 🆕 **2025/9/19** - Major Release
- ✅ **Added ASR**: 30+ region fine-tuned Whisper models
- ✅ **Added Denoiser**: MossFormer2_SE_48K
- ✅ **Added LLM Models**:
  - Qwen3-4B-Instruct-2507-abliterated
  - Qwen3-8B-abliterated-v2
  - Hunyuan-MT-7B-abliterated
  - Seed-X-PRO-7B
- ✅ **Performance Improvements**:
  - Applied Beam Search for Whisper-like ASR models
  - Applied ONNX Runtime IOBinding for maximum speed up (10%+ faster than normal ort_session_C.run())
  - Support for 20 seconds audio segment per single run inference
  - Improved multi-threads performance
- ✅ **Hardware Support Expansion**:
  - AMD-ROCm Execution Provider
  - AMD-MIGraphX Execution Provider
  - NVIDIA TensorRTX Execution Provider
  - *(Must config the env first or it will not work)*
- ✅ **Accuracy Improvements**:
  - SenseVoice
  - Paraformer
  - FireRedASR
  - Dolphin
  - ZipEnhancer
  - MossFormerGAN_SE_16K
  - NVIDIA-NeMo-VAD
- ✅ **Speed Improvements**:
  - MelBandRoformer (speed boost by converting to mono channel)
- ❌ **Removed Models**:
  - FSMN-VAD
  - Qwen3-4B-Official
  - Qwen3-8B-Official
  - Gemma3-4B-it
  - Gemma3-12B-it
  - InternLM3
  - Phi-4-Instruct

### **2025/7/5** - Noise Reduction Enhancement
- ✅ **Added noise reduction model**: MossFormerGAN_SE_16K

### **2025/6/11** - VAD Models Expansion
- ✅ **Added VAD Models**:
  - HumAware-VAD
  - NVIDIA-NeMo-VAD
  - TEN-VAD

### **2025/6/3** - Asian Language Support
- ✅ **Added Dolphin ASR model** to support Asian languages

### **2025/5/13** - GPU Acceleration
- ✅ **Added Float16/32 ASR models** to support CUDA/DirectML GPU usage
- ✅ **GPU Performance**: These models can achieve >99% GPU operator deployment

### **2025/5/9** - Major Feature Release
- ✅ **Flexibility Improvements**:
  - Added option to **not use** VAD (Voice Activity Detection)
- ✅ **Added Models**:
  - **Noise reduction**: MelBandRoformer
  - **ASR**: CrisperWhisper
  - **ASR**: Whisper-Large-v3.5-Distil (English fine-tuned)
  - **ASR**: FireRedASR-AED-L (Chinese + dialects support)
  - **Three Japanese anime fine-tuned Whisper models**
- ✅ **Performance Optimizations**:
  - Removed IPEX-LLM framework to enhance overall performance
  - Cancelled LLM quantization options, standardized on **Q4F32** format
  - Improved **Whisper** series inference speed by over 10%
- ✅ **Accuracy Improvements**:
  - Improved **FSMN-VAD** accuracy
  - Improved **Paraformer** recognition accuracy
  - Improved **SenseVoice** recognition accuracy
- ✅ **LLM Support with ONNX Runtime 100% GPU operator deployment**:
  - Qwen3-4B/8B
  - InternLM3-8B
  - Phi-4-mini-Instruct
  - Gemma3-4B/12B-it
- ✅ **Hardware Support Expansion**:
  - **Intel OpenVINO**
  - **NVIDIA CUDA GPU**
  - **Windows DirectML GPU** (supports integrated and discrete GPUs)

---

## 🚀 Quick Start

### Prerequisites
```bash
# Install FFmpeg
conda install ffmpeg

# Install Python dependencies
pip install -r requirements.txt
```

### Setup
1. **Download Models**: Get the required models from [HuggingFace](https://huggingface.co/H5N1AIDS/Transcribe_and_Translate_Subtitles)
2. **Download Script**: Place `run.py` in your `Transcribe_and_Translate_Subtitles` folder
3. **Add Media**: Place your videos in `Transcribe_and_Translate_Subtitles/Media/`
4. **Run**: Execute `python run.py` and open the web interface

### Results
Find your processed subtitles in:
```
Transcribe_and_Translate_Subtitles/Results/Subtitles/
```

---

## ✨ Features

### 🔇 Noise Reduction Models
- **[DFSMN](https://modelscope.cn/models/iic/speech_dfsmn_ans_psm_48k_causal)** - High-quality denoising
- **[GTCRN](https://github.com/Xiaobin-Rong/gtcrn)** - Real-time noise suppression
- **[ZipEnhancer](https://modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base)** - Advanced enhancement
- **[Mel-Band-Roformer](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model)** - Vocal isolation
- **[MossFormerGAN_SE_16K](https://www.modelscope.cn/models/alibabasglab/MossFormerGAN_SE_16K)** - 16kHz enhancement
- **[MossFormer2_SE_48K](https://www.modelscope.cn/models/alibabasglab/MossFormer2_SE_48K)** - 48kHz enhancement

### 🎤 Voice Activity Detection (VAD)
- **[Faster_Whisper-Silero](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/vad.py)** - Fast and accurate
- **[Official-Silero-v6](https://github.com/snakers4/silero-vad)** - Official implementation
- **[HumAware](https://huggingface.co/CuriousMonkey7/HumAware-VAD)** - Human-aware detection
- **[NVIDIA-NeMo-VAD-v2.0](https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0)** - Multilingual support
- **[TEN-VAD](https://github.com/TEN-framework/ten-vad)** - Lightweight detection
- **[Pyannote-Segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)** - Advanced segmentation
  - *Note: You need to accept Pyannote's terms of use and download the Pyannote `pytorch_model.bin` file. Place it in the `VAD/pyannote_segmentation` folder.*

### 🗣️ Speech Recognition (ASR)
#### Multilingual Models
- **[SenseVoice-Small](https://modelscope.cn/models/iic/SenseVoiceSmall)** - Compact multilingual
- **[Whisper-Large-V3](https://huggingface.co/openai/whisper-large-v3)** - State-of-the-art accuracy
- **[Whisper-Large-V3-Turbo](https://huggingface.co/openai/whisper-large-v3-turbo)** - Speed optimized
- **[CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper)** - Enhanced clarity
- **[Dolphin-Small](https://github.com/DataoceanAI/Dolphin)** - Asian language support

#### Chinese Models
- **[Paraformer-Small-Chinese](https://modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1)** - Compact Chinese
- **[Paraformer-Large-Chinese](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch)** - Advanced Chinese
- **[FireRedASR-AED-L](https://github.com/FireRedTeam/FireRedASR)** - Chinese dialects support

#### English Models
- **[Paraformer-Large-English](https://modelscope.cn/models/iic/speech_paraformer_asr-en-16k-vocab4199-pytorch)** - English specialized
- **[Whisper-Large-v3.5-Distil](https://huggingface.co/distil-whisper/distil-large-v3.5)** - Distilled efficiency

#### Japanese Models
- **[Whisper-Large-V3-Turbo-Japanese](https://huggingface.co/hhim8826/whisper-large-v3-turbo-ja)** - Japanese optimized
- **[Whisper-Large-V3-Anime-A](https://huggingface.co/efwkjn/whisper-ja-anime-v0.1)** - Anime specialized
- **[Whisper-Large-V3-Anime-B](https://huggingface.co/litagin/anime-whisper)** - Alternative anime model

### 🤖 Translation Models (LLM)
- **[Qwen-3-4B-Instruct-2507](https://huggingface.co/huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated)** - Efficient instruction following
- **[Qwen-3-8B-Fine_Tuned](https://huggingface.co/huihui-ai/Huihui-Qwen3-8B-abliterated-v2)** - Enhanced 8B variant
- **[Hunyuan-MT-7B](https://www.modelscope.cn/models/Tencent-Hunyuan/Hunyuan-MT-7B)** - Machine translation specialist
- **[Seed-X-PRO-7B](https://www.modelscope.cn/models/ByteDance-Seed/Seed-X-PPO-7B)** - Advanced reasoning

---

## 🖥️ Hardware Support

<table>
<tr>
<td align="center"><strong>💻 CPU</strong></td>
<td align="center"><strong>🎮 GPU</strong></td>
<td align="center"><strong>🧠 Specialized</strong></td>
</tr>
<tr>
<td>Intel • AMD • Apple Silicon</td>
<td>NVIDIA CUDA • AMD ROCm • DirectML</td>
<td>Intel OpenVINO • TensorRT • MIGraphX</td>
</tr>
</table>

**Currently Supported Platforms:**
- **Intel-OpenVINO-CPU-GPU-NPU**
- **Windows-AMD-GPU**
- **NVIDIA-GPU**
- **Apple-CPU**
- **AMD-CPU**

---

## 📊 Performance Benchmarks

*Test conditions: Ubuntu 24.04, Intel i3-12300, 7602-second video*

| OS | Backend | Denoiser | VAD | ASR | LLM | Real-Time Factor |
|:--:|:-------:|:--------:|:---:|:---:|:---:|:----------------:|
| Ubuntu-24.04 | CPU i3-12300 | - | Silero | SenseVoiceSmall | - | **0.08** |
| Ubuntu-24.04 | CPU i3-12300 | GTCRN | Silero | SenseVoiceSmall | Qwen2.5-7B-Instruct | **0.50** |
| Ubuntu-24.04 | CPU i3-12300 | GTCRN | FSMN | SenseVoiceSmall | - | **0.054** |
| Ubuntu-24.04 | CPU i3-12300 | ZipEnhancer | FSMN | SenseVoiceSmall | - | **0.39** |
| Ubuntu-24.04 | CPU i3-12300 | GTCRN | Silero | Whisper-Large-V3 | - | **0.20** |
| Ubuntu-24.04 | CPU i3-12300 | GTCRN | FSMN | Whisper-Large-V3-Turbo | - | **0.148** |

---

## 🛠️ Troubleshooting

### Common Issues
- **Silero VAD Error**: Simply restart the application on first run
- **libc++ Error** (Linux):
  ```bash
  sudo apt update
  sudo apt install libc++1
  ```
- **Apple Silicon**: Avoid installing `onnxruntime-openvino` as it will cause errors

---

## 🗺️ Roadmap

- [ ] **[Video Upscaling](https://github.com/sczhou/Upscale-A-Video)** - Enhance resolution
- [ ] **Real-time Player** - Live transcription and translation

---

<div align="center">

**Ready to get started?** 🎉

[Download Models](https://huggingface.co/H5N1AIDS/Transcribe_and_Translate_Subtitles) • [View Documentation](https://github.com/your-repo) • [Report Issues](https://github.com/your-repo/issues)

</div>

---
# 🎬 视频字幕转录和翻译工具

<div align="center">

**强大的隐私优先视频字幕转录翻译工具**

[![隐私优先](https://img.shields.io/badge/隐私-100%25%20本地-green.svg)](https://github.com/your-repo)
[![ONNX Runtime](https://img.shields.io/badge/基于-ONNX%20Runtime-blue.svg)](https://onnxruntime.ai/)
[![多平台](https://img.shields.io/badge/平台-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](https://github.com/your-repo)

</div>

---

## 🔒 隐私保障

> **🚨 所有处理完全离线运行** - 无需互联网连接，确保最大隐私和数据安全。

---

## 📋 完整更新历史

### 🆕 **2025/9/19** - 重大版本发布
- ✅ **新增 ASR**: 30+ 地区微调 Whisper 模型
- ✅ **新增降噪器**: MossFormer2_SE_48K
- ✅ **新增大语言模型**:
  - Qwen3-4B-Instruct-2507-abliterated
  - Qwen3-8B-abliterated-v2
  - Hunyuan-MT-7B-abliterated
  - Seed-X-PRO-7B
- ✅ **性能改进**:
  - 为 Whisper 类型 ASR 模型应用 Beam Search
  - 应用 ONNX Runtime IOBinding 实现最大速度提升（比普通 ort_session_C.run() 快 10%+）
  - 支持单次推理运行 20 秒音频片段
  - 改进多线程性能
- ✅ **硬件支持扩展**:
  - AMD-ROCm 执行提供器
  - AMD-MIGraphX 执行提供器
  - NVIDIA TensorRTX 执行提供器
  - *（必须先配置环境，否则无法工作）*
- ✅ **准确性改进**:
  - SenseVoice
  - Paraformer
  - FireRedASR
  - Dolphin
  - ZipEnhancer
  - MossFormerGAN_SE_16K
  - NVIDIA-NeMo-VAD
- ✅ **速度改进**:
  - MelBandRoformer（通过转换为单声道实现速度提升）
- ❌ **移除的模型**:
  - FSMN-VAD
  - Qwen3-4B-Official
  - Qwen3-8B-Official
  - Gemma3-4B-it
  - Gemma3-12B-it
  - InternLM3
  - Phi-4-Instruct

### **2025/7/5** - 降噪增强
- ✅ **新增降噪模型**: MossFormerGAN_SE_16K

### **2025/6/11** - VAD 模型扩展
- ✅ **新增 VAD 模型**:
  - HumAware-VAD
  - NVIDIA-NeMo-VAD
  - TEN-VAD

### **2025/6/3** - 亚洲语言支持
- ✅ **新增 Dolphin ASR 模型** 以支持亚洲语言

### **2025/5/13** - GPU 加速
- ✅ **新增 Float16/32 ASR 模型** 以支持 CUDA/DirectML GPU 使用
- ✅ **GPU 性能**: 这些模型可实现 >99% GPU 算子部署

### **2025/5/9** - 主要功能发布
- ✅ **灵活性改进**:
  - 新增选项可以**不使用** VAD（语音活动检测）
- ✅ **新增模型**:
  - **降噪**: MelBandRoformer
  - **ASR**: CrisperWhisper
  - **ASR**: Whisper-Large-v3.5-Distil（英语微调）
  - **ASR**: FireRedASR-AED-L（中文+方言支持）
  - **三个日语动漫微调 Whisper 模型**
- ✅ **性能优化**:
  - 移除 IPEX-LLM 框架以提高整体性能
  - 取消 LLM 量化选项，标准化为 **Q4F32** 格式
  - **Whisper** 系列推理速度提升超过 10%
- ✅ **准确性改进**:
  - 改进 **FSMN-VAD** 准确性
  - 改进 **Paraformer** 识别准确性
  - 改进 **SenseVoice** 识别准确性
- ✅ **支持 ONNX Runtime 100% GPU 算子部署的 LLM**:
  - Qwen3-4B/8B
  - InternLM3-8B
  - Phi-4-mini-Instruct
  - Gemma3-4B/12B-it
- ✅ **硬件支持扩展**:
  - **Intel OpenVINO**
  - **NVIDIA CUDA GPU**
  - **Windows DirectML GPU**（支持集成和独立 GPU）

---

## 🚀 快速开始

### 前置要求
```bash
# 安装 FFmpeg
conda install ffmpeg

# 安装 Python 依赖
pip install -r requirements.txt
```

### 设置步骤
1. **下载模型**: 从 [HuggingFace](https://huggingface.co/H5N1AIDS/Transcribe_and_Translate_Subtitles) 获取所需模型
2. **下载脚本**: 将 `run.py` 放入您的 `Transcribe_and_Translate_Subtitles` 文件夹
3. **添加媒体**: 将您的视频放入 `Transcribe_and_Translate_Subtitles/Media/`
4. **运行**: 执行 `python run.py` 并打开网页界面

### 结果输出
在以下位置找到您处理的字幕:
```
Transcribe_and_Translate_Subtitles/Results/Subtitles/
```

---

## ✨ 功能特性

### 🔇 降噪模型
- **[DFSMN](https://modelscope.cn/models/iic/speech_dfsmn_ans_psm_48k_causal)** - 高质量降噪
- **[GTCRN](https://github.com/Xiaobin-Rong/gtcrn)** - 实时噪声抑制
- **[ZipEnhancer](https://modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base)** - 高级增强
- **[Mel-Band-Roformer](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model)** - 人声分离
- **[MossFormerGAN_SE_16K](https://www.modelscope.cn/models/alibabasglab/MossFormerGAN_SE_16K)** - 16kHz 增强
- **[MossFormer2_SE_48K](https://www.modelscope.cn/models/alibabasglab/MossFormer2_SE_48K)** - 48kHz 增强

### 🎤 语音活动检测 (VAD)
- **[Faster_Whisper-Silero](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/vad.py)** - 快速准确
- **[Official-Silero-v6](https://github.com/snakers4/silero-vad)** - 官方实现
- **[HumAware](https://huggingface.co/CuriousMonkey7/HumAware-VAD)** - 人类感知检测
- **[NVIDIA-NeMo-VAD-v2.0](https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0)** - 多语言支持
- **[TEN-VAD](https://github.com/TEN-framework/ten-vad)** - 轻量级检测
- **[Pyannote-Segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)** - 高级分割
  - *注意：您需要接受 Pyannote 的使用条款并下载 Pyannote `pytorch_model.bin` 文件。将其放置在 `VAD/pyannote_segmentation` 文件夹中。*

### 🗣️ 语音识别 (ASR)
#### 多语言模型
- **[SenseVoice-Small](https://modelscope.cn/models/iic/SenseVoiceSmall)** - 紧凑多语言
- **[Whisper-Large-V3](https://huggingface.co/openai/whisper-large-v3)** - 最先进准确性
- **[Whisper-Large-V3-Turbo](https://huggingface.co/openai/whisper-large-v3-turbo)** - 速度优化
- **[CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper)** - 增强清晰度
- **[Dolphin-Small](https://github.com/DataoceanAI/Dolphin)** - 亚洲语言支持

#### 中文模型
- **[Paraformer-Small-Chinese](https://modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1)** - 紧凑中文
- **[Paraformer-Large-Chinese](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch)** - 高级中文
- **[FireRedASR-AED-L](https://github.com/FireRedTeam/FireRedASR)** - 中文方言支持

#### 英文模型
- **[Paraformer-Large-English](https://modelscope.cn/models/iic/speech_paraformer_asr-en-16k-vocab4199-pytorch)** - 英语专用
- **[Whisper-Large-v3.5-Distil](https://huggingface.co/distil-whisper/distil-large-v3.5)** - 蒸馏效率

#### 日语模型
- **[Whisper-Large-V3-Turbo-Japanese](https://huggingface.co/hhim8826/whisper-large-v3-turbo-ja)** - 日语优化
- **[Whisper-Large-V3-Anime-A](https://huggingface.co/efwkjn/whisper-ja-anime-v0.1)** - 动漫专用
- **[Whisper-Large-V3-Anime-B](https://huggingface.co/litagin/anime-whisper)** - 替代动漫模型

### 🤖 翻译模型 (LLM)
- **[Qwen-3-4B-Instruct-2507](https://huggingface.co/huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated)** - 高效指令遵循
- **[Qwen-3-8B-Fine_Tuned](https://huggingface.co/huihui-ai/Huihui-Qwen3-8B-abliterated-v2)** - 增强 8B 变体
- **[Hunyuan-MT-7B](https://www.modelscope.cn/models/Tencent-Hunyuan/Hunyuan-MT-7B)** - 机器翻译专家
- **[Seed-X-PRO-7B](https://www.modelscope.cn/models/ByteDance-Seed/Seed-X-PPO-7B)** - 高级推理

---

## 🖥️ 硬件支持

<table>
<tr>
<td align="center"><strong>💻 CPU</strong></td>
<td align="center"><strong>🎮 GPU</strong></td>
<td align="center"><strong>🧠 专用硬件</strong></td>
</tr>
<tr>
<td>Intel • AMD • Apple Silicon</td>
<td>NVIDIA CUDA • AMD ROCm • DirectML</td>
<td>Intel OpenVINO • TensorRT • MIGraphX</td>
</tr>
</table>

**当前支持的平台:**
- **Intel-OpenVINO-CPU-GPU-NPU**
- **Windows-AMD-GPU**
- **NVIDIA-GPU**
- **Apple-CPU**
- **AMD-CPU**

---

## 📊 性能基准测试

*测试条件: Ubuntu 24.04, Intel i3-12300, 7602 秒视频*

| 操作系统 | 后端 | 降噪器 | VAD | ASR | LLM | 实时因子 |
|:--:|:-------:|:--------:|:---:|:---:|:---:|:----------------:|
| Ubuntu-24.04 | CPU i3-12300 | - | Silero | SenseVoiceSmall | - | **0.08** |
| Ubuntu-24.04 | CPU i3-12300 | GTCRN | Silero | SenseVoiceSmall | Qwen2.5-7B-Instruct | **0.50** |
| Ubuntu-24.04 | CPU i3-12300 | GTCRN | FSMN | SenseVoiceSmall | - | **0.054** |
| Ubuntu-24.04 | CPU i3-12300 | ZipEnhancer | FSMN | SenseVoiceSmall | - | **0.39** |
| Ubuntu-24.04 | CPU i3-12300 | GTCRN | Silero | Whisper-Large-V3 | - | **0.20** |
| Ubuntu-24.04 | CPU i3-12300 | GTCRN | FSMN | Whisper-Large-V3-Turbo | - | **0.148** |

---

## 🛠️ 故障排除

### 常见问题
- **Silero VAD 错误**: 首次运行时简单重启应用程序即可
- **libc++ 错误** (Linux):
  ```bash
  sudo apt update
  sudo apt install libc++1
  ```
- **Apple Silicon**: 避免安装 `onnxruntime-openvino`，因为它会导致错误

---

## 🗺️ 路线图

- [ ] **[视频升级](https://github.com/sczhou/Upscale-A-Video)** - 提升分辨率
- [ ] **实时播放器** - 实时转录和翻译

---

<div align="center">

**准备开始了吗？** 🎉

[下载模型](https://huggingface.co/H5N1AIDS/Transcribe_and_Translate_Subtitles) • [查看文档](https://github.com/your-repo) • [报告问题](https://github.com/your-repo/issues)

</div>
