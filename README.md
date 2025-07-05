# Transcribe and Translate Subtitles

## 🚨 Important Note
- **Every task runs locally without internet, ensuring maximum privacy.**

---
## Updates
- 2025/7/5
    - Added a noise reduction model: MossFormerGAN_SE_16K
- 2025/6/11
    - Added HumAware-VAD, NVIDIA-NeMo-VAD, TEN-VAD
- 2025/6/3
    - Added Dolphin ASR model to support Asian languages.
- 2025/5/13
    - Added Float16/32 ASR models to support CUDA/DirectML GPU usage. These models can achieve >99% GPU operator deployment.
- 2025/5/9
    - Added an option to **not use** VAD (Voice Activity Detection), offering greater flexibility.
    - Added a noise reduction model: **MelBandRoformer**.
    - Added three Japanese anime fine-tuned Whisper models.
    - Added ASR model: **CrisperWhisper**.
    - Added English fine-tuned ASR model: **Whisper-Large-v3.5-Distil**.
    - Added ASR model supporting Chinese (including some dialects): **FireRedASR-AED-L**.
    - Removed the IPEX-LLM framework to enhance overall performance.
    - Cancelled LLM quantization options, standardizing on the **Q4F32** format.
    - Improved accuracy of **FSMN-VAD**.
    - Improved recognition accuracy of **Paraformer**.
    - Improved recognition accuracy of **SenseVoice**.
    - Improved inference speed of the **Whisper** series by over 10%.
    - Supported the following large language models (LLMs) with **ONNX Runtime 100% GPU operator deployment**:
        - Qwen3-4B/8B
        - InternLM3-8B
        - Phi-4-mini-Instruct
        - Gemma3-4B/12B-it
    - Expanded hardware support:
        - **Intel OpenVINO**
        - **NVIDIA CUDA GPU**
        - **Windows DirectML GPU** (supports integrated and discrete GPUs)  
---

## ✨ Features  
This project is built on ONNX Runtime framework.
- Deoiser Support:
  - [DFSMN](https://modelscope.cn/models/iic/speech_dfsmn_ans_psm_48k_causal)
  - [GTCRN](https://github.com/Xiaobin-Rong/gtcrn)
  - [ZipEnhancer](https://modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base)
  - [Mel-Band-Roformer](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model)
  - [MossFormerGAN_SE_16K](https://www.modelscope.cn/models/alibabasglab/MossFormerGAN_SE_16K)

- VAD Support:
  - [FSMN](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch)
  - [Faster_Whisper - Silero](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/vad.py)
  - [Official - Silero](https://github.com/snakers4/silero-vad)
  - [HumAware](https://huggingface.co/CuriousMonkey7/HumAware-VAD)
  - [NVIDIA-NeMo-VAD-v2.0](https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0)
  - [TEN-VAD](https://github.com/TEN-framework/ten-vad)
  - [Pyannote-Segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
    - You need to accept Pyannote's terms of use and download the Pyannote `pytorch_model.bin` file. Next, place it in the `VAD/pyannote_segmentation` folder.

- ASR Support:
  - [SenseVoice-Small](https://modelscope.cn/models/iic/SenseVoiceSmall)
  - [Paraformer-Small-Chinese](https://modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1)
  - [Paraformer-Large-Chinese](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch)
  - [Paraformer-Large-English](https://modelscope.cn/models/iic/speech_paraformer_asr-en-16k-vocab4199-pytorch)
  - [Whisper-Large-V3](https://huggingface.co/openai/whisper-large-v3)
  - [Whisper-Large-V3-Turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
  - [Whisper-Large-V3-Turbo-Japanese](https://huggingface.co/hhim8826/whisper-large-v3-turbo-ja)
  - [Whisper-Large-V3-Anime-A](https://huggingface.co/efwkjn/whisper-ja-anime-v0.1)
  - [Whisper-Large-V3-Anime-B](https://huggingface.co/litagin/anime-whisper)
  - [Whisper-Large-v3.5-Distil](https://huggingface.co/distil-whisper/distil-large-v3.5)
  - [CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper)
  - [FireRedASR-AED-L](https://github.com/FireRedTeam/FireRedASR)
  - [Dolphin-Small](https://github.com/DataoceanAI/Dolphin)

- LLM Supports: 
  - Qwen-3: [4B](https://modelscope.cn/models/Qwen/Qwen3-4B), [8B](https://modelscope.cn/models/Qwen/Qwen3-8B)
  - InternLM-3: [8B](https://huggingface.co/internlm/internlm3-8b-instruct)
  - Gemma-3-it: [4B](https://huggingface.co/google/gemma-3-4b-it), [12B](https://huggingface.co/google/gemma-3-12b-it)  
  - Phi-4-Instruct: [mini](https://huggingface.co/microsoft/Phi-4-mini-instruct)
---

## 📋 Setup Instructions

### ✅ Step 1: Install Dependencies
- Run the following command in your terminal to install the latest required Python packages:
- For Apple Silicon M-series chips, avoid installing `onnxruntime-openvino`, as it will cause errors.
```bash
conda install ffmpeg

pip install -r requirements.txt
```

### 📥 Step 2: Download Necessary Models
- Download the required models from HuggingFace: [Transcribe_and_Translate_Subtitles](https://huggingface.co/H5N1AIDS/Transcribe_and_Translate_Subtitles).


### 🖥️ Step 3: Download and Place `run.py`
- Download the `run.py` script from this repository.
- Place it in the `Transcribe_and_Translate_Subtitles` folder.

### 📁 Step 4: Place Target Videos in the Media Folder
- Place the videos you want to transcribe and translate in the following directory. The application will process the videos one by one.:
```
Transcribe_and_Translate_Subtitles/Media
```

### 🚀 Step 5: Run the Application
- Open your preferred terminal (PyCharm, CMD, PowerShell, etc.).
- Execute the following command to start the application:
```bash
python run.py
```
- Once the application starts, you will see a webpage open in your browser.
   ![screenshot](https://github.com/DakeQQ/Transcribe-and-Translate-Subtitles/blob/main/screen/Screenshot%20from%202025-05-08%2013-01-17.png)

### 🛠️ Step 6: Fix Error (if encountered)
- On the first run, you might encounter a **Silero-VAD error**. Simply restart the application, and it should be resolved.
- On the first run, you might encounter a **libc++1.so error**. Run the following commands in the terminal, and they should resolve the issue.
```bash
sudo apt update
sudo apt install libc++1
```

### 💻 Step 7: Device Support
- This project currently supports:
  - **Intel-OpenVINO-CPU-GPU-NPU**
  - **Windows-AMD-GPU**
  - **NVIDIA-GPU**
  - **Apple-CPU**
  - **AMD-CPU**

---

## 🎉 Enjoy the Application!
```
Transcribe_and_Translate_Subtitles/Results/Subtitles
```
---

## 📌 To-Do List
- [ ] [LLM-MiniCPM4](https://github.com/OpenBMB/MiniCPM)
- [ ] [Denoiser-MossFormer2-48K](https://www.modelscope.cn/models/alibabasglab/MossFormer2_SE_48K)
- [ ] [Upscale the Resolution of Audio-MossFormer2-48K](https://www.modelscope.cn/models/alibabasglab/MossFormer2_SR_48K)
- [ ] [Upscale the Resolution of Video](https://github.com/sczhou/Upscale-A-Video)
- [ ] AMD-ROCm Support
- [ ] Real-Time Translate & Trascribe Video Player

---
### 性能 Performance  
| OS           | Backend           | Denoiser          | VAD                  | ASR                  | LLM | Real-Time Factor<br>test_video.mp4<br>7602 seconds |
|:------------:|:-----------------:|:-----------------:|:--------------------:|:--------------------:|:----------------:|:----------------:|
| Ubuntu-24.04 | CPU <br> i3-12300 | -                 | Silero               | SenseVoiceSmall      |        -      |        0.08      |
| Ubuntu-24.04 | CPU <br> i3-12300 | GTCRN             | Silero               | SenseVoiceSmall      | Qwen2.5-7B-Instruct | 0.50       |
| Ubuntu-24.04 | CPU <br> i3-12300 | GTCRN             | FSMN                 | SenseVoiceSmall      |        -      |       0.054      |
| Ubuntu-24.04 | CPU <br> i3-12300 | ZipEnhancer       | FSMN                 | SenseVoiceSmall      |        -      |        0.39      |
| Ubuntu-24.04 | CPU <br> i3-12300 | GTCRN             | Silero               | Whisper-Large-V3     |        -      |        0.20      |
| Ubuntu-24.04 | CPU <br> i3-12300 | GTCRN             | FSMN                 | Whisper-Large-V3-Turbo |      -      |        0.148     |

---
# 转录和翻译字幕

## 🚨 重要提示  
- **所有任务均在本地运行，无需连接互联网，确保最大程度的隐私保护。**

---
## 最近更新与功能
- 2025/7/5
    - 新增 降噪 MossFormerGAN_SE_16K
- 2025/6/11
    - 新增 HumAware-VAD, NVIDIA-NeMo-VAD, TEN-VAD。
- 2025/6/3
    - 新增 Dolphin ASR 模型以支持亚洲语言。
- 2025/5/13
    - 新增 Float16/32 ASR 模型，支持 CUDA/DirectML GPU 使用。这些模型可实现 >99% 的 GPU 算子部署率。
- 2025/5/9
    - 新增 **不使用** VAD（语音活动检测）的选项，提供更多灵活性。
    - 新增降噪模型：**MelBandRoformer**。
    - 新增三款日语动漫微调Whisper模型。
    - 新增ASR模型：**CrisperWhisper**。
    - 新增英语微调ASR模型：**Whisper-Large-v3.5-Distil**。
    - 新增支持中文（包括部分方言）的ASR模型：**FireRedASR-AED-L**。
    - 移除IPEX-LLM框架，提升整体性能。
    - 取消LLM量化选项，统一采用**Q4F32**格式。
    - 改进了**FSMN-VAD**的准确率。
    - 改进了**Paraformer**的识别准确率。
    - 改进了**SenseVoice**的识别准确率。
    - 改进了**Whisper**系列的推理速度10%+。
    - 支持以下大语言模型（LLM），实现**ONNX Runtime 100% GPU算子部署**：
        - Qwen3-4B/8B  
        - InternLM3-8B  
        - Phi-4-mini-Instruct  
        - Gemma3-4B/12B-it
    - 扩展硬件支持：  
        - **Intel OpenVINO**  
        - **NVIDIA CUDA GPU**  
        - **Windows DirectML GPU**（支持集成显卡和独立显卡）
---

## ✨ 功能
这个项目基于 ONNX Runtime 框架。
- **去噪器 (Denoiser) 支持**：
  - [DFSMN](https://modelscope.cn/models/iic/speech_dfsmn_ans_psm_48k_causal)
  - [GTCRN](https://github.com/Xiaobin-Rong/gtcrn)
  - [ZipEnhancer](https://modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base)
  - [Mel-Band-Roformer](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model)
  - [MossFormerGAN_SE_16K](https://www.modelscope.cn/models/alibabasglab/MossFormerGAN_SE_16K)
 
- **语音活动检测（VAD）支持**：
  - [FSMN](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch)
  - [Faster_Whisper - Silero](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/vad.py)
  - [官方 - Silero](https://github.com/snakers4/silero-vad)
  - [HumAware](https://huggingface.co/CuriousMonkey7/HumAware-VAD)
  - [NVIDIA-NeMo-VAD-v2.0](https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0)
  - [TEN-VAD](https://github.com/TEN-framework/ten-vad)
  - [Pyannote-Segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
    - 需要接受Pyannote的使用条款，並自行下载 Pyannote `pytorch_model.bin` 文件，并将其放置在 `VAD/pyannote_segmentation` 文件夹中。

- **语音识别（ASR）支持**：
  - [SenseVoice-Small](https://modelscope.cn/models/iic/SenseVoiceSmall)
  - [Paraformer-Small-Chinese](https://modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1)
  - [Paraformer-Large-Chinese](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch)
  - [Paraformer-Large-English](https://modelscope.cn/models/iic/speech_paraformer_asr-en-16k-vocab4199-pytorch)
  - [Whisper-Large-V3](https://huggingface.co/openai/whisper-large-v3)
  - [Whisper-Large-V3-Turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
  - [Whisper-Large-V3-Turbo-Japanese](https://huggingface.co/hhim8826/whisper-large-v3-turbo-ja)
  - [Whisper-Large-V3-Anime-A](https://huggingface.co/efwkjn/whisper-ja-anime-v0.1)
  - [Whisper-Large-V3-Anime-B](https://huggingface.co/litagin/anime-whisper)
  - [Whisper-Large-v3.5-Distil](https://huggingface.co/distil-whisper/distil-large-v3.5)
  - [CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper)
  - [FireRedASR-AED-L](https://github.com/FireRedTeam/FireRedASR)
  - [Dolphin-Small](https://github.com/DataoceanAI/Dolphin)


- **大语言模型（LLM）支持**：  
  - Qwen-3: [4B](https://modelscope.cn/models/Qwen/Qwen3-4B), [8B](https://modelscope.cn/models/Qwen/Qwen3-8B)
  - InternLM-3: [8B](https://huggingface.co/internlm/internlm3-8b-instruct)
  - Gemma-3-it: [4B](https://huggingface.co/google/gemma-3-4b-it), [12B](https://huggingface.co/google/gemma-3-12b-it)  
  - Phi-4-Instruct: [mini](https://huggingface.co/microsoft/Phi-4-mini-instruct)

---

## 📋 设置指南

### ✅ 第一步：安装依赖项  
- 在终端中运行以下命令来安装所需的最新 Python 包：
- 对于苹果 M 系列芯片，请不要安装 `onnxruntime-openvino`，否则会导致错误。
```bash
conda install ffmpeg

pip install -r requirements.txt
```

### 📥 第二步：下载必要的模型  
- 从 HuggingFace 下载所需模型：[Transcribe_and_Translate_Subtitles](https://huggingface.co/H5N1AIDS/Transcribe_and_Translate_Subtitles)


### 🖥️ 第三步：下载并放置 `run.py`  
- 从此项目的仓库下载 `run.py` 脚本。  
- 将 `run.py` 放置在 `Transcribe_and_Translate_Subtitles` 文件夹中。

### 📁 第四步：将目标视频放入 Media 文件夹  
- 将你想要转录和翻译的视频放置在以下目录，应用程序将逐个处理这些视频：  
```
Transcribe_and_Translate_Subtitles/Media
```

### 🚀 第五步：运行应用程序  
- 打开你喜欢的终端工具（PyCharm、CMD、PowerShell 等）。  
- 运行以下命令来启动应用程序：  
```bash
python run.py
```
- 应用程序启动后，你的浏览器将自动打开一个网页。  
   ![screenshot](https://github.com/DakeQQ/Transcribe-and-Translate-Subtitles/blob/main/screen/Screenshot%20from%202025-05-08%2013-01-17.png)

### 🛠️ 第六步：修复错误（如有）  
- 首次运行时，你可能会遇到 **Silero-VAD 错误**。只需重启应用程序即可解决该问题。
- 首次运行时，你可能会遇到 **libc++1.so 错误**。在终端中运行以下命令，应该可以解决问题。
```bash
sudo apt update
sudo apt install libc++1
```

### 💻 第七步：支持设备  
- 此项目目前支持:
  - **Intel-OpenVINO-CPU-GPU-NPU**
  - **Windows-AMD-GPU**
  - **NVIDIA-GPU**
  - **Apple-CPU**
  - **AMD-CPU**

## 🎉 尽情享受应用程序吧！
```
Transcribe_and_Translate_Subtitles/Results/Subtitles
```
---

## 📌 待办事项  
- [ ] [LLM-MiniCPM4](https://github.com/OpenBMB/MiniCPM)
- [ ] [去噪-MossFormer2-48K](https://www.modelscope.cn/models/alibabasglab/MossFormer2_SE_48K)
- [ ] [提高音频质量-MossFormer2-48K](https://www.modelscope.cn/models/alibabasglab/MossFormer2_SR_48K)
- [ ] [提高视频分辨率](https://github.com/sczhou/Upscale-A-Video)
- [ ] 支持 AMD-ROCm
- [ ] 实现实时视频转录和翻译播放器
---
