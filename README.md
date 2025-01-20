# Transcribe and Translate Subtitles

## 🚨 Important Note
- **This project is for non-commercial use only!**
- **Every task runs locally without internet, ensuring maximum privacy.**

---

## ✨ Features  
This project is built on ONNX Runtime and the IPEX-LLM framework.
- VAD Support:
  - [Faster_Whisper - Silero](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/vad.py)
  - [Official - Silero](https://github.com/snakers4/silero-vad)
  - [FSMN](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch)

- Deoiser Support:
  - [ZipEnhancer](https://modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base)
  - [GTCRN](https://github.com/Xiaobin-Rong/gtcrn)
  - [DFSMN](https://modelscope.cn/models/iic/speech_dfsmn_ans_psm_48k_causal)

- ASR Support:
  - [SenseVoiceSmall](https://modelscope.cn/models/iic/SenseVoiceSmall)
  - [Paraformer](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch)
  - [Whisper-Large-V3](https://huggingface.co/openai/whisper-large-v3)
  - [Whisper-Large-V3-Turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
  - Custom-Whisper - V2 / V3 / Distil / Turbo (The model must be exported using this [tool](https://github.com/DakeQQ/Automatic-Speech-Recognition-ASR-ONNX))

- LLM Supports: 
  - Gemma-2-it: [2B](https://huggingface.co/google/gemma-2-2b-it), [9B](https://huggingface.co/google/gemma-2-9b-it)  
  - GLM4-Chat: [9B](https://github.com/THUDM/GLM-4)  
  - MiniCPM3: [4B](https://github.com/OpenBMB/MiniCPM)  
  - Phi3/3.5: [mini](https://huggingface.co/microsoft/Phi-3.5-mini-instruct), [medium](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct)  
  - Qwen2.5-Instruct: [3B](https://modelscope.cn/models/Qwen/Qwen2.5-3B-Instruct), [7B](https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct), [14B](https://modelscope.cn/models/Qwen/Qwen2.5-14B-Instruct), [32B](https://www.modelscope.cn/models/Qwen/Qwen2.5-32B-Instruct)  
  - GGUF format：iq1_s, iq2_xs, iq2_xxs, q4k_s, q4k_m


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
- Download the required models from Google Drive: [Transcribe_and_Translate_Subtitles](https://drive.google.com/drive/folders/1W5yqPm-FYD2r1KR7JrDwJ8jzuFALNr9O?usp=drive_link). After downloading, unzip the file.
- Or [Baidu Cloud](https://pan.baidu.com/s/1DSAYmbMX5lKj9oz8Uhmmwg?pwd=dake) (This link lacks a zip package; download files individually as Baidu Cloud doesn't support packages over 4GB.)


### 🤖 Step 3: Download a Preferred LLM Model (Optional, for translate task)
- Choose and download your preferred LLM model.
- The largest LLM size that can run on a 16GB RAM computer is 7 billion parameters (7B). For example: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

### 📂 Step 4: Place the LLM Model in the Correct Directory (Optional, for translate task)
- Move the downloaded LLM model to the following path:
```
Transcribe_and_Translate_Subtitles/LLM/Qwen/7B
```

### 🖥️ Step 5: Download and Place `run.py`
- Download the `run.py` script from this repository.
- Place it in the `Transcribe_and_Translate_Subtitles` folder.

### 📁 Step 6: Place Target Videos in the Media Folder
- Place the videos you want to transcribe and translate in the following directory:
```
Transcribe_and_Translate_Subtitles/Media
```
- The application will process the videos one by one.

### 🚀 Step 7: Run the Application
- Open your preferred terminal (PyCharm, CMD, PowerShell, etc.).
- Execute the following command to start the application:
```bash
python run.py
```
- Once the application starts, you will see a webpage open in your browser.
   ![screenshot](https://github.com/DakeQQ/Transcribe-and-Translate-Subtitles/blob/main/screen/Screenshot%20from%202025-01-13%2000-47-34.png)

### 🛠️ Step 8: Fix Silero-VAD Error (if encountered)
- On the first run, you might encounter a **Silero-VAD error**. Simply restart the application, and it should be resolved.

### 💻 Step 9: Device Support
- This project currently supports: **Intel-CPU-GPU-NPU**, **AMD-CPU** and **Apple-CPU** users.

### 🔧 Step 10: Enable Intel-GPU or Intel-NPU (Optional)
- The LLM integration is based on the [ipex-llm](https://github.com/intel-analytics/ipex-llm). To use Intel-GPU or Intel-NPU, follow the instructions in the `ipex-llm` repository to enable these devices (now only support python3.11). Without this setup, the application will not work on GPU/NPU hardware.

---

## 🎉 Enjoy the Application!
```
Transcribe_and_Translate_Subtitles/Results/Subtitles
```
---

## 📌 To-Do List
- [ ] Add support for NVIDIA devices
- [ ] Add support for AMD devices
- [ ] Real-Time Translate & Trascribe Video Player

---
### 性能 Performance  
| OS           | Backend           | Denoiser          | VAD                  | ASR                  | Real-Time Factor<br>test_video.mp4<br>7602 seconds |
|:------------:|:-----------------:|:-----------------:|:--------------------:|:--------------------:|:----------------:|
| Ubuntu-24.04 | CPU <br> i3-12300 | None              | Silero               | SenseVoiceSmall      |        0.08      |
| Ubuntu-24.04 | CPU <br> i3-12300 | GTCRN             | Silero               | SenseVoiceSmall      |        0.10      |
| Ubuntu-24.04 | CPU <br> i3-12300 | GTCRN             | FSMN                 | SenseVoiceSmall      |        0.054     |
| Ubuntu-24.04 | CPU <br> i3-12300 | ZipEnhancer       | FSMN                 | SenseVoiceSmall      |        0.39      |
| Ubuntu-24.04 | CPU <br> i3-12300 | GTCRN             | Silero               | Whisper-Large-V3     |        0.20      |
| Ubuntu-24.04 | CPU <br> i3-12300 | GTCRN             | FSMN                 | Whisper-Large-V3-Turbo |        0.148   |

---
# 转录和翻译字幕

## 🚨 重要提示  
- **此项目仅限非商业用途！**  
- **所有任务均在本地运行，无需连接互联网，确保最大程度的隐私保护。**

---

## ✨ 功能
这个项目基于ONNX Runtime和IPEX-LLM框架。
- **语音活动检测（VAD）支持**：
  - [Faster_Whisper - Silero](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/vad.py)
  - [官方 - Silero](https://github.com/snakers4/silero-vad)
  - [FSMN](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch)

- **去噪器 (Denoiser) 支持**：
  - [ZipEnhancer](https://modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base)
  - [GTCRN](https://github.com/Xiaobin-Rong/gtcrn)
  - [DFSMN](https://modelscope.cn/models/iic/speech_dfsmn_ans_psm_48k_causal)

- **语音识别（ASR）支持**：
  - [SenseVoiceSmall](https://modelscope.cn/models/iic/SenseVoiceSmall)
  - [Paraformer](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch)
  - [Whisper-Large-V3](https://huggingface.co/openai/whisper-large-v3)
  - [Whisper-Large-V3-Turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
  - 自定义 Whisper - V2 / V3 / Distil / Turbo（需要使用此[工具](https://github.com/DakeQQ/Automatic-Speech-Recognition-ASR-ONNX)导出模型）

- **大语言模型（LLM）支持**：  
  - Gemma-2-it: [2B](https://huggingface.co/google/gemma-2-2b-it), [9B](https://huggingface.co/google/gemma-2-9b-it)  
  - GLM4-Chat: [9B](https://github.com/THUDM/GLM-4)  
  - MiniCPM3: [4B](https://github.com/OpenBMB/MiniCPM)  
  - Phi3/3.5: [mini](https://huggingface.co/microsoft/Phi-3.5-mini-instruct), [medium](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct)  
  - Qwen2.5-Instruct: [3B](https://modelscope.cn/models/Qwen/Qwen2.5-3B-Instruct), [7B](https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct), [14B](https://modelscope.cn/models/Qwen/Qwen2.5-14B-Instruct), [32B](https://www.modelscope.cn/models/Qwen/Qwen2.5-32B-Instruct)  
  - GGUF 格式：iq1_s, iq2_xs, iq2_xxs, q4k_s, q4k_m  

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
- 从 Google Drive 下载所需模型：[Transcribe_and_Translate_Subtitles](https://drive.google.com/drive/folders/1W5yqPm-FYD2r1KR7JrDwJ8jzuFALNr9O?usp=drive_link), 下载完成后，解压文件。
- 或 [百度云](https://pan.baidu.com/s/1DSAYmbMX5lKj9oz8Uhmmwg?pwd=dake) (该链接没有提供压缩包；需要逐个下载文件夹，因为百度云不支持超过 4GB 的压缩包。)

### 🤖 第三步：下载你喜欢的 LLM 模型 （可选，用于翻译任务） 
- 选择并下载你偏好的 LLM 模型。
- 在16GB内存的电脑上可运行的最大LLM模型为70亿参数(7B)。例如：[Qwen2.5-7B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct)

### 📂 第四步：将 LLM 模型放置到正确的目录 （可选，用于翻译任务） 
- 将下载的 LLM 模型移动到以下路径：  
```
Transcribe_and_Translate_Subtitles/LLM/Qwen/7B
```

### 🖥️ 第五步：下载并放置 `run.py`  
- 从此项目的仓库下载 `run.py` 脚本。  
- 将 `run.py` 放置在 `Transcribe_and_Translate_Subtitles` 文件夹中。

### 📁 第六步：将目标视频放入 Media 文件夹  
- 将你想要转录和翻译的视频放置在以下目录：  
```
Transcribe_and_Translate_Subtitles/Media
```
- 应用程序将逐个处理这些视频。

### 🚀 第七步：运行应用程序  
- 打开你喜欢的终端工具（PyCharm、CMD、PowerShell 等）。  
- 运行以下命令来启动应用程序：  
```bash
python run.py
```
- 应用程序启动后，你的浏览器将自动打开一个网页。  
   ![screenshot](https://github.com/DakeQQ/Transcribe-and-Translate-Subtitles/blob/main/screen/Screenshot%20from%202025-01-13%2000-53-21.png)

### 🛠️ 第八步：修复 Silero-VAD 错误（如有）  
- 首次运行时，你可能会遇到 **Silero-VAD 错误**。只需重启应用程序即可解决该问题。

### 💻 第九步：支持设备  
- 此项目目前支持 **Intel-CPU-GPU-NPU**, **AMD-CPU** 和 **Apple-CPU** 用户。

### 🔧 第十步：启用 Intel-GPU 或 Intel-NPU（可选）  
- 此项目的 LLM 集成基于 [ipex-llm](https://github.com/intel-analytics/ipex-llm)。若要使用 Intel-GPU 或 Intel-NPU，请按照 `ipex-llm` 仓库中的说明来启用这些设备(现在只支持python3.11)。如果不进行此设置，应用程序将无法在 GPU/NPU 硬件上运行。

---

## 🎉 尽情享受应用程序吧！
```
Transcribe_and_Translate_Subtitles/Results/Subtitles
```
---

## 📌 待办事项  
- [ ] 添加对 NVIDIA 设备的支持  
- [ ] 添加对 AMD 设备的支持  
- [ ] 实现实时视频转录和翻译播放器
---
