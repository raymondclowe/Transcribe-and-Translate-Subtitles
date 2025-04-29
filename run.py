# Standard library imports
import os
import gc
import re
import time
import shutil
import site
import json
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Third-party imports
import torch
import numpy as np
import onnxruntime
import soundfile as sf
import gradio as gr
import psutil

from gradio.themes.utils import sizes
from pydub import AudioSegment
from transformers import AutoTokenizer
from sentencepiece import SentencePieceProcessor
from silero_vad import load_silero_vad, get_speech_timestamps
from faster_whisper.vad import get_speech_timestamps as get_speech_timestamps_FW, VadOptions
from ipex_llm.transformers import AutoModelForCausalLM
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model


MAX_SEQ_LEN = 64
STOP_TOKEN = 50257
MEDIA_EXTENSIONS = (
    # Audio formats
    '.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.alac', '.aiff', '.m4a',

    # Video formats
    '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.mpeg', '.3gp',

    # Movie formats (often overlaps with video)
    '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.mpeg', '.3gp',

    # Music formats (often overlaps with audio)
    '.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.alac', '.aiff', '.m4a'
)

PYTHON_PACKAGE = site.getsitepackages()[-1]
shutil.copyfile("./modeling_modified/utils_vad.py", PYTHON_PACKAGE + "/silero_vad/utils_vad.py")
shutil.copyfile("./modeling_modified/model.py", PYTHON_PACKAGE + "/silero_vad/model.py")
shutil.copyfile("./VAD/silero_vad.onnx", PYTHON_PACKAGE + "/silero_vad/data/silero_vad.onnx")

inv_16k = float(1.0 / 16000)
inv_1024 = float(1.0 / (1024 ** 3))


def update_ui(dropdown_ui_language):
    if "中文" in dropdown_ui_language:
        update_A = gr.update(
            label="输入视频/音频",
            info="输入您想要转录的视频/音频文件或文件夹的路径。"
        )
        update_B = gr.update(
            label="选择任务",
            info="选择要对所选视频/音频执行的操作。"
        )
        update_C = gr.update(
            label="运行测试",
            info="对视频/音频长度的10%运行短测试。"
        )
        update_D = gr.update(
            label="并行线程数",
            info="用于并行处理的核心数量。"
        )
        update_E = gr.update(
            label="使用硬件",
            info="选择任务的设备。"
        )
        update_F = gr.update(
            label="LLM翻译",
            info="选择用于翻译任务的模型。"
        )
        update_G = gr.update(
            label="LLM数据类型",
            info="用于LLM翻译的数据类型。"
        )
        update_H = gr.update(
            label="自定义GGUF-LLM路径",
            info="您的GGUF-LLM模型路径。"
        )
        update_I = gr.update(
            label="ASR",
            info="选择用于转录任务的模型。"
        )
        update_J = gr.update(
            label="自定义 Whisper 路径",
            info="指定您微调的 Whisper 模型文件夹路径，包括 Whisper_Encoder.ort, Whisper_Decoder.ort, tokenizer.json ..."
        )
        update_K = gr.update(
            label="转录语言",
            info="选择视频/音频所属的语言。"
        )
        update_L = gr.update(
            label="翻译语言",
            info="您想要翻译成哪种语言？",
        )
        update_M = gr.update(
            label="降噪器",
            info="选择用于视频/音频处理的降噪器。"
        )
        update_N = gr.update(
            label="使用缓存",
            info="决定是否使用之前的降噪结果以节省时间。"
        )
        update_O = gr.update(
            label="VAD",
            info="选择用于视频/音频处理的 VAD：Silero 在嘈杂音频中表现更好，而 FSMN 在中文音频环境中表现出色。"
        )
        update_P = gr.update(
            label="语音状态阈值",
            info="值越高，灵敏度越高，但可能误将噪声分类为语音。"
        )
        update_Q = gr.update(
            label="SNR 阈值",
            info="值越低，灵敏度越高，但可能误将噪声分类为语音。单位：dB"
        )
        update_R = gr.update(
            label="初始背景噪声",
            info="背景的初始值。较小的值表示更安静的环境。单位：dB。在使用降噪音频时，将此值设置为较小。"
        )
        update_S = gr.update(
            label="语音状态评分",
            info="用于判断是否在讲话状态的判断因子。值越大，激活越困难。"
        )
        update_T = gr.update(
            label="静音状态评分",
            info="用于判断是否在静音状态的判断因子。值越大，越容易切断讲话。"
        )
        update_U = gr.update(
            label="融合时间戳",
            info="用于合并时间戳的判断因子：如果两个语音段太接近，它们将合并为一个。单位：秒。"
        )
        update_V = gr.update(
            label="过滤短语音",
            info="用于过滤 VAD 结果的标准，当语音太短时。单位：秒。"
        )
        update_W = gr.update(
            label="过滤长语音",
            info="最大语音持续时间。单位：秒。"
        )
        update_X = gr.update(
            label="静音持续判断",
            info="最小静音持续时间。单位：毫秒。"
        )
        update_Y = gr.update(
            label="降噪因子",
            info="较大的值可以增强降噪效果。"
        )
        update_Z = gr.update(
            label="VAD 填充",
            info="为时间戳的起始和结束添加填充。单位：毫秒。"
        )
    elif "日本語" in dropdown_ui_language:
        update_A = gr.update(
            label="音声を入力",
            info="転写または翻訳したい音声ファイルまたはフォルダのパスを入力し。"
        )
        update_B = gr.update(
            label="タスクを選択",
            info="選択した音声に対して実行する操作を選択します。"
        )
        update_C = gr.update(
            label="テストを実行",
            info="音声の長さの10％で短いテストを実行します。"
        )
        update_D = gr.update(
            label="並列スレッド数",
            info="並列処理に使用するコア数。"
        )
        update_E = gr.update(
            label="ハードウェアを使用",
            info="タスクのデバイスを選択します。"
        )
        update_F = gr.update(
            label="LLM翻訳",
            info="翻訳タスクに使用するモデルを選択します。"
        )
        update_G = gr.update(
            label="LLM精度",
            info="LLMの翻訳で使用されるデータ型。"
        )
        update_H = gr.update(
            label="カスタムGGUF-LLMパス",
            info="GGUF-LLMモデルへのパス。"
        )
        update_I = gr.update(
            label="ASR",
            info="転写タスクに使用するモデルを選択します。"
        )
        update_J = gr.update(
            label="カスタム Whisper パス",
            info="微調整したWhisperモデルフォルダのパスを指定します（Whisper_Encoder.ort, Whisper_Decoder.ort, tokenizer.json ... などを含む）。"
        )
        update_K = gr.update(
            label="転写言語",
            info="音声の言語を選択します。"
        )
        update_L = gr.update(
            label="翻訳言語",
            info="どの言語に翻訳したいですか？"
        )
        update_M = gr.update(
            label="ノイズリダクション",
            info="音声処理に使用するノイズリダクションを選択します。"
        )
        update_N = gr.update(
            label="キャッシュを使用",
            info="以前のノイズリダクション結果を使用して時間を節約するかどうかを決定します。"
        )
        update_O = gr.update(
            label="VAD",
            info="音声処理に使用するVADを選択します：Sileroはノイズの多い音声でより優れたパフォーマンスを発揮し、FSMNは中国語音声環境で優れています。"
        )
        update_P = gr.update(
            label="音声しきい値",
            info="値が高いほど感度が高くなりますが、ノイズを音声と誤認する可能性があります。"
        )
        update_Q = gr.update(
            label="SNRしきい値",
            info="値が低いほど感度が高くなりますが、ノイズを音声と誤認する可能性があります。単位：dB"
        )
        update_R = gr.update(
            label="初期バックグラウンドノイズ",
            info="バックグラウンドの初期値。小さい値はより静かな環境を示します。単位：dB。ノイズリダクションされた音声を使用する場合は、この値を小さく設定します。"
        )
        update_S = gr.update(
            label="話者状態評価",
            info="話者状態であるかを判断する評価因子。値が大きいほど、活性化が困難になります。"
        )
        update_T = gr.update(
            label="無音状態評価",
            info="無音状態であるかを判断する評価因子。値が大きいほど、話者を切断しやすくなります。"
        )
        update_U = gr.update(
            label="タイムスタンプ統合",
            info="タイムスタンプを統合するための判断因子：2つの音声セグメントが近すぎる場合、それらは1つに統合されます。単位：秒。"
        )
        update_V = gr.update(
            label="短い音声をフィルタリング",
            info="VAD 結果をフィルタリングする基準で、音声が短すぎる場合。単位：秒。"
        )
        update_W = gr.update(
            label="長い音声をフィルタリング",
            info="最大音声継続時間。単位：秒。"
        )
        update_X = gr.update(
            label="無音判断",
            info="最小無音継続時間。単位：ミリ秒。"
        )
        update_Y = gr.update(
            label="ノイズ除去係数",
            info="大きな値はノイズ除去効果を高めます。"
        )
        update_Z = gr.update(
            label="VAD パディング",
            info="タイムスタンプの開始と終了にパディングを追加します。 単位：ミリ秒。"
        )
    else:
        update_A = gr.update(
            label="Input Video/Audio",
            info="Enter the path of the video/audio file or folder you want to transcribe."
        )
        update_B = gr.update(
            label="Select Task",
            info="Choose the operation to perform on the selected video/audio."
        )
        update_C = gr.update(
            label="Run Test",
            info="Run a short test on 10% of the video/audio length."
        )
        update_D = gr.update(
            label="Number of Parallel Threads",
            info="The number of cores used for parallel processing."
        )
        update_E = gr.update(
            label="Use Hardware",
            info="Select the device for the task."
        )
        update_F = gr.update(
            label="LLM Translation",
            info="Select the model used for translation tasks."
        )
        update_G = gr.update(
            label="LLM Accuracy",
            info="Dtype used in LLM translation."
        )
        update_H = gr.update(
            label="Custom GGUF-LLM Path",
            info="Path to your GGUF-LLM model.",
        )
        update_I = gr.update(
            label="ASR",
            info="Select the model used for transcription tasks."
        )
        update_J = gr.update(
            label="Custom Whisper Path",
            info="Specify the path to the folder of your fine-tuned Whisper model, including Whisper_Encoder.ort, Whisper_Decoder.ort, tokenizer.json ..."
        )
        update_K = gr.update(
            label="Transcription Language",
            info="Select the language the video/audio belongs to."
        )
        update_L = gr.update(
            label="Translation Language",
            info="Which language do you want to translate into?"
        )
        update_M = gr.update(
            label="Denoiser",
            info="Select the denoiser used for video/audio processing."
        )
        update_N = gr.update(
            label="Use Cache",
            info="Decide whether to use previous denoising results to save time."
        )
        update_O = gr.update(
            label="VAD",
            info="Select the VAD used for audio processing: Silero performs better in noisy audio, while FSMN excels in Chinese audio environments."
        )
        update_P = gr.update(
            label="Voice State Threshold",
            info="The higher the value, the higher the sensitivity, but it may mistakenly classify noise as voice."
        )
        update_Q = gr.update(
            label="SNR Threshold",
            info="The lower the value, the higher the sensitivity, but it may mistakenly classify noise as voice. Unit: dB"
        )
        update_R = gr.update(
            label="Initial Background Noise",
            info="Initial value of the background. Smaller values indicate a quieter environment. Unit: dB. Set this to a smaller value when using denoised audio."
        )
        update_S = gr.update(
            label="Voice State Score",
            info="A factor used to determine whether in a speaking state. The larger the value, the harder it is to activate."
        )
        update_T = gr.update(
            label="Silence State Score",
            info="A factor used to determine whether in a silence state. The larger the value, the easier it is to cut off speech."
        )
        update_U = gr.update(
            label="Merge Timestamps",
            info="A factor used for merging timestamps: if two voice segments are too close, they will be merged into one. Unit: seconds."
        )
        update_V = gr.update(
            label="Filter Short Voice",
            info="Criteria for filtering VAD results when the voice is too short. Unit: seconds."
        )
        update_W = gr.update(
            label="Filter Long Voice",
            info="Maximum voice duration. Unit: seconds."
        )
        update_X = gr.update(
            label="Silence Duration Judgment",
            info="Minimum silence duration. Unit: milliseconds."
        )
        update_Y = gr.update(
            label="Denoise Factor",
            info="A larger value enhances the denoising effect."
        )
        update_Z = gr.update(
            label="VAD Padding",
            info="Add padding to the start and end of the timestamps. Unit: milliseconds."
        )
    return update_A, update_B, update_C, update_D, update_E, update_F, update_G, update_H, update_I, update_J, update_K, update_L, update_M, update_N, update_O, update_P, update_Q, update_R, update_S, update_T, update_U, update_V, update_W, update_X, update_Y, update_Z


def update_task(dropdown_task):
    if "Translate" in dropdown_task:
        update_A = gr.update(visible=True)
        update_B = gr.update(visible=True)
        update_C = gr.update(visible=True)
        update_D = gr.update(visible=True)
    else:
        update_A = gr.update(visible=False)
        update_B = gr.update(visible=False)
        update_C = gr.update(visible=False)
        update_D = gr.update(visible=False)
    return update_A, update_B, update_C, update_D


def update_asr(dropdown_model_asr):
    if "Custom" in dropdown_model_asr:
        if "V2" in dropdown_model_asr:
            update_A = gr.update(visible=True, value="/ASR/Whisper/V2/Custom/")
        else:
            update_A = gr.update(visible=True, value="/ASR/Whisper/V3/Custom/")
    else:
        update_A = gr.update(visible=False)
    if "Whisper" in dropdown_model_asr:
        update_B = gr.update(
            choices=[
                "中文",         "English",          "日本語",         "한국인",
                "afrikaans",    "amharic",          "arabic",        "assamese",      "azerbaijani",
                "bashkir",      "belarusian",       "bulgarian",     "bengali",       "tibetan",
                "breton",       "bosnian",          "catalan",       "czech",         "welsh",
                "danish",       "german",           "greek",         "english",       "spanish",
                "estonian",     "basque",           "persian",       "finnish",       "faroese",
                "french",       "galician",         "gujarati",      "hawaiian",      "hausa",
                "hebrew",       "hindi",            "croatian",      "haitian creole","hungarian",
                "armenian",     "indonesian",       "icelandic",     "italian",       "japanese",
                "javanese",     "georgian",         "kazakh",        "khmer",         "kannada",
                "korean",       "latin",            "luxembourgish", "lingala",       "lao",
                "lithuanian",   "latvian",          "malagasy",      "maori",         "macedonian",
                "malayalam",    "mongolian",        "marathi",       "malay",         "maltese",
                "burmese",      "nepali",           "dutch",         "nynorsk",       "norwegian",
                "occitan",      "punjabi",          "polish",        "pashto",        "portuguese",
                "romanian",     "russian",          "sanskrit",      "sindhi",        "sinhala",
                "slovak",       "slovenian",        "shona",         "somali",        "albanian",
                "serbian",      "sundanese",        "swedish",       "swahili",       "tamil",
                "telugu",       "tajik",            "thai",          "turkmen",       "tagalog",
                "turkish",      "tatar",            "ukrainian",     "urdu",          "uzbek",
                "vietnamese",   "yiddish",          "yoruba",        "chinese"
            ]
        )
    elif "SenseVoiceSmall" in dropdown_model_asr:
        update_B = gr.update(choices=["日本語", "中文", "English", "粤语", "한국인", "Auto"])
    else:
        if "Small" in dropdown_model_asr:  # Paraformer
            update_B = gr.update(value="中文", choices=["中文"])
        else:
            update_B = gr.update(value="中文", choices=["中文", "English"])
    return update_A, update_B


def update_model_llm_accuracy(dropdown_hardware):
    if "CPU" == dropdown_hardware:
        update_A = gr.update(
            choices=["sym_int4", "asym_int4", "sym_int5", "asym_int5", "sym_int8", "gguf_iq2_xxs", "gguf_iq2_xs", "gguf_iq1_s", "gguf_q4k_s", "gguf_q4k_m"], value="sym_int4")
    else:
        update_A = gr.update(
            choices=["sym_int4", "asym_int4", "sym_int5", "asym_int5", "sym_int8", "nf3", "nf4", "fp4", "fp6", "fp8",
                     "fp8_e4m3", "fp8_e5m2", "fp16", "bf16", "q2_k", "q4_k", "q5_k", "q6_k", "fp6_kg", "mixed_fp4",
                     "mixed_fp8", "gguf_iq2_xxs", "gguf_iq2_xs", "gguf_iq1_s", "gguf_q4k_s", "gguf_q4k_m"],
            value="sym_int4")
    return update_A


def update_translate_language(dropdown_model_llm):
    if "Whisper" in dropdown_model_llm:
        update_A = gr.update(choices=["English"], value="English")
    else:
        update_A = gr.update(
            choices=[
                "中文",         "English",          "日本語",         "한국인",
                "afrikaans",    "amharic",          "arabic",        "assamese",      "azerbaijani",
                "bashkir",      "belarusian",       "bulgarian",     "bengali",       "tibetan",
                "breton",       "bosnian",          "catalan",       "czech",         "welsh",
                "danish",       "german",           "greek",         "english",       "spanish",
                "estonian",     "basque",           "persian",       "finnish",       "faroese",
                "french",       "galician",         "gujarati",      "hawaiian",      "hausa",
                "hebrew",       "hindi",            "croatian",      "haitian creole","hungarian",
                "armenian",     "indonesian",       "icelandic",     "italian",       "japanese",
                "javanese",     "georgian",         "kazakh",        "khmer",         "kannada",
                "korean",       "latin",            "luxembourgish", "lingala",       "lao",
                "lithuanian",   "latvian",          "malagasy",      "maori",         "macedonian",
                "malayalam",    "mongolian",        "marathi",       "malay",         "maltese",
                "burmese",      "nepali",           "dutch",         "nynorsk",       "norwegian",
                "occitan",      "punjabi",          "polish",        "pashto",        "portuguese",
                "romanian",     "russian",          "sanskrit",      "sindhi",        "sinhala",
                "slovak",       "slovenian",        "shona",         "somali",        "albanian",
                "serbian",      "sundanese",        "swedish",       "swahili",       "tamil",
                "telugu",       "tajik",            "thai",          "turkmen",       "tagalog",
                "turkish",      "tatar",            "ukrainian",     "urdu",          "uzbek",
                "vietnamese",   "yiddish",          "yoruba",        "chinese"
            ],
            value="中文"
        )
    if "Custom" in dropdown_model_llm:
        update_B = gr.update(visible=True, value=None)
        update_C = gr.update(choices=["gguf_iq2_xxs", "gguf_iq2_xs", "gguf_iq1_s", "gguf_q4k_s", "gguf_q4k_m"], value="gguf_q4k_s")
        update_D = gr.update(choices=["SenseVoiceSmall-Fast", "Whisper-Large-V2", "Whisper-Large-V3", "Whisper-Large-V3-Turbo", "Custom-Fine-Tune-Whisper-V2", "Custom-Fine-Tune-Whisper-V3", "Paraformer-Small", "Paraformer-Large"])
    else:
        update_B = gr.update(visible=False, value=None)
        if "Whisper" in dropdown_model_llm:
            update_C = gr.update(choices=["asym_int8"], value="asym_int8")
            update_D = gr.update(choices=["Whisper-Large-V2", "Whisper-Large-V3", "Custom-Fine-Tune-Whisper-V2", "Custom-Fine-Tune-Whisper-V3"], value="Whisper-Large-V3")
        else:
            update_C = gr.update(choices=["sym_int4", "asym_int4", "sym_int5", "asym_int5", "sym_int8", "gguf_iq2_xxs", "gguf_iq2_xs", "gguf_iq1_s", "gguf_q4k_s", "gguf_q4k_m"], value="sym_int4")
            update_D = gr.update(choices=["SenseVoiceSmall-Fast", "Whisper-Large-V2", "Whisper-Large-V3", "Whisper-Large-V3-Turbo", "Custom-Fine-Tune-Whisper-V2", "Custom-Fine-Tune-Whisper-V3", "Paraformer-Small", "Paraformer-Large"])
    return update_A, update_B, update_C, update_D


def update_denoiser(dropdown_model_denoiser):
    if "None" == dropdown_model_denoiser:
        return gr.update(visible=False), gr.update(visible=False)
    else:
        return gr.update(visible=True), gr.update(visible=True)


def update_vad(dropdown_model_vad):
    if "FSMN" in dropdown_model_vad:
        update_A = gr.update(visible=True)
        update_B = gr.update(visible=True)
        update_C = gr.update(visible=True)
        update_D = gr.update(visible=True)
        update_E = gr.update(visible=True)
        update_F = gr.update(visible=False)
        update_G = gr.update(visible=False)
        update_H = gr.update(value=1.0)
        update_I = gr.update(value=0.05)
        update_J = gr.update(value=400)
    elif "Pyannote" in dropdown_model_vad:
        update_A = gr.update(visible=False)
        update_B = gr.update(visible=False)
        update_C = gr.update(visible=False)
        update_D = gr.update(visible=False)
        update_E = gr.update(visible=False)
        update_F = gr.update(visible=False)
        update_G = gr.update(visible=False)
        update_H = gr.update(value=0.0)
        update_I = gr.update(value=0.05)
        update_J = gr.update(value=400)
    else:
        update_A = gr.update(visible=True)
        update_B = gr.update(visible=True)
        update_C = gr.update(visible=False)
        update_D = gr.update(visible=False)
        update_E = gr.update(visible=False)
        update_F = gr.update(visible=True)
        update_G = gr.update(visible=True)
        update_H = gr.update(value=0.0)
        update_I = gr.update(value=0.05)
        update_J = gr.update(value=400)
    return update_A, update_B, update_C, update_D, update_E, update_F, update_G, update_H, update_I, update_J


def get_language_id(language_input, is_whisper):
    language_input = language_input.lower()
    if is_whisper:
        language_map = {
            'af': 50327, 'am': 50334, 'ar': 50272, 'as': 50350, 'az': 50304,
            'ba': 50355, 'be': 50330, 'bg': 50292, 'bn': 50302, 'bo': 50347,
            'br': 50309, 'bs': 50315, 'ca': 50270, 'cs': 50283, 'cy': 50297,
            'da': 50285, 'de': 50261, 'el': 50281, 'en': 50259, 'es': 50262,
            'et': 50307, 'eu': 50310, 'fa': 50300, 'fi': 50277, 'fo': 50338,
            'fr': 50265, 'gl': 50319, 'gu': 50333, 'haw': 50352, 'ha': 50354,
            'he': 50279, 'hi': 50276, 'hr': 50291, 'ht': 50339, 'hu': 50286,
            'hy': 50312, 'id': 50275, 'is': 50311, 'it': 50274, 'ja': 50266,
            'jw': 50356, 'ka': 50329, 'kk': 50316, 'km': 50323, 'kn': 50306,
            'ko': 50264, 'la': 50294, 'lb': 50345, 'ln': 50353, 'lo': 50336,
            'lt': 50293, 'lv': 50301, 'mg': 50349, 'mi': 50295, 'mk': 50308,
            'ml': 50296, 'mn': 50314, 'mr': 50320, 'ms': 50282, 'mt': 50343,
            'my': 50346, 'ne': 50313, 'nl': 50271, 'nn': 50342, 'no': 50288,
            'oc': 50328, 'pa': 50321, 'pl': 50269, 'ps': 50340, 'pt': 50267,
            'ro': 50284, 'ru': 50263, 'sa': 50344, 'sd': 50332, 'si': 50322,
            'sk': 50298, 'sl': 50305, 'sn': 50324, 'so': 50326, 'sq': 50317,
            'sr': 50303, 'su': 50357, 'sv': 50273, 'sw': 50318, 'ta': 50287,
            'te': 50299, 'tg': 50331, 'th': 50289, 'tk': 50341, 'tl': 50348,
            'tr': 50268, 'tt': 50351, 'uk': 50280, 'ur': 50290, 'uz': 50337,
            'vi': 50278, 'yi': 50335, 'yo': 50325, 'zh': 50260
        }
        full_language_names = {
            '中文': 'zh',            '日本語': 'ja',          '한국인': 'ko',
            'afrikaans': 'af',      'amharic': 'am',        'arabic': 'ar',        'assamese': 'as',
            'azerbaijani': 'az',    'bashkir': 'ba',        'belarusian': 'be',    'bulgarian': 'bg',
            'bengali': 'bn',        'tibetan': 'bo',        'breton': 'br',        'bosnian': 'bs',
            'catalan': 'ca',        'czech': 'cs',          'welsh': 'cy',         'danish': 'da',
            'german': 'de',         'greek': 'el',          'english': 'en',       'spanish': 'es',
            'estonian': 'et',       'basque': 'eu',         'persian': 'fa',       'finnish': 'fi',
            'faroese': 'fo',        'french': 'fr',         'galician': 'gl',      'gujarati': 'gu',
            'hawaiian': 'haw',      'hausa': 'ha',          'hebrew': 'he',        'hindi': 'hi',
            'croatian': 'hr',       'haitian creole': 'ht', 'hungarian': 'hu',     'armenian': 'hy',
            'indonesian': 'id',     'icelandic': 'is',      'italian': 'it',       'japanese': 'ja',
            'javanese': 'jw',       'georgian': 'ka',       'kazakh': 'kk',        'khmer': 'km',
            'kannada': 'kn',        'korean': 'ko',         'latin': 'la',         'luxembourgish': 'lb',
            'lingala': 'ln',        'lao': 'lo',            'lithuanian': 'lt',    'latvian': 'lv',
            'malagasy': 'mg',       'maori': 'mi',          'macedonian': 'mk',    'malayalam': 'ml',
            'mongolian': 'mn',      'marathi': 'mr',        'malay': 'ms',         'maltese': 'mt',
            'burmese': 'my',        'nepali': 'ne',         'dutch': 'nl',         'nynorsk': 'nn',
            'norwegian': 'no',      'occitan': 'oc',        'punjabi': 'pa',       'polish': 'pl',
            'pashto': 'ps',         'portuguese': 'pt',     'romanian': 'ro',      'russian': 'ru',
            'sanskrit': 'sa',       'sindhi': 'sd',         'sinhala': 'si',       'slovak': 'sk',
            'slovenian': 'sl',      'shona': 'sn',          'somali': 'so',        'albanian': 'sq',
            'serbian': 'sr',        'sundanese': 'su',      'swedish': 'sv',       'swahili': 'sw',
            'tamil': 'ta',          'telugu': 'te',         'tajik': 'tg',         'thai': 'th',
            'turkmen': 'tk',        'tagalog': 'tl',        'turkish': 'tr',       'tatar': 'tt',
            'ukrainian': 'uk',      'urdu': 'ur',           'uzbek': 'uz',         'vietnamese': 'vi',
            'yiddish': 'yi',        'yoruba': 'yo',         'chinese': 'zh'
        }
    else:
        language_map = {'auto': 0, 'zh': 1, 'en': 2, 'yue': 3, 'ja': 4, 'ko': 5}
        full_language_names = {
            'auto': 'auto', '中文': 'zh', 'english': 'en', '粤语': 'yue', '日本語': 'ja', '한국인': 'ko'
        }
    if language_input in full_language_names:
        language_input = full_language_names[language_input]
    return language_map.get(language_input)


def get_task_id(task_input, use_v3):
    task_input = task_input.lower()
    if use_v3:
        task_map = {
            'translate': 50359,
            'transcribe': 50360
        }
        return task_map[task_input], 50363, 50364
    else:
        task_map = {
            'translate': 50358,
            'transcribe': 50359
        }
        return task_map[task_input], 50362, 50363


def remove_repeated_parts(ids, repeat_words_threshold):
    ids_len = len(ids)
    if ids_len <= repeat_words_threshold:
        return np.array([ids], dtype=np.int32)
    side_L = repeat_words_threshold // 2
    side_R = side_L + 1
    boundary = ids_len - side_L
    for i in range(side_L, boundary):
        for j in range(i + repeat_words_threshold, boundary):
            check = []
            for k in range(-side_L, side_R):
                if ids[j + k] == ids[i + k]:
                    check.append(True)
                else:
                    check.append(False)
                    break
            if False not in check:
                return np.array([ids[: j - side_L]], dtype=np.int32)
    return np.array([ids], dtype=np.int32)


def handle_sentence(text_list):
    do_again = False
    for i in range(len(text_list) - 1):
        if text_list[i]:
            if "@" in text_list[i]:
                text_list[i] = text_list[i].replace("@", "") + text_list[i + 1]
                text_list[i + 1] = ""
            if "@" in text_list[i]:
                do_again = True
    text_list = [word for word in text_list if word != '']
    return text_list, do_again


def process_timestamps(timestamps, fusion_threshold=1.0, min_duration=0.5):
    # Filter out short durations
    if min_duration > 0.0:
        filtered_timestamps = [(start, end) for start, end in timestamps if (end - start) > min_duration]
    else:
        filtered_timestamps = timestamps
    del timestamps 
    
    # Fuse and filter timestamps
    if fusion_threshold > 0.0:
        fused_timestamps = []
        for start, end in filtered_timestamps:
            if fused_timestamps and (start - fused_timestamps[-1][1] <= fusion_threshold):
                fused_timestamps[-1] = (
                    fused_timestamps[-1][0],
                    max(end, fused_timestamps[-1][1])
                )
            else:
                fused_timestamps.append((start, end))
        return fused_timestamps
    else:
        return filtered_timestamps


def vad_to_timestamps(vad_output, frame_duration):
    timestamps = []
    start = None
    # Extract raw timestamps
    for i, silence in enumerate(vad_output):
        if silence:
            if start is not None:  # End of the current speaking segment
                end = i * frame_duration + frame_duration
                timestamps.append((start, end))
                start = None
        else:
            if start is None:  # Start of a new speaking segment
                start = i * frame_duration
    # Handle the case where speech continues until the end
    if start is not None:
        timestamps.append((start, len(vad_output) * frame_duration))
    return timestamps


def format_time(seconds):
    """Convert seconds to VTT time format 'hh:mm:ss.mmm'."""
    td = timedelta(seconds=seconds)
    td_sec = td.total_seconds()
    total_seconds = int(td_sec)
    milliseconds = int((td_sec - total_seconds) * 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)


def handle_inputs(
        file_path_input,
        task,
        switcher_run_test,
        parallel_threads,
        hardware,
        model_llm,
        model_llm_accuracy,
        model_llm_custom_path,
        model_asr,
        model_whisper_custom_path,
        transcribe_language,
        translate_language,
        model_denoiser,
        switcher_denoiser_cache,
        model_vad,
        slider_vad_ONE_MINUS_SPEECH_THRESHOLD,
        slider_vad_SNR_THRESHOLD,
        slider_vad_BACKGROUND_NOISE_dB_INIT,
        slider_vad_SPEAKING_SCORE,
        slider_vad_SILENCE_SCORE,
        slider_vad_FUSION_THRESHOLD,
        slider_vad_MIN_SPEECH_DURATION,
        slider_vad_MAX_SPEECH_DURATION,
        slider_vad_MIN_SILENCE_DURATION,
        slider_denoise_factor,
        slider_vad_pad
):
    total_process_time = time.time()

    task_queue = []
    if os.path.isfile(file_path_input):
        if file_path_input.endswith(MEDIA_EXTENSIONS):
            task_queue.append(file_path_input)
        else:
            print(f"The specified path or file '{file_path_input}' does not exist or is not in a legal media format.")
    elif os.path.isdir(file_path_input):
        for file in os.listdir(file_path_input):
            if file.endswith(MEDIA_EXTENSIONS):
                task_queue.append(f"{file_path_input}/{file}")
            else:
                print(f"The specified path or file '{file_path_input}/{file}' does not exist or is not in a legal media format.")
    else:
        print(f"The specified path or file '{file_path_input}' does not exist or is not in a legal media format.")

    if len(task_queue) < 1:
        return f"The specified path or file '{file_path_input}' does not exist or is not in a legal media format."

    SAMPLE_RATE = 16000
    USE_DENOISED = True
    USE_V3 = False
    FIRST_RUN = True

    slider_denoise_factor_minus = float(1.0 - slider_denoise_factor)
    slider_denoise_factor = float(slider_denoise_factor)
    if "ZipEnhancer" in model_denoiser:
        denoiser_name = "ZipEnhancer"
    elif "GTCRN" in model_denoiser:
        denoiser_name = "GTCRN"
    elif "DFSMN" in model_denoiser:
        denoiser_name = "DFSMN"
        SAMPLE_RATE = 48000
    else:
        denoiser_name = "None"
        USE_DENOISED = False

    if os.name == 'nt':
        if hardware == "CPU":
            llm_special_set = False
        else:
            llm_special_set = True
        denoiser_os = "Windows"
    else:
        llm_special_set = False
        denoiser_os = "Linux"

    if USE_DENOISED:
        onnx_model_A = f"./Denoiser/{denoiser_os}/{denoiser_name}.ort"
        if os.path.isfile(onnx_model_A):
            print(f"\nFound the Denoiser-{denoiser_name}.")
        else:
            print(f"\nThe Denoiser-{denoiser_name} doesn't exist.\nPlease export it first.")
    else:
        onnx_model_A = None
        print("\nThis task is running without the denoiser.")

    if "FSMN" in model_vad:
        onnx_model_B = "./VAD/FSMN.ort"
        if os.path.isfile(onnx_model_B):
            vad_type = 0
            print("\nFound the VAD-FSMN.")
        else:
            print("\nThe VAD-FSMN doesn't exist.\nPlease export it first.")
            return "\nThe VAD-FSMN doesn't exist.\nPlease export it first."
    elif 'Faster_Whisper' in model_vad:
        if os.path.isdir(PYTHON_PACKAGE + "/faster_whisper"):
            vad_type = 1
            onnx_model_B = None
            print(f"\nFound the VAD-Faster_Whisper-Silero.")
        else:
            print("\nThe VAD-Faster_Whisper-Silero doesn't exist. Please run 'pip install faster-whisper --upgrade'")
            return "\nThe VAD-Faster_Whisper-Silero doesn't exist. Please run 'pip install faster-whisper --upgrade'"
    elif 'Official' in model_vad:
        if os.path.isdir(PYTHON_PACKAGE + "/silero_vad"):
            vad_type = 2
            onnx_model_B = None
            print(f"\nFound the VAD-Official_Silero.")
        else:
            print("\nThe VAD-Official_Silero doesn't exist. Please run 'pip install silero-vad --upgrade'")
            return "\nThe VAD-Official_Silero doesn't exist. Please run 'pip install silero-vad --upgrade'"
    else:
        if os.path.isfile("./VAD/pyannote_segmentation_3/pytorch_model.bin"):
            vad_type = 3
            onnx_model_B = None
            print(f"\nFound the VAD-Pyannote_Segmentation_3.0.")
        else:
            print("\nThe VAD-Pyannote_Segmentation_3.0 doesn't exist. Please run 'pip install pyannote.audio --upgrade and Download the pytorch_model.bin from https://huggingface.co/pyannote/segmentation-3.0'")
            return "\nThe VAD-Pyannote_Segmentation_3.0 doesn't exist. Please run 'pip install pyannote.audio --upgrade and Download the pytorch_model.bin from https://huggingface.co/pyannote/segmentation-3.0'"

    if "Whisper" in model_asr:
        asr_type = 0
        if "V3" in model_asr:
            ver = "V3"
            USE_V3 = True
        else:
            ver = "V2"
        if "Custom" in model_asr:
            if model_whisper_custom_path:
                path = model_whisper_custom_path
            else:
                print("\nPlease assign your custom Whisper path which contains Encoder.ort and Decoder.ort")
                return "\nPlease assign your custom Whisper path which contains Encoder.ort and Decoder.ort"
        elif "Turbo" in model_asr:
            path = f"./ASR/Whisper/{ver}/Turbo"
        else:
            path = f"./ASR/Whisper/{ver}/General"
        tokenizer = AutoTokenizer.from_pretrained(path)
        target_language_id = get_language_id(transcribe_language, True)
        if "Whisper" in model_llm:
            target_task_id = get_task_id('translate', USE_V3)[0]
        else:
            target_task_id = get_task_id('transcribe', USE_V3)[0]
        onnx_model_C = f"{path}/Whisper_Encoder.ort"
        onnx_model_D = f"{path}/Whisper_Decoder.ort"
        if os.path.isfile(onnx_model_C) and os.path.isfile(onnx_model_D):
            print("\nFound the ASR-Whisper.")
        else:
            print("\nThe ASR-Whisper doesn't exist.\nPlease export it first.")
            return "\nThe ASR-Whisper doesn't exist.\nPlease export it first."
    elif "SenseVoiceSmall" in model_asr:
        asr_type = 1
        tokenizer = SentencePieceProcessor()
        tokenizer.Load("./ASR/SenseVoiceSmall/chn_jpn_yue_eng_ko_spectok.bpe.model")
        target_language_id = get_language_id(transcribe_language, False)
        target_task_id = None
        onnx_model_C = f"./ASR/SenseVoiceSmall/SenseVoiceSmall.ort"
        onnx_model_D = None
        if os.path.isfile(onnx_model_C):
            print("\nFound the ASR-SenseVoiceSmall.")
        else:
            print("\nThe ASR-SenseVoiceSmall doesn't exist.\nPlease export it first.")
            return "\nThe ASR-SenseVoiceSmall doesn't exist.\nPlease export it first."
    else:
        if "Large" in model_asr:
            if "English" in transcribe_language:
                tokens_path = "./ASR/Paraformer/English/Large/tokens.json"
                onnx_model_C = "./ASR/Paraformer/English/Large/Paraformer_English.onnx"
                asr_type = 2
            else:
                tokens_path = "./ASR/Paraformer/Chinese/Large/tokens.json"
                onnx_model_C = "./ASR/Paraformer/Chinese/Large/Paraformer_Chinese.onnx"
                asr_type = 3
        else:
            tokens_path = "./ASR/Paraformer/Chinese/Small/tokens.json"
            onnx_model_C = "./ASR/Paraformer/Chinese/Small/Paraformer_Chinese.onnx"
            asr_type = 3
        with open(tokens_path, 'r', encoding='UTF-8') as json_file:
            tokenizer = np.array(json.load(json_file), dtype=np.str_)
        target_language_id = get_language_id(transcribe_language, False)
        target_task_id = None
        onnx_model_D = None
        if os.path.isfile(onnx_model_C):
            print("\nFound the ASR-Paraformer.")
        else:
            print("\nThe ASR-Paraformer doesn't exist.\nPlease export it first.")
            return "\nThe ASR-Paraformer doesn't exist.\nPlease export it first."

    # ONNX Runtime settings
    session_opts = onnxruntime.SessionOptions()
    session_opts.log_severity_level = 3                      # error level, it an adjustable value.
    session_opts.inter_op_num_threads = parallel_threads     # Run different nodes with num_threads. Set 0 for auto.
    session_opts.intra_op_num_threads = parallel_threads     # Under the node, execute the operators with num_threads. Set 0 for auto.
    session_opts.enable_cpu_mem_arena = True                 # True for execute speed; False for less memory usage.
    session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
    session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
    session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")

    if hardware == "CPU":
        device_type = "CPU"
        ORT_Accelerate_Providers = ['OpenVINOExecutionProvider']
    elif hardware == "Intel-GPU":
        device_type = 'GPU'
        ORT_Accelerate_Providers = ['OpenVINOExecutionProvider']
    elif hardware == "Intel-NPU":
        device_type = 'NPU'
        ORT_Accelerate_Providers = ['OpenVINOExecutionProvider']
    elif hardware == "Intel-AUTO_NPU_CPU":
        device_type = 'AUTO:NPU,CPU'
        ORT_Accelerate_Providers = ['OpenVINOExecutionProvider']
    elif hardware == "Intel-AUTO_GPU_CPU":
        device_type = 'AUTO:GPU,CPU'
        ORT_Accelerate_Providers = ['OpenVINOExecutionProvider']
    elif hardware == "Intel-AUTO_NPU_GPU":
        device_type = 'AUTO:NPU,GPU'
        ORT_Accelerate_Providers = ['OpenVINOExecutionProvider']
    elif hardware == "Intel-AUTO_ALL":
        device_type = 'AUTO:NPU,GPU,CPU'
        ORT_Accelerate_Providers = ['OpenVINOExecutionProvider']
    elif hardware == "Intel-HETERO_NPU_CPU":
        device_type = 'HETERO:NPU,CPU'
        ORT_Accelerate_Providers = ['OpenVINOExecutionProvider']
    elif hardware == "Intel-HETERO_GPU_CPU":
        device_type = 'HETERO:GPU,CPU'
        ORT_Accelerate_Providers = ['OpenVINOExecutionProvider']
    elif hardware == "Intel-HETERO_NPU_GPU":
        device_type = 'HETERO:NPU,GPU'
        ORT_Accelerate_Providers = ['OpenVINOExecutionProvider']
    elif hardware == "Intel-HETERO_ALL":
        device_type = 'HETERO:NPU,GPU,CPU'
        ORT_Accelerate_Providers = ['OpenVINOExecutionProvider']
    else:
        device_type = None
        ORT_Accelerate_Providers = ['CPUExecutionProvider']  # Apple-CPU

    if 'OpenVINOExecutionProvider' in ORT_Accelerate_Providers:
        provider_options = \
            [{
                'device_type': device_type,
                'precision': 'ACCURACY',
                'num_of_threads': parallel_threads,
                'num_streams': 1,
                'enable_opencl_throttling': True,
                'enable_qdq_optimizer': False
            }]
    else:
        provider_options = None

    print("----------------------------------------------------------------------------------------------------------")
    print("\nNow loading the required models.")
    if vad_type == 0:
        ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=['CPUExecutionProvider'], provider_options=None)
        print(f"\nVAD - Usable Providers: {ort_session_B.get_providers()}")
        in_name_B = ort_session_B.get_inputs()
        out_name_B = ort_session_B.get_outputs()
        in_name_B0 = in_name_B[0].name
        in_name_B1 = in_name_B[1].name
        in_name_B2 = in_name_B[2].name
        in_name_B3 = in_name_B[3].name
        in_name_B4 = in_name_B[4].name
        in_name_B5 = in_name_B[5].name
        in_name_B6 = in_name_B[6].name
        out_name_B0 = out_name_B[0].name
        out_name_B1 = out_name_B[1].name
        out_name_B2 = out_name_B[2].name
        out_name_B3 = out_name_B[3].name
        out_name_B4 = out_name_B[4].name
        out_name_B5 = out_name_B[5].name
    else:
        if vad_type == 2:
            silero_vad = load_silero_vad(session_opts=session_opts, providers=['CPUExecutionProvider'], provider_options=None)
        else:
            pyannote_vad = Model.from_pretrained("./VAD/pyannote_segmentation_3/pytorch_model.bin")
            pyannote_vad_pipeline = VoiceActivityDetection(segmentation=pyannote_vad)
            HYPER_PARAMETERS = {
                "min_duration_on": slider_vad_MIN_SPEECH_DURATION,
                "min_duration_off": slider_vad_FUSION_THRESHOLD
            }
            pyannote_vad_pipeline.instantiate(HYPER_PARAMETERS)
        print(f"\nVAD - Usable Providers: ['CPUExecutionProvider']")
        
    if asr_type == 0:
        ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=['CPUExecutionProvider'], provider_options=None)  # Use CPUExecutionProvider is better for ORT-int8
        ort_session_D = onnxruntime.InferenceSession(onnx_model_D, sess_options=session_opts, providers=['CPUExecutionProvider'], provider_options=None)  # Use CPUExecutionProvider is better for ORT-int8

        input_shape_C = ort_session_C._inputs_meta[0].shape[-1]
        in_name_C = ort_session_C.get_inputs()
        out_name_C = ort_session_C.get_outputs()
        in_name_C0 = in_name_C[0].name
        out_name_C0 = out_name_C[0].name
        out_name_C1 = out_name_C[1].name

        in_name_D = ort_session_D.get_inputs()
        out_name_D = ort_session_D.get_outputs()
        in_name_D0 = in_name_D[0].name
        in_name_D1 = in_name_D[1].name
        in_name_D2 = in_name_D[2].name
        in_name_D3 = in_name_D[3].name
        in_name_D4 = in_name_D[4].name
        in_name_D5 = in_name_D[5].name
        in_name_D6 = in_name_D[6].name
        in_name_D7 = in_name_D[7].name
        out_name_D0 = out_name_D[0].name
        out_name_D1 = out_name_D[1].name
        out_name_D2 = out_name_D[2].name
    else:
        if asr_type != 1:
            ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
            input_shape_C = ort_session_C._inputs_meta[0].shape[-1]
            in_name_C = ort_session_C.get_inputs()
            out_name_C = ort_session_C.get_outputs()
            in_name_C0 = in_name_C[0].name
            out_name_C0 = out_name_C[0].name
        else:
            ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=['CPUExecutionProvider'], provider_options=None)
            input_shape_C = ort_session_C._inputs_meta[0].shape[-1]
            in_name_C = ort_session_C.get_inputs()
            out_name_C = ort_session_C.get_outputs()
            in_name_C0 = in_name_C[0].name
            in_name_C1 = in_name_C[1].name
            out_name_C0 = out_name_C[0].name
    print(f"\nASR - Usable Providers: {ort_session_C.get_providers()}")

    for input_audio in task_queue:
        print(f"\nLoading the Input Media: {input_audio}")
        file_name = os.path.basename(input_audio).split(".")[0]
        if USE_DENOISED:
            if switcher_denoiser_cache and Path(f"./Cache/{file_name}_{denoiser_name}.wav").exists():
                print("\nThe denoised audio file already exists. Using the cache instead.")
                USE_DENOISED = False
                SAMPLE_RATE = 16000
                de_audio = np.array(AudioSegment.from_file(f"./Cache/{file_name}_{denoiser_name}.wav").set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
                audio = np.array(AudioSegment.from_file(input_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
                min_len = min(audio.shape[-1], de_audio.shape[-1])
                audio = (audio[:min_len] * slider_denoise_factor_minus + de_audio[:min_len] * slider_denoise_factor).clip(min=-32768.0, max=32767.0).astype(np.float32)
                audio = normalize_to_int16(audio)
                del de_audio
                if vad_type == 3:
                    sf.write(f"./Cache/{file_name}_vad.wav", audio, SAMPLE_RATE, format='WAVEX')
            else:
                audio = np.array(AudioSegment.from_file(input_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
                audio = normalize_to_int16(audio)
                if FIRST_RUN:
                    if denoiser_name == "ZipEnhancer":
                        ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
                    else:
                        ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'], provider_options=None)
                    in_name_A = ort_session_A.get_inputs()
                    out_name_A = ort_session_A.get_outputs()
                    in_name_A0 = in_name_A[0].name
                    out_name_A0 = out_name_A[0].name
                    print(f"\nDenoise - Usable Providers: {ort_session_A.get_providers()}")
        else:
            audio = np.array(AudioSegment.from_file(input_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
            audio = normalize_to_int16(audio)
            if vad_type == 3:
                sf.write(f"./Cache/{file_name}_vad.wav", audio, SAMPLE_RATE, format='WAVEX')
        if FIRST_RUN:
            print(f"\nModels have been successfully loaded.")
            print("----------------------------------------------------------------------------------------------------------")

        def process_segment_A(_inv_audio_len, _slice_start, _slice_end, _audio):
            return _slice_start * _inv_audio_len, ort_session_A.run([out_name_A0], {in_name_A0: _audio[:, :, _slice_start: _slice_end]})[0]

        def process_segment_C_sensevoice(_start, _end, _inv_audio_len, _audio, sample_rate, _language_idx):
            start_indices = _start * sample_rate
            audio_segment = _audio[:, :, int(start_indices): int(_end * sample_rate)]
            audio_len = audio_segment.shape[-1]
            if isinstance(input_shape_C, str):
                INPUT_AUDIO_LENGTH = min(320000, audio_len)  # You can adjust it.
            else:
                INPUT_AUDIO_LENGTH = input_shape_C
            stride_step = INPUT_AUDIO_LENGTH
            if audio_len > INPUT_AUDIO_LENGTH:
                num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
                total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
                pad_amount = total_length_needed - audio_len
                final_slice = audio_segment[:, :, -pad_amount:].astype(np.float32)
                white_noise = (np.sqrt(np.mean(final_slice * final_slice, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount))).astype(audio_segment.dtype)
                audio_segment = np.concatenate((audio_segment, white_noise), axis=-1)
            elif audio_len < INPUT_AUDIO_LENGTH:
                audio_segment_float = audio_segment.astype(np.float32)
                white_noise = (np.sqrt(np.mean(audio_segment_float * audio_segment_float, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH - audio_len))).astype(audio_segment.dtype)
                audio_segment = np.concatenate((audio_segment, white_noise), axis=-1)
            aligned_len = audio_segment.shape[-1]
            slice_start = 0
            slice_end = INPUT_AUDIO_LENGTH
            text = ""
            while slice_end <= aligned_len:
                token_ids = ort_session_C.run([out_name_C0], {in_name_C0: audio_segment[:, :, slice_start: slice_end], in_name_C1: _language_idx})[0]
                text += tokenizer.decode(token_ids.tolist())[0]
                slice_start += stride_step
                slice_end = slice_start + INPUT_AUDIO_LENGTH
            return start_indices * _inv_audio_len, text.strip() + ";", (_start, _end)

        def process_segment_C_paraformer_chinese(_start, _end, _inv_audio_len, _audio, sample_rate):
            start_indices = _start * sample_rate
            audio_segment = _audio[:, :, int(start_indices): int(_end * sample_rate)]
            audio_len = audio_segment.shape[-1]
            if isinstance(input_shape_C, str):
                INPUT_AUDIO_LENGTH = min(320000, audio_len)  # You can adjust it.
            else:
                INPUT_AUDIO_LENGTH = input_shape_C
            stride_step = INPUT_AUDIO_LENGTH
            if audio_len > INPUT_AUDIO_LENGTH:
                num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
                total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
                pad_amount = total_length_needed - audio_len
                final_slice = audio_segment[:, :, -pad_amount:].astype(np.float32)
                white_noise = (np.sqrt(np.mean(final_slice * final_slice, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount))).astype(audio_segment.dtype)
                audio_segment = np.concatenate((audio_segment, white_noise), axis=-1)
            elif audio_len < INPUT_AUDIO_LENGTH:
                audio_segment_float = audio_segment.astype(np.float32)
                white_noise = (np.sqrt(np.mean(audio_segment_float * audio_segment_float, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH - audio_len))).astype(audio_segment.dtype)
                audio_segment = np.concatenate((audio_segment, white_noise), axis=-1)
            aligned_len = audio_segment.shape[-1]
            slice_start = 0
            slice_end = INPUT_AUDIO_LENGTH
            save_text = ""
            while slice_end <= aligned_len:
                token_ids = ort_session_C.run([out_name_C0], {in_name_C0: audio_segment[:, :, slice_start: slice_end]})[0]
                text = tokenizer[token_ids[0]].tolist()
                if '</s>' in text:
                    text.remove('</s>')
                save_text += ''.join(text)
                slice_start += stride_step
                slice_end = slice_start + INPUT_AUDIO_LENGTH
            return start_indices * _inv_audio_len, save_text + ";", (_start, _end)

        def process_segment_C_paraformer_english(_start, _end, _inv_audio_len, _audio, sample_rate):
            start_indices = _start * sample_rate
            audio_segment = _audio[:, :, int(start_indices): int(_end * sample_rate)]
            audio_len = audio_segment.shape[-1]
            if isinstance(input_shape_C, str):
                INPUT_AUDIO_LENGTH = min(320000, audio_len)  # You can adjust it.
            else:
                INPUT_AUDIO_LENGTH = input_shape_C
            stride_step = INPUT_AUDIO_LENGTH
            if audio_len > INPUT_AUDIO_LENGTH:
                num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
                total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
                pad_amount = total_length_needed - audio_len
                final_slice = audio_segment[:, :, -pad_amount:].astype(np.float32)
                white_noise = (np.sqrt(np.mean(final_slice * final_slice, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount))).astype(audio_segment.dtype)
                audio_segment = np.concatenate((audio_segment, white_noise), axis=-1)
            elif audio_len < INPUT_AUDIO_LENGTH:
                audio_segment_float = audio_segment.astype(np.float32)
                white_noise = (np.sqrt(np.mean(audio_segment_float * audio_segment_float, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH - audio_len))).astype(audio_segment.dtype)
                audio_segment = np.concatenate((audio_segment, white_noise), axis=-1)
            aligned_len = audio_segment.shape[-1]
            slice_start = 0
            slice_end = INPUT_AUDIO_LENGTH
            save_text = ""
            while slice_end <= aligned_len:
                token_ids = ort_session_C.run([out_name_C0], {in_name_C0: audio_segment[:, :, slice_start: slice_end]})[0]
                text = tokenizer[token_ids[0]].tolist()
                if '</s>' in text:
                    text.remove('</s>')
                text, do_again = handle_sentence(text)
                while do_again:
                    text, do_again = handle_sentence(text)
                save_text += ' '.join(text)
                slice_start += stride_step
                slice_end = slice_start + INPUT_AUDIO_LENGTH
            return start_indices * _inv_audio_len, save_text + ";", (_start, _end)

        def process_segment_CD(_start, _end, _inv_audio_len, input_ids_, past_key_de_, past_value_de_, _audio, sample_rate):
            start_indices = _start * sample_rate
            audio_segment = _audio[:, :, int(start_indices): int(_end * sample_rate)]
            audio_len = audio_segment.shape[-1]
            if isinstance(input_shape_C, str):
                INPUT_AUDIO_LENGTH = min(327680, audio_len)  # You can adjust it.
            else:
                INPUT_AUDIO_LENGTH = input_shape_C
            stride_step = INPUT_AUDIO_LENGTH
            if audio_len > INPUT_AUDIO_LENGTH:
                num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
                total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
                pad_amount = total_length_needed - audio_len
                final_slice = audio_segment[:, :, -pad_amount:].astype(np.float32)
                white_noise = (np.sqrt(np.mean(final_slice * final_slice, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount))).astype(audio_segment.dtype)
                audio_segment = np.concatenate((audio_segment, white_noise), axis=-1)
            elif audio_len < INPUT_AUDIO_LENGTH:
                audio_segment_float = audio_segment.astype(np.float32)
                white_noise = (np.sqrt(np.mean(audio_segment_float * audio_segment_float, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH - audio_len))).astype(audio_segment.dtype)
                audio_segment = np.concatenate((audio_segment, white_noise), axis=-1)
            aligned_len = audio_segment.shape[-1]
            slice_start = 0
            slice_end = INPUT_AUDIO_LENGTH
            save_token = []
            while slice_end <= aligned_len:
                _input_ids = input_ids_
                _past_key_de = past_key_de_
                _past_value_de = past_value_de_
                _ids_len = np.array([input_ids_.shape[0]], dtype=np.int64)
                _history_len = np.array([0], dtype=np.int64)
                _attention_mask = np.array([-65504.0], dtype=np.float32)
                first_run = True
                save_encoder_key, save_encoder_value = ort_session_C.run([out_name_C0, out_name_C1], {in_name_C0: audio_segment[:, :, slice_start: slice_end]})
                while _history_len < MAX_SEQ_LEN:
                    _input_ids, _past_key_de, _past_value_de = ort_session_D.run(
                        [out_name_D0, out_name_D1, out_name_D2], {
                            in_name_D0: _input_ids,
                            in_name_D1: save_encoder_key,
                            in_name_D2: save_encoder_value,
                            in_name_D3: _past_key_de,
                            in_name_D4: _past_value_de,
                            in_name_D5: _ids_len,
                            in_name_D6: _history_len,
                            in_name_D7: _attention_mask
                        })
                    if STOP_TOKEN in _input_ids:
                        break
                    if first_run:
                        _history_len += _ids_len
                        _ids_len[0] = 1
                        _attention_mask[0] = 0.0
                        first_run = False
                    else:
                        _history_len += 1
                    save_token.append(_input_ids)
                    _input_ids = np.array([_input_ids], dtype=np.int32)
                slice_start += stride_step
                slice_end = slice_start + INPUT_AUDIO_LENGTH
            save_token_array = remove_repeated_parts(save_token, 4)  # To handle "over-talking".
            text, _ = tokenizer._decode_asr(
                [{
                    "tokens": save_token_array
                }],
                return_timestamps=None,  # Do not support return timestamps
                return_language=None,
                time_precision=0
            )
            return start_indices * _inv_audio_len, text + ";", (_start, _end)

        # Process audio
        audio_len = audio.shape[-1]
        if switcher_run_test:
            audio_len = audio_len // 10
            audio = audio[:audio_len]
        inv_audio_len = float(100.0 / audio_len)
        audio = audio.reshape(1, 1, -1)
        if USE_DENOISED:
            shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
            shape_value_out = ort_session_A._outputs_meta[0].shape[-1]
            if isinstance(shape_value_in, str):
                INPUT_AUDIO_LENGTH = min(32000, audio_len)  # You can adjust it.
            else:
                INPUT_AUDIO_LENGTH = shape_value_in
            stride_step = INPUT_AUDIO_LENGTH
            if audio_len > INPUT_AUDIO_LENGTH:
                if (shape_value_in != shape_value_out) & isinstance(shape_value_in, int) & isinstance(shape_value_out, int):
                    stride_step = shape_value_out
                num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
                total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
                pad_amount = total_length_needed - audio_len
                final_slice = audio[:, :, -pad_amount:].astype(np.float32)
                white_noise = (np.sqrt(np.mean(final_slice * final_slice, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount))).astype(audio.dtype)
                audio = np.concatenate((audio, white_noise), axis=-1)
            elif audio_len < INPUT_AUDIO_LENGTH:
                audio_float = audio.astype(np.float32)
                white_noise = (np.sqrt(np.mean(audio_float * audio_float, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH - audio_len))).astype(audio.dtype)
                audio = np.concatenate((audio, white_noise), axis=-1)
            aligned_len = audio.shape[-1]
            print("----------------------------------------------------------------------------------------------------------")
            print("\nDenoising the audio.")
            results = []
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
                futures = []
                slice_start = 0
                slice_end = INPUT_AUDIO_LENGTH
                while slice_end <= aligned_len:
                    futures.append(executor.submit(process_segment_A, inv_audio_len, slice_start, slice_end, audio))
                    slice_start += stride_step
                    slice_end = slice_start + INPUT_AUDIO_LENGTH
                for future in futures:
                    results.append(future.result())
                    print(f"Denoising: {results[-1][0]:.3f}%")
            end_time = time.time()
            results.sort(key=lambda x: x[0])
            saved = [result[1] for result in results]
            de_audio = (np.concatenate(saved, axis=-1))
            de_audio = de_audio[:, :, :audio_len]
            audio = audio[:, :, :audio_len].astype(np.float32) * slider_denoise_factor_minus + de_audio.astype(np.float32) * slider_denoise_factor
            if denoiser_name == "DFSMN":
                SAMPLE_RATE = 16000
                audio_len = audio_len // 3
                audio_len_3 = audio_len + audio_len + audio_len
                audio = np.sum(audio[:, :, :audio_len_3].reshape(-1, 3), axis=-1, dtype=np.float32).reshape(1, 1, -1)
                de_audio = np.sum(de_audio[:, :, :audio_len_3].reshape(-1, 3), axis=-1, dtype=np.float32).clip(min=-32768.0, max=32767.0).astype(np.int16)
                inv_audio_len = float(100.0 / audio_len)
            audio = audio.clip(min=-32768.0, max=32767.0).astype(np.int16)
            sf.write(f"./Cache/{file_name}_{denoiser_name}.wav", de_audio.reshape(-1), SAMPLE_RATE, format='WAVEX')
            if vad_type == 3:
                sf.write(f"./Cache/{file_name}_vad.wav", audio.reshape(-1), SAMPLE_RATE, format='WAVEX')
            print(f"Denoising Complete 100.00%.\nTime Cost: {(end_time - start_time):.3f} seconds.")
            del saved
            del results
            del de_audio

        # VAD parts.
        print("----------------------------------------------------------------------------------------------------------")
        print("\nNext, use the VAD model to extract key segments.")
        start_time = time.time()
        if vad_type == 0:
            shape_value_in = ort_session_B._inputs_meta[0].shape[-1]
            if isinstance(shape_value_in, str):
                INPUT_AUDIO_LENGTH = min(512, audio_len)  # You can adjust it.
            else:
                INPUT_AUDIO_LENGTH = shape_value_in
            stride_step = INPUT_AUDIO_LENGTH
            if audio_len > INPUT_AUDIO_LENGTH:
                num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
                total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
                pad_amount = total_length_needed - audio_len
                final_slice = audio[:, :, -pad_amount:].astype(np.float32)
                white_noise = (np.sqrt(np.mean(final_slice * final_slice, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount))).astype(audio.dtype)
                audio = np.concatenate((audio, white_noise), axis=-1)
            elif audio_len < INPUT_AUDIO_LENGTH:
                audio_float = audio.astype(np.float32)
                white_noise = (np.sqrt(np.mean(audio_float * audio_float, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH - audio_len))).astype(audio.dtype)
                audio = np.concatenate((audio, white_noise), axis=-1)
            audio_len = audio.shape[-1]
            inv_audio_len = float(100.0 / audio_len)
            cache_0 = np.zeros((1, 128, 19, 1), dtype=np.float32)  # FSMN_VAD model fixed cache shape. Do not edit it.
            noise_average_dB = np.array([slider_vad_BACKGROUND_NOISE_dB_INIT + slider_vad_SNR_THRESHOLD], dtype=np.float32)
            one_minus_speech_threshold = np.array([slider_vad_ONE_MINUS_SPEECH_THRESHOLD], dtype=np.float32)
            cache_1 = cache_0
            cache_2 = cache_0
            cache_3 = cache_0
            slider_vad_SNR_THRESHOLD_half = slider_vad_SNR_THRESHOLD * 0.5
            silence = True
            saved = []
            slice_start = 0
            slice_end = INPUT_AUDIO_LENGTH
            while slice_end <= audio_len:
                score, cache_0, cache_1, cache_2, cache_3, noisy_dB = ort_session_B.run(
                    [out_name_B0, out_name_B1, out_name_B2, out_name_B3, out_name_B4, out_name_B5],
                    {
                        in_name_B0: audio[:, :, slice_start: slice_end],
                        in_name_B1: cache_0,
                        in_name_B2: cache_1,
                        in_name_B3: cache_2,
                        in_name_B4: cache_3,
                        in_name_B5: one_minus_speech_threshold,
                        in_name_B6: noise_average_dB
                    })
                if silence:
                    if score >= slider_vad_SPEAKING_SCORE:
                        silence = False
                else:
                    if score <= slider_vad_SILENCE_SCORE:
                        silence = True
                saved.append(silence)
                noise_average_dB = 0.5 * (noise_average_dB + noisy_dB) + slider_vad_SNR_THRESHOLD_half
                print(f"VAD: {slice_start * inv_audio_len:.3f}%")
                slice_start += stride_step
                slice_end = slice_start + INPUT_AUDIO_LENGTH
            timestamps = vad_to_timestamps(saved, INPUT_AUDIO_LENGTH / SAMPLE_RATE)
            del saved
            del cache_0
            del cache_1
            del cache_2
            del cache_3
        else:
            if vad_type == 1:
                print("\nThe VAD-Faster_Whisper-Silero does not provide the running progress for visualization.")
                vad_options = {
                    'threshold': slider_vad_SPEAKING_SCORE,
                    'neg_threshold': slider_vad_SILENCE_SCORE,
                    'max_speech_duration_s': slider_vad_MAX_SPEECH_DURATION,
                    'min_speech_duration_ms': int(slider_vad_MIN_SPEECH_DURATION * 1000),
                    'min_silence_duration_ms': slider_vad_MIN_SILENCE_DURATION,
                    'speech_pad_ms': slider_vad_pad
                }
                timestamps = get_speech_timestamps_FW(
                    (audio.reshape(-1).astype(np.float32) * 0.000030517578),  # 1/32768
                    vad_options=VadOptions(**vad_options),
                    sampling_rate=SAMPLE_RATE
                )
                timestamps = [(item['start'] * inv_16k, item['end'] * inv_16k) for item in timestamps]
            elif vad_type == 2:
                print("\nThe VAD-Official-Silero does not provide the running progress for visualization.")
                timestamps = get_speech_timestamps(
                    torch.from_numpy(audio.reshape(-1).astype(np.float32) * 0.000030517578),  # 1/32768
                    model=silero_vad,
                    threshold=slider_vad_SPEAKING_SCORE,
                    neg_threshold=slider_vad_SILENCE_SCORE,
                    max_speech_duration_s=slider_vad_MAX_SPEECH_DURATION,
                    min_speech_duration_ms=int(slider_vad_MIN_SPEECH_DURATION * 1000),
                    min_silence_duration_ms=slider_vad_MIN_SILENCE_DURATION,
                    speech_pad_ms=slider_vad_pad,
                    return_seconds=True
                )
                timestamps = [(item['start'], item['end']) for item in timestamps]
            else:
                print("\nThe VAD-Pyannote_Segmentation_3.0 does not provide the running progress for visualization.")
                with torch.inference_mode():
                    timestamps = pyannote_vad_pipeline(f"./Cache/{file_name}_vad.wav")
                    segments = list(timestamps._tracks.keys())
                    total_seconds = audio_len * inv_16k
                    timestamps = []
                    slider_vad_pad_s = slider_vad_pad * 0.001
                    for segment in segments:
                        segment_start = segment.start - slider_vad_pad_s
                        segment_end = segment.end + slider_vad_pad_s
                        if segment_start < 0:
                            segment_start = 0
                        if segment_end > total_seconds:
                            segment_end = total_seconds
                        timestamps.append((segment_start, segment_end))
        timestamps = process_timestamps(timestamps, slider_vad_FUSION_THRESHOLD, slider_vad_MIN_SPEECH_DURATION)
        gc.collect()
        print(f"VAD Complete 100.00%.\nTime Cost: {(time.time() - start_time):.3f} seconds.")

        # ASR parts
        print("\nStart to transcribe task.")
        results = []
        start_time = time.time()
        if asr_type == 0:
            input_ids = np.array([50258, target_language_id, target_task_id], dtype=np.int32)
            past_key_de = np.zeros((ort_session_D._inputs_meta[3].shape[0], ort_session_D._inputs_meta[3].shape[1], 0, ort_session_D._inputs_meta[3].shape[-1]), dtype=np.float16)
            past_value_de = np.zeros((ort_session_D._inputs_meta[4].shape[0], ort_session_D._inputs_meta[4].shape[1], 0, ort_session_D._inputs_meta[4].shape[-1]), dtype=np.float16)
            with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
                futures = [executor.submit(process_segment_CD, start, end, inv_audio_len, input_ids, past_key_de, past_value_de, audio, SAMPLE_RATE) for start, end in timestamps]
                for future in futures:
                    results.append(future.result())
                    print(f"ASR: {results[-1][0]:.3f}%")
        elif asr_type == 1:
            language_idx = np.array([target_language_id], dtype=np.int32)
            with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
                futures = [executor.submit(process_segment_C_sensevoice, start, end, inv_audio_len, audio, SAMPLE_RATE, language_idx) for start, end in timestamps]
                for future in futures:
                    results.append(future.result())
                    print(f"ASR: {results[-1][0]:.3f}%")
        elif asr_type == 2:
            with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
                futures = [executor.submit(process_segment_C_paraformer_english, start, end, inv_audio_len, audio, SAMPLE_RATE) for start, end in timestamps]
                for future in futures:
                    results.append(future.result())
                    print(f"ASR: {results[-1][0]:.3f}%")
        else:
            with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
                futures = [executor.submit(process_segment_C_paraformer_chinese, start, end, inv_audio_len, audio, SAMPLE_RATE) for start, end in timestamps]
                for future in futures:
                    results.append(future.result())
                    print(f"ASR: {results[-1][0]:.3f}%")
        results.sort(key=lambda x: x[0])
        save_text = [result[1] for result in results]
        save_timestamps = [result[2] for result in results]
        print(f"ASR Complete 100.00%.\nTime Cost: {time.time() - start_time:.3f} Seconds")
        del audio
        del timestamps
        gc.collect()
        print("----------------------------------------------------------------------------------------------------------")
        print(f"\nSaving Results.")

        with open(f"./Results/Timestamps/{file_name}.txt", "w", encoding='UTF-8') as time_file, \
                open(f"./Results/Text/{file_name}.txt", "w", encoding='UTF-8') as text_file, \
                open(f"./Results/Subtitles/{file_name}.vtt", "w", encoding='UTF-8') as subtitles_file:

            subtitles_file.write("WEBVTT\n\n")  # Correct VTT header
            idx = 0
            for text, t_stamp in zip(save_text, save_timestamps):
                text = text.replace("\n", "")
                if asr_type == 1:
                    text = text.split("<|withitn|>")
                    transcription = ""
                    for i in range(1, len(text)):
                        if "<|zh|>" in text[i]:
                            text[i] = text[i].split("<|zh|>")[0]
                        elif "<|en|>" in text[i]:
                            text[i] = text[i].split("<|en|>")[0]
                        elif "<|yue|>" in text[i]:
                            text[i] = text[i].split("<|yue|>")[0]
                        elif "<|ja|>" in text[i]:
                            text[i] = text[i].split("<|ja|>")[0]
                        elif "<|ko|>" in text[i]:
                            text[i] = text[i].split("<|ko|>")[0]
                        transcription += text[i]
                else:
                    transcription = text

                start_sec = t_stamp[0]
                if t_stamp[1] - start_sec > 8.0:
                    markers = re.split(r'([。、,.!?？;])', transcription)  # Keep markers in results
                    text_chunks = ["".join(markers[i:i + 2]) for i in range(0, len(markers), 2)]
                    time_per_chunk = (t_stamp[1] - start_sec) / len(text_chunks)
                    if len(text_chunks) > 2:
                        for i, chunk in enumerate(text_chunks):
                            chunk_start = start_sec + i * time_per_chunk
                            chunk_end = chunk_start + time_per_chunk
                            chunk = chunk.replace(";", "")
                            if chunk and chunk != "。" and chunk != ".":
                                start_time = format_time(chunk_start)
                                end_time = format_time(chunk_end)
                                timestamp = f"{start_time} --> {end_time}\n"
                                time_file.write(timestamp)
                                text_file.write(f"{chunk}\n")
                                subtitles_file.write(f"{idx}\n{timestamp}{chunk}\n\n")
                                idx += 1
                    else:
                        transcription = transcription.replace(";", "")
                        if transcription and transcription != "。" and transcription != ".":
                            start_time = format_time(start_sec)
                            end_time = format_time(t_stamp[1])
                            timestamp = f"{start_time} --> {end_time}\n"
                            time_file.write(timestamp)
                            text_file.write(f"{transcription}\n")
                            subtitles_file.write(f"{idx}\n{timestamp}{transcription}\n\n")
                            idx += 1
                else:
                    transcription = transcription.replace(";", "")
                    if transcription and transcription != "。" and transcription != ".":
                        start_time = format_time(start_sec)
                        end_time = format_time(t_stamp[1])
                        timestamp = f"{start_time} --> {end_time}\n"
                        time_file.write(timestamp)
                        text_file.write(f"{transcription}\n")
                        subtitles_file.write(f"{idx}\n{timestamp}{transcription}\n\n")
                        idx += 1
            del save_text
            del save_timestamps
        print(f"\nTranscribe Tasks Complete.\n\nTranscribe Time: {(time.time() - total_process_time):.3f} Seconds.\n\nThe subtitles are saved in the folder ./Result/Subtitles\n")

        if "Translate" not in task:
            continue
        else:
            start_time = time.time()
            if FIRST_RUN:
                system_ram = psutil.virtual_memory().total * inv_1024
                low_mem = False
                if system_ram < 17:
                    if (("8" in model_llm_accuracy) or ("16" in model_llm_accuracy)) and ("Whisper" not in model_llm):
                        print("\nWarning for the low memory system with 8 bit or 16 bit LLM accuracy. Try to using the 4 bit or lower bit instead.")
                        low_mem = True
                if model_llm == "Custom-GGUF-LLM":
                    MAX_TRANSLATE_LINES = 28  # Default
                    if system_ram < 9:
                        low_mem = True
                elif model_llm == "Gemma-2-2B-it":
                    llm_path = "./LLM/Gemma/2B"
                    MAX_TRANSLATE_LINES = 8
                elif model_llm == "Gemma-2-9B-it":
                    model_llm = "./LLM/Gemma/9B"
                    MAX_TRANSLATE_LINES = 28
                    if system_ram < 9:
                        low_mem = True
                elif model_llm == "GLM-4-9B-Chat":
                    llm_path = "./LLM/GLM/9B"
                    MAX_TRANSLATE_LINES = 28
                    if system_ram < 9:
                        low_mem = True
                elif model_llm == "MiniCPM3-4B":
                    llm_path = "./LLM/MiniCPM/4B"
                    MAX_TRANSLATE_LINES = 12
                elif model_llm == "Phi-3.5-mini-Instruct":
                    llm_path = "./LLM/Phi/mini"
                    MAX_TRANSLATE_LINES = 12
                elif model_llm == "Phi-3-medium-128k-Instruct":
                    llm_path = "./LLM/Phi/medium"
                    MAX_TRANSLATE_LINES = 24
                    if system_ram < 9:
                        low_mem = True
                elif model_llm == "Qwen2.5-3B-Instruct":
                    llm_path = "./LLM/Qwen/3B"
                    MAX_TRANSLATE_LINES = 8
                elif model_llm == "Qwen2.5-7B-Instruct":
                    llm_path = "./LLM/Qwen/7B"
                    MAX_TRANSLATE_LINES = 24
                    if system_ram < 9:
                        low_mem = True
                elif model_llm == "Qwen2.5-14B-Instruct":
                    llm_path = "./LLM/Qwen/14B"
                    MAX_TRANSLATE_LINES = 36
                    if system_ram < 9:
                        low_mem = True
                elif model_llm == "Qwen2.5-32B-Instruct":
                    llm_path = "./LLM/Qwen/32B"
                    MAX_TRANSLATE_LINES = 60
                    if system_ram < 9:
                        low_mem = True
                elif model_llm == "Whisper":
                    print(f"\nTranslate tasks complete.")
                    continue
                else:
                    print("Can not find the LLM model for translation task.")
                    return "Can not find the LLM model for translation task."

                TRANSLATE_OVERLAP = MAX_TRANSLATE_LINES // 4
                MAX_TOKENS_PER_CHUNK = MAX_TRANSLATE_LINES * MAX_SEQ_LEN

                if translate_language == "中文":
                    translate_language = "chinese"
                elif translate_language == '日本語':
                    translate_language = 'japanese'
                elif translate_language == '한국인':
                    translate_language = 'korean'
                translate_language = translate_language[0].upper() + translate_language[1:]

                if transcribe_language == "中文":
                    transcribe_language = "chinese"
                elif transcribe_language == '日本語':
                    transcribe_language = 'japanese'
                elif transcribe_language == '한국인':
                    transcribe_language = 'korean'
                elif transcribe_language == 'Auto':
                    transcribe_language = "unknown language"
                transcribe_language = transcribe_language[0].upper() + transcribe_language[1:]

                if os.path.isfile(model_llm_custom_path) and (("gguf" in model_llm_custom_path) or ("GGUF" in model_llm_custom_path)):
                    translation_model = AutoModelForCausalLM.from_gguf(model_llm_custom_path, cpu_embedding=False if llm_special_set else True)
                else:
                    translation_model = AutoModelForCausalLM.from_pretrained(
                        llm_path,
                        trust_remote_code=True,
                        use_cache=True,
                        load_in_low_bit=model_llm_accuracy,
                        optimize_model=True,
                        cpu_embedding=False if llm_special_set else True,
                        speculative=False,
                        disk_embedding=low_mem,
                        lightweight_bmm=True if llm_special_set else False,
                        embedding_qtype='q2_k' if low_mem else 'q4_k',
                        mixed_precision=False,
                        pipeline_parallel_stages=1
                    )
                tokenizer_llm = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
                FIRST_RUN = False

            with open(f"./Results/Text/{file_name}.txt", 'r', encoding='utf-8') as asr_file:
                asr_lines = asr_file.readlines()

            with open(f"./Results/Timestamps/{file_name}.txt", 'r', encoding='utf-8') as timestamp_file:
                timestamp_lines = timestamp_file.readlines()

            for line_index in range(len(asr_lines)):
                asr_lines[line_index] = f"{line_index}-{asr_lines[line_index]}"

            total_lines = len(asr_lines)
            if total_lines < 1:
                print("\nEmpty content for translation task.")
                continue

            print("\nStart to LLM Translate.\n")
            inv_total_lines = float(100.0 / total_lines)
            step_size = MAX_TRANSLATE_LINES - TRANSLATE_OVERLAP
            translated_responses = []
            with torch.inference_mode():
                for chunk_start in range(0, total_lines, step_size):
                    chunk_end = min(total_lines, chunk_start + MAX_TRANSLATE_LINES)
                    translation_prompt = "".join(asr_lines[chunk_start:chunk_end])
                    conversation_context = [
                        {
                            "role": "system",
                            "content": f"Translate the provided subtitles from {transcribe_language} to {translate_language}, and return them strictly in the defined 'ID-translation_results' format. Fix transcription errors (missing punctuation, homophones, omitted words) and enhance fluency using context from preceding and succeeding lines. Ensure translations are logical, natural, emotionally rich, and maintain smooth transitions both within and across lines. Preserve and enrich the intended meaning and tone."
                        },
                        {
                            "role": "user",
                            "content": translation_prompt
                        }
                    ]
                    tokenized_input = tokenizer_llm.apply_chat_template(
                        conversation=conversation_context,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    model_inputs = tokenizer_llm([tokenized_input], return_tensors="pt")
                    generated_ids = translation_model.generate(model_inputs.input_ids, max_new_tokens=MAX_TOKENS_PER_CHUNK)
                    decoded_response = tokenizer_llm.batch_decode(generated_ids, skip_special_tokens=True)[0].split("assistant", 1)[-1]
                    if chunk_start > 0:
                        decoded_response = "\n".join(decoded_response.split("\n")[TRANSLATE_OVERLAP + 1:])
                    translated_responses.append(decoded_response)
                    print(f"\nTranslating - {chunk_end * inv_total_lines:.3f}%")
                    print(decoded_response)
                    if chunk_end == total_lines - 1:
                        break
                print(f"\nTranslating - 100.00%")
                merged_responses = "\n".join(translated_responses).split("\n")
                with open(f"./Results/Subtitles/{file_name}_translated.vtt", "w", encoding='UTF-8') as subtitles_file:
                    subtitles_file.write("WEBVTT\n\n")
                    idx = 0
                    timestamp_len = len(timestamp_lines)
                    for i in range(len(merged_responses)):
                        response_line = merged_responses[i]
                        if response_line:
                            parts = response_line.split("-")
                            if len(parts) > 1:
                                if parts[0].isdigit():
                                    line_index = int(parts[0])
                                    if line_index < timestamp_len:
                                        subtitles_file.write(f"{idx}\n{timestamp_lines[line_index]}{parts[-1]}\n\n")
                                        idx += 1
            print(f"\nLLM Translate Complete. Processing time: {time.time() - start_time:.3f} seconds")
    print(f"All tasks complete.\n\nTotal Time: {(time.time() - total_process_time):.3f} seconds.\n\nThe subtitles are saved in the folder ./Result/Subtitles\n")
    return f"All tasks complete.\n\nTotal Time: {(time.time() - total_process_time):.3f} seconds.\n\nThe subtitles are saved in the folder ./Result/Subtitles\n"


with gr.Blocks(css=".gradio-container { background-color: black; }", fill_height=False, fill_width=False, theme=gr.themes.Citrus(text_size=sizes.text_lg)) as GUI:
    gr.Markdown("<span style='font-size: 32px; font-weight: bold; color: #fcfc1e;'>Subtitles is All You Need</span>")
    with gr.Row():
        with gr.Column():
            gr.Image("./Icon/psyduck.png", type="filepath", show_download_button=False, show_fullscreen_button=False)
        with gr.Column():
            ui_language = gr.Dropdown(
                choices=["English", "中文", "日本語"],
                label="User Interface / 用户界面 / ユーザーインターフェース",
                info="Choose a language you are comfortable using. / 选择一种觉得舒适使用的语言。 / 使用していて快適な言語を選んでください。",
                value="English",
                visible=True
            )
        with gr.Column():
            task = gr.Dropdown(
                choices=["Transcribe / 转录 / 転写", "Transcribe+Translate / 转录+翻译 / 転写+翻訳"],
                label="Task",
                info="Choose the operation to perform on the audio.",
                value="Transcribe+Translate / 转录+翻译 / 転写+翻訳",
                visible=True
            )
        with gr.Column():
            switcher_run_test = gr.Checkbox(
                label="Run Test",
                info="Run a short test on 10% of the audio length.",
                value=False
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("<span style='font-size: 24px; font-weight: bold; color: #68fc1e;'>System Settings</span>")
        with gr.Column():
            file_path_input = gr.Textbox(
                label="Video / Audio File Path",
                info="Enter the path of the video/audio file or folder you want to transcribe.",
                value="./Media",
                visible=True
            )
        with gr.Column():
            parallel_threads = gr.Slider(
                1, 64, step=1, label="Parallel Threads",
                info="Number of cores for parallel processing.",
                value=4,
                visible=True
            )
        with gr.Column():
            hardware = gr.Dropdown(
                choices=["CPU", "Intel-GPU", "Intel-NPU", "Intel-AUTO_NPU_CPU", "Intel-AUTO_GPU_CPU", "Intel-AUTO_NPU_GPU",
                         "Intel-AUTO_ALL", "Intel-HETERO_NPU_CPU", "Intel-HETERO_GPU_CPU", "Intel-HETERO_NPU_GPU", "Intel-HETERO_ALL"],
                label="Hardware",
                info="Select the device for the task.",
                value="CPU",
                visible=True
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("<span style='font-size: 24px; font-weight: bold; color: #fb7450;'>Model Selection</span>")
        with gr.Column():
            model_llm = gr.Dropdown(
                choices=["Custom-GGUF-LLM", "Gemma-2-2B-it", "Gemma-2-9B-it", "GLM-4-9B-Chat", "MiniCPM3-4B", "Phi-3.5-mini-Instruct", "Phi-3-medium-128k-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct", "Whisper"],
                label="LLM Model",
                info="Model used for translation.",
                value="Qwen2.5-7B-Instruct",
                visible=True
            )
            model_llm_custom_path = gr.Textbox(
                label="Custom GGUF-LLM Path",
                info="Path to your GGUF-LLM model.",
                value="",
                visible=False
            )
        with gr.Column():
            model_llm_accuracy = gr.Dropdown(
                choices=["sym_int4", "asym_int4", "sym_int5", "asym_int5", "sym_int8", "gguf_iq2_xxs", "gguf_iq2_xs", "gguf_iq1_s", "gguf_q4k_s", "gguf_q4k_m"],
                label="LLM Accuracy",
                info="Dtype used in LLM translation.",
                value="sym_int4",
                visible=True
            )
        with gr.Column():
            model_asr = gr.Dropdown(
                choices=["SenseVoiceSmall-Fast", "Whisper-Large-V2", "Whisper-Large-V3", "Whisper-Large-V3-Turbo", "Custom-Fine-Tune-Whisper-V2", "Custom-Fine-Tune-Whisper-V3", "Paraformer-Small", "Paraformer-Large"],
                label="ASR Model",
                info="Model used for transcription.",
                value="SenseVoiceSmall-Fast",
                visible=True
            )
            model_whisper_custom_path = gr.Textbox(
                label="Custom Whisper Path",
                info="Path to your fine-tuned Whisper model. Including: Whisper_Encoder.ort, Whisper_Decoder.ort, tokenizer.json ...",
                value="",
                visible=False
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("<span style='font-size: 24px; font-weight: bold; color: #50ccfb;'>Target Language</span>")
        with gr.Column():
            transcribe_language = gr.Dropdown(
                choices=["日本語", "中文", "English", "粤语", "한국인", "Auto"],
                label="Transcription Language",
                info="Language of the input media.",
                value="日本語",
                visible=True
            )
        with gr.Column():
            pass
        with gr.Column():
            translate_language = gr.Dropdown(
                choices=[
                    "中文",         "English",          "日本語",         "한국인",
                    "afrikaans",    "amharic",          "arabic",        "assamese",      "azerbaijani",
                    "bashkir",      "belarusian",       "bulgarian",     "bengali",       "tibetan",
                    "breton",       "bosnian",          "catalan",       "czech",         "welsh",
                    "danish",       "german",           "greek",         "english",       "spanish",
                    "estonian",     "basque",           "persian",       "finnish",       "faroese",
                    "french",       "galician",         "gujarati",      "hawaiian",      "hausa",
                    "hebrew",       "hindi",            "croatian",      "haitian creole","hungarian",
                    "armenian",     "indonesian",       "icelandic",     "italian",       "japanese",
                    "javanese",     "georgian",         "kazakh",        "khmer",         "kannada",
                    "korean",       "latin",            "luxembourgish", "lingala",       "lao",
                    "lithuanian",   "latvian",          "malagasy",      "maori",         "macedonian",
                    "malayalam",    "mongolian",        "marathi",       "malay",         "maltese",
                    "burmese",      "nepali",           "dutch",         "nynorsk",       "norwegian",
                    "occitan",      "punjabi",          "polish",        "pashto",        "portuguese",
                    "romanian",     "russian",          "sanskrit",      "sindhi",        "sinhala",
                    "slovak",       "slovenian",        "shona",         "somali",        "albanian",
                    "serbian",      "sundanese",        "swedish",       "swahili",       "tamil",
                    "telugu",       "tajik",            "thai",          "turkmen",       "tagalog",
                    "turkish",      "tatar",            "ukrainian",     "urdu",          "uzbek",
                    "vietnamese",   "yiddish",          "yoruba",        "chinese"
                ],
                label="Translation Language",
                info="Language to translate into.",
                value="中文",
                visible=True
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("<span style='font-size: 24px; font-weight: bold; color: #f968f3;'>Audio Processor</span>")
        with gr.Column():
            model_denoiser = gr.Dropdown(
                choices=["None", "ZipEnhancer-Time Consume", "GTCRN", "DFSMN"],
                label="Denoiser",
                info="Choose a denoiser for audio processing.",
                value="None",
                visible=True
            )
        with gr.Column():
            switcher_denoiser_cache = gr.Checkbox(
                label="Use Cache",
                info="Use previous denoising results to save time.",
                value=True,
                visible=False
            )
        with gr.Column():
            model_vad = gr.Dropdown(
                choices=["FSMN", 'Faster_Whisper-Silero', "Official-Silero", "Pyannote_3.0-Time Consume"],
                label="VAD",
                info="Select the VAD used for audio processing: Silero performs better in noisy audio, while FSMN excels in Chinese audio environments.",
                value="Faster_Whisper-Silero",
                visible=True
            )
        slider_denoise_factor = gr.Slider(
            0.1, 1.0, step=0.025, label="Denoise Factor",
            info="A larger value enhances the denoising effect.",
            value=0.5,
            visible=False
        )
    with gr.Row():
        with gr.Column():
            gr.Markdown("<span style='font-size: 24px; font-weight: bold; color: #fdfefe;'>VAD Configurations</span>")
            slider_vad_pad = gr.Slider(
                0, 1000, step=10, label="VAD Padding",
                info="Add padding to the start and end of the timestamps. Unit: milliseconds.",
                value=400,
                visible=True
            )
            slider_vad_ONE_MINUS_SPEECH_THRESHOLD = gr.Slider(
                0, 1, step=0.025, label="Voice State Threshold",
                info="FSMN VAD parameter for sensitivity.",
                value=1.0,
                visible=False
            )
            slider_vad_SNR_THRESHOLD = gr.Slider(
                0, 60, step=1, label="SNR Threshold",
                info="FSMN VAD parameter for noise classification.",
                value=10,
                visible=False
            )
            slider_vad_BACKGROUND_NOISE_dB_INIT = gr.Slider(
                0, 100, step=1, label="Initial Background Noise",
                info="Initial background noise value in dB.",
                value=40,
                visible=False
            )
            slider_vad_SPEAKING_SCORE = gr.Slider(
                0, 1, step=0.025, label="Voice State Score",
                info="A larger value makes activation more difficult.",
                value=0.4,
                visible=True
            )
            slider_vad_SILENCE_SCORE = gr.Slider(
                0, 1, step=0.025, label="Silence State Score",
                info="A larger value makes it easier to cut off speaking.",
                value=0.25,
                visible=True
            )
            slider_vad_FUSION_THRESHOLD = gr.Slider(
                0, 5, step=0.025, label="Merge Timestamps",
                info="If two voice segments are too close, they will be merged into one. Unit: seconds.",
                value=0.0,
                visible=True
            )
            slider_vad_MIN_SPEECH_DURATION = gr.Slider(
                0, 2, step=0.025, label="Filter Short Voice",
                info="Minimum duration for voice filtering. Unit: seconds.",
                value=0.05,
                visible=True
            )
            slider_vad_MAX_SPEECH_DURATION = gr.Slider(
                1, 30, step=1, label="Filter Long Voice",
                info="Maximum voice duration. Unit: seconds.",
                value=20,
                visible=True
            )
            slider_vad_MIN_SILENCE_DURATION = gr.Slider(
                100, 3000, step=50, label="Silence Duration Judgment",
                info="Minimum silence duration. Unit: milliseconds.",
                value=1500,
                visible=True
            )

    task_state = gr.Textbox(
        label="Task State",
        value="Click the Run and Wait a Moment.",
        interactive=False,
        visible=True
    )

    # Add a submit button to handle inputs
    submit_button = gr.Button("Run Task")
    submit_button.click(
        fn=handle_inputs,
        inputs=[
            file_path_input,
            task,
            switcher_run_test,
            parallel_threads,
            hardware,
            model_llm,
            model_llm_accuracy,
            model_llm_custom_path,
            model_asr,
            model_whisper_custom_path,
            transcribe_language,
            translate_language,
            model_denoiser,
            switcher_denoiser_cache,
            model_vad,
            slider_vad_ONE_MINUS_SPEECH_THRESHOLD,
            slider_vad_SNR_THRESHOLD,
            slider_vad_BACKGROUND_NOISE_dB_INIT,
            slider_vad_SPEAKING_SCORE,
            slider_vad_SILENCE_SCORE,
            slider_vad_FUSION_THRESHOLD,
            slider_vad_MIN_SPEECH_DURATION,
            slider_vad_MAX_SPEECH_DURATION,
            slider_vad_MIN_SILENCE_DURATION,
            slider_denoise_factor,
            slider_vad_pad
        ],
        outputs=task_state
    )
    ui_language.change(
        fn=update_ui,
        inputs=ui_language,
        outputs=[
            file_path_input,
            task,
            switcher_run_test,
            parallel_threads,
            hardware,
            model_llm,
            model_llm_accuracy,
            model_llm_custom_path,
            model_asr,
            model_whisper_custom_path,
            transcribe_language,
            translate_language,
            model_denoiser,
            switcher_denoiser_cache,
            model_vad,
            slider_vad_ONE_MINUS_SPEECH_THRESHOLD,
            slider_vad_SNR_THRESHOLD,
            slider_vad_BACKGROUND_NOISE_dB_INIT,
            slider_vad_SPEAKING_SCORE,
            slider_vad_SILENCE_SCORE,
            slider_vad_FUSION_THRESHOLD,
            slider_vad_MIN_SPEECH_DURATION,
            slider_vad_MAX_SPEECH_DURATION,
            slider_vad_MIN_SILENCE_DURATION,
            slider_denoise_factor,
            slider_vad_pad
        ]
    )
    task.change(
        fn=update_task,
        inputs=task,
        outputs=[model_llm, translate_language, model_llm_custom_path, model_llm_accuracy]
    )
    hardware.change(
        fn=update_model_llm_accuracy,
        inputs=hardware,
        outputs=model_llm_accuracy
    )
    model_llm.change(
        fn=update_translate_language,
        inputs=model_llm,
        outputs=[translate_language, model_llm_custom_path, model_llm_accuracy, model_asr]
    )
    model_asr.change(
        fn=update_asr,
        inputs=model_asr,
        outputs=[model_whisper_custom_path, transcribe_language]
    )
    model_denoiser.change(
        fn=update_denoiser,
        inputs=model_denoiser,
        outputs=[switcher_denoiser_cache, slider_denoise_factor]
    )
    model_vad.change(
        fn=update_vad,
        inputs=model_vad,
        outputs=[
            slider_vad_SPEAKING_SCORE,
            slider_vad_SILENCE_SCORE,
            slider_vad_ONE_MINUS_SPEECH_THRESHOLD,
            slider_vad_SNR_THRESHOLD,
            slider_vad_BACKGROUND_NOISE_dB_INIT,
            slider_vad_MAX_SPEECH_DURATION,
            slider_vad_MIN_SILENCE_DURATION,
            slider_vad_FUSION_THRESHOLD,
            slider_vad_MIN_SPEECH_DURATION,
            slider_vad_pad
        ]
    )

# Launch the app
GUI.launch()
