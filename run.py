# Standard library imports
import os
import gc
import re
import json
import time
import site
import shutil
import platform
from pathlib import Path
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor

# Third-party imports
import psutil
import onnxruntime
import numpy as np
import librosa
import gradio as gr
import soundfile as sf
from pydub import AudioSegment
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from transformers import AutoTokenizer
from sentencepiece import SentencePieceProcessor
from silero_vad import load_silero_vad, get_speech_timestamps
from faster_whisper.vad import get_speech_timestamps as get_speech_timestamps_FW, VadOptions
from ASR.FireRedASR.AED.L.aed_tokenizer import ChineseCharEnglishSpmTokenizer
from VAD.TEN.include.ten_vad import TenVad


physical_cores = psutil.cpu_count(logical=False)
print(f"\n找到 {physical_cores} 个物理 CPU 核心。Found {physical_cores} physical CPU cores.\n")


def update_task(dropdown_task):
    if "转录 + 翻译 Transcribe + Translate" == dropdown_task:
        update_A = gr.update(visible=True)
        update_B = gr.update(visible=True)
    else:
        update_A = gr.update(visible=False)
        update_B = gr.update(visible=False)
    return update_A, update_B


def update_translate_language(dropdown_model_llm):
    if "Whisper" in dropdown_model_llm:
        update_A = gr.update(choices=["English"], value="English")
    else:
        update_A = gr.update(
            choices=[
                "中文",         "English",          "日語",         "한국인",
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
    return update_A


def update_transcribe_language(dropdown_model_asr):
    if "English" in dropdown_model_asr:
        update_A = gr.update(value="English", choices=["English"])
    elif ("Japanese" in dropdown_model_asr) or ("Anime" in dropdown_model_asr):
        update_A = gr.update(value="日語", choices=["日語"])
    elif "Whisper" in dropdown_model_asr:
        update_A = gr.update(
            value="日語",
            choices=[
                "中文",         "English",          "日語",         "한국인",
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
    elif "SenseVoice-Small" == dropdown_model_asr:
        update_A = gr.update(value="日語", choices=["日語", "中文", "English", "粤语", "한국인", "自动 auto"])
    elif ("Paraformer-Small" == dropdown_model_asr) or ("FireRedASR-AED-L" == dropdown_model_asr):
        update_A = gr.update(value="中文", choices=["中文"])
    elif "Paraformer-Large" == dropdown_model_asr:
        update_A = gr.update(value="中文", choices=["中文", "English"])
    elif "Dolphin" in dropdown_model_asr:
        update_A = gr.update(
            value="Auto-Auto",
            choices=[
                # Auto-detection options
                "Auto-Auto",             "Mandarin-Auto",          "Yue-Auto",              "Tamil-Auto",
                "Urdu-Auto",             "Arabic-Auto",
                "自动-自动",               "中文-自动",               "粤语-自动",               "泰米尔语-自动",
                "乌尔都语-自动",            "阿拉伯语-自动",

                # Chinese varieties (English)
                "Chinese-Mandarin",       "Chinese-Taiwan",        "Chinese-Wuyu",          "Chinese-Sichuan",
                "Chinese-Shanxi",         "Chinese-Anhui",         "Chinese-Tianjin",       "Chinese-Ningxia",
                "Chinese-Shaanxi",        "Chinese-Hebei",         "Chinese-Shandong",      "Chinese-Guangdong",
                "Chinese-Shanghai",       "Chinese-Hubei",         "Chinese-Liaoning",      "Chinese-Gansu",
                "Chinese-Fujian",         "Chinese-Hunan",         "Chinese-Henan",         "Chinese-Yunnan",
                "Chinese-Minnan",         "Chinese-Wenzhou",

                # Chinese varieties (Chinese)
                "中文-普通话",             "中文-台湾",               "中文-吴语",              "中文-四川话",
                "中文-山西话",             "中文-安徽话",             "中文-天津话",             "中文-宁夏话",
                "中文-陕西话",             "中文-河北话",             "中文-山东话",             "中文-广东话",
                "中文-上海话",             "中文-湖北话",             "中文-辽宁话",             "中文-甘肃话",
                "中文-福建话",             "中文-湖南话",             "中文-河南话",             "中文-云南话",
                "中文-闽南语",             "中文-温州话",

                # Cantonese and East Asian languages
                "Yue-Unknown",           "Yue-Hongkong",           "Yue-Guangdong",         "Japanese",
                "Korean",                "Thai",                   "Vietnamese",
                "粤语-未知",               "粤语-香港",               "粤语-广东",              "日语",
                "韩语",                   "泰语",                    "越南语",

                # Southeast Asian languages
                "Indonesian",            "Malay",                  "Burmese",               "Tagalog",
                "Khmer",                 "Javanese",               "Lao",                   "Filipino",            "Sundanese",
                "印度尼西亚语",            "马来语",                  "缅甸语",                 "塔加洛语",
                "高棉语",                 "爪哇语",                  "老挝语",                 "菲律宾语",              "巽他语",

                # South Asian languages
                "Hindi",                 "Bengali",                "Tamil-Singaporean",     "Tamil-Sri Lankan",
                "Tamil-India",           "Tamil-Malaysia",         "Telugu",                "Gujarati",
                "Oriya",                 "Odia",                   "Nepali",                "Sinhala",
                "Panjabi",               "Kashmiri",               "Marathi",
                "印地语",                 "孟加拉语",                "泰米尔语-新加坡",          "泰米尔语-斯里兰卡",
                "泰米尔语-印度",           "泰米尔语-马来西亚",         "泰卢固语",                "古吉拉特语",
                "奥里亚语",               "尼泊尔语",                 "僧伽罗语",                "旁遮普语",
                "克什米尔语",              "马拉地语",

                # West Asian languages
                "Urdu",                  "Urdu-India",             "Persian",               "Pushto",
                "乌尔都语",               "乌尔都语-印度",             "波斯语",                 "普什图语",

                # Arabic varieties
                "Arabic",                "Arabic-Morocco",         "Arabic-Saudi Arabia",   "Arabic-Egypt",
                "Arabic-Kuwait",         "Arabic-Libya",           "Arabic-Jordan",         "Arabic-U.A.E.",
                "Arabic-Levant",
                "阿拉伯语",               "阿拉伯语-摩洛哥",           "阿拉伯语-沙特",            "阿拉伯语-埃及",
                "阿拉伯语-科威特",         "阿拉伯语-利比亚",           "阿拉伯语-约旦",            "阿拉伯语-阿联酋",
                "阿拉伯语-黎凡特",

                # Central Asian and Turkic languages
                "Uighur",               "Uzbek",                  "Kazakh",                 "Mongolian",
                "Kabyle",               "Bashkir",                "Tajik",                  "Kirghiz",             "Azerbaijani",
                "维吾尔语",               "乌兹别克语",              "哈萨克语",                 "蒙古语",
                "卡拜尔语",               "巴什基尔语",              "塔吉克语",                 "吉尔吉斯语",            "阿塞拜疆语",

                # Other languages
                "Russian",              "俄语"
            ])
    else:
        update_A = gr.update(visible=False)
    return update_A


def update_denoiser(dropdown_model_denoiser):
    if "None" == dropdown_model_denoiser:
        return gr.update(visible=False), gr.update(visible=False)
    else:
        return gr.update(visible=True), gr.update(visible=True)


def update_vad(dropdown_model_vad):
    if "FSMN" == dropdown_model_vad:
        update_A = gr.update(visible=True, value=0.3)
        update_B = gr.update(visible=True, value=0.3)
        update_C = gr.update(visible=True)
        update_D = gr.update(visible=True)
        update_E = gr.update(visible=True)
        update_F = gr.update(visible=False)
        update_G = gr.update(visible=False)
        update_H = gr.update(visible=True, value=0.5)
        update_I = gr.update(visible=True, value=0.05)
        update_J = gr.update(visible=True, value=400)
    elif "Pyannote" in dropdown_model_vad:
        update_A = gr.update(visible=False, value=0.4)
        update_B = gr.update(visible=False, value=0.4)
        update_C = gr.update(visible=False)
        update_D = gr.update(visible=False)
        update_E = gr.update(visible=False)
        update_F = gr.update(visible=False)
        update_G = gr.update(visible=False)
        update_H = gr.update(visible=True, value=0.2)
        update_I = gr.update(visible=True, value=0.05)
        update_J = gr.update(visible=True, value=400)
    elif "Silero" in dropdown_model_vad:
        update_A = gr.update(visible=True, value=0.4)
        update_B = gr.update(visible=True, value=0.4)
        update_C = gr.update(visible=False)
        update_D = gr.update(visible=False)
        update_E = gr.update(visible=False)
        update_F = gr.update(visible=True)
        update_G = gr.update(visible=True)
        update_H = gr.update(visible=True, value=0.2)
        update_I = gr.update(visible=True, value=0.05)
        update_J = gr.update(visible=True, value=400)
    elif "HumAware" in dropdown_model_vad:
        update_A = gr.update(visible=True, value=0.4)
        update_B = gr.update(visible=True, value=0.4)
        update_C = gr.update(visible=False)
        update_D = gr.update(visible=False)
        update_E = gr.update(visible=False)
        update_F = gr.update(visible=True)
        update_G = gr.update(visible=True)
        update_H = gr.update(visible=True, value=0.2)
        update_I = gr.update(visible=True, value=0.05)
        update_J = gr.update(visible=True, value=400)
    elif "MarbleNet" in dropdown_model_vad:
        update_A = gr.update(visible=True, value=0.4)
        update_B = gr.update(visible=True, value=0.4)
        update_C = gr.update(visible=False)
        update_D = gr.update(visible=False)
        update_E = gr.update(visible=False)
        update_F = gr.update(visible=True)
        update_G = gr.update(visible=True)
        update_H = gr.update(visible=True, value=0.2)
        update_I = gr.update(visible=True, value=0.05)
        update_J = gr.update(visible=True, value=400)
    elif dropdown_model_vad == "TEN":
        update_A = gr.update(visible=True, value=0.4)
        update_B = gr.update(visible=True, value=0.4)
        update_C = gr.update(visible=False)
        update_D = gr.update(visible=False)
        update_E = gr.update(visible=False)
        update_F = gr.update(visible=True)
        update_G = gr.update(visible=True)
        update_H = gr.update(visible=True, value=0.2)
        update_I = gr.update(visible=True, value=0.05)
        update_J = gr.update(visible=True, value=400)
    else:
        update_A = gr.update(visible=False)
        update_B = gr.update(visible=False)
        update_C = gr.update(visible=False)
        update_D = gr.update(visible=False)
        update_E = gr.update(visible=False)
        update_F = gr.update(visible=False)
        update_G = gr.update(visible=False)
        update_H = gr.update(visible=False)
        update_I = gr.update(visible=False)
        update_J = gr.update(visible=False)
    return update_A, update_B, update_C, update_D, update_E, update_F, update_G, update_H, update_I, update_J


def get_language_id(language_input, is_whisper):
    language_input = language_input.lower()
    if is_whisper:
        language_input = LANGUAGE_MAP[0][1][language_input]
        return LANGUAGE_MAP[0][0].get(language_input)
    else:
        language_input = LANGUAGE_MAP[1][1][language_input]
        return LANGUAGE_MAP[1][0].get(language_input)


def get_task_id(task_input, is_v3):
    task_input = task_input.lower()
    if is_v3:
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


def vad_to_timestamps(vad_output, frame_duration, pad=0.0):
    timestamps = []
    start = None
    max_len = len(vad_output) * frame_duration
    frame_duration_plus = frame_duration + pad
    # Extract raw timestamps
    for i, silence in enumerate(vad_output):
        if silence:
            if start is not None:  # End of the current speaking segment
                end = i * frame_duration + frame_duration_plus
                if end > max_len:
                    end = max_len
                timestamps.append((start, end))
                start = None
        else:
            if start is None:  # Start of a new speaking segment
                start = i * frame_duration - pad
                if start < 0.0:
                    start = 0.0
    # Handle the case where speech continues until the end
    if start is not None:
        timestamps.append((start, max_len))
    return timestamps


def format_time(seconds):
    """Convert Seconds to VTT time format 'hh:mm:ss.mmm'."""
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


class Dolphin_Tokenizer:
    def __init__(self, filename=None):
        self.str_to_idx = {}
        self.idx_to_str = {}
        if filename:
            self.load_from_file(filename)

    def load_from_file(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                token = line.rstrip('\n')
                self.str_to_idx[token] = idx
                self.idx_to_str[idx] = token

    def encode(self, token):
        return self.str_to_idx.get(token)

    def decode(self, idx):
        return self.idx_to_str.get(idx)

    def __len__(self):
        return len(self.str_to_idx)


def MAIN_PROCESS(
        task,
        hardware,
        parallel_threads,
        file_path_input,
        translate_language,
        transcribe_language,
        model_asr,
        model_vad,
        model_denoiser,
        model_llm,
        switcher_run_test,
        switcher_denoiser_cache,
        slider_vad_pad,
        slider_denoise_factor,
        slider_vad_ONE_MINUS_SPEECH_THRESHOLD,
        slider_vad_SNR_THRESHOLD,
        slider_vad_BACKGROUND_NOISE_dB_INIT,
        slider_vad_SPEAKING_SCORE,
        slider_vad_SILENCE_SCORE,
        slider_vad_FUSION_THRESHOLD,
        slider_vad_MIN_SPEECH_DURATION,
        slider_vad_MAX_SPEECH_DURATION,
        slider_vad_MIN_SILENCE_DURATION
):
    def process_segment_A(_inv_audio_len, _slice_start, _slice_end, _audio):
        return _slice_start * _inv_audio_len, ort_session_A.run([out_name_A0], {in_name_A0: _audio[:, :, _slice_start: _slice_end]})[0]

    def process_segment_C_sensevoice(_start, _end, _inv_audio_len, _audio, _sample_rate, _language_idx):
        start_indices = _start * _sample_rate
        audio_segment = _audio[:, :, int(start_indices): int(_end * _sample_rate)]
        audio_segment_len = audio_segment.shape[-1]
        INPUT_AUDIO_LENGTH = min(MAX_ASR_SEGMENT, audio_segment_len)
        stride_step = INPUT_AUDIO_LENGTH
        slice_start = 0
        slice_end = INPUT_AUDIO_LENGTH
        text = ""
        while slice_start < audio_segment_len:
            token_ids = ort_session_C.run([out_name_C0], {in_name_C0: audio_segment[:, :, slice_start: slice_end], in_name_C1: _language_idx})[0]
            text += tokenizer.decode(token_ids.tolist())[0]
            slice_start += stride_step
            slice_end = slice_start + INPUT_AUDIO_LENGTH
        return start_indices * _inv_audio_len, text + ";", (_start, _end)

    def process_segment_C_paraformer(_start, _end, _inv_audio_len, _audio, _sample_rate, _is_english):
        start_indices = _start * _sample_rate
        audio_segment = _audio[:, :, int(start_indices): int(_end * _sample_rate)]
        audio_segment_len = audio_segment.shape[-1]
        INPUT_AUDIO_LENGTH = min(MAX_ASR_SEGMENT, audio_segment_len)  # You can adjust it.
        stride_step = INPUT_AUDIO_LENGTH
        slice_start = 0
        slice_end = INPUT_AUDIO_LENGTH
        text = np.array([], dtype=np.str_)
        while slice_start < audio_segment_len:
            token_ids = ort_session_C.run([out_name_C0], {in_name_C0: audio_segment[:, :, slice_start: slice_end]})[0]
            text = np.concatenate((text, tokenizer[token_ids[0]]))
            slice_start += stride_step
            slice_end = slice_start + INPUT_AUDIO_LENGTH
        if _is_english:
            text = ' '.join(text).replace("</s>", "").replace("@@ ", "")
        else:
            text = ''.join(text).replace("</s>", "")
        return start_indices * _inv_audio_len, text + ";", (_start, _end)

    def process_segment_CD(_start, _end, _inv_audio_len, _audio, _sample_rate, _init_input_ids, _init_history_len, _init_ids_len, _init_ids_len_1, _init_attention_mask_D_0, _init_attention_mask_D_1, _init_past_keys_D, _init_past_values_D, _is_whisper):
        start_indices = _start * _sample_rate
        audio_segment = _audio[:, :, int(start_indices): int(_end * _sample_rate)]
        audio_segment_len = audio_segment.shape[-1]
        INPUT_AUDIO_LENGTH = min(MAX_ASR_SEGMENT, audio_segment_len)  # You can adjust it.
        stride_step = INPUT_AUDIO_LENGTH
        slice_start = 0
        slice_end = INPUT_AUDIO_LENGTH
        save_token = []
        while slice_start < audio_segment_len:
            input_feed_D = {
                in_name_D[-1]: _init_attention_mask_D_1,
                in_name_D[num_layers_2]: _init_input_ids,
                in_name_D[num_layers_2_plus_1]: _init_history_len,
                in_name_D[num_layers_2_plus_2]: _init_ids_len
            }
            for i in range(num_layers):
                input_feed_D[in_name_D[i]] = _init_past_keys_D
            for i in range(num_layers, num_layers_2):
                input_feed_D[in_name_D[i]] = _init_past_values_D
            num_decode = 0
            all_outputs_C = ort_session_C.run_with_ort_values(output_names_C, {in_name_C0: onnxruntime.OrtValue.ortvalue_from_numpy(audio_segment[:, :, slice_start: slice_end], device_type, DEVICE_ID)})
            for i in range(num_layers_2):
                input_feed_D[in_name_D[layer_indices[i]]] = all_outputs_C[i]
            while num_decode < generate_limit:
                all_outputs_D = ort_session_D.run_with_ort_values(out_name_D, input_feed_D)
                max_logit_ids = onnxruntime.OrtValue.numpy(all_outputs_D[-2])[0][0]
                num_decode += 1
                if max_logit_ids in ASR_STOP_TOKEN:
                    break
                for i in range(amount_of_outputs_D):
                    input_feed_D[in_name_D[i]] = all_outputs_D[i]
                if num_decode < 2:
                    input_feed_D[in_name_D[-1]] = _init_attention_mask_D_0
                    if _is_whisper:
                        input_feed_D[in_name_D[num_layers_2_plus_2]] = _init_ids_len_1
                save_token.append(max_logit_ids)
            slice_start += stride_step
            slice_end = slice_start + INPUT_AUDIO_LENGTH
        if _is_whisper:
            save_token_array = remove_repeated_parts(save_token, 4)  # To handle "over-talking".
            text, _ = tokenizer._decode_asr(
                [{
                    "tokens": save_token_array
                }],
                return_timestamps=None,  # Do not support return timestamps
                return_language=None,
                time_precision=0
            )
        else:
            text = ("".join([tokenizer.dict[int(id)] for id in save_token])).replace(tokenizer.SPM_SPACE, ' ').strip()
        return start_indices * _inv_audio_len, text + ";", (_start, _end)

    def process_segment_CD_dolphin(_start, _end, _inv_audio_len, _audio, _sample_rate, _init_input_ids, _init_history_len, _init_ids_len, _init_ids_len_1, _init_attention_mask_D_0, _init_attention_mask_D_1, _init_past_keys_D, _init_past_values_D, _lang_id, _region_id):
        start_indices = _start * _sample_rate
        audio_segment = _audio[:, :, int(start_indices): int(_end * _sample_rate)]
        audio_segment_len = audio_segment.shape[-1]
        INPUT_AUDIO_LENGTH = min(MAX_ASR_SEGMENT, audio_segment_len)  # You can adjust it.
        stride_step = INPUT_AUDIO_LENGTH
        save_token = []
        slice_start = 0
        slice_end = INPUT_AUDIO_LENGTH
        while slice_start < audio_segment_len:
            all_outputs_C = ort_session_C.run_with_ort_values(output_names_C, {in_name_C0: onnxruntime.OrtValue.ortvalue_from_numpy(audio_segment[:, :, slice_start: slice_end], device_type, DEVICE_ID)})
            input_feed_D = {
                in_name_D[-1]: _init_attention_mask_D_1,
                in_name_D[num_layers_2_plus_1]: _init_history_len,
            }
            for i in range(num_layers):
                input_feed_D[in_name_D[i]] = _init_past_keys_D
                input_feed_D[in_name_D[layer_indices[i]]] = all_outputs_C[i]
            for i in range(num_layers, num_layers_2):
                input_feed_D[in_name_D[i]] = _init_past_values_D
                input_feed_D[in_name_D[layer_indices[i]]] = all_outputs_C[i]

            if detect_language:
                input_feed_D[in_name_D[-3]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([7], dtype=np.int64), device_type, DEVICE_ID)
                input_feed_D[in_name_D[-2]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([145], dtype=np.int64), device_type, DEVICE_ID)
                input_feed_D[in_name_D[num_layers_2]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([[39999]], dtype=np.int32), device_type, DEVICE_ID)
                input_feed_D[in_name_D[num_layers_2_plus_2]] = _init_ids_len_1
                all_outputs_D = ort_session_D.run_with_ort_values(out_name_D, input_feed_D)
                _lang_id = onnxruntime.OrtValue.numpy(all_outputs_D[-2])[0][0] + 7
                for i in range(num_layers):
                    input_feed_D[in_name_D[i]] = _init_past_keys_D
                for i in range(num_layers, num_layers_2):
                    input_feed_D[in_name_D[i]] = _init_past_values_D
            if detect_region:
                input_feed_D[in_name_D[-3]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([145], dtype=np.int64), device_type, DEVICE_ID)
                input_feed_D[in_name_D[-2]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([324], dtype=np.int64), device_type, DEVICE_ID)
                input_feed_D[in_name_D[num_layers_2]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([[39999, _lang_id]], dtype=np.int32), device_type, DEVICE_ID)
                input_feed_D[in_name_D[num_layers_2_plus_2]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([2], dtype=np.int64), device_type, DEVICE_ID)
                all_outputs_D = ort_session_D.run_with_ort_values(out_name_D, input_feed_D)
                _region_id = onnxruntime.OrtValue.numpy(all_outputs_D[-2])[0][0] + 145
                for i in range(num_layers):
                    input_feed_D[in_name_D[i]] = _init_past_keys_D
                for i in range(num_layers, num_layers_2):
                    input_feed_D[in_name_D[i]] = _init_past_values_D
            input_ids = np.array([[39999, _lang_id, _region_id, 6, 324]], dtype=np.int32)  # start_id = 39999; itn = 5; asr = 6; no_timestamp = 324
            ids_len = np.array([input_ids.shape[1]], dtype=np.int64)
            input_feed_D[in_name_D[num_layers_2]] = onnxruntime.OrtValue.ortvalue_from_numpy(input_ids, device_type, DEVICE_ID)
            input_feed_D[in_name_D[num_layers_2_plus_2]] = onnxruntime.OrtValue.ortvalue_from_numpy(ids_len, device_type, DEVICE_ID)
            input_feed_D[in_name_D[-3]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type, DEVICE_ID)
            input_feed_D[in_name_D[-2]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([40002], dtype=np.int64), device_type, DEVICE_ID)
            num_decode = 0
            while num_decode < generate_limit:
                all_outputs_D = ort_session_D.run_with_ort_values(out_name_D, input_feed_D)
                max_logit_ids = onnxruntime.OrtValue.numpy(all_outputs_D[-2])[0][0]
                num_decode += 1
                if max_logit_ids in ASR_STOP_TOKEN:
                    break
                for i in range(amount_of_outputs_D):
                    input_feed_D[in_name_D[i]] = all_outputs_D[i]
                if num_decode < 2:
                    input_feed_D[in_name_D[-1]] = _init_attention_mask_D_0
                    input_feed_D[in_name_D[num_layers_2_plus_2]] = _init_ids_len_1
                save_token.append(max_logit_ids)
            slice_start += stride_step
            slice_end = slice_start + INPUT_AUDIO_LENGTH
        text = ""
        for i in save_token:
            text += tokenizer.decode(i)
        text = text.replace("▁", " ")
        return start_indices * _inv_audio_len, text + ";", (_start, _end)

    total_process_time = time.time()
    print("----------------------------------------------------------------------------------------------------------")

    USE_V3 = True  # In the current version, only release the Whisper-V3 series.
    FIRST_RUN = True
    USE_DENOISED = True
    HAS_CACHE = False
    SAMPLE_RATE = 16000
    sys_os = platform.system()

    if (sys_os != 'Windows') and (sys_os != 'Darwin') and (sys_os != 'Linux'):
        return f"Unsupported OS: {sys_os}"

    task_queue = []
    if os.path.isfile(file_path_input):
        if file_path_input.endswith(MEDIA_EXTENSIONS):
            task_queue.append(file_path_input)
        else:
            error = f"\n指定的路径或文件不存在，或是不合规的媒体格式。\nThe specified path or file '{file_path_input}' does not exist or is not in a legal media format."
            print(error)
    elif os.path.isdir(file_path_input):
        for file in os.listdir(file_path_input):
            if file.endswith(MEDIA_EXTENSIONS):
                task_queue.append(f"{file_path_input}/{file}")
            else:
                error = f"\n指定的路径或文件不存在，或是不合规的媒体格式。\nThe specified path or file '{file_path_input}/{file}' does not exist or is not in a legal media format."
                print(error)
    else:
        error = f"\n指定的路径或文件不存在，或是不合规的媒体格式。\nThe specified path or file '{file_path_input}' does not exist or is not in a legal media format."
        print(error)
        return error

    total_task = len(task_queue)
    if total_task < 1:
        error = f"\n指定的路径或文件不存在，或是不合规的媒体格式。\nThe specified path or file '{file_path_input}' does not exist or is not in a legal media format."
        print(error)
        return error
    else:
        print(f"\n找到了 {total_task} 个媒体文件。Totally {total_task} media found.")

    if hardware == "Intel-OpenVINO-CPU":
        asr_format = "F16"
        device_type = 'CPU'
        ORT_Accelerate_Providers = ['OpenVINOExecutionProvider']
    elif hardware == "Intel-OpenVINO-GPU":
        asr_format = "F16"
        device_type = 'GPU'
        ORT_Accelerate_Providers = ['OpenVINOExecutionProvider']
    elif hardware == "Intel-OpenVINO-NPU":
        asr_format = "F16"
        device_type = 'NPU'
        ORT_Accelerate_Providers = ['OpenVINOExecutionProvider']
    elif hardware == "Intel-OpenVINO-AUTO_ALL":
        asr_format = "F16"
        device_type = 'AUTO:NPU,GPU,CPU'
        ORT_Accelerate_Providers = ['OpenVINOExecutionProvider']
    elif hardware == "Intel-OpenVINO-HETERO_ALL":
        asr_format = "F16"
        device_type = 'HETERO:NPU,GPU,CPU'
        ORT_Accelerate_Providers = ['OpenVINOExecutionProvider']
    elif hardware == "NVIDIA-CUDA-GPU":
        asr_format = "F16"
        device_type = 'cuda'
        ORT_Accelerate_Providers = ['CUDAExecutionProvider']
    elif hardware == "Windows-DirectML-GPU-NPU":
        asr_format = "F16"
        device_type = 'npu'
        ORT_Accelerate_Providers = ['DmlExecutionProvider']
    else:
        asr_format = "Q8F32"
        device_type = "cpu"
        ORT_Accelerate_Providers = ['CPUExecutionProvider']

    if 'OpenVINOExecutionProvider' in ORT_Accelerate_Providers:
        provider_options = [
            {
                'device_type': device_type,
                'precision': 'ACCURACY',
                'model_priority': 'HIGH',
                'num_of_threads': parallel_threads,
                'num_streams': 1,
                'enable_opencl_throttling': True,
                'enable_qdq_optimizer': True,
                'disable_dynamic_shapes': False
            }
        ]
        device_type = 'cpu'
        max_workers = parallel_threads
    elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
        provider_options = [
            {
                'device_id': DEVICE_ID,
                'gpu_mem_limit': 24 * 1024 * 1024 * 1024,      # 24 GB
                'arena_extend_strategy': 'kNextPowerOfTwo',    # ["kNextPowerOfTwo", "kSameAsRequested"]
                'cudnn_conv_algo_search': 'EXHAUSTIVE',        # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
                'sdpa_kernel': '2',                            # ["0", "1", "2"]
                'use_tf32': '1',
                'fuse_conv_bias': '0',
                'cudnn_conv_use_max_workspace': '1',
                'cudnn_conv1d_pad_to_nc1d': '1',
                'tunable_op_enable': '1',
                'tunable_op_tuning_enable': '1',
                'tunable_op_max_tuning_duration_ms': 10000,
                'do_copy_in_default_stream': '0',
                'enable_cuda_graph': '0',                      # Set to '0' to avoid potential errors when enabled.
                'prefer_nhwc': '0',
                'enable_skip_layer_norm_strict_mode': '0',
                'use_ep_level_unified_stream': '0',
            }
        ]
        max_workers = parallel_threads
    elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
        if sys_os != "Windows":
            print("\nDirectML-GPU-NPU 仅支持 Windows 系统。回退到 CPU 硬件。\nThe DirectML-GPU-NPU only support the Windows System. Fallback to the CPU providers.")
            device_type = 'cpu'
            ORT_Accelerate_Providers = ['CPUExecutionProvider']
            provider_options = None
        else:
            provider_options = [
                {
                    'device_id': DEVICE_ID,
                    'performance_preference': 'high_performance',  # [high_performance, default, minimum_power]
                    'device_filter': device_type                   # [any, npu, gpu]
                }
            ]
            device_type = 'dml'
            max_workers = 1
    else:
        device_type = 'cpu'
        provider_options = None
        ORT_Accelerate_Providers = ['CPUExecutionProvider']
        max_workers = parallel_threads

    slider_denoise_factor_minus = float(1.0 - slider_denoise_factor)
    slider_denoise_factor = float(slider_denoise_factor)
    denoiser_format = ""
    if model_denoiser == "DFSMN":
        SAMPLE_RATE = 48000
    elif model_denoiser == "GTCRN":
        if 'CPUExecutionProvider' in ORT_Accelerate_Providers:
            denoiser_format = "/F32"
        else:
            denoiser_format = "/F16"
    elif model_denoiser == "ZipEnhancer":
        slider_vad_pad += 200  # Over denoise
    elif model_denoiser == "MelBandRoformer":
        if 'CPUExecutionProvider' in ORT_Accelerate_Providers:
            denoiser_format = "/Q8F32"
        else:
            denoiser_format = "/F16"
        SAMPLE_RATE = 44100
    elif model_denoiser == "None":
        USE_DENOISED = False

    if USE_DENOISED:
        onnx_model_A = f"./Denoiser/{model_denoiser}{denoiser_format}/{model_denoiser}.onnx"
        if os.path.isfile(onnx_model_A):
            print(f"\n找到了降噪器。Found the Denoiser-{model_denoiser}.")
        else:
            error = f"\n降噪器不存在。The Denoiser-{model_denoiser} doesn't exist."
            print(error)
            return error
    else:
        onnx_model_A = None
        print("\n此任务不使用降噪器。\nThis task is running without the denoiser.")

    if model_vad == "FSMN":
        onnx_model_B = "./VAD/FSMN/FSMN.onnx"
        if os.path.isfile(onnx_model_B):
            vad_type = 0
            print("\n找到了 VAD-FSMN。Found the VAD-FSMN.")
        else:
            error = "\nVAD-FSMN不存在。The VAD-FSMN doesn't exist."
            print(error)
            return error
    elif model_vad == "Faster_Whisper-Silero":
        if os.path.isdir(PYTHON_PACKAGE + "/faster_whisper"):
            vad_type = 1
            onnx_model_B = None
            print(f"\n找到了 VAD-Faster_Whisper-Silero。Found the VAD-Faster_Whisper-Silero.")
        else:
            error = "\nVAD-Faster_Whisper-Silero 不存在。请运行 pip install fastest-whisper --upgrade。\nThe VAD-Faster_Whisper-Silero doesn't exist. Please run 'pip install faster-whisper --upgrade'"
            print(error)
            return error
    elif model_vad == "Official-Silero":
        if os.path.isdir(PYTHON_PACKAGE + "/silero_vad"):
            vad_type = 2
            onnx_model_B = None
            print(f"\n找到了 VAD-Official_Silero。Found the VAD-Official_Silero.")
        else:
            error = "\nVAD-Official_Silero不存在。请运行 pip install silero-vad --upgrade。\nThe VAD-Official_Silero doesn't exist. Please run 'pip install silero-vad --upgrade'"
            print(error)
            return error
    elif "Pyannote" in model_vad:
        if os.path.isfile("./VAD/Pyannote_Segmentation/pytorch_model.bin"):
            vad_type = 3
            onnx_model_B = None
            print(f"\n找到了 VAD-Pyannote_Segmentation。Found the VAD-Pyannote_Segmentation.")
        else:
            error = "\nVAD-Pyannote_Segmentation 不存在。请运行'pip install pyannote.audio --upgrade' 并从 https://huggingface.co/pyannote/segmentation-3.0 下载 pytorch_model.bin。\nThe VAD-Pyannote_Segmentation doesn't exist. Please run 'pip install pyannote.audio' --upgrade and Download the pytorch_model.bin from https://huggingface.co/pyannote/segmentation-3.0"
            print(error)
            return error
    elif model_vad == "HumAware":
        if os.path.isfile("./VAD/HumAware/HumAwareVAD.jit"):
            vad_type = 4
            onnx_model_B = None
            print(f"\n找到了 VAD-HumAware。Found the VAD-HumAware.")
        else:
            error = "\nVAD-HumAware不存在。"
            print(error)
            return error
    elif "MarbleNet" in model_vad:
        if 'CPUExecutionProvider' in ORT_Accelerate_Providers:
            onnx_model_B = "./VAD/NVIDIA_Frame_VAD_Multilingual_MarbleNet/F32/NVIDIA_MarbleNet.onnx"
        else:
            onnx_model_B = "./VAD/NVIDIA_Frame_VAD_Multilingual_MarbleNet/F16/NVIDIA_MarbleNet.onnx"
        if os.path.isfile(onnx_model_B):
            vad_type = 5
            print(f"\n找到了 VAD-NVIDIA_Frame_VAD_Multilingual_MarbleNet。Found the VAD-NVIDIA_Frame_VAD_Multilingual_MarbleNet.")
        else:
            error = "\nVAD-NVIDIA_Frame_VAD_Multilingual_MarbleNet 不存在。\nThe VAD-NVIDIA_Frame_VAD_Multilingual_MarbleNet doesn't exist. "
            print(error)
            return error
    elif model_vad == "TEN":
        if sys_os == "Darwin":
            # Current not support MAC + Python. Fallback to Default.
            print("\nTAN-VAD 目前不支持 MAC + Python。回退到 Faster_Whisper-Silero。\nThe TAN-VAD vurrently doesn't support MAC + Python. Fallback to Faster_Whisper-Silero.")
            if os.path.isdir(PYTHON_PACKAGE + "/faster_whisper"):
                vad_type = 1
                onnx_model_B = None
                print(f"\n找到了 VAD-Faster_Whisper-Silero。Found the VAD-Faster_Whisper-Silero.")
            else:
                error = "\nVAD-Faster_Whisper-Silero 不存在。请运行 pip install fastest-whisper --upgrade。\nThe VAD-Faster_Whisper-Silero doesn't exist. Please run 'pip install faster-whisper --upgrade'"
                print(error)
                return error
        else:
            if sys_os != "Windows":
                lib_path = "./VAD/TEN/lib/Linux/x64/libten_vad.so"
                print("\n您正在使用 Linux 操作系统的 TEN-VAD。首次启动前，请运行以下命令。\nYou are using the TEN-VAD with Linux OS. Please run the following commands before the first launch.\n\nsudo apt update\nsudo apt install libc++1")
            else:
                lib_path = "./VAD/TEN/lib/Windows/x64/libten_vad.dll"
            if os.path.isfile(lib_path):
                vad_type = 6
                onnx_model_B = None
                print(f"\n找到了 VAD-TEN。Found the VAD-TEN.")
            else:
                error = "\nVAD-TEN 不存在。\nThe VAD-TEN doesn't exist. "
                print(error)
                return error
    else:
        vad_type = -1
        onnx_model_B = None

    if "Whisper" in model_asr:
        if model_asr == "Whisper-Large-V3":
            path = "./ASR/Whisper/Official_Large"
        elif model_asr == "Whisper-Large-V3-Turbo":
            path = "./ASR/Whisper/Official_Turbo"
        elif model_asr == "Whisper-Large-V3-Turbo-Japanese":
            path = "./ASR/Whisper/Fine_Tune_Large_Turbo_Japanese"
        elif model_asr == "Whisper-Large-V3-Anime-A":
            path = "./ASR/Whisper/Fine_Tune_Large_Anime_A"
        elif model_asr == "Whisper-Large-V3-Anime-B":
            path = "./ASR/Whisper/Fine_Tune_Large_Anime_B"
        elif model_asr == "Whisper-Large-CrisperWhisper":
            path = "./ASR/Whisper/Fine_Tune_Large_CrisperWhisper"
        elif model_asr == "Whisper-Large-V3.5-Distil-English":
            path = "./ASR/Whisper/Fine_Tune_Large_Distill_V3.5"
        else:
            error = f"\n未找到模型。The {model_asr} doesn't exist."
            print(error)
            return error
        onnx_model_C = f"{path}/{asr_format}/Whisper_Encoder.onnx"
        onnx_model_D = f"{path}/{asr_format}/Whisper_Decoder.onnx"
        if os.path.isfile(onnx_model_C) and os.path.isfile(onnx_model_D):
            print(f"\n找到了 ASR。Found the {model_asr}.")
        else:
            error = f"\n未找到模型。The {model_asr} doesn't exist."
            print(error)
            return error
        asr_type = 0
        tokenizer = AutoTokenizer.from_pretrained(path)
        target_language_id = get_language_id(transcribe_language, True)
        if (model_llm == "Whisper") and ("Translate" in task):
            target_task_id = get_task_id('translate', USE_V3)[0]
        else:
            target_task_id = get_task_id('transcribe', USE_V3)[0]
    elif "SenseVoice" in model_asr:
        onnx_model_C = f"./ASR/SenseVoice/Small/F32/SenseVoice.onnx"
        if os.path.isfile(onnx_model_C):
            print(f"\n找到了 ASR。Found the {model_asr}.")
        else:
            error = f"\n未找到模型 The {model_asr} doesn't exist."
            print(error)
            return error
        asr_type = 1
        tokenizer = SentencePieceProcessor()
        tokenizer.Load("./ASR/SenseVoice/Small/chn_jpn_yue_eng_ko_spectok.bpe.model")
        target_language_id = get_language_id(transcribe_language, False)
        target_task_id = None
        onnx_model_D = None
    elif "Paraformer" in model_asr:
        if "Large" in model_asr:
            if "English" in transcribe_language or "english" in transcribe_language:
                is_english = True
                tokens_path = "./ASR/Paraformer/English/Large/tokens.json"
                onnx_model_C = f"./ASR/Paraformer/English/Large/F32/Paraformer.onnx"
            else:
                is_english = False
                tokens_path = "./ASR/Paraformer/Chinese/Large/tokens.json"
                onnx_model_C = f"./ASR/Paraformer/Chinese/Large/F32/Paraformer.onnx"
        else:
            is_english = False
            tokens_path = "./ASR/Paraformer/Chinese/Small/tokens.json"
            onnx_model_C = f"./ASR/Paraformer/Chinese/Small/F32/Paraformer.onnx"
        if os.path.isfile(onnx_model_C):
            print(f"\n找到了 ASR。Found the {model_asr}.")
        else:
            error = f"\n未找到模型。The {model_asr} doesn't exist."
            print(error)
            return error
        asr_type = 2
        with open(tokens_path, 'r', encoding='UTF-8') as json_file:
            tokenizer = np.array(json.load(json_file), dtype=np.str_)
        target_language_id = get_language_id(transcribe_language, False)
        target_task_id = None
        onnx_model_D = None
    elif "FireRedASR" in model_asr:
        onnx_model_C = f"./ASR/FireRedASR/AED/L/{asr_format}/FireRedASR_AED_L-Encoder.onnx"
        onnx_model_D = f"./ASR/FireRedASR/AED/L/{asr_format}/FireRedASR_AED_L-Decoder.onnx"
        if os.path.isfile(onnx_model_C) and os.path.isfile(onnx_model_D):
            print(f"\n找到了 ASR。Found the {model_asr}.")
        else:
            error = f"\n未找到模型。The {model_asr} doesn't exist."
            print(error)
            return error
        asr_type = 3
        tokenizer = ChineseCharEnglishSpmTokenizer("./ASR/FireRedASR/AED/L/dict.txt", "./ASR/FireRedASR/AED/L/train_bpe1000.model")
    elif "Dolphin" in model_asr:
        if 'CPUExecutionProvider' in ORT_Accelerate_Providers:
            onnx_model_C = f"./ASR/Dolphin/Small/F32/Dolphin_Encoder.onnx"
            onnx_model_D = f"./ASR/Dolphin/Small/F32/Dolphin_Decoder.onnx"
        else:
            onnx_model_C = f"./ASR/Dolphin/Small/F16/Dolphin_Encoder.onnx"
            onnx_model_D = f"./ASR/Dolphin/Small/F16/Dolphin_Decoder.onnx"
        if os.path.isfile(onnx_model_C) and os.path.isfile(onnx_model_D):
            print(f"\n找到了 ASR。Found the {model_asr}.")
        else:
            error = f"\n未找到模型。The {model_asr} doesn't exist."
            print(error)
            return error
        asr_type = 4
        tokenizer = Dolphin_Tokenizer(r"./ASR/Dolphin/Small/vocab_Dolphin.txt")
    else:
        error = f"\n未找到模型。The {model_asr} doesn't exist."
        print(error)
        return error

    # ONNX Runtime settings
    session_opts = onnxruntime.SessionOptions()
    session_opts.log_severity_level = 4                      # Fatal level, it an adjustable value.
    session_opts.log_verbosity_level = 4                     # Fatal level, it an adjustable value.
    session_opts.inter_op_num_threads = parallel_threads     # Run different nodes with num_threads. Set 0 for auto.
    session_opts.intra_op_num_threads = parallel_threads     # Under the node, execute the operators with num_threads. Set 0 for auto.
    session_opts.enable_cpu_mem_arena = True                 # True for execute speed; False for less memory usage.
    session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
    session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
    session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
    session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
    session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
    session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
    session_opts.add_session_config_entry("disable_synchronize_execution_providers", "1")
    session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
    session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")

    print("----------------------------------------------------------------------------------------------------------")
    print("\n正在加载所需的模型和目标文件。Now loading the required models and target files.")
    # Load VAD model
    if vad_type == 0:
        init_cache = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, 128, 19, 1), dtype=np.float32), 'cpu', DEVICE_ID)  # FSMN_VAD model fixed cache shape. Do not edit it.
        noise_average_dB = np.array([slider_vad_BACKGROUND_NOISE_dB_INIT + slider_vad_SNR_THRESHOLD], dtype=np.float32) * float(0.1)
        slider_vad_SNR_THRESHOLD = float(slider_vad_SNR_THRESHOLD * 0.1)
        one_minus_speech_threshold = np.array([slider_vad_ONE_MINUS_SPEECH_THRESHOLD], dtype=np.float32)
        ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=['CPUExecutionProvider'], provider_options=None)
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
        print("\nVAD 可用的硬件 VAD Usable Providers: ['CPUExecutionProvider']")
    elif vad_type == 2:
        import torch
        torch.set_num_threads(parallel_threads)
        silero_vad = load_silero_vad(session_opts=session_opts, providers=['CPUExecutionProvider'], provider_options=None)
        print("\nVAD 可用的硬件 VAD Usable Providers: ['CPUExecutionProvider']")
    elif vad_type == 3:
        import torch
        torch.set_num_threads(parallel_threads)
        pyannote_vad = Model.from_pretrained("./VAD/Pyannote_Segmentation/pytorch_model.bin")
        pyannote_vad_pipeline = VoiceActivityDetection(segmentation=pyannote_vad)
        HYPER_PARAMETERS = {
            "min_duration_on": slider_vad_MIN_SPEECH_DURATION,
            "min_duration_off": slider_vad_FUSION_THRESHOLD
        }
        pyannote_vad_pipeline.instantiate(HYPER_PARAMETERS)
        print("\nVAD 可用的硬件 VAD Usable Providers: ['CPUExecutionProvider']")
    elif vad_type == 4:
        import torch
        torch.set_num_threads(parallel_threads)
        humaware_vad = torch.jit.load("./VAD/HumAware/HumAwareVAD.jit", map_location='cpu')
        humaware_vad.eval()
        print("\nVAD 可用的硬件 VAD Usable Providers: ['CPUExecutionProvider']")
    elif vad_type == 5:
        ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
        print(f"\nVAD 可用的硬件 VAD Usable Providers: {ort_session_B.get_providers()}")
        in_name_B = ort_session_B.get_inputs()
        out_name_B = ort_session_B.get_outputs()
        in_name_B0 = in_name_B[0].name
        out_name_B0 = out_name_B[0].name
        out_name_B1 = out_name_B[1].name
        out_name_B2 = out_name_B[2].name
    elif vad_type == 6:
        ten_vad = TenVad(TEN_VAD_FRAME_LENGTH, 0.5, lib_path)  # TEN_VAD_FRAME_LENGTH = 256, standard threshold = 0.5
        print("\nVAD 可用的硬件 VAD Usable Providers: ['CPUExecutionProvider']")

    # Load ASR model
    if (asr_type == 0) or (asr_type == 3) or (asr_type == 4):  # Whisper & FireRedASR & Dolphin
        ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
        ort_session_D = onnxruntime.InferenceSession(onnx_model_D, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
        input_shape_C = ort_session_C._inputs_meta[0].shape[-1]
        in_name_C = ort_session_C.get_inputs()
        out_name_C = ort_session_C.get_outputs()
        in_name_C0 = in_name_C[0].name
        output_names_C = []
        for i in range(len(out_name_C)):
            output_names_C.append(out_name_C[i].name)
        model_D_dtype = ort_session_D._inputs_meta[0].type
        if 'float16' in model_D_dtype:
            model_D_dtype = np.float16
        else:
            model_D_dtype = np.float32
        in_name_D = ort_session_D.get_inputs()
        out_name_D = ort_session_D.get_outputs()
        amount_of_outputs_D = len(out_name_D)
        in_name_D = [in_name_D[i].name for i in range(len(in_name_D))]
        out_name_D = [out_name_D[i].name for i in range(amount_of_outputs_D)]
        if asr_type == 0:
            ASR_STOP_TOKEN = [50257]
            input_ids = np.array([[50258, target_language_id, target_task_id]], dtype=np.int32)
            generate_limit = MAX_SEQ_LEN - 5  # 5 = length of initial input_ids
        elif asr_type == 3:
            ASR_STOP_TOKEN = [4]
            input_ids = np.array([[3]], dtype=np.int32)
            generate_limit = MAX_SEQ_LEN - 1  # 1 = length of initial input_ids
        else:
            ASR_STOP_TOKEN = [40000]
            input_ids = np.array([[39999]], dtype=np.int32)
            language_region = DOLPHIN_LANGUAGE_REGION.get(transcribe_language, "NONE")
            language_region = language_region.split("-")
            lang_id = f"<{language_region[0]}>"
            region_id = f"<{language_region[1]}>"
            if lang_id != "<auto>":
                detect_language = False
                lang_id = tokenizer.encode(lang_id)
            else:
                detect_language = True
            if not detect_language:
                if region_id != "<auto>":
                    detect_region = False
                    region_id = tokenizer.encode(region_id)
                else:
                    detect_region = True
            else:
                detect_region = True
            if detect_language or detect_region:
                lang_str = tokenizer.decode(lang_id)
                region_str = tokenizer.decode(region_id)
                transcribe_language = INV_DOLPHIN_REGION_LANGUAGE.get(f"{lang_str}-{region_str}", "Unknown")
            generate_limit = MAX_SEQ_LEN - 5  # 5 = length of initial input_ids
        num_layers = (amount_of_outputs_D - 2) // 2
        num_layers_2 = num_layers + num_layers
        num_layers_4 = num_layers_2 + num_layers_2
        num_layers_2_plus_1 = num_layers_2 + 1
        num_layers_2_plus_2 = num_layers_2 + 2
        layer_indices = np.arange(num_layers_2, num_layers_4, dtype=np.int32) + 3
        init_attention_mask_D_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), device_type, DEVICE_ID)
        init_attention_mask_D_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), device_type, DEVICE_ID)
        init_history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type, DEVICE_ID)
        init_ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([input_ids.shape[-1]], dtype=np.int64), device_type, DEVICE_ID)
        init_ids_len_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)
        init_input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(input_ids, device_type, DEVICE_ID)
        if device_type != 'dml':
            init_past_keys_D = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_D._inputs_meta[0].shape[0], ort_session_D._inputs_meta[0].shape[1], 0), dtype=model_D_dtype), device_type, DEVICE_ID)
            init_past_values_D = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_D._inputs_meta[num_layers].shape[0], 0, ort_session_D._inputs_meta[num_layers].shape[2]), dtype=model_D_dtype), device_type, DEVICE_ID)
        else:
            init_past_keys_D = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_D._inputs_meta[0].shape[0], ort_session_D._inputs_meta[0].shape[1], 0), dtype=model_D_dtype), 'cpu', DEVICE_ID)
            init_past_values_D = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_D._inputs_meta[num_layers].shape[0], 0, ort_session_D._inputs_meta[num_layers].shape[2]), dtype=model_D_dtype), 'cpu', DEVICE_ID)
        c_provider = ort_session_C.get_providers()
        print(f"\nASR 可用的硬件 ASR-Usable Providers: {c_provider}")
    else:
        ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts,  providers=ORT_Accelerate_Providers, provider_options=provider_options)
        in_name_C = ort_session_C.get_inputs()
        out_name_C = ort_session_C.get_outputs()
        in_name_C0 = in_name_C[0].name
        if asr_type == 1:  # SenseVoice
            in_name_C1 = in_name_C[1].name
        else:
            in_name_C1 = None
        out_name_C0 = out_name_C[0].name
        c_provider = ort_session_C.get_providers()
        print(f"\nASR 可用的硬件 ASR-Usable Providers: {c_provider}")

    # Process Loop
    for input_audio in task_queue:
        print(f"\n加载音频文件 Loading the Input Media: {input_audio}")
        file_name = Path(input_audio).stem
        audio = np.array(AudioSegment.from_file(input_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
        if USE_DENOISED:
            if switcher_denoiser_cache and Path(f"./Cache/{file_name}_{model_denoiser}.wav").exists():
                print("\n降噪音频文件已存在，改用缓存文件。The denoised audio file already exists. Using the cache instead.")
                USE_DENOISED = False
                HAS_CACHE = True
                SAMPLE_RATE = 16000
                de_audio = np.array(AudioSegment.from_file(f"./Cache/{file_name}_{model_denoiser}.wav").set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
                min_len = min(audio.shape[-1], de_audio.shape[-1])
                audio = audio[:min_len] * slider_denoise_factor_minus + de_audio[:min_len] * slider_denoise_factor
                del de_audio
            else:
                if FIRST_RUN:
                    # Load Denoiser model
                    if model_denoiser == "ZipEnhancer":
                        if device_type == 'cpu':
                            if "CPUExecutionProvider" in ORT_Accelerate_Providers:
                                ORT_Accelerate_Providers = ["OpenVINOExecutionProvider"]
                                provider_options = [
                                    {
                                        'device_type': "CPU",
                                        'precision': 'ACCURACY',
                                        'model_priority': 'HIGH',
                                        'num_of_threads': parallel_threads,
                                        'num_streams': 1,
                                        'enable_opencl_throttling': True,
                                        'enable_qdq_optimizer': True,
                                        'disable_dynamic_shapes': True
                                    }
                                ]
                            else:
                                provider_options[0]['disable_dynamic_shapes'] = True
                            ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
                            if "CPUExecutionProvider" in ORT_Accelerate_Providers:
                                provider_options = None
                                ORT_Accelerate_Providers = ["CPUExecutionProvider"]
                            else:
                                provider_options[0]['disable_dynamic_shapes'] = False
                        elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
                            provider_options[0]['cudnn_conv_algo_search'] = "DEFAULT"
                            ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
                            provider_options[0]['cudnn_conv_algo_search'] = "EXHAUSTIVE"
                        else:
                            ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
                    else:
                        ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
                    in_name_A = ort_session_A.get_inputs()
                    out_name_A = ort_session_A.get_outputs()
                    in_name_A0 = in_name_A[0].name
                    out_name_A0 = out_name_A[0].name
                    a_provider = ort_session_A.get_providers()
                    print(f"\n降噪可用的硬件 Denoise-Usable Providers: {a_provider}")

        if FIRST_RUN:
            print(f"\n所有模型已成功加载。All Models have been successfully loaded.")
            print("----------------------------------------------------------------------------------------------------------")

        # Process audio
        audio = normalize_to_int16(audio)
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
            if model_denoiser == "MelBandRoformer":
                audio = np.concatenate((audio, audio), axis=1)
            aligned_len = audio.shape[-1]
            print("----------------------------------------------------------------------------------------------------------")
            print("\n对音频进行降噪。Denoising the audio.")
            print("\n如果遇到 RAM 不足的情况，请尝试减少并行线程的数量。If you encounter low RAM conditions, try reducing the number of parallel threads.\n")
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
            de_audio = np.concatenate(saved, axis=-1)
            de_audio = de_audio[:, :, :audio_len]
            audio = audio[:, :, :audio_len].astype(np.float32) * slider_denoise_factor_minus + de_audio.astype(np.float32) * slider_denoise_factor
            if model_denoiser == "DFSMN":
                SAMPLE_RATE = 16000
                audio_len = audio_len // 3
                audio_len_3 = audio_len + audio_len + audio_len
                audio = np.mean(audio[:, :, :audio_len_3].reshape(-1, 3), axis=-1, dtype=np.float32).reshape(1, 1, -1)
                de_audio = np.mean(de_audio[:, :, :audio_len_3].reshape(-1, 3), axis=-1, dtype=np.float32).clip(min=-32768.0, max=32767.0).astype(np.int16)
                inv_audio_len = float(100.0 / audio_len)
            elif model_denoiser == "MelBandRoformer":
                SAMPLE_RATE = 16000
                de_audio = librosa.resample(
                    de_audio.astype(np.float32).reshape(-1),
                    orig_sr=44100,
                    target_sr=SAMPLE_RATE
                ).clip(min=-32768.0, max=32767.0).astype(np.int16)
                audio = librosa.resample(
                    audio.reshape(-1),
                    orig_sr=44100,
                    target_sr=SAMPLE_RATE
                ).reshape(1, 1, -1)
                inv_audio_len = float(100.0 / audio_len)
            audio = normalize_to_int16(audio.clip(min=-32768.0, max=32767.0))
            sf.write(f"./Cache/{file_name}_{model_denoiser}.wav", de_audio.reshape(-1), SAMPLE_RATE, format='WAVEX')
            print(f"Denoising: 100.00%\n降噪完成。Complete.\nTime Cost: {(end_time - start_time):.3f} Seconds.")
            del saved
            del results
            del de_audio

        # VAD parts.
        print("----------------------------------------------------------------------------------------------------------")
        print("\n接下来利用VAD模型提取语音片段。Next, use the VAD model to extract speech segments.")
        start_time = time.time()
        if vad_type != -1:
            if USE_DENOISED or HAS_CACHE:
                waveform = np.array(AudioSegment.from_file(f"./Cache/{file_name}_{model_denoiser}.wav").set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
                waveform = normalize_to_int16(waveform)
                waveform = waveform.reshape(1, 1, -1)
            else:
                waveform = audio
        
        if vad_type == 0:
            shape_value_in = ort_session_B._inputs_meta[0].shape[-1]
            if isinstance(shape_value_in, str):
                INPUT_AUDIO_LENGTH = min(8000, audio_len)  # You can adjust it.
            else:
                INPUT_AUDIO_LENGTH = shape_value_in
            stride_step = INPUT_AUDIO_LENGTH
            if audio_len > INPUT_AUDIO_LENGTH:
                num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
                total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
                pad_amount = total_length_needed - audio_len
                final_slice = waveform[:, :, -pad_amount:].astype(np.float32)
                white_noise = (np.sqrt(np.mean(final_slice * final_slice, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount))).astype(waveform.dtype)
                waveform = np.concatenate((waveform, white_noise), axis=-1)
            elif audio_len < INPUT_AUDIO_LENGTH:
                audio_float = waveform.astype(np.float32)
                white_noise = (np.sqrt(np.mean(audio_float * audio_float, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH - audio_len))).astype(waveform.dtype)
                waveform = np.concatenate((waveform, white_noise), axis=-1)
            audio_len = waveform.shape[-1]
            inv_audio_len = float(100.0 / audio_len)
            cache_0 = init_cache
            cache_1 = init_cache
            cache_2 = init_cache
            cache_3 = init_cache
            silence = True
            saved = []
            slice_start = 0
            slice_end = INPUT_AUDIO_LENGTH
            while slice_end <= audio_len:
                score, cache_0, cache_1, cache_2, cache_3, noisy_dB = ort_session_B.run(
                    [out_name_B0, out_name_B1, out_name_B2, out_name_B3, out_name_B4, out_name_B5],
                    {
                        in_name_B0: waveform[:, :, slice_start: slice_end],
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
                if noisy_dB > 0.0:
                    noise_average_dB = 0.5 * (noise_average_dB + noisy_dB + slider_vad_SNR_THRESHOLD)
                print(f"VAD: {slice_start * inv_audio_len:.3f}%")
                slice_start += stride_step
                slice_end = slice_start + INPUT_AUDIO_LENGTH
            timestamps = vad_to_timestamps(saved, INPUT_AUDIO_LENGTH * inv_16k, float(slider_vad_pad * 0.001))
            del saved
            del waveform
            del cache_0
            del cache_1
            del cache_2
            del cache_3
            gc.collect()
        elif vad_type == 1:
            print("\nVAD-Faster_Whisper-Silero 不提供可视化的运行进度。\nThe VAD-Faster_Whisper-Silero does not provide the running progress for visualization.\n")
            vad_options = {
                'threshold': slider_vad_SPEAKING_SCORE,
                'neg_threshold': slider_vad_SILENCE_SCORE,
                'max_speech_duration_s': slider_vad_MAX_SPEECH_DURATION,
                'min_speech_duration_ms': int(slider_vad_MIN_SPEECH_DURATION * 1000),
                'min_silence_duration_ms': slider_vad_MIN_SILENCE_DURATION,
                'speech_pad_ms': slider_vad_pad
            }
            timestamps = get_speech_timestamps_FW(
                (waveform.reshape(-1).astype(np.float32) * inv_int16),
                vad_options=VadOptions(**vad_options),
                sampling_rate=SAMPLE_RATE
            )
            timestamps = [(item['start'] * inv_16k, item['end'] * inv_16k) for item in timestamps]
            del waveform
        elif vad_type == 2:
            print("\nVAD-Official-Silero 不提供可视化的运行进度。\nThe VAD-Official-Silero does not provide the running progress for visualization.\n")
            with torch.inference_mode():
                timestamps = get_speech_timestamps(
                    torch.from_numpy(waveform.reshape(-1).astype(np.float32) * inv_16k),
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
                del waveform
        elif vad_type == 3:
            print("\nVAD-Pyannote_Segmentation 不提供可视化的运行进度。\nThe VAD-Pyannote_Segmentation does not provide the running progress for visualization.\n")
            with torch.inference_mode():
                timestamps = pyannote_vad_pipeline(f"./Cache/{file_name}_{model_denoiser}.wav")
                segments = list(timestamps._tracks.keys())
                total_seconds = audio_len * inv_16k
                timestamps = []
                slider_vad_pad_s = float(slider_vad_pad * 0.001)
                for segment in segments:
                    segment_start = segment.start - slider_vad_pad_s
                    segment_end = segment.end + slider_vad_pad_s
                    if segment_start < 0:
                        segment_start = 0
                    if segment_end > total_seconds:
                        segment_end = total_seconds
                    timestamps.append((segment_start, segment_end))
                del waveform
        elif vad_type == 4:
            print("\nVAD-HumAware 不提供可视化的运行进度。\nThe VAD-HumAware does not provide the running progress for visualization.\n")
            with torch.inference_mode():
                waveform = torch.from_numpy(waveform.reshape(-1).astype(np.float32) * inv_16k)
                waveform_len = len(waveform)
                INPUT_AUDIO_LENGTH = 512
                stride_step = INPUT_AUDIO_LENGTH
                if waveform_len > INPUT_AUDIO_LENGTH:
                    num_windows = int(torch.ceil(torch.tensor((waveform_len - INPUT_AUDIO_LENGTH) / stride_step))) + 1
                    total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
                    pad_amount = total_length_needed - waveform_len
                    final_slice = waveform[-pad_amount:]
                    rms = torch.sqrt(torch.mean(final_slice * final_slice))
                    white_noise = rms * torch.randn(pad_amount, device=waveform.device, dtype=waveform.dtype)
                    waveform = torch.cat((waveform, white_noise), dim=-1)
                    waveform_len = len(waveform)
                elif waveform_len < INPUT_AUDIO_LENGTH:
                    rms = torch.sqrt(torch.mean(waveform * waveform))
                    white_noise = rms * torch.randn(INPUT_AUDIO_LENGTH - audio_len, device=waveform.device, dtype=waveform.dtype)
                    waveform = torch.cat((waveform, white_noise), dim=-1)
                    waveform_len = len(waveform)
                silence = True
                saved = []
                for i in range(0, waveform_len, INPUT_AUDIO_LENGTH):
                    score = humaware_vad(waveform[i: i+INPUT_AUDIO_LENGTH], SAMPLE_RATE)
                    if silence:
                        if score >= slider_vad_SPEAKING_SCORE:
                            silence = False
                    else:
                        if score <= slider_vad_SILENCE_SCORE:
                            silence = True
                    saved.append(silence)
                timestamps = vad_to_timestamps(saved, HumAware_param, float(slider_vad_pad * 0.001))
                del saved
                del waveform
                gc.collect()
        elif vad_type == 5:
            print("\nVAD-NVIDIA_Frame_VAD_Multilingual_MarbleNet 不提供可视化的运行进度。\nThe VAD-NVIDIA_Frame_VAD_Multilingual_MarbleNet does not provide the running progress for visualization.\n")
            inv_slider_vad_SILENCE_SCORE = float(1.0 - slider_vad_SILENCE_SCORE)
            score_silence, score_active, signal_len = ort_session_B.run([out_name_B0, out_name_B1, out_name_B2], {in_name_B0: waveform})
            silence = True
            saved = []
            for i in range(signal_len[0]):
                if silence:
                    if score_active[:, i] >= slider_vad_SPEAKING_SCORE:
                        silence = False
                else:
                    if score_silence[:, i] >= inv_slider_vad_SILENCE_SCORE:
                        silence = True
                saved.append(silence)
            timestamps = vad_to_timestamps(saved, NVIDIA_VAD_param, float(slider_vad_pad * 0.001))
            del saved
            del waveform
            del score_silence
            del score_active
            del signal_len
            gc.collect()
        elif vad_type == 6:
            waveform = waveform.reshape(-1)
            INPUT_AUDIO_LENGTH = TEN_VAD_FRAME_LENGTH
            stride_step = INPUT_AUDIO_LENGTH
            if audio_len > INPUT_AUDIO_LENGTH:
                num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
                total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
                pad_amount = total_length_needed - audio_len
                final_slice = waveform[-pad_amount:].astype(np.float32)
                white_noise = (np.sqrt(np.mean(final_slice * final_slice, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(pad_amount))).astype(waveform.dtype)
                waveform = np.concatenate((waveform, white_noise), axis=-1)
            elif audio_len < INPUT_AUDIO_LENGTH:
                audio_float = waveform.astype(np.float32)
                white_noise = (np.sqrt(np.mean(audio_float * audio_float, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(INPUT_AUDIO_LENGTH - audio_len))).astype(waveform.dtype)
                waveform = np.concatenate((waveform, white_noise), axis=-1)
            audio_len = waveform.shape[-1]
            inv_audio_len = float(100.0 / audio_len)
            silence = True
            saved = []
            slice_start = 0
            slice_end = INPUT_AUDIO_LENGTH
            while slice_end <= audio_len:
                score, _ = ten_vad.process(waveform[slice_start: slice_end])
                if silence:
                    if score >= slider_vad_SPEAKING_SCORE:
                        silence = False
                else:
                    if score <= slider_vad_SILENCE_SCORE:
                        silence = True
                saved.append(silence)
                print(f"VAD: {slice_start * inv_audio_len:.3f}%")
                slice_start += stride_step
                slice_end = slice_start + INPUT_AUDIO_LENGTH
            timestamps = vad_to_timestamps(saved, TEN_VAD_param, float(slider_vad_pad * 0.001))
        else:
            print("\n这个任务不使用 VAD。This task does not use VAD.\n")
        if vad_type != -1:
            timestamps = process_timestamps(timestamps, slider_vad_FUSION_THRESHOLD, slider_vad_MIN_SPEECH_DURATION)
            print(f"VAD: 100.00%\n完成提取语音片段。Complete.\nTime Cost: {(time.time() - start_time):.3f} Seconds.")
        else:
            timestamps = [(0.0, audio_len * inv_16k)]
        print("----------------------------------------------------------------------------------------------------------")

        # ASR parts
        print("\n开始转录任务。Start to transcribe task.")
        print("\n如果遇到 RAM 不足的情况，请尝试减少并行线程的数量。If you encounter low RAM conditions, try reducing the number of parallel threads.\n")
        results = []
        start_time = time.time()
        if asr_type == 0:
            if max_workers != 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(process_segment_CD, start, end, inv_audio_len, audio, SAMPLE_RATE, init_input_ids, init_history_len, init_ids_len, init_ids_len_1, init_attention_mask_D_0, init_attention_mask_D_1, init_past_keys_D, init_past_values_D, True) for start, end in timestamps]
                    for future in futures:
                        results.append(future.result())
                        print(f"ASR: {results[-1][0]:.3f}%")
            else:
                for start, end in timestamps:
                    results.append(process_segment_CD(start, end, inv_audio_len, audio, SAMPLE_RATE, init_input_ids, init_history_len, init_ids_len, init_ids_len_1, init_attention_mask_D_0, init_attention_mask_D_1, init_past_keys_D, init_past_values_D, True))
                    print(f"ASR: {results[-1][0]:.3f}%")
        elif asr_type == 1:
            language_idx = np.array([target_language_id], dtype=np.int32)
            if max_workers != 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(process_segment_C_sensevoice, start, end, inv_audio_len, audio, SAMPLE_RATE, language_idx) for start, end in timestamps]
                    for future in futures:
                        results.append(future.result())
                        print(f"ASR: {results[-1][0]:.3f}%")
            else:
                for start, end in timestamps:
                    results.append(process_segment_C_sensevoice(start, end, inv_audio_len, audio, SAMPLE_RATE, language_idx))
                    print(f"ASR: {results[-1][0]:.3f}%")
        elif asr_type == 2:
            if max_workers != 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(process_segment_C_paraformer, start, end, inv_audio_len, audio, SAMPLE_RATE, is_english) for start, end in timestamps]
                    for future in futures:
                        results.append(future.result())
                        print(f"ASR: {results[-1][0]:.3f}%")
            else:
                for start, end in timestamps:
                    results.append(process_segment_C_paraformer(start, end, inv_audio_len, audio, SAMPLE_RATE, is_english))
                    print(f"ASR: {results[-1][0]:.3f}%")
        elif asr_type == 3:
            if max_workers != 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(process_segment_CD, start, end, inv_audio_len, audio, SAMPLE_RATE, init_input_ids, init_history_len, init_ids_len, init_ids_len_1, init_attention_mask_D_0, init_attention_mask_D_1, init_past_keys_D, init_past_values_D, False) for start, end in timestamps]
                    for future in futures:
                        results.append(future.result())
                        print(f"ASR: {results[-1][0]:.3f}%")
            else:
                for start, end in timestamps:
                    results.append(process_segment_CD(start, end, inv_audio_len, audio, SAMPLE_RATE, init_input_ids, init_history_len, init_ids_len, init_ids_len_1, init_attention_mask_D_0, init_attention_mask_D_1, init_past_keys_D, init_past_values_D, False))
                    print(f"ASR: {results[-1][0]:.3f}%")
        else:
            if max_workers != 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(process_segment_CD_dolphin, start, end, inv_audio_len, audio, SAMPLE_RATE, init_input_ids, init_history_len, init_ids_len, init_ids_len_1, init_attention_mask_D_0, init_attention_mask_D_1, init_past_keys_D, init_past_values_D, lang_id, region_id) for start, end in timestamps]
                    for future in futures:
                        results.append(future.result())
                        print(f"ASR: {results[-1][0]:.3f}%")
            else:
                for start, end in timestamps:
                    results.append(process_segment_CD_dolphin(start, end, inv_audio_len, audio, SAMPLE_RATE, init_input_ids, init_history_len, init_ids_len, init_ids_len_1, init_attention_mask_D_0, init_attention_mask_D_1, init_past_keys_D, init_past_values_D, lang_id, region_id))
                    print(f"ASR: {results[-1][0]:.3f}%")
        results.sort(key=lambda x: x[0])
        save_text = [result[1] for result in results]
        save_timestamps = [result[2] for result in results]
        print(f"ASR: 100.00%\n完成转录任务。Complete.\nTime Cost: {time.time() - start_time:.3f} Seconds")
        del audio
        del timestamps
        gc.collect()
        print("----------------------------------------------------------------------------------------------------------")

        print(f"\n保存转录结果。Saving ASR Results.")
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
                if t_stamp[1] - start_sec > 10.0:
                    markers = re.split(r'([。、，！？；：,.!?:;])', transcription)  # Keep markers in results
                    text_chunks = ["".join(markers[i:i + 2]) for i in range(0, len(markers), 2)]
                    time_per_chunk = (t_stamp[1] - start_sec) / len(text_chunks)
                    if len(text_chunks) > 3:
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
        print(f"\n转录任务完成。Transcribe Tasks Complete.\n\n原文字幕保存在文件夹: ./Result/Subtitles\nThe original subtitles are saved in the folder: ./Result/Subtitles\n\nTranscribe Time: {(time.time() - total_process_time):.3f} Seconds.")
        print("----------------------------------------------------------------------------------------------------------")

        if "Translate" not in task:
            continue
        else:
            print("\n开始 LLM 翻译任务。Start to LLM Translate.")
            start_time = time.time()
            if FIRST_RUN:
                print("\n加载 LLM 模型。Loading the LLM model.")
                if model_llm == "Qwen-3-4B":
                    MAX_TRANSLATE_LINES = 4
                    llm_path = f"./LLM/Qwen/4B/Qwen.onnx"
                elif model_llm == "Qwen-3-8B":
                    MAX_TRANSLATE_LINES = 8
                    llm_path = f"./LLM/Qwen/8B/Qwen.onnx"
                elif model_llm == "InternLM-3-8B-Instruct":
                    MAX_TRANSLATE_LINES = 8
                    llm_path = f"./LLM/Intern/8B/InternLM.onnx"
                elif model_llm == "Gemma-3-4B-it":
                    MAX_TRANSLATE_LINES = 4
                    llm_path = f"./LLM/Gemma/4B/Gemma.onnx"
                elif model_llm == "Gemma-3-12B-it":
                    MAX_TRANSLATE_LINES = 16
                    llm_path = f"./LLM/Gemma/12B/Gemma.onnx"
                elif model_llm == "Phi-4-mini-Instruct":
                    MAX_TRANSLATE_LINES = 4
                    llm_path = f"./LLM/Phi/mini/Phi.onnx"
                elif model_llm == "Whisper":
                    print(f"\n翻译任务完成。Translate tasks completed.")
                    continue
                else:
                    error = "\n找不到翻译任务的 LLM 模型。Can not find the LLM model for translation task."
                    print(error)
                    return error

                TRANSLATE_OVERLAP = MAX_TRANSLATE_LINES // 4
                MAX_TOKENS_PER_CHUNK = MAX_TRANSLATE_LINES * MAX_SEQ_LEN

                if (translate_language == '中文') or (translate_language == 'chinese'):
                    translate_language = 'simplified Chinese'
                elif translate_language == '日語':
                    translate_language = 'japanese'
                elif translate_language == '한국인':
                    translate_language = 'korean'
                translate_language = translate_language[0].upper() + translate_language[1:]

                if (translate_language == '中文') or (translate_language == 'chinese'):
                    transcribe_language = 'simplified Chinese'
                elif transcribe_language == '日語':
                    transcribe_language = 'japanese'
                elif transcribe_language == '한국인':
                    transcribe_language = 'korean'
                elif transcribe_language == '自动 auto':
                    transcribe_language = 'unknown language'
                transcribe_language = transcribe_language[0].upper() + transcribe_language[1:]

                # Load the LLM
                system_prompt = f"Translate every subtitle line from {transcribe_language} to {translate_language}, fixing contextual errors for fluent, vivid flow, and output only the exact ‘ID-translated’ line per entry—nothing else."
                if "Qwen" in model_llm:
                    LLM_STOP_TOKEN = [151643, 151645]
                    prompt_head = f'<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nThe given subtitles:\n'
                    prompt_tail = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                    tokenizer_llm = AutoTokenizer.from_pretrained(f"./LLM/Qwen/Tokenizer", trust_remote_code=True)
                    is_Intern = False
                elif "InternLM" in model_llm:
                    LLM_STOP_TOKEN = [2, 128131]
                    prompt_head = f'<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nThe given subtitles:\n'
                    prompt_tail = "<|im_end|>\n<|im_start|>assistant\n"
                    tokenizer_llm = AutoTokenizer.from_pretrained(f"./LLM/Intern/Tokenizer", trust_remote_code=True)
                    is_Intern = True
                elif "Gemma" in model_llm:
                    LLM_STOP_TOKEN = [106, 1]
                    prompt_head = f'<bos><start_of_turn>user\n{system_prompt}\n\nThe given subtitles:\n'
                    prompt_tail = "<end_of_turn>\n<start_of_turn>model\n"
                    tokenizer_llm = AutoTokenizer.from_pretrained(f"./LLM/Gemma/Tokenizer", trust_remote_code=True)
                    is_Intern = False
                elif "Phi" in model_llm:
                    LLM_STOP_TOKEN = [200020, 199999]
                    prompt_head = f'<|system|>{system_prompt}<|end|><|user|>The given subtitles:\n'
                    prompt_tail = "<|end|><|assistant|>"
                    tokenizer_llm = AutoTokenizer.from_pretrained(f"./LLM/Phi/Tokenizer", trust_remote_code=True)
                    is_Intern = False
                else:
                    error = "\n未找到指定的 LLM 模型。The specified LLM model was not found."
                    print(error)
                    return error

                prompt_head = tokenizer_llm(prompt_head, return_tensors="np")['input_ids'].astype(np.int32)
                prompt_tail = tokenizer_llm(prompt_tail, return_tensors="np")['input_ids'].astype(np.int32)
                if device_type == 'cpu':
                    ORT_Accelerate_Providers = ['CPUExecutionProvider']  # Currently, OpenVINO will crash with Int4 onnx model.
                    provider_options = None
                ort_session_E = onnxruntime.InferenceSession(llm_path, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
                in_name_E = ort_session_E.get_inputs()
                out_name_E = ort_session_E.get_outputs()
                amount_of_outputs_E = len(out_name_E)
                in_name_E = [in_name_E[i].name for i in range(len(in_name_E))]
                out_name_E = [out_name_E[i].name for i in range(amount_of_outputs_E)]
                num_layers = (amount_of_outputs_E - 2) // 2
                num_keys_values = num_layers + num_layers
                init_attention_mask_E_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), device_type, DEVICE_ID)
                init_attention_mask_E_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), device_type, DEVICE_ID)
                init_history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type, DEVICE_ID)
                init_ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)
                if device_type != 'dml':
                    init_past_keys_E = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_E._inputs_meta[0].shape[0], 1, ort_session_E._inputs_meta[0].shape[2], 0), dtype=np.float32), device_type, DEVICE_ID)
                    init_past_values_E = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_E._inputs_meta[num_layers].shape[0], 1, 0, ort_session_E._inputs_meta[num_layers].shape[3]), dtype=np.float32), device_type, DEVICE_ID)
                else:
                    init_past_keys_E = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_E._inputs_meta[0].shape[0], 1, ort_session_E._inputs_meta[0].shape[2], 0), dtype=np.float32), 'cpu', DEVICE_ID)
                    init_past_values_E = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_E._inputs_meta[num_layers].shape[0], 1, 0, ort_session_E._inputs_meta[num_layers].shape[3]), dtype=np.float32), 'cpu', DEVICE_ID)
                input_feed_E = {
                    in_name_E[-1]: init_attention_mask_E_1,
                    in_name_E[-3]: init_history_len
                }
                for i in range(num_layers):
                    input_feed_E[in_name_E[i]] = init_past_keys_E
                for i in range(num_layers, num_keys_values):
                    input_feed_E[in_name_E[i]] = init_past_values_E
                print("\nLLM 模型加载完成。LLM loading completed")
                FIRST_RUN = False

            with open(f"./Results/Text/{file_name}.txt", 'r', encoding='utf-8') as asr_file:
                asr_lines = asr_file.readlines()

            with open(f"./Results/Timestamps/{file_name}.txt", 'r', encoding='utf-8') as timestamp_file:
                timestamp_lines = timestamp_file.readlines()

            for line_index in range(len(asr_lines)):
                asr_lines[line_index] = f"{line_index}-{asr_lines[line_index]}"

            total_lines = len(asr_lines)
            if total_lines < 1:
                print("\n翻译内容为空。Empty content for translation task.")
                continue

            print("----------------------------------------------------------------------------------------------------------")
            inv_total_lines = float(100.0 / total_lines)
            step_size = MAX_TRANSLATE_LINES - TRANSLATE_OVERLAP
            translated_responses = []
            for chunk_start in range(0, total_lines, step_size):
                chunk_end = min(total_lines, chunk_start + MAX_TRANSLATE_LINES)
                translation_prompt = "".join(asr_lines[chunk_start:chunk_end])
                tokens = np.concatenate((prompt_head, tokenizer_llm(translation_prompt, return_tensors="np")['input_ids'].astype(np.int32), prompt_tail), axis=1)
                input_feed_E[in_name_E[-4]] = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device_type, DEVICE_ID)
                input_feed_E[in_name_E[-2]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([tokens.shape[-1]], dtype=np.int64), device_type, DEVICE_ID)
                num_decode = 0
                save_text = ""
                start_time = time.time()
                while num_decode < MAX_TOKENS_PER_CHUNK:
                    all_outputs = ort_session_E.run_with_ort_values(
                        out_name_E,
                        input_feed_E
                    )
                    max_logit_ids = onnxruntime.OrtValue.numpy(all_outputs[-2])
                    num_decode += 1
                    if max_logit_ids in LLM_STOP_TOKEN:
                        break
                    for i in range(amount_of_outputs_E):
                        input_feed_E[in_name_E[i]] = all_outputs[i]
                    if num_decode < 2:
                        input_feed_E[in_name_E[-1]] = init_attention_mask_E_0
                        input_feed_E[in_name_E[-2]] = init_ids_len
                    text = tokenizer_llm.decode(max_logit_ids[0], skip_special_tokens=True)
                    if is_Intern:
                        text += " "
                    save_text += text
                    print(text, end="", flush=True)
                print(f"\n\nDecode: {(num_decode / (time.time() - start_time)):.3f} token/s")
                if chunk_start > 0:
                    save_text = "\n".join(save_text.split("\n")[TRANSLATE_OVERLAP + 1:])
                translated_responses.append(save_text)
                print(f"Translating: - {chunk_end * inv_total_lines:.3f}%")
                print("----------------------------------------------------------------------------------------------------------")
                if chunk_end == total_lines - 1:
                    break
                input_feed_E[in_name_E[-1]] = init_attention_mask_E_1
                input_feed_E[in_name_E[-3]] = init_history_len
                for i in range(num_layers):
                    input_feed_E[in_name_E[i]] = init_past_keys_E
                for i in range(num_layers, num_keys_values):
                    input_feed_E[in_name_E[i]] = init_past_values_E
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
                                    text = "".join(parts[1:])
                                    subtitles_file.write(f"{idx}\n{timestamp_lines[line_index]}{text}\n\n")
                                    idx += 1
            print(f"\n翻译完成。LLM Translate Complete.\nTime Cost: {time.time() - start_time:.3f} Seconds")
            print("----------------------------------------------------------------------------------------------------------")
    success = f"\n所有任务已完成。翻译字幕保存在文件夹: ./Result/Subtitles\nAll tasks complete. The translated subtitles are saved in the folder: ./Result/Subtitles\n\nTotal Time: {(time.time() - total_process_time):.3f} Seconds.\n"
    print(success)
    return success


################################################################################
# ----------------------------  CUSTOM  THEME  ---------------------------------
################################################################################
CUSTOM_CSS = """
/* ===== base =============================================================== */
html, body, .gradio-container{
    background:#0d0d0d;
    color:#f4f4f4;
    font-family:"Segoe UI",sans-serif;
    font-size:18px;                /* ——— bigger global font                */
}
h1,h2,h3,h4,h5,h6,.markdown{color:#f4f4f4;}

input,textarea,select,.input-container{
    background:#111;
    border:1px solid #333;
    color:#f4f4f4;
    border-radius:6px;
    font-size:18px;
}
label{
    color:#1e90ff !important;
    font-size:20px !important;
    font-weight:600 !important;
}
.dropdown,.slider,.checkbox,.radio{
    background:#111;
    border:1px solid #333;
    border-radius:6px;
}
.slider .noUi-connect{background:#1e90ff;}
.slider .noUi-handle {background:#0d8bfd;border:1px solid #80d0ff;}

/* ===== buttons ============================================================ */
button,.button-primary{
    font-size:20px;
    font-weight:700;
    background:linear-gradient(90deg,#0d8bfd 0%,#31d2ff 100%);
    border:none;
    color:#0d0d0d;
    box-shadow:0 0 12px #0d8bfd,0 0 6px #31d2ff inset;
    transition:all .15s;
}
button:hover{
    transform:translateY(-2px) scale(1.02);
    box-shadow:0 0 20px #31d2ff;
}

/* ===== big page title ===================================================== */
.big-title{
    font-size:40px;
    font-weight:900;
    text-align:center;
    margin:16px 0 30px 0;
    background:linear-gradient(90deg,#31d2ff 0%,#ffffff 50%,#31d2ff 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    text-shadow:0 0 20px rgba(49,210,255,.5);
}

/* ===== section blocks ===================================================== */
.section{
    padding:18px 18px 12px 18px;
    border-radius:14px;
    margin-bottom:20px;
    border:1px solid transparent;
    box-shadow:0 0 8px rgba(0,0,0,.35);
}
.section-title{
    font-size:26px;
    font-weight:800;
    margin:0 0 16px 0;
}
.section-sys   {background:rgba( 30,144,255,.12);border-color:#1e90ff55;}
.section-model {background:rgba(255,165,  0,.12);border-color:#ffa50055;}
.section-tgt   {background:rgba( 50,205, 50,.12);border-color:#32cd3255;}
.section-aud   {background:rgba(255, 20,147,.12);border-color:#ff149355;}
.section-vad   {background:rgba( 75,  0,130,.12);border-color:#4b008255;}

/* ===== “coding-style” state box =========================================== */
.task-state textarea{
    background:#000 !important;
    color:#00c853 !important;
    font-family:Consolas,monospace;
    font-size:16px;
}
"""


with gr.Blocks(css=CUSTOM_CSS, title="Subtitles is All You Need") as GUI:

    # --------------------------------------------------------------------- #
    # Header                                                                #
    # --------------------------------------------------------------------- #
    gr.Markdown("<div class='big-title'>Subtitles is All You Need</div>")

    # --------------------------------------------------------------------- #
    # Quick controls                                                        #
    # --------------------------------------------------------------------- #
    with gr.Row():
        with gr.Column(scale=1):
            gr.Image(
                "./Icon/psyduck.jpg",
                type="filepath",
                show_download_button=False,
                show_fullscreen_button=False
            )
        with gr.Column(scale=6):
            with gr.Row():
                task = gr.Dropdown(
                    choices=[
                        "转录 Transcribe",
                        "转录 + 翻译 Transcribe + Translate",
                    ],
                    label="Task",
                    info="选择操作\nSelect an operation for the audio.",
                    value="转录 + 翻译 Transcribe + Translate",
                    interactive=True,
                )
                switcher_run_test = gr.Checkbox(
                    label="简短测试 Run Test",
                    info="对音频的 10% 时长运行简短测试。\nRun a short test on 10% of the audio length.",
                    value=False,
                    interactive=True,
                )
            file_path_input = gr.Textbox(
                label="视频 / 音频文件路径 Video / Audio File Path",
                info="输入要转录的视频/音频文件或文件夹的路径。\nEnter the path of the video / audio file or folder you want to transcribe.",
                value="./Media",
                interactive=True,
            )

    # --------------------------------------------------------------------- #
    # System settings                                                       #
    # --------------------------------------------------------------------- #
    with gr.Column(elem_classes=["section", "section-sys"]):
        gr.Markdown("<div class='section-title'>🖥️  系统设置 / System Settings</div>")
        with gr.Row():
            parallel_threads = gr.Slider(
                1, 64, step=1,
                label="并行处理 Parallel Threads",
                info="用于并行处理的 CPU 内核数。\nNumber of CPU cores.",
                value=physical_cores,
                interactive=True
            )
            hardware = gr.Dropdown(
                choices=[
                    "CPU",
                    "Intel-OpenVINO-CPU",
                    "Intel-OpenVINO-GPU",
                    "Intel-OpenVINO-NPU",
                    "Intel-OpenVINO-AUTO_ALL",
                    "Intel-OpenVINO-HETERO_ALL",
                    "NVIDIA-CUDA-GPU",
                    "Windows-DirectML-GPU-NPU"
                ],
                label="硬件设备 Hardware Device",
                info="选择用于运行任务的设备。\nSelect the device for running the task.",
                value="CPU",
                visible=True,
                interactive=True
            )

    # --------------------------------------------------------------------- #
    # Model selection                                                       #
    # --------------------------------------------------------------------- #
    with gr.Column(elem_classes=["section", "section-model"]):
        gr.Markdown("<div class='section-title'>🧠  模型选择 / Model Selection</div>")
        with gr.Row():
            model_llm = gr.Dropdown(
                choices=[
                    "Qwen-3-4B",
                    "Qwen-3-8B",
                    "InternLM-3-8B-Instruct",
                    "Gemma-3-4B-it",
                    "Gemma-3-12B-it",
                    "Phi-4-mini-Instruct",
                    "Whisper"
                ],
                label="大型语言模型 LLM Model",
                info="用于翻译的模型。\nModel used for translation.",
                value="Qwen-3-8B",
                visible=True,
                interactive=True
            )
            model_asr = gr.Dropdown(
                choices=[
                    "SenseVoice-Small",
                    "Whisper-Large-V3",
                    "Whisper-Large-V3-Anime-A",
                    "Whisper-Large-V3-Anime-B",
                    "Whisper-Large-V3-Turbo",
                    "Whisper-Large-V3-Turbo-Japanese",
                    "Whisper-Large-V3.5-Distil-English",
                    "Whisper-Large-CrisperWhisper",
                    "Paraformer-Small",
                    "Paraformer-Large",
                    "FireRedASR-AED-L",
                    "Dolphin-Small"
                ],
                label="ASR模型 ASR Model",
                info="用于转录的模型。\nModel used for transcription.",
                value="SenseVoice-Small",
                visible=True,
                interactive=True
            )

    # --------------------------------------------------------------------- #
    # Target language                                                       #
    # --------------------------------------------------------------------- #
    with gr.Column(elem_classes=["section", "section-tgt"]):
        gr.Markdown("<div class='section-title'>🌐  目标语言 / Target Language</div>")
        with gr.Row():
            transcribe_language = gr.Dropdown(
                choices=[
                    "日語",
                    "中文",
                    "English",
                    "粤语",
                    "한국인",
                    "自动 auto"
                ],
                label="转录语言 Transcription Language",
                info="源媒体的语言。\nLanguage of the input media.",
                value="日語",
                visible=True,
                interactive=True
            )
            translate_language = gr.Dropdown(
                choices=[
                    "中文",         "English",          "日語",         "한국인",
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
                label="翻译语言 Translation Language",
                info="要翻译成的语言。\nLanguage to translate into.",
                value="中文",
                visible=True,
                interactive=True
            )

    # --------------------------------------------------------------------- #
    # Audio processor                                                      #
    # --------------------------------------------------------------------- #
    with gr.Column(elem_classes=["section", "section-aud"]):
        gr.Markdown("<div class='section-title'>🎙️  音频处理 / Audio Processor</div>")
        with gr.Row():
            model_denoiser = gr.Dropdown(
                choices=[
                    "None",
                    "GTCRN",
                    "DFSMN",
                    "ZipEnhancer",
                    "MelBandRoformer"
                ],
                label="降噪器 Denoiser",
                info="选择用于增强人声的降噪器。\nChoose a denoiser for audio processing.",
                value="GTCRN",
                visible=True,
                interactive=True
            )
            slider_denoise_factor = gr.Slider(
                0.1, 1.0, step=0.025,
                label="降噪系数 Denoise Factor",
                info="较大的值可增强降噪效果。\nHigher = stronger denoise.",
                value=0.5,
                visible=True,
                interactive=True
            )
            switcher_denoiser_cache = gr.Checkbox(
                label="使用缓存 Use Cache",
                info="使用以前的降噪结果以节省时间。\nUse previous results.",
                value=True,
                visible=True,
                interactive=True
            )

    # --------------------------------------------------------------------- #
    # VAD configuration                                                     #
    # --------------------------------------------------------------------- #
    with gr.Column(elem_classes=["section", "section-vad"]):
        gr.Markdown("<div class='section-title'>🔊  VAD 配置 / VAD Configurations</div>")
        model_vad = gr.Dropdown(
            choices=[
                "FSMN",
                'Faster_Whisper-Silero',
                "Official-Silero",
                "Pyannote-3.0",
                "HumAware",
                "NVIDIA-NeMo-MarbleNet-v2.0",
                "TEN",
                "None"
            ],
            label="语音活动检测 VAD",
            info="选择用于音频处理的 VAD。Select the VAD used for audio processing.",
            value="Faster_Whisper-Silero",
            visible=True,
            interactive=True
        )
        slider_vad_pad = gr.Slider(
            0, 3000, step=10,
            label="VAD 填充 VAD Padding",
            info="在时间戳的开头和结尾添加填充。单位：毫秒。\nAdd padding to the start and end of the timestamps. Unit: Milliseconds.",
            value=400,
            visible=True,
            interactive=True
        )
        with gr.Row():
            slider_vad_ONE_MINUS_SPEECH_THRESHOLD = gr.Slider(
                0, 1, step=0.025,
                label="语音激活阈值 Voice State Threshold",
                info="数值越小，字幕越少。\nThe smaller the value, the harder it is to activate.",
                value=1.0,
                visible=False,
                interactive=True
            )
            slider_vad_SNR_THRESHOLD = gr.Slider(
                0, 60, step=1,
                label="信噪比值 SNR Value",
                info="低 SNR：音频中噪声成分较大; 高 SNR：意味着信号强、噪声弱。\nLow SNR: The noise component in the audio is large; High SNR: Means that the signal is strong and the noise is weak",
                value=10,
                visible=False,
                interactive=True
            )
            slider_vad_BACKGROUND_NOISE_dB_INIT = gr.Slider(
                0, 100, step=1,
                label="初始背景噪声 Initial Background Noise",
                info="初始背景噪声，单位：分贝。\nInitial background noise value in dB.",
                value=40,
                visible=False,
                interactive=True
            )
        with gr.Row():
            slider_vad_SPEAKING_SCORE = gr.Slider(
                0, 1, step=0.025,
                label="语音状态分数 Voice State Score",
                info="值越大，判断语音状态越困难。\nThe higher the value, the more difficult it is to determine the state of the speech",
                value=0.4,
                visible=True,
                interactive=True
            )
            slider_vad_SILENCE_SCORE = gr.Slider(
                0, 1, step=0.025,
                label="静音状态分数 Silence State Score",
                info="值越大，越容易截断语音。\nA larger value makes it easier to cut off speaking.",
                value=0.4,
                visible=True,
                interactive=True
            )
        with gr.Row():
            slider_vad_FUSION_THRESHOLD = gr.Slider(
                0, 5, step=0.025,
                label="合并时间戳 Merge Timestamps",
                info="如果两个语音段间隔太近，它们会被合并成一个。 单位：秒。\nIf two voice segments are too close, they will be merged into one. Unit: Seconds.",
                value=0.2,
                visible=True,
                interactive=True
            )
            slider_vad_MIN_SPEECH_DURATION = gr.Slider(
                0, 2, step=0.025,
                label="过滤短语音段 Filter Short Voice Segment",
                info="最短语音时长。单位：秒。\nMinimum duration for voice filtering. Unit: Seconds.",
                value=0.05,
                visible=True,
                interactive=True
            )
            slider_vad_MAX_SPEECH_DURATION = gr.Slider(
                1, 15, step=1,
                label="过滤长语音段 Filter Long Voice Segment",
                info="最大语音时长。单位：秒。\nMaximum voice duration. Unit: Seconds.",
                value=15,
                visible=True,
                interactive=True
            )
            slider_vad_MIN_SILENCE_DURATION = gr.Slider(
                100, 3000, step=25,
                label="静音时长判断 Silence Duration Judgment",
                info="最短静音时长。单位：毫秒。\nMinimum silence duration. Unit: Milliseconds.",
                value=1500,
                visible=True,
                interactive=True
            )

    # --------------------------------------------------------------------- #
    # Status & Run                                                          #
    # --------------------------------------------------------------------- #
    task_state = gr.Textbox(
        label="任务状态 Task State",
        value="点击运行任务并稍等片刻。Click the Run button and wait a moment.",
        interactive=False,
        elem_classes="task-state")

    submit_button = gr.Button("🚀  运行任务  |  Run Task", variant="primary", )

    submit_button.click(
        fn=MAIN_PROCESS,
        inputs=[
            task,
            hardware,
            parallel_threads,
            file_path_input,
            translate_language,
            transcribe_language,
            model_asr,
            model_vad,
            model_denoiser,
            model_llm,
            switcher_run_test,
            switcher_denoiser_cache,
            slider_vad_pad,
            slider_denoise_factor,
            slider_vad_ONE_MINUS_SPEECH_THRESHOLD,
            slider_vad_SNR_THRESHOLD,
            slider_vad_BACKGROUND_NOISE_dB_INIT,
            slider_vad_SPEAKING_SCORE,
            slider_vad_SILENCE_SCORE,
            slider_vad_FUSION_THRESHOLD,
            slider_vad_MIN_SPEECH_DURATION,
            slider_vad_MAX_SPEECH_DURATION,
            slider_vad_MIN_SILENCE_DURATION
        ],
        outputs=task_state
    )
    task.change(
        fn=update_task,
        inputs=task,
        outputs=[model_llm, translate_language]
    )
    model_llm.change(
        fn=update_translate_language,
        inputs=model_llm,
        outputs=translate_language
    )
    model_asr.change(
        fn=update_transcribe_language,
        inputs=model_asr,
        outputs=transcribe_language
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
if __name__ == "__main__":
    DEVICE_ID = 0
    MAX_SEQ_LEN = 64
    MAX_ASR_SEGMENT = 240000  # The exported setting of ASR models. Do not edit it.
    TEN_VAD_FRAME_LENGTH = 256
    inv_16k = float(1.0 / 16000.0)
    inv_int16 = float(1.0 / 32768.0)
    HumAware_param = 512.0 * inv_16k
    NVIDIA_VAD_param = 320.0 * inv_16k
    TEN_VAD_param = TEN_VAD_FRAME_LENGTH * inv_16k
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
    language_map_A = {
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
    full_language_names_A = {
            '中文': 'zh',            '日語': 'ja',          '한국인': 'ko',
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
    language_map_B = {'auto': 0, 'zh': 1, 'en': 2, 'yue': 3, 'ja': 4, 'ko': 5}
    full_language_names_B = {
        '自动 auto': 'auto',
        '中文': 'zh',
        'english': 'en',
        '粤语': 'yue',
        '日語': 'ja',
        '한국인': 'ko'
    }
    
    DOLPHIN_LANGUAGE_REGION = {
        # ───────────────────────────── Auto Detection ─────────────────────────────
        "Auto-Auto"                    : "auto-auto",
        "Mandarin-Auto"                : "zh-auto",
        "Yue-Auto"                     : "ct-NULL",
        "Tamil-Auto"                   : "ta-auto",
        "Urdu-Auto"                    : "ur-auto",
        "Arabic-Auto"                  : "ar-auto",

        "自动-自动"                      : "auto-auto",
        "中文-自动"                      : "zh-auto",
        "粤语-自动"                      : "ct-NULL",
        "泰米尔语-自动"                   : "ta-auto",
        "乌尔都语-自动"                   : "ur-auto",
        "阿拉伯语-自动"                   : "ar-auto",

        # ───────────────────────────── Chinese variants ─────────────────────────────
        "Chinese-Mandarin"              : "zh-CN",
        "Chinese-Taiwan"                : "zh-TW",
        "Chinese-Wuyu"                  : "zh-WU",
        "Chinese-Sichuan"               : "zh-SICHUAN",
        "Chinese-Shanxi"                : "zh-SHANXI",
        "Chinese-Anhui"                 : "zh-ANHUI",
        "Chinese-Tianjin"               : "zh-TIANJIN",
        "Chinese-Ningxia"               : "zh-NINGXIA",
        "Chinese-Shaanxi"               : "zh-SHAANXI",
        "Chinese-Hebei"                 : "zh-HEBEI",
        "Chinese-Shandong"              : "zh-SHANDONG",
        "Chinese-Guangdong"             : "zh-GUANGDONG",
        "Chinese-Shanghai"              : "zh-SHANGHAI",
        "Chinese-Hubei"                 : "zh-HUBEI",
        "Chinese-Liaoning"              : "zh-LIAONING",
        "Chinese-Gansu"                 : "zh-GANSU",
        "Chinese-Fujian"                : "zh-FUJIAN",
        "Chinese-Hunan"                 : "zh-HUNAN",
        "Chinese-Henan"                 : "zh-HENAN",
        "Chinese-Yunnan"                : "zh-YUNNAN",
        "Chinese-Minnan"                : "zh-MINNAN",
        "Chinese-Wenzhou"               : "zh-WENZHOU",

        "中文-普通话"                    : "zh-CN",
        "中文-台湾"                      : "zh-TW",
        "中文-吴语"                      : "zh-WU",
        "中文-四川话"                    : "zh-SICHUAN",
        "中文-山西话"                    : "zh-SHANXI",
        "中文-安徽话"                    : "zh-ANHUI",
        "中文-天津话"                    : "zh-TIANJIN",
        "中文-宁夏话"                    : "zh-NINGXIA",
        "中文-陕西话"                    : "zh-SHAANXI",
        "中文-河北话"                    : "zh-HEBEI",
        "中文-山东话"                    : "zh-SHANDONG",
        "中文-广东话"                    : "zh-GUANGDONG",
        "中文-上海话"                    : "zh-SHANGHAI",
        "中文-湖北话"                    : "zh-HUBEI",
        "中文-辽宁话"                    : "zh-LIAONING",
        "中文-甘肃话"                    : "zh-GANSU",
        "中文-福建话"                    : "zh-FUJIAN",
        "中文-湖南话"                    : "zh-HUNAN",
        "中文-河南话"                    : "zh-HENAN",
        "中文-云南话"                    : "zh-YUNNAN",
        "中文-闽南语"                    : "zh-MINNAN",
        "中文-温州话"                    : "zh-WENZHOU",

        # ───────────────────────────── Yue-Cantonese variants ───────────────────────────
        "Yue-Unknown"                  : "ct-NULL",
        "Yue-Hongkong"                 : "ct-HK",
        "Yue-Guangdong"                : "ct-GZ",

        "粤语-未知"                     : "ct-NULL",
        "粤语-香港"                     : "ct-HK",
        "粤语-广东"                     : "ct-GZ",

        # ───────────────────────────── East-Asian languages ──────────────────────────────
        "Japanese"                      : "ja-JP",
        "Korean"                        : "ko-KR",

        "日语"                           : "ja-JP",
        "韩语"                           : "ko-KR",

        # ───────────────────────────── South-East Asian languages ─────────────────────────
        "Thai"                          : "th-TH",
        "Indonesian"                    : "id-ID",
        "Vietnamese"                    : "vi-VN",
        "Malay"                         : "ms-MY",
        "Burmese"                       : "my-MM",
        "Tagalog"                       : "tl-PH",
        "Khmer"                         : "km-KH",
        "Javanese"                      : "jv-ID",
        "Lao"                           : "lo-LA",
        "Filipino"                      : "fil-PH",
        "Sundanese"                     : "su-ID",

        "泰语"                            : "th-TH",
        "印度尼西亚语"                     : "id-ID",
        "越南语"                          : "vi-VN",
        "马来语"                          : "ms-MY",
        "缅甸语"                          : "my-MM",
        "塔加洛语"                        : "tl-PH",
        "高棉语"                          : "km-KH",
        "爪哇语"                          : "jv-ID",
        "老挝语"                          : "lo-LA",
        "菲律宾语"                        : "fil-PH",
        "巽他语"                          : "su-ID",

        # ───────────────────────────── South-Asian languages ──────────────────────────────
        "Hindi"                         : "hi-IN",
        "Bengali"                       : "bn-BD",
        "Tamil-Singaporean"             : "ta-SG",
        "Tamil-Sri Lankan"              : "ta-LK",
        "Tamil-India"                   : "ta-IN",
        "Tamil-Malaysia"                : "ta-MY",
        "Telugu"                        : "te-IN",
        "Gujarati"                      : "gu-IN",
        "Oriya"                         : "or-IN",
        "Odia"                          : "or-IN",
        "Nepali"                        : "ne-NP",
        "Sinhala"                       : "si-LK",
        "Panjabi"                       : "pa-IN",
        "Kashmiri"                      : "ks-IN",
        "Marathi"                       : "mr-IN",

        "印地语"                         : "hi-IN",
        "孟加拉语"                       : "bn-BD",
        "泰米尔语-新加坡"                 : "ta-SG",
        "泰米尔语-斯里兰卡"                : "ta-LK",
        "泰米尔语-印度"                   : "ta-IN",
        "泰米尔语-马来西亚"                : "ta-MY",
        "泰卢固语"                        : "te-IN",
        "古吉拉特语"                      : "gu-IN",
        "奥里亚语"                        : "or-IN",
        "尼泊尔语"                        : "ne-NP",
        "僧伽罗语"                        : "si-LK",
        "旁遮普语"                        : "pa-IN",
        "克什米尔语"                      : "ks-IN",
        "马拉地语"                        : "mr-IN",

        # ───────────────────────────── Middle-Eastern languages ───────────────────────────
        "Urdu"                          : "ur-PK",
        "Urdu-India"                    : "ur-IN",
        "Persian"                       : "fa-IR",
        "Pushto"                        : "ps-AF",

        "乌尔都语"                        : "ur-PK",
        "乌尔都语-印度"                    : "ur-IN",
        "波斯语"                          : "fa-IR",
        "普什图语"                        : "ps-AF",

        # ───────────────────────────── Arabic variants ──────────────────────────────
        "Arabic"                        : "ar-GLA",
        "Arabic-Morocco"                : "ar-MA",
        "Arabic-Saudi Arabia"           : "ar-SA",
        "Arabic-Egypt"                  : "ar-EG",
        "Arabic-Kuwait"                 : "ar-KW",
        "Arabic-Libya"                  : "ar-LY",
        "Arabic-Jordan"                 : "ar-JO",
        "Arabic-U.A.E."                 : "ar-AE",
        "Arabic-Levant"                 : "ar-LVT",

        "阿拉伯语"                        : "ar-GLA",
        "阿拉伯语-摩洛哥"                  : "ar-MA",
        "阿拉伯语-沙特"                    : "ar-SA",
        "阿拉伯语-埃及"                    : "ar-EG",
        "阿拉伯语-科威特"                  : "ar-KW",
        "阿拉伯语-利比亚"                  : "ar-LY",
        "阿拉伯语-约旦"                    : "ar-JO",
        "阿拉伯语-阿联酋"                  : "ar-AE",
        "阿拉伯语-黎凡特"                  : "ar-LVT",

        # ───────────────────────────── Central-Asian languages ────────────────────────────
        "Uighur"                        : "ug-CN",
        "Uzbek"                         : "uz-UZ",
        "Kazakh"                        : "kk-KZ",
        "Mongolian"                     : "mn-MN",
        "Kabyle"                        : "kab-NULL",
        "Bashkir"                       : "ba-NULL",
        "Tajik"                         : "tg-TJ",
        "Kirghiz"                       : "ky-KG",
        "Azerbaijani"                   : "az-AZ",

        "维吾尔语"                        : "ug-CN",
        "乌兹别克语"                      : "uz-UZ",
        "哈萨克语"                        : "kk-KZ",
        "蒙古语"                          : "mn-MN",
        "卡拜尔语"                        : "kab-NULL",
        "巴什基尔语"                      : "ba-NULL",
        "塔吉克语"                        : "tg-TJ",
        "吉尔吉斯语"                      : "ky-KG",
        "阿塞拜疆语"                      : "az-AZ",

        # ───────────────────────────── Eastern-European languages ─────────────────────────
        "Russian"                       : "ru-RU",
        "俄语"                           : "ru-RU"
    }
    
    INV_DOLPHIN_REGION_LANGUAGE = {
        # ───────────────────────────── Chinese variants ─────────────────────────────
        "zh-CN"                         : "Chinese-Mandarin",
        "zh-TW"                         : "Chinese-Taiwan",
        "zh-WU"                         : "Chinese-Wuyu",
        "zh-SICHUAN"                    : "Chinese-Sichuan",
        "zh-SHANXI"                     : "Chinese-Shanxi",
        "zh-ANHUI"                      : "Chinese-Anhui",
        "zh-TIANJIN"                    : "Chinese-Tianjin",
        "zh-NINGXIA"                    : "Chinese-Ningxia",
        "zh-SHAANXI"                    : "Chinese-Shaanxi",
        "zh-HEBEI"                      : "Chinese-Hebei",
        "zh-SHANDONG"                   : "Chinese-Shandong",
        "zh-GUANGDONG"                  : "Chinese-Guangdong",
        "zh-SHANGHAI"                   : "Chinese-Shanghai",
        "zh-HUBEI"                      : "Chinese-Hubei",
        "zh-LIAONING"                   : "Chinese-Liaoning",
        "zh-GANSU"                      : "Chinese-Gansu",
        "zh-FUJIAN"                     : "Chinese-Fujian",
        "zh-HUNAN"                      : "Chinese-Hunan",
        "zh-HENAN"                      : "Chinese-Henan",
        "zh-YUNNAN"                     : "Chinese-Yunnan",
        "zh-MINNAN"                     : "Chinese-Minnan",
        "zh-WENZHOU"                    : "Chinese-Wenzhou",

        # ───────────────────────────── Yue-Cantonese variants ───────────────────────────
        "ct-NULL"                       : "Yue-Unknown",
        "ct-HK"                         : "Yue-Hongkong",
        "ct-GZ"                         : "Yue-Guangdong",

        # ───────────────────────────── East-Asian languages ──────────────────────────────
        "ja-JP"                         : "Japanese",
        "ko-KR"                         : "Korean",

        # ───────────────────────────── South-East Asian languages ─────────────────────────
        "th-TH"                         : "Thai",
        "id-ID"                         : "Indonesian",
        "vi-VN"                         : "Vietnamese",
        "ms-MY"                         : "Malay",
        "my-MM"                         : "Burmese",
        "tl-PH"                         : "Tagalog",
        "km-KH"                         : "Khmer",
        "jv-ID"                         : "Javanese",
        "lo-LA"                         : "Lao",
        "fil-PH"                        : "Filipino",
        "su-ID"                         : "Sundanese",

        # ───────────────────────────── South-Asian languages ──────────────────────────────
        "hi-IN"                         : "Hindi",
        "bn-BD"                         : "Bengali",
        "ta-SG"                         : "Tamil-Singaporean",
        "ta-LK"                         : "Tamil-Sri Lankan",
        "ta-IN"                         : "Tamil-India",
        "ta-MY"                         : "Tamil-Malaysia",
        "te-IN"                         : "Telugu",
        "gu-IN"                         : "Gujarati",
        "or-IN"                         : "Odia",  # Note: "Oriya" would be overwritten by "Odia"
        "ne-NP"                         : "Nepali",
        "si-LK"                         : "Sinhala",
        "pa-IN"                         : "Panjabi",
        "ks-IN"                         : "Kashmiri",
        "mr-IN"                         : "Marathi",

        # ───────────────────────────── Middle-Eastern languages ───────────────────────────
        "ur-PK"                         : "Urdu-Islamic Republic of Pakistan",  # Note: "Urdu" would be overwritten
        "ur-IN"                         : "Urdu-India",
        "fa-IR"                         : "Persian",
        "ps-AF"                         : "Pushto",

        # ───────────────────────────── Arabic variants ──────────────────────────────
        "ar-GLA"                        : "Arabic",
        "ar-MA"                         : "Arabic-Morocco",
        "ar-SA"                         : "Arabic-Saudi Arabia",
        "ar-EG"                         : "Arabic-Egypt",
        "ar-KW"                         : "Arabic-Kuwait",
        "ar-LY"                         : "Arabic-Libya",
        "ar-JO"                         : "Arabic-Jordan",
        "ar-AE"                         : "Arabic-U.A.E.",
        "ar-LVT"                        : "Arabic-Levant",

        # ───────────────────────────── Central-Asian languages ────────────────────────────
        "ug-CN"                         : "Uighur",
        "uz-UZ"                         : "Uzbek",
        "kk-KZ"                         : "Kazakh",
        "mn-MN"                         : "Mongolian",
        "kab-NULL"                      : "Kabyle",
        "ba-NULL"                       : "Bashkir",
        "tg-TJ"                         : "Tajik",
        "ky-KG"                         : "Kirghiz",
        "az-AZ"                         : "Azerbaijani",

        # ───────────────────────────── Eastern-European languages ─────────────────────────
        "ru-RU"                         : "Russian",
    }

    LANGUAGE_MAP = [(language_map_A, full_language_names_A), (language_map_B, full_language_names_B)]

    PYTHON_PACKAGE = site.getsitepackages()[-1]
    shutil.copyfile("./VAD/Silero/utils_vad.py", PYTHON_PACKAGE + "/silero_vad/utils_vad.py")
    shutil.copyfile("./VAD/Silero/model.py", PYTHON_PACKAGE + "/silero_vad/model.py")
    shutil.copyfile("./VAD/Silero/silero_vad.onnx", PYTHON_PACKAGE + "/silero_vad/data/silero_vad.onnx")

    GUI.launch()
    
