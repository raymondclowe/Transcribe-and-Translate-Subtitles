LEARNINGS FROM SESSION - Transcribe-and-Translate-Subtitles
=========================================================

This document captures the key learnings, decisions, and next steps from the interactive session where a CPU-only demo was prepared to transcribe a user-provided audio file.

1) Goal and approach
---------------------
- Goal: produce a working CPU-only demo that can transcribe an audio file quickly, expose a CLI and a REST endpoint, and report performance (RTF).
- Pragmatic choice: use the OpenAI "whisper" PyTorch implementation on CPU for a fast, practical demo instead of fully populating the repository's ONNX model path.

2) What I changed / added
-------------------------
- Patched `run.py` to avoid failing when Silero VAD helper files are missing (safe conditional copy + warnings).
- Added small helper scripts (top-level):
  - `quick_transcribe.py` — one-off script that transcribed a WAV and wrote VTT.
  - `transcribe_cli.py` — fast CLI wrapper using Whisper (default `tiny`) that reports load time, inference time, audio length and RTF, and writes VTT.
  - `rest_server.py` — FastAPI REST endpoint (`/transcribe`) that accepts an uploaded audio file and returns transcription + timing metrics.
  - `benchmark_transcribe.py` — runs `tiny`, `base`, and `small` Whisper models to measure load & inference times and computes RTF.

3) Environment and dependencies
-------------------------------
- Python venv used: `./venv` (Python 3.12 in this session).
- System packages installed: `ffmpeg`, `libc++1` (so FFmpeg conversions work and optional C++ libs are available).
- Key Python packages (CPU-focused): `onnxruntime==1.23.1`, `transformers`, `sentencepiece`, `tokenizers`, `psutil`, `py-cpuinfo`, `openai-whisper`, and CPU builds of `torch`/`torchaudio`.

4) Media and outputs
--------------------
- The user-provided `Memo 251021_080406.m4a` was moved to `Media/` and converted to `Media/test.wav` using FFmpeg.
- Quick transcription produced `Results/Subtitles/test.vtt` (VTT subtitle file) using Whisper `small`.

5) Performance (example results from this session)
-------------------------------------------------
- File: `Media/test.wav` (~15.1 seconds).
- Measured Real-Time Factors (RTF):
  - `tiny`: inference_time ~1.06s → RTF ≈ 0.07
  - `base`: inference_time ~1.03s → RTF ≈ 0.07
  - `small`: inference_time ~2.88s → RTF ≈ 0.19

Notes: RTF = inference_time / audio_length (lower is faster than real time). Tiny/base are fastest on CPU and suitable for low-latency CLI/REST. Small gives better accuracy at higher cost but still < 1.0 RTF on this machine.

6) Repo layout changes (safe & minimal)
--------------------------------------
- Created placeholders for ONNX model files to avoid early crashes when `run.py` checks for model presence. These are only placeholders — the actual ONNX models must be downloaded and placed in the expected `ASR/*` folders for full ONNX runtime usage.

7) Git remotes and pushing
--------------------------
- Original remotes before change: `origin` pointed at the upstream repo and `upstream` pointed at a fork. The user provided their fork URL and requested the repository be pointed to it.

8) Next steps and recommendations
---------------------------------
- If you want to continue using the repo's ONNX path (for potentially faster inference with onnxruntime), download and place the actual model ONNX artifacts into `ASR/Whisper/FP32/Official-Whisper-v3/` and the tokenizer assets into `ASR/Whisper/Tokenizer/Official-Whisper-v3/`.
- For production REST usage, add batching/queueing, limit concurrency, and consider running inference in a worker pool (torch/onnx inference can be CPU-bound). Containerize with resource limits and expose metrics.
- If you want to publish these changes, push the branch to your fork (I will set `origin` to your fork and push the new `LEARNINGS.md`).

9) Repro commands (copy-paste)
------------------------------
Create venv & activate:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # or install the CPU-only set used in this session
```

Convert the provided m4a and run quick CLI transcription:

```bash
mkdir -p Media
# move your file into Media/
ffmpeg -y -i Media/Memo\ 251021_080406.m4a Media/test.wav
python transcribe_cli.py Media/test.wav --model tiny
```

Run the REST server (FastAPI + uvicorn):

```bash
python rest_server.py  # accessible at http://127.0.0.1:8000
```

10) Session artifacts (files added/edited)
----------------------------------------
- Edited: `run.py` (made Silero helper copy logic tolerant)
- Added: `quick_transcribe.py`, `transcribe_cli.py`, `rest_server.py`, `benchmark_transcribe.py`, `LEARNINGS.md`

11) Contact & follow-ups
------------------------
- If you'd like, I can now:
  - push this branch to your fork (I've been given your fork URL) and open a PR to the original repo,
  - download and wire actual ONNX model files into the `ASR/` layout (requires large downloads), or
  - harden the REST server for production (TLS, Dockerfile, systemd/uvicorn setup, queueing).

-- End of session learnings
