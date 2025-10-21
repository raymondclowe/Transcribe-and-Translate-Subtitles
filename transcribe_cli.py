#!/usr/bin/env python3
"""Simple fast CLI to transcribe an audio file using OpenAI Whisper.
Default model is 'tiny' (fast, lower accuracy). Saves a VTT file and prints text.
"""
import argparse
import time
from pathlib import Path

import whisper


def save_vtt(segments, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('WEBVTT\n\n')
        for seg in segments:
            start = seg['start']
            end = seg['end']
            def fmt(s):
                h = int(s // 3600)
                m = int((s % 3600) // 60)
                sec = int(s % 60)
                ms = int((s - int(s)) * 1000)
                return f"{h:02}:{m:02}:{sec:02}.{ms:03}"
            f.write(f"{fmt(start)} --> {fmt(end)}\n{seg['text'].strip()}\n\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input', help='Path to audio file')
    p.add_argument('--model', default='tiny', help='Whisper model to use (tiny, base, small, etc)')
    p.add_argument('--out', default='Results/Subtitles/quick.vtt', help='Output VTT path')
    args = p.parse_args()

    t0 = time.time()
    model = whisper.load_model(args.model)
    t1 = time.time()
    print(f"Loaded model {args.model} in {t1-t0:.2f}s")

    t_start = time.time()
    result = model.transcribe(args.input)
    t_end = time.time()

    audio_len = 0.0
    if 'segments' in result and result['segments']:
        audio_len = result['segments'][-1]['end']

    elapsed = t_end - t_start
    rtf = elapsed / audio_len if audio_len > 0 else None

    print('Transcription:')
    print(result.get('text',''))
    print(f'Inference time: {elapsed:.2f}s; audio length: {audio_len:.2f}s; RTF={rtf:.3f}')

    save_vtt(result.get('segments', []), Path(args.out))
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
