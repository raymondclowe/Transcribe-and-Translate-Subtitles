import time
import whisper
from pathlib import Path
import sys

def bench(model_name, audio_path):
    t0 = time.time()
    model = whisper.load_model(model_name)
    t1 = time.time()
    load_time = t1 - t0

    t_start = time.time()
    result = model.transcribe(audio_path)
    t_end = time.time()

    audio_len = 0.0
    if 'segments' in result and result['segments']:
        audio_len = result['segments'][-1]['end']

    inference_time = t_end - t_start
    rtf = inference_time / audio_len if audio_len>0 else None

    return {
        'model': model_name,
        'load_time_s': load_time,
        'inference_time_s': inference_time,
        'audio_length_s': audio_len,
        'rtf': rtf,
        'text': result.get('text','')[:200]
    }


if __name__ == '__main__':
    audio = sys.argv[1] if len(sys.argv)>1 else 'Media/test.wav'
    models = ['tiny','base','small']
    for m in models:
        r = bench(m, audio)
        print(r)
