import whisper
from datetime import timedelta
from pathlib import Path

model = whisper.load_model('small')
result = model.transcribe('Media/test.wav', language='en')
print('Transcription:')
print(result['text'])
# create simple VTT
segments = result.get('segments', [])
Path('Results/Subtitles').mkdir(parents=True, exist_ok=True)
with open('Results/Subtitles/test.vtt','w',encoding='utf-8') as f:
    f.write('WEBVTT\n\n')
    for i, seg in enumerate(segments):
        start = timedelta(seconds=seg['start'])
        end = timedelta(seconds=seg['end'])
        def fmt(td):
            total = int(td.total_seconds())
            ms = int((td.total_seconds() - total) * 1000)
            h = total // 3600
            m = (total % 3600) // 60
            s = total % 60
            return f"{h:02}:{m:02}:{s:02}.{ms:03}"
        f.write(f"{fmt(start)} --> {fmt(end)}\n{seg['text'].strip()}\n\n")
print('Saved Results/Subtitles/test.vtt')
