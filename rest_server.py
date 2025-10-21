from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import whisper
import time
from pathlib import Path

app = FastAPI()
model = None


@app.on_event('startup')
async def startup_event():
    global model
    model = whisper.load_model('tiny')


@app.post('/transcribe')
async def transcribe_file(file: UploadFile = File(...)):
    contents = await file.read()
    tmp = Path('tmp_upload.wav')
    tmp.write_bytes(contents)
    t0 = time.time()
    result = model.transcribe(str(tmp))
    t1 = time.time()
    audio_len = 0.0
    if 'segments' in result and result['segments']:
        audio_len = result['segments'][-1]['end']
    return JSONResponse({
        'text': result.get('text',''),
        'segments': result.get('segments', []),
        'inference_time_s': t1-t0,
        'audio_length_s': audio_len,
        'rtf': (t1-t0)/audio_len if audio_len>0 else None
    })


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
