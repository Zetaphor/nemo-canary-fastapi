from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import urllib.request
import os
from nemo.collections.asr.models import EncDecMultiTaskModel
import json
import time
import soundfile as sf
from urllib.parse import urlparse
import tempfile
import io
import requests

# Initialize FastAPI app
app = FastAPI(title="Canary ASR API")

class TranscriptionRequest(BaseModel):
    audio_source: str  # Can be either a file path or URL
    target_lang: str | None = None  # Optional target language for translation

    @validator('audio_source')
    def validate_audio_source(cls, v):
        if not v:
            raise ValueError("audio_source cannot be empty")
        return v

# Load model at startup
print("Loading Canary model...")
# nvidia/canary-1b, nvidia/canary-1b-flash, or nvidia/canary-180m-flash
canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-180m-flash')

# Configure decoding parameters
decode_cfg = canary_model.cfg.decoding
decode_cfg.beam.beam_size = 1
canary_model.change_decoding_strategy(decode_cfg)
print("Model loaded and configured")

def download_audio(url: str) -> tuple:
    """Download audio from URL and return the audio data and filename"""
    try:
        # Extract filename from URL or generate one
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = f"audio_{int(time.time())}.wav"

        # Download the file into memory
        response = requests.get(url)
        response.raise_for_status()
        audio_data = io.BytesIO(response.content)

        return audio_data, filename
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download audio: {str(e)}")

@app.post("/transcribe")
async def transcribe_audio(request: TranscriptionRequest):
    try:
        # Start timing for the entire process
        total_start_time = time.time()
        io_time = 0

        # Determine if input is URL or file path
        is_url = urlparse(request.audio_source).scheme in ['http', 'https']

        # Start timing for I/O operations
        io_start_time = time.time()

        if is_url:
            print(f"Downloading audio from URL: {request.audio_source}")
            audio_data, filename = download_audio(request.audio_source)
        else:
            # Read file into memory
            if not os.path.exists(request.audio_source):
                raise HTTPException(status_code=404, detail=f"Audio file not found: {request.audio_source}")

            with open(request.audio_source, 'rb') as f:
                audio_data = io.BytesIO(f.read())
            filename = os.path.basename(request.audio_source)

        print(f"Processing audio: {filename}")
        if request.target_lang:
            print(f"Will translate to: {request.target_lang}")

        # Get audio duration directly from the in-memory data
        audio_data.seek(0)
        audio_info = sf.info(audio_data)
        audio_duration = audio_info.duration

        # Create a temporary file for the audio data
        # (NeMo requires a file path for transcription)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio_file:
            audio_data.seek(0)
            temp_audio_file.write(audio_data.read())
            temp_audio_path = temp_audio_file.name

        # Create manifest in memory
        manifest_entry = {
            "audio_filepath": temp_audio_path,
            "duration": audio_duration,
            "taskname": "s2t_translation" if request.target_lang else "asr",
            "source_lang": "en",
            "target_lang": request.target_lang if request.target_lang else "en",
            "pnc": "yes",
            "prompt_format": "canary2" if request.target_lang else None
        }

        # Create a temporary file for the manifest
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_manifest_file:
            json.dump(manifest_entry, temp_manifest_file)
            manifest_path = temp_manifest_file.name

        # Calculate I/O time
        io_time += time.time() - io_start_time

        try:
            # Start timing for inference
            inference_start_time = time.time()

            # Run transcription/translation
            results = canary_model.transcribe(
                audio=manifest_path,
                batch_size=1
            )

            # Calculate inference time
            inference_time = time.time() - inference_start_time

            # Start timing for cleanup I/O
            io_start_time = time.time()

            # Get the text from the first result
            result_text = results[0].text

            # Calculate total elapsed time
            total_elapsed_time = time.time() - total_start_time

            # Calculate RTF (Real-Time Factor) based on inference time only
            rtf = inference_time / audio_duration

            response = {
                "text": result_text,
                "total_time_seconds": round(total_elapsed_time, 2),
                "inference_time_seconds": round(inference_time, 2),
                "io_time_seconds": round(io_time, 2),
                "audio_duration_seconds": round(audio_duration, 2),
                "rtf": round(rtf, 2)
            }

            # Add translation info if applicable
            if request.target_lang:
                response["source_lang"] = "en"
                response["target_lang"] = request.target_lang

            return response

        finally:
            # Clean up temporary files
            io_start_time = time.time()
            try:
                os.unlink(temp_audio_path)
                os.unlink(manifest_path)
            except:
                pass
            io_time += time.time() - io_start_time

    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        print(f"Exception type: {type(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
