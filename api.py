from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nemo.collections.asr.models import EncDecMultiTaskModel
import json
import time
import soundfile as sf

"""
curl -X POST "http://localhost:8000/transcribe" \
    -H "Content-Type: application/json" \
    -d '{"audio_path": "audio.wav"}'
"""

# Initialize FastAPI app
app = FastAPI(title="Canary ASR API")

# Define request model
class TranscriptionRequest(BaseModel):
    audio_path: str

# Load model at startup
print("Loading Canary model...")
# nvidia/canary-1b, nvidia/canary-1b-flash, or nvidia/canary-180m-flash
canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')

# Configure decoding parameters
decode_cfg = canary_model.cfg.decoding
decode_cfg.beam.beam_size = 1
canary_model.change_decoding_strategy(decode_cfg)
print("Model loaded and configured")

@app.post("/transcribe")
async def transcribe_audio(request: TranscriptionRequest):
    try:
        print(f"Transcribing audio from: {request.audio_path}")

        # Get audio duration
        audio_info = sf.info(request.audio_path)
        audio_duration = audio_info.duration

        # Update the manifest file with the new audio path
        manifest_entry = {
            "audio_filepath": request.audio_path,
            "duration": audio_duration,  # Now using actual duration
            "taskname": "asr",
            "source_lang": "en",
            "target_lang": "en",
            "pnc": "yes"
        }

        # Write to the existing manifest file
        with open("asr_manifest.json", "w") as f:
            json.dump(manifest_entry, f)

        # Start timing
        start_time = time.time()

        # Run transcription and get the text from the first result
        transcriptions = canary_model.transcribe(
            audio="asr_manifest.json",
            batch_size=1
        )

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Get the text property from the first result
        transcription_text = transcriptions[0].text

        # Calculate RTF (Real-Time Factor)
        rtf = elapsed_time / audio_duration

        return {
            "transcription": transcription_text,
            "processing_time_seconds": round(elapsed_time, 2),
            "audio_duration_seconds": round(audio_duration, 2),
            "rtf": round(rtf, 2)
        }

    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        print(f"Exception type: {type(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)