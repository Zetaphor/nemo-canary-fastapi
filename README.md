# Canary ASR API

A FastAPI-based REST API for NVIDIA's family of Canary-1B ASR (Automatic Speech Recognition) models with support for translation.

## Features

- Fast and accurate speech-to-text transcription
- Support for both local audio files and URLs
- Optional translation to other languages
- Real-time factor (RTF) calculation
- Processing time metrics
- Simple REST API interface

## Installation Options

### Option 1: Quick Start with Docker

1. Clone the repository:
```bash
git clone https://github.com/zetaphor/nemo-canary-fastapi.git
cd nemo-canary-fastapi
```

2. Start the service:
```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`

### Option 2: Local Installation

#### Prerequisites
- Python 3.10
- NVIDIA GPU with CUDA support
- [UV package manager](https://github.com/astral-sh/uv)

1. Clone the repository:
```bash
git clone https://github.com/zetaphor/nemo-canary-fastapi.git
cd nemo-canary-fastapi
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
uv pip install -e .
```

4. Change the model in `api.py` if desired:

Available models:
- [`nvidia/canary-1b`](https://huggingface.co/nvidia/canary-1b)
- [`nvidia/canary-1b-flash`](https://huggingface.co/nvidia/canary-1b-flash)
- [`nvidia/canary-180m-flash`](https://huggingface.co/nvidia/canary-180m-flash)

```python
canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')
```

5. Run the API:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Usage

### Transcribe Endpoint

**POST** `/transcribe`

The endpoint accepts JSON with the following parameters:
- `audio_source`: Path to local audio file or URL (required)
- `target_lang`: Target language for translation (optional)

#### Examples

1. Transcribe from local file:
```bash
curl -X POST "http://localhost:8000/transcribe" \
    -H "Content-Type: application/json" \
    -d '{"audio_source": "/path/to/audio.wav"}'
```

2. Transcribe from URL:
```bash
curl -X POST "http://localhost:8000/transcribe" \
    -H "Content-Type: application/json" \
    -d '{"audio_source": "https://example.com/audio.wav"}'
```

3. Translate from local file:
```bash
curl -X POST "http://localhost:8000/transcribe" \
    -H "Content-Type: application/json" \
    -d '{"audio_source": "/path/to/audio.wav", "target_lang": "de"}'
```

### Response Format

For ASR:
```json
{
    "text": "Hello, world!",
    "processing_time_seconds": 0.12,
    "audio_duration_seconds": 1.0,
    "rtf": 0.12
}
```

For translation:
```json
{
    "text": "Hallo, Welt!",
    "processing_time_seconds": 0.15,
    "audio_duration_seconds": 1.0,
    "rtf": 0.15,
    "source_lang": "en",
    "target_lang": "de"
}
```

## Development

To access the interactive API documentation, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`