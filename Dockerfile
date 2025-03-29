# Use NVIDIA's PyTorch container as base
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Install UV and ffmpeg
RUN apt-get update && apt-get install -y ffmpeg
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY api.py .

# Create and activate venv using UV with Python 3.10
RUN uv venv --python=3.10 /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# Install dependencies using UV
RUN uv pip install -e .

# Expose port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]