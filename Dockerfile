# Application image - based on pre-built dependencies
FROM whisper-subtitles-base:latest

# Copy application code only
COPY app/ ./app/

# Safety net: ensure diarization dependency exists even with stale base image
RUN pip3 install --no-cache-dir pyannote.audio "huggingface_hub<1.0"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
