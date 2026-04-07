# Application image - based on pre-built dependencies
FROM whisper-subtitles-base:latest

# Copy application code only
COPY app/ ./app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
