#!/usr/bin/env python3
"""
Standalone transcription worker that can be killed.
Used for proper job cancellation support.

Outputs JSON progress updates to stdout:
  {"status": "downloading_model", "progress": 0, "message": "Downloading model..."}
  {"status": "loading_model", "progress": 5, "message": "Loading model to memory..."}
  {"status": "loading_audio", "progress": 10, "message": "Loading audio file..."}
  {"status": "transcribing", "progress": 15, "message": "Starting transcription..."}
  {"status": "completed", "progress": 100, "output": "/path/to/file.srt"}
  {"status": "error", "error": "error message"}
"""
import sys
import json
import os
import signal

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings


def format_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt_from_result(result) -> str:
    """Generate SRT content from Whisper result."""
    srt_lines = []
    segment_idx = 1

    # Whisper returns segments with timestamps
    if "segments" in result and result["segments"]:
        for segment in result["segments"]:
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"].strip()

            if text:
                srt_lines.append(str(segment_idx))
                srt_lines.append(f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}")
                srt_lines.append(text)
                srt_lines.append("")
                segment_idx += 1
    else:
        # Fallback: no timestamps, use full text
        text = result.get("text", "").strip()
        if text:
            srt_lines.append("1")
            srt_lines.append("00:00:00,000 --> 00:10:00,000")
            srt_lines.append(text)
            srt_lines.append("")

    return "\n".join(srt_lines)


def emit_status(status: str, progress: int, message: str = None, **kwargs):
    """Emit a JSON status line to stdout."""
    output = {
        "status": status,
        "progress": progress,
    }
    if message:
        output["message"] = message
    output.update(kwargs)
    print(json.dumps(output), flush=True)


def main():
    if len(sys.argv) < 5:
        emit_status("error", 0, message="Usage: transcribe_worker.py <audio_path> <output_srt> <model_name> [language] [device]")
        sys.exit(1)

    audio_path = sys.argv[1]
    output_srt_path = sys.argv[2]
    model_name = sys.argv[3]
    language = sys.argv[4] if len(sys.argv) > 4 else "auto"
    device = sys.argv[5] if len(sys.argv) > 5 else "auto"

    # Handle termination signals gracefully
    def signal_handler(signum, frame):
        emit_status("cancelled", 0, message="Transcription cancelled")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        import whisper
        import torch

        # Determine device from argument
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            emit_status("warning", 0, message="CUDA not available, falling back to CPU")
            device = "cpu"

        emit_status("loading_model", 0, message=f"Loading model '{model_name}' to {device}...")

        # Load model (cache to /app/models for volume persistence)
        model = whisper.load_model(model_name, device=device, download_root="/app/models")

        emit_status("model_loaded", 5, message=f"Model '{model_name}' loaded successfully")
        emit_status("loading_audio", 10, message=f"Loading audio: {os.path.basename(audio_path)}")

        # Get audio duration for progress estimation
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)
        duration = len(audio) / sr
        
        # Free audio buffer
        del audio
        import gc
        gc.collect()

        emit_status("transcribing", 15, 
                   message=f"Transcribing audio ({duration:.1f}s) with {device.upper()}...")

        # Configure transcription
        options = {
            "task": "transcribe",
        }
        
        if language != "auto" and language:
            options["language"] = language

        # Run transcription
        result = model.transcribe(audio_path, **options)

        emit_status("generating_srt", 90, message="Generating SRT subtitles...")

        # Generate SRT
        srt_content = generate_srt_from_result(result)

        with open(output_srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)

        emit_status("completed", 100, message=f"Transcription complete: {os.path.basename(output_srt_path)}", 
                   output=output_srt_path)

    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_info = traceback.format_exc()
        emit_status("error", 0, message=error_msg, traceback=traceback_info)
        sys.exit(1)


if __name__ == "__main__":
    main()
