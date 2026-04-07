#!/usr/bin/env python3
"""
Standalone transcription worker that can be killed.
Used for proper job cancellation support.

Outputs JSON progress updates to stdout:
  {"status": "loading_model", "progress": 0}
  {"status": "transcribing", "progress": 50}
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


def main():
    if len(sys.argv) < 5:
        print(json.dumps({"status": "error", "error": "Usage: transcribe_worker.py <audio_path> <output_srt> <model_name> [language] [device]"}), flush=True)
        sys.exit(1)

    audio_path = sys.argv[1]
    output_srt_path = sys.argv[2]
    model_name = sys.argv[3]
    language = sys.argv[4] if len(sys.argv) > 4 else "auto"
    device = sys.argv[5] if len(sys.argv) > 5 else "auto"

    # Handle termination signals gracefully
    def signal_handler(signum, frame):
        print(json.dumps({"status": "cancelled", "progress": 0}), flush=True)
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
            print(json.dumps({"status": "warning", "message": "CUDA not available, falling back to CPU"}), flush=True)
            device = "cpu"

        print(json.dumps({"status": "loading_model", "progress": 0, "device": device, "model": model_name}), flush=True)

        # Load model (cache to /app/models for volume persistence)
        model = whisper.load_model(model_name, device=device, download_root="/app/models")

        print(json.dumps({"status": "loading_model", "progress": 10}), flush=True)
        print(json.dumps({"status": "loading_audio", "progress": 15}), flush=True)

        # Get audio duration for progress estimation
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)
        duration = len(audio) / sr
        
        # Free audio buffer
        del audio
        import gc
        gc.collect()

        print(json.dumps({"status": "transcribing", "progress": 20}), flush=True)

        # Configure transcription
        options = {
            "task": "transcribe",
        }
        
        if language != "auto" and language:
            options["language"] = language

        # Run transcription
        result = model.transcribe(audio_path, **options)

        # Generate SRT
        srt_content = generate_srt_from_result(result)

        with open(output_srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)

        print(json.dumps({"status": "completed", "progress": 100, "output": output_srt_path}), flush=True)

    except Exception as e:
        import traceback
        error_msg = str(e)
        print(json.dumps({"status": "error", "error": error_msg, "traceback": traceback.format_exc()}), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
