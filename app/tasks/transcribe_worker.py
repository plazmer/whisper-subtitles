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
import re
import signal
import time

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


_real_stdout = None


def emit_status(status: str, progress: int, message: str = None, **kwargs):
    """Emit a JSON status line to stdout (or saved real stdout)."""
    payload = {"status": status, "progress": progress}
    if message:
        payload["message"] = message
    payload.update(kwargs)
    target = _real_stdout if _real_stdout is not None else sys.stdout
    target.write(json.dumps(payload) + "\n")
    target.flush()


def _fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"{s // 60:02d}:{s % 60:02d}"


def _parse_whisper_ts(ts: str) -> float:
    """Parse Whisper verbose timestamp like '01:23.456' to seconds."""
    parts = ts.split(":")
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    return 0.0


class WhisperOutputCapture:
    """Intercept Whisper verbose output, parse segments, emit JSON progress."""

    _TS_RE = re.compile(r"\[(\d+:\d+\.\d+)\s*-->\s*(\d+:\d+\.\d+)\]\s*(.*)")

    def __init__(self, duration: float, throttle_sec: float = 2.0):
        self.duration = max(duration, 1.0)
        self.throttle_sec = throttle_sec
        self._last_emit = 0.0
        self._buf = ""

    def write(self, text: str):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._handle(line)

    def flush(self):
        pass

    def _handle(self, line: str):
        m = self._TS_RE.match(line.strip())
        if not m:
            return
        end_sec = _parse_whisper_ts(m.group(2))
        text = m.group(3).strip()
        ratio = min(end_sec / self.duration, 1.0)
        progress = min(89, int(15 + ratio * 74))

        now = time.time()
        if now - self._last_emit < self.throttle_sec:
            return
        self._last_emit = now

        snippet = (text[:80] + "...") if len(text) > 80 else text
        msg = f"[{_fmt_time(end_sec)}/{_fmt_time(self.duration)}] {snippet}"
        emit_status("transcribing", progress, message=msg)


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

        from app.tasks.gpu_utils import resolve_device, format_device_message
        from app.tasks.errors import make_error

        device, diag = resolve_device(device)

        emit_status("device_info", 0, message=format_device_message("Inference", diag), diagnostics=diag)

        if diag.get("fallback"):
            emit_status("warning", 0, message=f"CUDA not available ({diag.get('fallback_reason')}), using CPU")

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
                   message=format_device_message(f"Transcribing audio ({duration:.1f}s)", diag))

        options = {
            "task": "transcribe",
            "verbose": True,
        }

        if language != "auto" and language:
            options["language"] = language

        global _real_stdout
        _real_stdout = sys.stdout
        sys.stdout = WhisperOutputCapture(duration)
        try:
            result = model.transcribe(audio_path, **options)
        finally:
            sys.stdout = _real_stdout
            _real_stdout = None

        emit_status("generating_srt", 90, message="Generating SRT subtitles...")

        # Generate SRT
        srt_content = generate_srt_from_result(result)

        with open(output_srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)

        segments_json_path = output_srt_path.rsplit('.', 1)[0] + '.segments.json'
        whisper_segments = []
        for segment in result.get("segments", []):
            whisper_segments.append(
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                }
            )

        with open(segments_json_path, 'w', encoding='utf-8') as f:
            json.dump(whisper_segments, f, ensure_ascii=False, indent=2)

        emit_status(
            "completed",
            100,
            message=f"Transcription complete: {os.path.basename(output_srt_path)}",
            output=output_srt_path,
            segments_path=segments_json_path,
        )

    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_info = traceback.format_exc()

        try:
            from app.tasks.errors import make_error
            if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                structured = make_error("cuda_out_of_memory", error_msg,
                                        details=traceback_info,
                                        hint="Попробуйте более лёгкую модель или переключитесь на CPU")
            else:
                structured = make_error("transcription_failed", error_msg,
                                        details=traceback_info)
            emit_status("error", 0, message=error_msg, traceback=traceback_info,
                       structured_error=structured)
        except Exception:
            emit_status("error", 0, message=error_msg, traceback=traceback_info)
        sys.exit(1)


if __name__ == "__main__":
    main()
