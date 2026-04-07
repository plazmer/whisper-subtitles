import asyncio
import os
from typing import Optional, Callable

import whisper

from app.config import settings
from app.auth import get_config_data
from app.tasks.gpu_utils import resolve_device, format_device_message


_pipeline_cache = {}


def get_whisper_model(model_name: str, device: str = None):
    """Get or cache a Whisper model."""
    if model_name not in _pipeline_cache:
        print(f"Loading Whisper model: {model_name}")

        requested = device if device else settings.device
        actual_device, diag = resolve_device(requested)
        print(format_device_message(f"Loading {model_name}", diag))

        model = whisper.load_model(model_name, device=actual_device, download_root="/app/models")
        print(f"Model loaded on device: {actual_device}")

        _pipeline_cache[model_name] = (model, actual_device)

    return _pipeline_cache[model_name]


async def transcribe_audio(
    audio_path: str,
    output_srt_path: str,
    model_name: str = "large-v3",
    language: str = "auto",
    device: str = None,
    progress_callback: Optional[Callable] = None
) -> str:
    """
    Transcribe audio file to SRT subtitle format using OpenAI Whisper.
    Returns path to generated SRT file.
    """

    loop = asyncio.get_event_loop()

    def do_transcribe():
        model, actual_device = get_whisper_model(model_name, device)

        # Transcribe with whisper
        options = {
            "task": "transcribe",
        }
        
        # Set language if specified
        if language != "auto" and language:
            options["language"] = language
        
        result = model.transcribe(audio_path, **options)

        return result

    result = await loop.run_in_executor(None, do_transcribe)

    # Parse result and generate SRT
    srt_content = generate_srt_from_result(result)

    with open(output_srt_path, 'w', encoding='utf-8') as f:
        f.write(srt_content)

    return output_srt_path


def generate_srt_from_result(result) -> str:
    """Generate SRT content from Whisper result."""
    srt_lines = []

    # Check if result has segments with timestamps
    if "segments" in result and result["segments"]:
        for i, segment in enumerate(result["segments"], 1):
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"].strip()

            if text:
                srt_lines.append(f"{i}")
                srt_lines.append(f"{start_time} --> {end_time}")
                srt_lines.append(text)
                srt_lines.append("")
    else:
        # Fallback: single text without timestamps
        text = result.get("text", "").strip()
        if text:
            srt_lines.append("1")
            srt_lines.append("00:00:00,000 --> 99:59:59,999")
            srt_lines.append(text)
            srt_lines.append("")

    return "\n".join(srt_lines)


def generate_srt(segments) -> str:
    """Generate SRT content from segments (compatibility function)."""
    srt_lines = []

    for i, segment in enumerate(segments, 1):
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        text = segment["text"].strip()

        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(text)
        srt_lines.append("")

    return "\n".join(srt_lines)


def format_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    if seconds is None:
        return "00:00:00,000"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


async def transcribe_with_progress(
    audio_path: str,
    output_srt_path: str,
    model_name: str = "large-v3",
    language: str = "auto",
    device: str = None,
    progress_callback: Optional[Callable] = None,
    job_id: str = None,
    is_cancelled: Optional[Callable] = None
) -> str:
    """
    Transcribe audio with progress tracking using OpenAI Whisper.
    Progress is estimated based on audio duration.
    """
    from app.tasks.extractor import get_video_duration

    # Get audio duration for progress estimation
    duration = await get_video_duration(audio_path)

    # Check if cancelled
    if is_cancelled and is_cancelled():
        raise Exception("Transcription cancelled")

    # Initial progress
    if progress_callback:
        await progress_callback(5)

    def do_transcribe():
        model, actual_device = get_whisper_model(model_name, device)

        # Transcribe with whisper
        options = {
            "task": "transcribe",
        }
        
        # Set language if specified
        if language != "auto" and language:
            options["language"] = language
        
        result = model.transcribe(audio_path, **options)

        return result

    loop = asyncio.get_event_loop()
    
    # Run transcription with progress updates
    result = await loop.run_in_executor(None, do_transcribe)

    # Check if cancelled after transcription
    if is_cancelled and is_cancelled():
        raise Exception("Transcription cancelled")

    # Generate SRT file
    srt_content = generate_srt_from_result(result)

    with open(output_srt_path, 'w', encoding='utf-8') as f:
        f.write(srt_content)

    if progress_callback:
        await progress_callback(100)

    return output_srt_path
