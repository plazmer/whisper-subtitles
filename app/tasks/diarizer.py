"""Speaker diarization helpers based on pyannote.audio."""
from typing import Any, Dict, List, Tuple

from app.tasks.errors import make_error
from app.tasks.gpu_utils import resolve_device, format_device_message


DEFAULT_COLORS = [
    "#FF6B6B",
    "#4ECDC4",
    "#FFB347",
    "#A78BFA",
    "#60A5FA",
    "#F472B6",
    "#34D399",
    "#FBBF24",
    "#FB923C",
    "#818CF8",
]


def _format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS for speaker preview snippets."""
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _parse_hf_error(exc: Exception, model_name: str) -> str:
    """Turn a HuggingFace / pyannote exception into a structured error JSON string."""
    error_str = str(exc)
    model_url = f"https://huggingface.co/{model_name}"
    lower = error_str.lower()

    if "401" in lower or "unauthorized" in lower:
        return make_error(
            "hf_gated_access_denied",
            f"Токен HuggingFace невалиден или отсутствует для модели {model_name}",
            details=error_str,
            hint="Создайте токен на huggingface.co/settings/tokens и укажите его в настройках",
            url=model_url,
        )

    if "403" in lower or "access" in lower or "gated" in lower:
        return make_error(
            "hf_gated_access_denied",
            f"Нет доступа к модели {model_name}. Требуется принять условия использования.",
            details=error_str,
            hint="Откройте страницу модели, примите условия, и убедитесь что токен указан в настройках",
            url=model_url,
        )

    if "404" in lower or "not found" in lower:
        return make_error(
            "hf_model_not_found",
            f"Модель {model_name} не найдена на HuggingFace",
            details=error_str,
            hint="Проверьте название модели в настройках",
            url=model_url,
        )

    return make_error(
        "diarization_failed",
        f"Ошибка диаризации: {model_name}",
        details=error_str,
    )


def diarize(
    audio_path: str,
    model_name: str,
    hf_token: str,
    device: str = "cuda",
) -> Tuple[List[Dict[str, Any]], dict]:
    """Run pyannote diarization and return ``(segments, diagnostics)``.

    Raises ``RuntimeError`` with a structured-error JSON payload on failure.
    """
    from pyannote.audio import Pipeline
    import torch

    if not hf_token:
        raise RuntimeError(
            make_error(
                "hf_token_missing",
                "Для диаризации требуется токен HuggingFace",
                hint="Создайте токен на huggingface.co/settings/tokens и укажите его в настройках приложения",
                url="https://huggingface.co/settings/tokens",
            )
        )

    actual_device, diag = resolve_device(device)

    try:
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
    except Exception as exc:
        raise RuntimeError(_parse_hf_error(exc, model_name)) from exc

    if pipeline is None:
        raise RuntimeError(
            make_error(
                "hf_gated_access_denied",
                f"Не удалось загрузить модель {model_name}",
                hint="Проверьте HF-токен и примите условия модели на HuggingFace",
                url=f"https://huggingface.co/{model_name}",
            )
        )

    try:
        pipeline.to(torch.device(actual_device))
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            make_error(
                "cuda_out_of_memory",
                f"Недостаточно видеопамяти для загрузки модели {model_name}",
                details=str(exc),
                hint="Попробуйте использовать более лёгкую модель или переключиться на CPU",
            )
        ) from exc

    diarization = pipeline(audio_path)

    segments: List[Dict[str, Any]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "speaker": speaker,
            }
        )
    return segments, diag


def merge_transcription_with_diarization(
    whisper_segments: List[Dict[str, Any]],
    diarization_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Assign speakers to whisper segments by max overlap."""
    merged: List[Dict[str, Any]] = []

    for segment in whisper_segments:
        ws_start = segment["start"]
        ws_end = segment["end"]
        text = segment["text"].strip()
        if not text:
            continue

        best_speaker = "SPEAKER_00"
        best_overlap = 0.0
        for diar_segment in diarization_segments:
            overlap_start = max(ws_start, diar_segment["start"])
            overlap_end = min(ws_end, diar_segment["end"])
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar_segment["speaker"]

        merged.append(
            {
                "start": ws_start,
                "end": ws_end,
                "text": text,
                "speaker": best_speaker,
            }
        )

    return merged


def assign_default_speakers(merged_segments: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """Create default speaker mapping with names and colors."""
    unique_speakers: List[str] = []
    for segment in merged_segments:
        speaker = segment["speaker"]
        if speaker not in unique_speakers:
            unique_speakers.append(speaker)

    speakers: Dict[str, Dict[str, str]] = {}
    for index, speaker in enumerate(unique_speakers):
        speakers[speaker] = {
            "name": f"Speaker {index + 1}",
            "color": DEFAULT_COLORS[index % len(DEFAULT_COLORS)],
        }
    return speakers


def get_speaker_examples(
    merged_segments: List[Dict[str, Any]],
    speakers: Dict[str, Dict[str, str]],
    max_examples: int = 3,
    max_chars: int = 80,
) -> Dict[str, List[str]]:
    """Return up to max_examples utterances per speaker."""
    examples: Dict[str, List[str]] = {speaker_id: [] for speaker_id in speakers}

    for segment in merged_segments:
        speaker = segment["speaker"]
        if speaker not in examples or len(examples[speaker]) >= max_examples:
            continue

        text = segment["text"]
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        start = _format_timestamp(segment.get("start", 0.0))
        end = _format_timestamp(segment.get("end", 0.0))
        examples[speaker].append(f"[{start} - {end}] {text}")

    return examples
