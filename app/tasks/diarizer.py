"""Speaker diarization helpers based on pyannote.audio."""
from typing import Any, Dict, List


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


def diarize(audio_path: str, model_name: str, hf_token: str, device: str = "cuda") -> List[Dict[str, Any]]:
    """Run pyannote diarization and return normalized segments."""
    from pyannote.audio import Pipeline
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
    if pipeline is None:
        raise RuntimeError(
            "Не удалось загрузить модель диаризации. "
            "Проверьте HF-токен и примите условия модели на HuggingFace."
        )
    pipeline.to(torch.device(device))
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
    return segments


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
        examples[speaker].append(text)

    return examples
