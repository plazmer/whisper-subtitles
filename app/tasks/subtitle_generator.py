"""Subtitle generation for diarized output formats."""
from typing import Any, Dict, List


def _format_srt_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_vtt_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _format_ass_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _hex_to_ass_color(hex_color: str) -> str:
    color = hex_color.lstrip("#")
    r = color[0:2]
    g = color[2:4]
    b = color[4:6]
    return f"&H00{b}{g}{r}"


def generate_ass(segments: List[Dict[str, Any]], speakers: Dict[str, Dict[str, str]]) -> str:
    lines: List[str] = [
        "[Script Info]",
        "Title: Diarized Subtitles",
        "ScriptType: v4.00+",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
    ]

    for _, info in speakers.items():
        style_name = info["name"].replace(",", " ")
        ass_color = _hex_to_ass_color(info["color"])
        lines.append(
            f"Style: {style_name},Arial,20,{ass_color},&H000000FF,&H00000000,&H80000000,"
            "0,0,0,0,100,100,0,0,1,2,1,2,10,10,10,1"
        )

    lines.extend(
        [
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
    )

    for segment in segments:
        speaker = segment["speaker"]
        info = speakers.get(speaker, {"name": "Unknown", "color": "#FFFFFF"})
        style_name = info["name"].replace(",", " ")
        start = _format_ass_timestamp(segment["start"])
        end = _format_ass_timestamp(segment["end"])
        text = f"[{info['name']}] {segment['text']}"
        lines.append(f"Dialogue: 0,{start},{end},{style_name},{info['name']},0,0,0,,{text}")

    return "\n".join(lines) + "\n"


def generate_srt_with_speakers(segments: List[Dict[str, Any]], speakers: Dict[str, Dict[str, str]]) -> str:
    lines: List[str] = []
    for idx, segment in enumerate(segments, 1):
        speaker = segment["speaker"]
        info = speakers.get(speaker, {"name": "Unknown"})
        start = _format_srt_timestamp(segment["start"])
        end = _format_srt_timestamp(segment["end"])
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(f"[{info['name']}] {segment['text']}")
        lines.append("")
    return "\n".join(lines)


def generate_vtt_with_speakers(segments: List[Dict[str, Any]], speakers: Dict[str, Dict[str, str]]) -> str:
    lines: List[str] = ["WEBVTT", "", "STYLE"]
    speaker_ids = list(speakers.keys())

    for idx, speaker in enumerate(speaker_ids):
        lines.append(f"::cue(.speaker{idx}) {{ color: {speakers[speaker]['color']}; }}")
    lines.append("")

    speaker_index = {speaker: idx for idx, speaker in enumerate(speaker_ids)}
    for segment in segments:
        speaker = segment["speaker"]
        info = speakers.get(speaker, {"name": "Unknown"})
        idx = speaker_index.get(speaker, 0)
        start = _format_vtt_timestamp(segment["start"])
        end = _format_vtt_timestamp(segment["end"])
        lines.append(f"{start} --> {end}")
        lines.append(f"<c.speaker{idx}>[{info['name']}] {segment['text']}</c>")
        lines.append("")

    return "\n".join(lines)
