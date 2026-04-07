# Speaker Diarization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add speaker diarization via pyannote.audio so subtitles identify who is speaking, with a UI for renaming speakers and colored subtitle output in ASS/WebVTT/SRT formats.

**Architecture:** pyannote.audio runs sequentially after Whisper on the same WAV file. Results are merged by timestamp overlap. A new `AWAITING_SPEAKERS` status pauses the pipeline for user review. After confirmation, ASS/SRT/VTT files are generated and embedded in MKV.

**Tech Stack:** pyannote.audio, openai-whisper, FastAPI, SQLite (aiosqlite), vanilla JS frontend, ffmpeg, Docker + CUDA

---

## File Structure

### New files
- `app/tasks/diarizer.py` — pyannote pipeline, merge logic, speaker assignment
- `app/tasks/subtitle_generator.py` — ASS/SRT/VTT generation from merged segments + speakers

### Modified files
- `app/models.py` — new statuses, new fields on Job, new request models
- `app/database.py` — migration for new columns, update CRUD
- `app/config.py` — diarization model list, HF_HOME setting
- `app/auth.py` — expose/persist diarization_model and hf_token in settings
- `app/tasks/transcribe_worker.py` — return Whisper segments as JSON (not just SRT)
- `app/tasks/extractor.py` — embed ASS+SRT in MKV, generate colored VTT
- `app/main.py` — new API endpoints, updated process_job pipeline
- `app/static/index.html` — speaker editing modal
- `app/static/app.js` — speaker editing logic, colored subtitles
- `app/static/styles.css` — speaker card styles, color picker
- `app/static/locales/en.yml` — new i18n keys (English)
- `app/static/locales/ru.yml` — new i18n keys (Russian)
- `app/static/locales/*.yml` — new i18n keys (remaining 13 locales)
- `Dockerfile.base` — add pyannote.audio, set HF_HOME
- `docker-compose.yml` — add hf_cache volume
- `requirements.txt` — add pyannote.audio

---

### Task 1: Модели и статусы

**Files:**
- Modify: `app/models.py:7-17` (JobStatus enum)
- Modify: `app/models.py:53-89` (Job model)
- Modify: `app/models.py:110-114` (SettingsUpdateRequest)

- [ ] **Step 1: Добавить новые статусы в JobStatus**

В `app/models.py`, в enum `JobStatus` добавить два новых значения после `TRANSCRIBING`:

```python
class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    EXTRACTING = "extracting"
    AWAITING_TRACK = "awaiting_track"
    TRANSCRIBING = "transcribing"
    AWAITING_SPEAKERS = "awaiting_speakers"
    GENERATING = "generating"
    EMBEDDING = "embedding"
    CONVERTING = "converting"
    COMPLETED = "completed"
    FAILED = "failed"
```

- [ ] **Step 2: Добавить новые поля в Job**

В модель `Job` добавить поля для спикеров и сегментов после поля `model`:

```python
class Job(BaseModel):
    """Main job model."""
    id: str
    type: JobType
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    created_at: datetime
    updated_at: datetime
    
    source: str
    files: List[JobFile] = []
    
    embed_subtitles: bool = False
    language: str = "auto"
    model: str = "large-v3-int8"
    
    # Diarization data
    speakers: Optional[str] = None        # JSON: {"SPEAKER_00": {"name": "Speaker 1", "color": "#FF6B6B"}, ...}
    diarization_segments: Optional[str] = None  # JSON: [{"start": 0.0, "end": 3.5, "speaker": "SPEAKER_00"}, ...]
    merged_segments: Optional[str] = None  # JSON: [{"start": 0.0, "end": 3.5, "text": "...", "speaker": "SPEAKER_00"}, ...]
    
    error: Optional[str] = None
    status_message: Optional[str] = None
    
    is_group: bool = False
    group_name: Optional[str] = None
    selected_indices: Optional[List[int]] = None
    download_speed: Optional[str] = None
    eta: Optional[str] = None
    is_paused: bool = False
```

- [ ] **Step 3: Добавить новые Request-модели**

Добавить в конец `app/models.py`:

```python
class SpeakerUpdateRequest(BaseModel):
    """Request to update speaker names and colors."""
    speakers: dict  # {"SPEAKER_00": {"name": "Иван", "color": "#FF6B6B"}, ...}


```

- [ ] **Step 4: Обновить SettingsUpdateRequest**

```python
class SettingsUpdateRequest(BaseModel):
    """Settings update request model."""
    model: Optional[str] = None
    device: Optional[str] = None
    language: Optional[str] = None
    diarization_model: Optional[str] = None
    hf_token: Optional[str] = None
```

- [ ] **Step 5: Коммит**

```bash
git add app/models.py
git commit -m "Добавлены модели и статусы для диаризации спикеров"
```

---

### Task 2: База данных — миграция

**Files:**
- Modify: `app/database.py:9-55` (init_db migrations)
- Modify: `app/database.py:58-79` (create_job)
- Modify: `app/database.py:102-124` (update_job)
- Modify: `app/database.py:148-196` (_row_to_job)

- [ ] **Step 1: Добавить миграции для новых колонок**

В `app/database.py`, в функции `init_db()`, после существующих миграций (строка 54) добавить:

```python
        try:
            await db.execute("ALTER TABLE jobs ADD COLUMN speakers TEXT")
        except:
            pass
        try:
            await db.execute("ALTER TABLE jobs ADD COLUMN diarization_segments TEXT")
        except:
            pass
        try:
            await db.execute("ALTER TABLE jobs ADD COLUMN merged_segments TEXT")
        except:
            pass
```

- [ ] **Step 2: Обновить create_job**

В функции `create_job`, добавить новые поля в INSERT-запрос. Полный обновлённый запрос:

```python
async def create_job(job: Job) -> Job:
    """Create a new job in the database."""
    async with aiosqlite.connect(settings.db_path) as db:
        files_json = json.dumps([f.model_dump() for f in job.files], default=str)
        indices_json = json.dumps(job.selected_indices) if job.selected_indices else None
        status_message = job.status_message or job.files[0].status_message if job.files else None
        await db.execute("""
            INSERT INTO jobs (id, type, status, progress, created_at, updated_at,
                            source, files, embed_subtitles, language, model, error,
                            is_group, group_name, selected_indices, download_speed, eta, is_paused,
                            status_message, speakers, diarization_segments, merged_segments)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job.id, job.type.value, job.status.value, job.progress,
            job.created_at.isoformat(), job.updated_at.isoformat(),
            job.source, files_json, int(job.embed_subtitles),
            job.language, job.model, job.error, int(job.is_group), job.group_name,
            indices_json, job.download_speed, job.eta, int(job.is_paused),
            status_message, job.speakers, job.diarization_segments, job.merged_segments
        ))
        await db.commit()
    return job
```

- [ ] **Step 3: Обновить update_job**

В функции `update_job`, добавить новые поля в UPDATE-запрос:

```python
async def update_job(job: Job) -> Job:
    """Update a job in the database."""
    async with aiosqlite.connect(settings.db_path) as db:
        files_json = json.dumps([f.model_dump() for f in job.files], default=str)
        indices_json = json.dumps(job.selected_indices) if job.selected_indices else None
        job.updated_at = datetime.utcnow()
        status_message = job.status_message or (job.files[0].status_message if job.files else None)
        await db.execute("""
            UPDATE jobs SET
                status = ?, progress = ?, updated_at = ?, files = ?,
                embed_subtitles = ?, language = ?, model = ?, error = ?,
                is_group = ?, group_name = ?, selected_indices = ?,
                download_speed = ?, eta = ?, is_paused = ?, status_message = ?,
                speakers = ?, diarization_segments = ?, merged_segments = ?
            WHERE id = ?
        """, (
            job.status.value, job.progress, job.updated_at.isoformat(),
            files_json, int(job.embed_subtitles), job.language, job.model,
            job.error, int(job.is_group), job.group_name, indices_json,
            job.download_speed, job.eta, int(job.is_paused), status_message,
            job.speakers, job.diarization_segments, job.merged_segments,
            job.id
        ))
        await db.commit()
    return job
```

- [ ] **Step 4: Обновить _row_to_job**

В функции `_row_to_job`, добавить чтение новых полей в конструктор Job:

```python
    return Job(
        id=row['id'],
        type=JobType(row['type']),
        status=JobStatus(row['status']),
        progress=row['progress'],
        created_at=datetime.fromisoformat(row['created_at']),
        updated_at=datetime.fromisoformat(row['updated_at']),
        source=row['source'],
        files=files,
        embed_subtitles=bool(row.get('embed_subtitles', 1)),
        language=row['language'],
        model=row['model'],
        error=row['error'],
        status_message=row.get('status_message'),
        is_group=bool(row.get('is_group', 0)),
        group_name=row.get('group_name'),
        selected_indices=selected_indices,
        download_speed=row.get('download_speed'),
        eta=row.get('eta'),
        is_paused=bool(row.get('is_paused', 0)),
        speakers=row.get('speakers'),
        diarization_segments=row.get('diarization_segments'),
        merged_segments=row.get('merged_segments'),
    )
```

- [ ] **Step 5: Коммит**

```bash
git add app/database.py
git commit -m "Миграция БД: добавлены колонки для диаризации"
```

---

### Task 3: Конфигурация и настройки

**Files:**
- Modify: `app/config.py:1-109`
- Modify: `app/auth.py:128-151`

- [ ] **Step 1: Добавить модели диаризации и HF_HOME в config.py**

В `app/config.py`, после `default_language` (строка 16) добавить:

```python
    # Diarization settings
    default_diarization_model: str = "pyannote/speaker-diarization-3.1"
    hf_home: str = "/root/.cache/huggingface"
```

После `available_models` property (строка 75) добавить новое свойство:

```python
    @property
    def available_diarization_models(self) -> dict:
        return {
            "pyannote/speaker-diarization-3.1": {
                "name": "Speaker Diarization 3.1",
                "description": "Latest, highest accuracy",
            },
            "pyannote/speaker-diarization-3.0": {
                "name": "Speaker Diarization 3.0",
                "description": "Previous version, slightly lighter",
            },
            "pyannote/speaker-diarization": {
                "name": "Speaker Diarization 2.x",
                "description": "Base version, minimal resources",
            },
        }
```

- [ ] **Step 2: Обновить get_app_settings в auth.py**

В `app/auth.py`, функция `get_app_settings()`:

```python
def get_app_settings() -> dict:
    """Get current application settings."""
    config = get_config_data()
    return {
        "model": config.get("model", settings.default_model),
        "device": config.get("device", settings.device),
        "language": config.get("language", settings.default_language),
        "available_models": settings.available_models,
        "diarization_model": config.get("diarization_model", settings.default_diarization_model),
        "hf_token": config.get("hf_token", ""),
        "available_diarization_models": settings.available_diarization_models,
    }
```

- [ ] **Step 3: Обновить update_app_settings в auth.py**

```python
def update_app_settings(
    model: Optional[str] = None,
    device: Optional[str] = None,
    language: Optional[str] = None,
    diarization_model: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> dict:
    """Update application settings."""
    config = get_config_data()

    if model and model in settings.available_models:
        config["model"] = model
    if device is not None and device in ["cuda", "cpu", "auto"]:
        config["device"] = device
    if language:
        config["language"] = language
    if diarization_model and diarization_model in settings.available_diarization_models:
        config["diarization_model"] = diarization_model
    if hf_token is not None:
        config["hf_token"] = hf_token

    save_config_data(config)
    return get_app_settings()
```

- [ ] **Step 4: Обновить get_config_data defaults**

В `get_config_data()`, обновить дефолтный словарь:

```python
    return {
        "password_hash": pwd_context.hash(DEFAULT_PASSWORD),
        "model": settings.default_model,
        "device": settings.device,
        "language": settings.default_language,
        "diarization_model": settings.default_diarization_model,
        "hf_token": "",
    }
```

- [ ] **Step 5: Коммит**

```bash
git add app/config.py app/auth.py
git commit -m "Конфигурация: добавлены настройки диаризации и HuggingFace-токен"
```

---

### Task 4: Модуль диаризации

**Files:**
- Create: `app/tasks/diarizer.py`

- [ ] **Step 1: Создать app/tasks/diarizer.py**

```python
"""Speaker diarization using pyannote.audio."""
import json
from typing import List, Dict, Any, Optional

DEFAULT_COLORS = [
    "#FF6B6B", "#4ECDC4", "#FFB347", "#A78BFA", "#60A5FA",
    "#F472B6", "#34D399", "#FBBF24", "#FB923C", "#818CF8",
]


def diarize(audio_path: str, model_name: str, hf_token: str, device: str = "cuda") -> List[Dict[str, Any]]:
    """Run pyannote speaker diarization pipeline.
    
    Returns list of segments: [{"start": 0.0, "end": 3.5, "speaker": "SPEAKER_00"}, ...]
    """
    from pyannote.audio import Pipeline
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
    pipeline.to(torch.device(device))

    diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker,
        })

    return segments


def merge_transcription_with_diarization(
    whisper_segments: List[Dict[str, Any]],
    diarization_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Assign a speaker to each Whisper segment based on maximum time overlap.
    
    Returns merged list: [{"start": ..., "end": ..., "text": ..., "speaker": "SPEAKER_00"}, ...]
    """
    merged = []
    for ws in whisper_segments:
        ws_start = ws["start"]
        ws_end = ws["end"]
        text = ws["text"].strip()
        if not text:
            continue

        best_speaker = "SPEAKER_00"
        best_overlap = 0.0

        for ds in diarization_segments:
            overlap_start = max(ws_start, ds["start"])
            overlap_end = min(ws_end, ds["end"])
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = ds["speaker"]

        merged.append({
            "start": ws_start,
            "end": ws_end,
            "text": text,
            "speaker": best_speaker,
        })

    return merged


def assign_default_speakers(merged_segments: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """Build speaker dict with default names and colors from palette.
    
    Returns: {"SPEAKER_00": {"name": "Speaker 1", "color": "#FF6B6B"}, ...}
    """
    unique_speakers = []
    for seg in merged_segments:
        if seg["speaker"] not in unique_speakers:
            unique_speakers.append(seg["speaker"])

    speakers = {}
    for i, spk_id in enumerate(unique_speakers):
        speakers[spk_id] = {
            "name": f"Speaker {i + 1}",
            "color": DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
        }

    return speakers


def get_speaker_examples(
    merged_segments: List[Dict[str, Any]],
    speakers: Dict[str, Dict[str, str]],
    max_examples: int = 3,
    max_chars: int = 80,
) -> Dict[str, list]:
    """Get example utterances for each speaker for the UI.
    
    Returns: {"SPEAKER_00": ["Добрый день, коллеги...", "Переходим к вопросу..."], ...}
    """
    examples: Dict[str, list] = {spk_id: [] for spk_id in speakers}

    for seg in merged_segments:
        spk_id = seg["speaker"]
        if spk_id in examples and len(examples[spk_id]) < max_examples:
            text = seg["text"]
            if len(text) > max_chars:
                text = text[:max_chars].rstrip() + "..."
            examples[spk_id].append(text)

    return examples
```

- [ ] **Step 2: Коммит**

```bash
git add app/tasks/diarizer.py
git commit -m "Добавлен модуль диаризации (pyannote.audio)"
```

---

### Task 5: Генератор субтитров

**Files:**
- Create: `app/tasks/subtitle_generator.py`

- [ ] **Step 1: Создать app/tasks/subtitle_generator.py**

```python
"""Generate ASS, SRT, and WebVTT subtitles with speaker information."""
from typing import List, Dict, Any


def _format_srt_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_vtt_timestamp(seconds: float) -> str:
    """Format seconds to WebVTT timestamp: HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _format_ass_timestamp(seconds: float) -> str:
    """Format seconds to ASS timestamp: H:MM:SS.cc (centiseconds)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _hex_to_ass_color(hex_color: str) -> str:
    """Convert #RRGGBB to ASS color format &H00BBGGRR (BGR, reversed)."""
    hex_color = hex_color.lstrip("#")
    r = hex_color[0:2]
    g = hex_color[2:4]
    b = hex_color[4:6]
    return f"&H00{b}{g}{r}"


def generate_ass(
    segments: List[Dict[str, Any]],
    speakers: Dict[str, Dict[str, str]],
) -> str:
    """Generate ASS subtitle content with colored styles per speaker."""
    lines = []

    lines.append("[Script Info]")
    lines.append("Title: Diarized Subtitles")
    lines.append("ScriptType: v4.00+")
    lines.append("PlayResX: 1920")
    lines.append("PlayResY: 1080")
    lines.append("")

    lines.append("[V4+ Styles]")
    lines.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding")

    for spk_id, info in speakers.items():
        ass_color = _hex_to_ass_color(info["color"])
        style_name = info["name"].replace(",", " ")
        lines.append(
            f"Style: {style_name},Arial,20,{ass_color},&H000000FF,&H00000000,&H80000000,"
            f"0,0,0,0,100,100,0,0,1,2,1,2,10,10,10,1"
        )

    lines.append("")
    lines.append("[Events]")
    lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")

    for seg in segments:
        spk_id = seg["speaker"]
        info = speakers.get(spk_id, {"name": "Unknown", "color": "#FFFFFF"})
        style_name = info["name"].replace(",", " ")
        start = _format_ass_timestamp(seg["start"])
        end = _format_ass_timestamp(seg["end"])
        text = f"[{info['name']}] {seg['text']}"
        lines.append(f"Dialogue: 0,{start},{end},{style_name},{info['name']},0,0,0,,{text}")

    return "\n".join(lines) + "\n"


def generate_srt_with_speakers(
    segments: List[Dict[str, Any]],
    speakers: Dict[str, Dict[str, str]],
) -> str:
    """Generate SRT subtitle content with speaker name prefix."""
    lines = []

    for i, seg in enumerate(segments, 1):
        spk_id = seg["speaker"]
        info = speakers.get(spk_id, {"name": "Unknown"})
        start = _format_srt_timestamp(seg["start"])
        end = _format_srt_timestamp(seg["end"])
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(f"[{info['name']}] {seg['text']}")
        lines.append("")

    return "\n".join(lines)


def generate_vtt_with_speakers(
    segments: List[Dict[str, Any]],
    speakers: Dict[str, Dict[str, str]],
) -> str:
    """Generate WebVTT subtitle content with CSS classes for speaker colors."""
    lines = ["WEBVTT", ""]

    lines.append("STYLE")
    speaker_list = list(speakers.keys())
    for i, (spk_id, info) in enumerate(speakers.items()):
        lines.append(f"::cue(.speaker{i}) {{ color: {info['color']}; }}")
    lines.append("")

    spk_index = {spk_id: i for i, spk_id in enumerate(speaker_list)}

    for seg in segments:
        spk_id = seg["speaker"]
        info = speakers.get(spk_id, {"name": "Unknown"})
        idx = spk_index.get(spk_id, 0)
        start = _format_vtt_timestamp(seg["start"])
        end = _format_vtt_timestamp(seg["end"])
        lines.append(f"{start} --> {end}")
        lines.append(f"<c.speaker{idx}>[{info['name']}] {seg['text']}</c>")
        lines.append("")

    return "\n".join(lines)
```

- [ ] **Step 2: Коммит**

```bash
git add app/tasks/subtitle_generator.py
git commit -m "Добавлен генератор субтитров ASS/SRT/VTT со спикерами"
```

---

### Task 6: Обновление transcribe_worker — возврат сегментов

**Files:**
- Modify: `app/tasks/transcribe_worker.py`

- [ ] **Step 1: Обновить воркер для возврата сегментов JSON**

Сейчас воркер генерирует SRT и записывает в файл. Нужно дополнительно сохранять сырые сегменты Whisper в JSON-файл рядом с SRT, чтобы потом объединить их с диаризацией.

В `app/tasks/transcribe_worker.py`, заменить блок после `result = model.transcribe(...)` (строки 140–149):

```python
        emit_status("generating_srt", 90, message="Generating SRT subtitles...")

        # Generate SRT
        srt_content = generate_srt_from_result(result)

        with open(output_srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)

        # Save raw Whisper segments as JSON for diarization merge
        segments_json_path = output_srt_path.rsplit('.', 1)[0] + '.segments.json'
        whisper_segments = []
        for seg in result.get("segments", []):
            whisper_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            })

        import json as json_mod
        with open(segments_json_path, 'w', encoding='utf-8') as f:
            json_mod.dump(whisper_segments, f, ensure_ascii=False, indent=2)

        emit_status("completed", 100, 
                   message=f"Transcription complete: {os.path.basename(output_srt_path)}", 
                   output=output_srt_path,
                   segments_path=segments_json_path)
```

- [ ] **Step 2: Коммит**

```bash
git add app/tasks/transcribe_worker.py
git commit -m "Воркер транскрипции: сохранение сегментов в JSON для диаризации"
```

---

### Task 7: Обновление extractor.py — ASS в MKV

**Files:**
- Modify: `app/tasks/extractor.py:146-209` (embed_subtitles)
- Modify: `app/tasks/extractor.py:327-380` (convert_srt_to_vtt)

- [ ] **Step 1: Создать функцию embed_subtitles_diarized**

Добавить новую функцию в `app/tasks/extractor.py` после `embed_subtitles` (после строки 209):

```python
async def embed_subtitles_diarized(
    video_path: str,
    ass_path: str,
    srt_path: str,
    output_path: str,
    progress_callback=None
) -> str:
    """Embed ASS (colored) + SRT (compatibility) subtitles into MKV.
    
    Creates two subtitle tracks: ASS as default, SRT as fallback.
    """
    cmd = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-i', ass_path,
        '-i', srt_path,
        '-map', '0:v',
        '-map', '0:a',
        '-map', '1:0',
        '-map', '2:0',
        '-c:v', 'copy',
        '-c:a', 'copy',
        '-c:s:0', 'ass',
        '-c:s:1', 'srt',
        '-metadata:s:s:0', 'language=rus',
        '-metadata:s:s:0', 'title=AI Generated (colored)',
        '-metadata:s:s:1', 'language=rus',
        '-metadata:s:s:1', 'title=AI Generated (plain)',
        '-disposition:s:0', 'default',
        '-disposition:s:1', '0',
        '-progress', 'pipe:1',
        output_path
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    duration = await get_video_duration(video_path)

    stderr_lines = []
    stderr_task = asyncio.create_task(_drain_stderr(process.stderr, stderr_lines))

    while True:
        line = await process.stdout.readline()
        if not line:
            break
        line_str = line.decode('utf-8', errors='ignore').strip()
        if line_str.startswith('out_time_ms='):
            try:
                current_ms = int(line_str.split('=')[1])
                current_sec = current_ms / 1000000
                if duration > 0 and progress_callback:
                    progress = min(100, (current_sec / duration) * 100)
                    await progress_callback(progress)
            except:
                pass

    await stderr_task
    await process.wait()

    if process.returncode != 0:
        raise Exception(f"ffmpeg failed: {''.join(stderr_lines)}")

    return output_path
```

- [ ] **Step 2: Коммит**

```bash
git add app/tasks/extractor.py
git commit -m "extractor: функция встраивания ASS+SRT субтитров в MKV"
```

---

### Task 8: Обновление main.py — пайплайн и API

**Files:**
- Modify: `app/main.py`

Это самый большой таск. Изменения в `main.py`:

- [ ] **Step 1: Обновить импорты**

В начале `app/main.py` (строки 16–30), добавить новые импорты:

```python
from app.models import (
    Job, JobFile, JobStatus, JobType, AudioTrack,
    LoginRequest, LoginResponse, PasswordChangeRequest,
    SettingsUpdateRequest, JobCreateRequest, TrackSelectionRequest,
    TorrentFileInfo, FileSelectionRequest,
    SpeakerUpdateRequest
)
from app.tasks.downloader import download_url, download_torrent, get_torrent_files
from app.tasks.extractor import get_audio_tracks, extract_audio, embed_subtitles, embed_subtitles_diarized, create_streaming_version
from app.tasks.transcriber import transcribe_with_progress
```

- [ ] **Step 2: Обновить resume_pending_jobs — добавить AWAITING_SPEAKERS и GENERATING**

В функции `resume_pending_jobs` (строка 733), обновить список статусов для возобновления:

```python
        if job.status in [JobStatus.PENDING, JobStatus.DOWNLOADING, JobStatus.EXTRACTING, 
                          JobStatus.TRANSCRIBING, JobStatus.CONVERTING, JobStatus.GENERATING]:
```

- [ ] **Step 3: Обновить process_job — диаризация после транскрипции**

В `process_job`, после блока транскрипции (после строки 958, `await run_transcription_subprocess(...)`) и записи `file.srt_path = srt_path`, заменить блок от строки 960 до строки 1017.

Новый код вместо прежнего блока встраивания (строки 960–1017):

```python
                file.srt_path = srt_path
                
                # Run diarization if HF token is set
                segments_json_path = srt_path.rsplit('.', 1)[0] + '.segments.json'
                
                hf_token = current_settings.get("hf_token", "")
                if hf_token and os.path.exists(segments_json_path):
                    file.status_message = f"Diarizing speakers: {file.filename}"
                    job.status_message = f"Diarizing speakers: {file.filename}"
                    await update_job(job)
                    
                    diar_model = current_settings.get("diarization_model", "pyannote/speaker-diarization-3.1")
                    diar_device = current_settings.get("device", "auto")
                    
                    from app.tasks.diarizer import diarize, merge_transcription_with_diarization, assign_default_speakers
                    
                    # Run diarization in executor to not block event loop
                    loop = asyncio.get_event_loop()
                    diar_segments = await loop.run_in_executor(
                        None, diarize, audio_path, diar_model, hf_token, diar_device
                    )
                    
                    # Load Whisper segments
                    with open(segments_json_path, 'r', encoding='utf-8') as f:
                        whisper_segments = json.load(f)
                    
                    # Merge
                    merged = merge_transcription_with_diarization(whisper_segments, diar_segments)
                    speakers_dict = assign_default_speakers(merged)
                    
                    # Store in job
                    job.diarization_segments = json.dumps(diar_segments, ensure_ascii=False)
                    job.merged_segments = json.dumps(merged, ensure_ascii=False)
                    job.speakers = json.dumps(speakers_dict, ensure_ascii=False)
                    
                    file.status = JobStatus.AWAITING_SPEAKERS
                    job.status = JobStatus.AWAITING_SPEAKERS
                    await update_job(job)
                    return  # Pause pipeline, wait for user confirmation
                else:
                    # No diarization — generate subtitles directly and continue with embedding
                    await _generate_and_embed(job, file, video_path, current_settings)
```

- [ ] **Step 4: Вынести логику генерации/встраивания в отдельную функцию**

Добавить перед `process_job` новую async-функцию `_generate_and_embed`:

```python
async def _generate_and_embed(job: Job, file: JobFile, video_path: str, current_settings: dict):
    """Generate subtitle files and embed into MKV."""
    output_dir = os.path.join(settings.output_dir, job.id)
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(file.filename)[0]
    srt_path = file.srt_path

    if job.merged_segments and job.speakers:
        merged = json.loads(job.merged_segments)
        speakers_dict = json.loads(job.speakers)

        from app.tasks.subtitle_generator import generate_ass, generate_srt_with_speakers, generate_vtt_with_speakers

        # Generate ASS
        ass_path = os.path.join(output_dir, f"{base_name}.ass")
        with open(ass_path, 'w', encoding='utf-8') as f:
            f.write(generate_ass(merged, speakers_dict))

        # Regenerate SRT with speaker names
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(generate_srt_with_speakers(merged, speakers_dict))

        # Embed ASS + SRT into MKV
        if job.embed_subtitles:
            file.status_message = f"Embedding subtitles: {file.filename}"
            job.status = JobStatus.EMBEDDING
            await update_job(job)

            output_video = os.path.join(output_dir, f"{base_name}_subtitled.mkv")
            from app.tasks.extractor import embed_subtitles_diarized
            await embed_subtitles_diarized(video_path, ass_path, srt_path, output_video)
            file.output_path = output_video

            file.status = JobStatus.EMBEDDING
            file.status_message = None
            await update_job(job)

            # Create streaming version (this also generates a plain VTT via convert_srt_to_vtt)
            job.status = JobStatus.CONVERTING
            file.status = JobStatus.CONVERTING
            file.status_message = f"Creating streaming version: {file.filename}"
            file.progress = 0
            await update_job(job)

            streaming_video = os.path.join(output_dir, f"{base_name}_streaming.mp4")
            
            async def update_conversion_progress(progress):
                file.progress = progress
                await update_job(job)

            try:
                await create_streaming_version(
                    output_video, srt_path, streaming_video,
                    max_height=1080,
                    progress_callback=update_conversion_progress,
                    is_cancelled=lambda: job.id in cancelled_jobs
                )
                file.streaming_path = streaming_video
            except Exception as e:
                print(f"[STREAMING] Failed: {e}")

        # Generate colored VTT AFTER create_streaming_version
        # (create_streaming_version calls convert_srt_to_vtt which writes a plain VTT;
        #  we overwrite it with the colored version)
        vtt_path = srt_path.rsplit('.', 1)[0] + '.vtt'
        with open(vtt_path, 'w', encoding='utf-8') as f:
            f.write(generate_vtt_with_speakers(merged, speakers_dict))

    else:
        # No diarization data — original flow with plain SRT
        if job.embed_subtitles:
            file.status_message = f"Embedding subtitles: {file.filename}"
            job.status = JobStatus.EMBEDDING
            await update_job(job)

            output_video = os.path.join(output_dir, f"{base_name}_subtitled.mkv")
            await embed_subtitles(video_path, srt_path, output_video)
            file.output_path = output_video

            file.status = JobStatus.EMBEDDING
            file.status_message = None
            await update_job(job)

            job.status = JobStatus.CONVERTING
            file.status = JobStatus.CONVERTING
            file.status_message = f"Creating streaming version: {file.filename}"
            file.progress = 0
            await update_job(job)

            streaming_video = os.path.join(output_dir, f"{base_name}_streaming.mp4")
            
            async def update_conversion_progress(progress):
                file.progress = progress
                await update_job(job)

            try:
                await create_streaming_version(
                    output_video, srt_path, streaming_video,
                    max_height=1080,
                    progress_callback=update_conversion_progress,
                    is_cancelled=lambda: job.id in cancelled_jobs
                )
                file.streaming_path = streaming_video
            except Exception as e:
                print(f"[STREAMING] Failed: {e}")

    file.status = JobStatus.COMPLETED
    file.progress = 100
```

- [ ] **Step 5: Добавить API-эндпоинты для спикеров**

Добавить в `app/main.py` новые эндпоинты (перед функцией `process_job_queue`):

```python
@app.get("/api/jobs/{job_id}/speakers")
async def get_speakers(job_id: str, user=Depends(get_current_user)):
    """Get speakers with example utterances."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.speakers or not job.merged_segments:
        raise HTTPException(status_code=400, detail="No diarization data")

    speakers_dict = json.loads(job.speakers)
    merged = json.loads(job.merged_segments)

    from app.tasks.diarizer import get_speaker_examples
    examples = get_speaker_examples(merged, speakers_dict)

    return {
        "speakers": speakers_dict,
        "examples": examples,
    }


@app.put("/api/jobs/{job_id}/speakers")
async def update_speakers(job_id: str, req: SpeakerUpdateRequest, user=Depends(get_current_user)):
    """Update speaker names and colors."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.AWAITING_SPEAKERS:
        raise HTTPException(status_code=400, detail="Job is not awaiting speaker confirmation")

    job.speakers = json.dumps(req.speakers, ensure_ascii=False)
    await update_job(job)
    return {"status": "ok"}


@app.post("/api/jobs/{job_id}/speakers/confirm")
async def confirm_speakers(job_id: str, user=Depends(get_current_user)):
    """Confirm speakers and start subtitle generation."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.AWAITING_SPEAKERS:
        raise HTTPException(status_code=400, detail="Job is not awaiting speaker confirmation")

    job.status = JobStatus.GENERATING
    await update_job(job)

    # Resume processing in background
    asyncio.create_task(_finalize_diarized_job(job))
    return {"status": "ok"}


async def _finalize_diarized_job(job: Job):
    """Generate subtitles and embed after speaker confirmation."""
    try:
        current_settings = get_app_settings()

        for file in job.files:
            if file.status != JobStatus.AWAITING_SPEAKERS:
                continue

            video_path = find_video_file(job, file)
            if not video_path:
                file.status = JobStatus.FAILED
                file.error = "Video file not found"
                continue

            await _generate_and_embed(job, file, video_path, current_settings)
            await update_job(job)

        all_completed = all(f.status == JobStatus.COMPLETED for f in job.files)
        any_failed = any(f.status == JobStatus.FAILED for f in job.files)

        if all_completed:
            job.status = JobStatus.COMPLETED
            job.progress = 100
        elif any_failed:
            failed_count = sum(1 for f in job.files if f.status == JobStatus.FAILED)
            if failed_count == len(job.files):
                job.status = JobStatus.FAILED
                job.error = "All files failed"
            else:
                job.status = JobStatus.COMPLETED
                job.error = f"{failed_count} file(s) failed"

        await update_job(job)

    except Exception as e:
        print(f"[ERROR] Finalize diarized job failed: {e}")
        import traceback
        traceback.print_exc()
        job.status = JobStatus.FAILED
        job.error = str(e)
        await update_job(job)
```

- [ ] **Step 6: Добавить эндпоинт скачивания ASS**

```python
@app.get("/api/jobs/{job_id}/files/{file_id}/download/ass")
async def download_ass(job_id: str, file_id: str, user=Depends(get_current_user)):
    """Download ASS subtitle file."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    for file in job.files:
        if file.id == file_id and file.srt_path:
            ass_path = file.srt_path.rsplit('.', 1)[0] + '.ass'
            if os.path.exists(ass_path):
                return FileResponse(
                    ass_path,
                    media_type="text/x-ssa",
                    filename=os.path.basename(ass_path)
                )

    raise HTTPException(status_code=404, detail="ASS file not found")
```

- [ ] **Step 7: Обновить эндпоинт настроек — передача diarization_model и hf_token**

Найти существующий эндпоинт `PUT /api/settings` и обновить вызов `update_app_settings`, чтобы он передавал новые поля:

```python
@app.put("/api/settings")
async def update_settings(req: SettingsUpdateRequest, user=Depends(get_current_user)):
    """Update application settings."""
    result = update_app_settings(
        model=req.model,
        device=req.device,
        language=req.language,
        diarization_model=req.diarization_model,
        hf_token=req.hf_token,
    )
    return result
```

- [ ] **Step 8: Коммит**

```bash
git add app/main.py
git commit -m "API и пайплайн: диаризация, управление спикерами, генерация субтитров"
```

---

### Task 9: Фронтенд — HTML

**Files:**
- Modify: `app/static/index.html`

- [ ] **Step 1: Добавить модальное окно редактирования спикеров**

В `app/static/index.html`, перед закрывающим `</body>`, добавить новую модалку:

```html
    <!-- Speaker editing modal -->
    <div id="speakers-modal" class="modal" style="display:none">
        <div class="modal-content modal-large">
            <div class="modal-header">
                <h2 data-i18n="speakers.title">Speaker Editing</h2>
                <button class="close-btn" onclick="closeSpeakersModal()">&times;</button>
            </div>
            <div id="speakers-list" class="speakers-list">
                <!-- Filled dynamically -->
            </div>
            <div class="modal-footer">
                <button class="btn btn-secondary" onclick="confirmSpeakers(false)" data-i18n="speakers.keep_as_is">Keep as is</button>
                <button class="btn btn-primary" onclick="confirmSpeakers(true)" data-i18n="speakers.confirm">Confirm</button>
            </div>
        </div>
    </div>
```

- [ ] **Step 2: Коммит**

```bash
git add app/static/index.html
git commit -m "HTML: модальное окно редактирования спикеров"
```

---

### Task 10: Фронтенд — CSS

**Files:**
- Modify: `app/static/styles.css`

- [ ] **Step 1: Добавить стили для редактирования спикеров**

Добавить в конец `app/static/styles.css`:

```css
/* Speaker editing */
.speakers-list {
    padding: 1rem;
    max-height: 60vh;
    overflow-y: auto;
}

.speaker-card {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
    display: flex;
    gap: 1rem;
    align-items: flex-start;
}

.speaker-color {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    border: 2px solid var(--border);
    cursor: pointer;
    flex-shrink: 0;
    padding: 0;
    position: relative;
}

.speaker-color input[type="color"] {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    opacity: 0;
    cursor: pointer;
}

.speaker-info {
    flex: 1;
}

.speaker-name-input {
    width: 100%;
    padding: 0.5rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    font-size: 1rem;
    background: var(--bg);
    color: var(--text);
    margin-bottom: 0.5rem;
}

.speaker-examples {
    font-size: 0.85rem;
    color: var(--text-secondary);
}

.speaker-examples p {
    margin: 0.25rem 0;
    font-style: italic;
}

.modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: 0.5rem;
    padding: 1rem;
    border-top: 1px solid var(--border);
}

.modal-large {
    max-width: 600px;
}
```

- [ ] **Step 2: Коммит**

```bash
git add app/static/styles.css
git commit -m "CSS: стили модалки редактирования спикеров"
```

---

### Task 11: Фронтенд — JavaScript

**Files:**
- Modify: `app/static/app.js`

- [ ] **Step 1: Добавить обработку статуса AWAITING_SPEAKERS в рендеринге задачи**

Найти в `app.js` место, где рендерятся кнопки действий для задач (рядом с обработкой `awaiting_track`). Добавить аналогичную логику для `awaiting_speakers`:

```javascript
if (job.status === 'awaiting_speakers') {
    actionsHtml += `<button class="btn btn-primary btn-sm" onclick="openSpeakersModal('${job.id}')" data-i18n="speakers.edit_speakers">Edit Speakers</button>`;
}
```

Также добавить статус в отображение:

В маппинг статусов (где `awaiting_track` → `jobs.status.awaiting_track`), добавить:

```javascript
'awaiting_speakers': t('jobs.status.awaiting_speakers'),
'generating': t('jobs.status.generating'),
```

- [ ] **Step 2: Добавить функции для модалки спикеров**

В конец `app.js` добавить:

```javascript
let currentSpeakersJobId = null;
let currentSpeakers = {};

async function openSpeakersModal(jobId) {
    currentSpeakersJobId = jobId;
    try {
        const resp = await apiCall(`/api/jobs/${jobId}/speakers`);
        currentSpeakers = resp.speakers;
        const examples = resp.examples;

        const list = document.getElementById('speakers-list');
        list.innerHTML = '';

        for (const [spkId, info] of Object.entries(currentSpeakers)) {
            const examplesHtml = (examples[spkId] || [])
                .map(ex => `<p>«${ex}»</p>`)
                .join('');

            list.innerHTML += `
                <div class="speaker-card" data-speaker-id="${spkId}">
                    <div class="speaker-color" style="background-color: ${info.color}">
                        <input type="color" value="${info.color}"
                               onchange="updateSpeakerColor('${spkId}', this.value, this.parentElement)">
                    </div>
                    <div class="speaker-info">
                        <input type="text" class="speaker-name-input"
                               value="${info.name}"
                               data-speaker-id="${spkId}"
                               onchange="updateSpeakerName('${spkId}', this.value)">
                        <div class="speaker-examples">${examplesHtml}</div>
                    </div>
                </div>
            `;
        }

        document.getElementById('speakers-modal').style.display = 'flex';
    } catch (e) {
        showToast(t('toast.error.load_speakers'), 'error');
    }
}

function updateSpeakerColor(spkId, color, el) {
    currentSpeakers[spkId].color = color;
    el.style.backgroundColor = color;
}

function updateSpeakerName(spkId, name) {
    currentSpeakers[spkId].name = name;
}

function closeSpeakersModal() {
    document.getElementById('speakers-modal').style.display = 'none';
    currentSpeakersJobId = null;
}

async function confirmSpeakers(applyChanges) {
    if (!currentSpeakersJobId) return;

    try {
        if (applyChanges) {
            await apiCall(`/api/jobs/${currentSpeakersJobId}/speakers`, {
                method: 'PUT',
                body: JSON.stringify({ speakers: currentSpeakers }),
            });
        }
        await apiCall(`/api/jobs/${currentSpeakersJobId}/speakers/confirm`, {
            method: 'POST',
        });

        closeSpeakersModal();
        showToast(t('toast.success.speakers_confirmed'), 'success');
        loadJobs();
    } catch (e) {
        showToast(t('toast.error.confirm_speakers'), 'error');
    }
}
```

- [ ] **Step 3: Добавить кнопку скачивания ASS в действия завершённых задач**

В рендеринге завершённых задач, рядом с кнопкой `download_srt`, добавить:

```javascript
actionsHtml += `<button class="btn btn-sm" onclick="downloadAss('${job.id}', '${file.id}')" data-i18n="jobs.actions.download_ass">Download ASS</button>`;
```

И функцию:

```javascript
function downloadAss(jobId, fileId) {
    window.location.href = `/api/jobs/${jobId}/files/${fileId}/download/ass`;
}
```

- [ ] **Step 4: Добавить настройки диаризации в модалку настроек**

В функцию, которая рендерит настройки, добавить поля для модели диаризации и HF-токена. Найти то место, где рендерится `settings.model` dropdown, и после него добавить:

```javascript
// Diarization model select
let diarModelOptions = '';
for (const [key, val] of Object.entries(data.available_diarization_models || {})) {
    const selected = key === data.diarization_model ? 'selected' : '';
    diarModelOptions += `<option value="${key}" ${selected}>${val.name} — ${val.description}</option>`;
}

// Add to settings form HTML:
settingsHtml += `
    <div class="form-group">
        <label data-i18n="settings.diarization_model.label">Diarization Model</label>
        <select id="diarization-model">${diarModelOptions}</select>
    </div>
    <div class="form-group">
        <label data-i18n="settings.hf_token.label">HuggingFace Token</label>
        <input type="password" id="hf-token" value="${data.hf_token || ''}" placeholder="hf_...">
        <small data-i18n="settings.hf_token.hint">Required for pyannote models</small>
    </div>
`;
```

В функцию сохранения настроек добавить:

```javascript
const diarModel = document.getElementById('diarization-model')?.value;
const hfToken = document.getElementById('hf-token')?.value;

// Add to request body:
body.diarization_model = diarModel;
body.hf_token = hfToken;
```

- [ ] **Step 5: Коммит**

```bash
git add app/static/app.js
git commit -m "JS: модалка спикеров, настройки диаризации, скачивание ASS"
```

---

### Task 12: Локализация

**Files:**
- Modify: `app/static/locales/en.yml`
- Modify: `app/static/locales/ru.yml`
- Modify: `app/static/locales/*.yml` (remaining 13 locales)

- [ ] **Step 1: Обновить en.yml**

Добавить в конец (перед блоком `toast`):

```yaml
# Speakers
speakers:
  title: "Speaker Editing"
  edit_speakers: "Edit Speakers"
  keep_as_is: "Keep as is"
  confirm: "Confirm"
  speaker_n: "Speaker {n}"
```

В блок `jobs.status` добавить:

```yaml
    awaiting_speakers: "Speaker editing"
    generating: "Generating subtitles"
```

В блок `jobs.actions` добавить:

```yaml
    download_ass: "Download ASS"
```

В блок `settings` добавить:

```yaml
  diarization_model:
    label: "Diarization Model"
    descriptions:
      "pyannote/speaker-diarization-3.1": "Latest, highest accuracy"
      "pyannote/speaker-diarization-3.0": "Previous version, slightly lighter"
      "pyannote/speaker-diarization": "Base version, minimal resources"
  hf_token:
    label: "HuggingFace Token"
    hint: "Required for speaker diarization (pyannote models)"
    warning: "Token not set. Speaker diarization will be disabled."
```

В блок `toast.success` добавить:

```yaml
    speakers_confirmed: "Speakers confirmed, generating subtitles"
```

В блок `toast.error` добавить:

```yaml
    load_speakers: "Error loading speakers"
    confirm_speakers: "Error confirming speakers"
```

- [ ] **Step 2: Обновить ru.yml**

Аналогичные ключи на русском:

```yaml
# Speakers
speakers:
  title: "Редактирование спикеров"
  edit_speakers: "Редактировать спикеров"
  keep_as_is: "Оставить как есть"
  confirm: "Подтвердить"
  speaker_n: "Спикер {n}"
```

```yaml
    awaiting_speakers: "Редактирование спикеров"
    generating: "Генерация субтитров"
```

```yaml
    download_ass: "Скачать ASS"
```

```yaml
  diarization_model:
    label: "Модель диаризации"
    descriptions:
      "pyannote/speaker-diarization-3.1": "Новейшая, максимальная точность"
      "pyannote/speaker-diarization-3.0": "Предыдущая версия, чуть легче"
      "pyannote/speaker-diarization": "Базовая версия, минимальные ресурсы"
  hf_token:
    label: "Токен HuggingFace"
    hint: "Необходим для диаризации спикеров (модели pyannote)"
    warning: "Токен не задан. Диаризация спикеров будет отключена."
```

```yaml
    speakers_confirmed: "Спикеры подтверждены, генерируем субтитры"
```

```yaml
    load_speakers: "Ошибка загрузки спикеров"
    confirm_speakers: "Ошибка подтверждения спикеров"
```

- [ ] **Step 3: Обновить оставшиеся 13 локалей**

Для каждого файла из `app/static/locales/` (ar, de, es, fr, hi, it, ja, kk, ko, pt, tr, uk, zh) добавить аналогичные ключи. Для минимизации усилий можно использовать английские значения — носители языка смогут перевести позже.

- [ ] **Step 4: Коммит**

```bash
git add app/static/locales/
git commit -m "Локализация: ключи для диаризации спикеров (15 языков)"
```

---

### Task 13: Docker и зависимости

**Files:**
- Modify: `Dockerfile.base`
- Modify: `docker-compose.yml`
- Modify: `requirements.txt`

- [ ] **Step 1: Обновить Dockerfile.base**

После строки с `openai-whisper` (строка 40) добавить новый stage:

```dockerfile
# Stage 5.5: Install pyannote.audio for speaker diarization
RUN pip3 install --no-cache-dir pyannote.audio
```

После `ENV PYTHONDONTWRITEBYTECODE=1` (строка 55) добавить:

```dockerfile
ENV HF_HOME=/root/.cache/huggingface
```

- [ ] **Step 2: Обновить docker-compose.yml**

Добавить volume для кэша HuggingFace. В секции `volumes`:

```yaml
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./hf_cache:/root/.cache/huggingface
```

- [ ] **Step 3: Обновить requirements.txt**

Добавить в конец:

```
pyannote.audio
```

- [ ] **Step 4: Коммит**

```bash
git add Dockerfile.base docker-compose.yml requirements.txt
git commit -m "Docker: pyannote.audio, кэш HuggingFace, зависимости"
```

---

### Task 14: Интеграционная проверка

- [ ] **Step 1: Проверить, что приложение запускается**

```bash
cd d:\GIT\whisper-subtitles
python -c "from app.models import JobStatus; print(JobStatus.AWAITING_SPEAKERS)"
python -c "from app.tasks.diarizer import diarize, merge_transcription_with_diarization, assign_default_speakers; print('OK')"
python -c "from app.tasks.subtitle_generator import generate_ass, generate_srt_with_speakers, generate_vtt_with_speakers; print('OK')"
```

- [ ] **Step 2: Проверить генерацию субтитров на тестовых данных**

```bash
python -c "
from app.tasks.subtitle_generator import generate_ass, generate_srt_with_speakers, generate_vtt_with_speakers
segments = [
    {'start': 0.0, 'end': 3.5, 'text': 'Hello everyone', 'speaker': 'SPEAKER_00'},
    {'start': 4.0, 'end': 6.2, 'text': 'Thank you', 'speaker': 'SPEAKER_01'},
]
speakers = {
    'SPEAKER_00': {'name': 'Ivan', 'color': '#FF6B6B'},
    'SPEAKER_01': {'name': 'Maria', 'color': '#4ECDC4'},
}
print('=== ASS ===')
print(generate_ass(segments, speakers)[:500])
print('=== SRT ===')
print(generate_srt_with_speakers(segments, speakers))
print('=== VTT ===')
print(generate_vtt_with_speakers(segments, speakers))
"
```

- [ ] **Step 3: Финальный коммит**

```bash
git add -A
git commit -m "Диаризация спикеров: полная реализация"
```
