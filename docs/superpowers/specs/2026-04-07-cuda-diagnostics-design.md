# CUDA-ускорение: диагностика, structured errors, smoke-тест — спецификация

## Цель

Гарантировать корректную работу CUDA-ускорения (Whisper + pyannote) в Docker и локально, обеспечить прозрачную доставку GPU-диагностики и структурированных ошибок на фронтенд, включая HuggingFace gated-ошибки с URL репозитория и подсказкой про токен. Реальный smoke-тест GPU с загрузкой tiny-модели Whisper.

## Требования

1. **Structured Error Payload** — единый JSON-формат для всех ошибок (CUDA, HF, транскрипция, диаризация) вместо плоских строк.
2. **CUDA fallback** — при недоступной CUDA мягкий переход на CPU с диагностикой, во всех точках (transcriber, diarizer, diagnostics).
3. **HF gated parsing** — при ошибках HuggingFace формировать ошибку с конкретным URL модели, подсказкой про токен и оригинальным текстом исключения.
4. **GPU Diagnostics endpoint** — `POST /api/diagnostics/gpu` с реальным smoke-run (tiny-модель Whisper на GPU).
5. **Volume validation** — проверка что `/app/models` реально смонтирован как volume (writable, persistent), перечисление закешированных моделей.
6. **Единый volume для всех моделей** — Whisper + HF кеш в одном `./models:/app/models`.
7. **Docker base image validation** — validation step при сборке базового образа.
8. **Фронтенд** — structured errors в карточках задач, GPU Diagnostics в настройках, device info в прогрессе.
9. **Тестирование** — end-to-end проверка с `R:\output.mp4` (>1 спикера).

## Structured Error Payload

### Схема

```json
{
    "code": "hf_gated_access_denied",
    "message": "Нет доступа к модели pyannote/speaker-diarization-3.1",
    "details": "403 Client Error: Forbidden for url: https://huggingface.co/api/models/...",
    "hint": "Откройте страницу модели и примите условия, затем укажите токен в настройках",
    "url": "https://huggingface.co/pyannote/speaker-diarization-3.1"
}
```

- `code` — машиночитаемый код ошибки (строка).
- `message` — человекочитаемое описание (основной текст на фронтенде).
- `details` — оригинальный текст исключения / traceback (скрыт под «Подробности»).
- `hint` — подсказка что делать (курсивом на фронтенде).
- `url` — ссылка (кликабельная на фронтенде, открывается в новой вкладке). Необязательное поле.

Хранение: `job.error` и `file.error` содержат `json.dumps(payload)`. Обратная совместимость: фронтенд пробует `JSON.parse()`, при неудаче показывает как текст.

### Коды ошибок

| Код | Когда |
|-----|-------|
| `cuda_unavailable` | Запрошен `cuda`, но `torch.cuda.is_available() == False` |
| `cuda_out_of_memory` | OOM при загрузке модели на GPU |
| `hf_gated_access_denied` | 401/403 от HuggingFace (gated model) |
| `hf_token_missing` | Диаризация запрошена, но `hf_token` пустой |
| `hf_model_not_found` | Модель не найдена на HuggingFace |
| `diarization_failed` | Общая ошибка диаризации |
| `transcription_failed` | Общая ошибка транскрипции |
| `diagnostic_failed` | Smoke-тест GPU не прошёл |

### Утилита формирования ошибок

Новый модуль `app/tasks/errors.py`:

```python
def make_error(code: str, message: str, details: str = None, hint: str = None, url: str = None) -> str:
    """Формирует JSON-строку structured error для записи в job.error / file.error."""

def parse_error(error_raw: str) -> dict:
    """Парсит error-строку. Возвращает dict с полями code/message/details/hint/url.
    Если не JSON — возвращает {"code": "unknown", "message": error_raw}."""
```

## CUDA Fallback и resolve_device

### Новый модуль `app/tasks/gpu_utils.py`

```python
def resolve_device(requested: str = "auto") -> tuple[str, dict]:
    """
    Определяет фактическое устройство и собирает диагностику.
    
    Returns:
        (actual_device, diagnostics) где diagnostics:
        {
            "requested": "cuda",
            "resolved": "cuda",
            "gpu_name": "NVIDIA GeForce RTX 3060",
            "vram_total_mb": 12288,
            "vram_free_mb": 10500,
            "cuda_version": "12.1",
            "driver_version": "535.129.03",
            "torch_version": "2.5.1+cu121",
            "fallback": False,
            "fallback_reason": None
        }
    """
```

Логика:
- `requested == "auto"` → `cuda` если `torch.cuda.is_available()`, иначе `cpu`.
- `requested == "cuda"` и CUDA недоступна → fallback на `cpu`, `fallback=True`, `fallback_reason="CUDA not available"`.
- `requested == "cuda"` и CUDA доступна → `cuda`, заполняем gpu_name/vram/versions из `torch.cuda`.
- `requested == "cpu"` → `cpu`, без диагностики GPU.

### Точки интеграции

1. **`app/tasks/transcribe_worker.py`** — заменить inline-проверку CUDA на `resolve_device()`. Результат diagnostics отправлять через `emit_status("device_info", ...)`.
2. **`app/tasks/diarizer.py`** — добавить `resolve_device()` перед `pipeline.to()`. Сейчас проверки CUDA fallback нет — будет падать.
3. **`app/tasks/transcriber.py`** — заменить inline-проверку в `get_whisper_model()`.
4. **`POST /api/diagnostics/gpu`** — использовать для отчёта.

### Прокидывание device info в status_message

При транскрипции и диаризации `status_message` включает информацию об устройстве:
- `"Transcribing on CUDA (RTX 3060, 10.5 GB free)"`
- `"Diarizing speakers on CUDA (RTX 3060)"`
- `"Transcribing on CPU (CUDA fallback: not available)"` — если произошёл fallback

## HF Gated Error Parsing

### В `app/tasks/diarizer.py`

Функция `_parse_hf_error(exc, model_name)` парсит исключения от `huggingface_hub`/`pyannote`:

```python
def _parse_hf_error(exc: Exception, model_name: str) -> dict:
    error_str = str(exc)
    model_url = f"https://huggingface.co/{model_name}"
    lower = error_str.lower()

    if "401" in lower or "unauthorized" in lower:
        return make_error("hf_gated_access_denied",
            f"Токен HuggingFace невалиден или отсутствует для модели {model_name}",
            details=error_str,
            hint="Создайте токен на huggingface.co/settings/tokens и укажите его в настройках",
            url=model_url)

    if "403" in lower or "access" in lower or "gated" in lower:
        return make_error("hf_gated_access_denied",
            f"Нет доступа к модели {model_name}. Требуется принять условия использования.",
            details=error_str,
            hint="Откройте страницу модели, примите условия, и убедитесь что токен указан в настройках",
            url=model_url)

    if "404" in lower or "not found" in lower:
        return make_error("hf_model_not_found",
            f"Модель {model_name} не найдена на HuggingFace",
            details=error_str,
            hint="Проверьте название модели в настройках",
            url=model_url)

    return make_error("diarization_failed",
        f"Ошибка диаризации: {model_name}",
        details=error_str)
```

### Превентивная проверка

Перед вызовом `Pipeline.from_pretrained()` — если `hf_token` пустой, сразу:
```python
make_error("hf_token_missing",
    "Для диаризации требуется токен HuggingFace",
    hint="Создайте токен на huggingface.co/settings/tokens и укажите его в настройках приложения",
    url="https://huggingface.co/settings/tokens")
```

## GPU Diagnostics Endpoint

### `POST /api/diagnostics/gpu`

Требует авторизации. Параметр `?force=true` — сбрасывает кеш.

Этапы:
1. **Device info** (~1 мс): `resolve_device()` с текущим `settings.device`.
2. **Volume check** (~1 мс): для `/app/models` — записать/прочитать/удалить `.probe` файл, перечислить закешированные Whisper-модели (поддиректории), посчитать размер `huggingface/` поддиректории.
3. **Smoke-run** (~3-30 сек): загрузить `tiny` модель на GPU, транскрибировать 5-секундный синтетический WAV (numpy синусоида → soundfile → whisper.transcribe), замерить время. Выполняется в `run_in_executor`, чтобы не блокировать event loop.

### Ответ

```json
{
    "cuda_available": true,
    "gpu_name": "NVIDIA GeForce RTX 3060",
    "vram_total_mb": 12288,
    "vram_free_mb": 10500,
    "cuda_version": "12.1",
    "driver_version": "535.129.03",
    "torch_version": "2.5.1+cu121",
    "smoke_test": {
        "status": "passed",
        "model": "tiny",
        "duration_sec": 3.2,
        "device_used": "cuda",
        "error": null
    },
    "volumes": {
        "models_path": "/app/models",
        "writable": true,
        "cached_whisper_models": ["large-v3", "tiny"],
        "hf_cache_path": "/app/models/huggingface",
        "hf_cache_size_mb": 1240
    }
}
```

### Кеширование

Результат хранится в памяти 5 минут. Параметр `?force=true` сбрасывает. Smoke-run не выполняется, если GPU занят транскрипцией (возвращает `smoke_test.status: "skipped"`).

## Docker

### `Dockerfile.base` — изменения

1. Whisper устанавливается полноценно (без `--no-deps`), чтобы deps были синхронны:
```dockerfile
RUN pip3 install --no-cache-dir openai-whisper
```

2. pyannote с pin huggingface_hub:
```dockerfile
RUN pip3 install --no-cache-dir "pyannote.audio<4" "huggingface_hub>=0.20,<1.0"
```

3. torchaudio aligned:
```dockerfile
RUN pip3 install --no-cache-dir --force-reinstall \
    torchaudio==2.5.1 --no-deps \
    --index-url https://download.pytorch.org/whl/cu121
```

4. `HF_HOME` перенаправлен в models volume:
```dockerfile
ENV HF_HOME=/app/models/huggingface
```

5. Validation step в конце:
```dockerfile
RUN python3 -c "\
import whisper; import pyannote.audio; import torch; \
print(f'torch={torch.__version__} cuda_build={torch.version.cuda}'); \
print('whisper OK'); print('pyannote OK')"
```

### `Dockerfile` — упрощение

```dockerfile
FROM whisper-subtitles-base:latest
COPY app/ ./app/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Убран `pip install pyannote.audio` safety net. Основной образ собирается за 1-2 секунды.

### `docker-compose.yml` — изменения

```yaml
volumes:
  - ./data:/app/data
  - ./models:/app/models
# Убрана строка: - ./hf_cache:/root/.cache/huggingface
```

### `app/config.py` — изменения

```python
hf_home: str = "/app/models/huggingface"
```

### Структура volume `./models` на хосте

```
./models/
├── large-v3.pt          # Whisper (или поддиректория, зависит от версии)
├── tiny.pt              # Whisper (для smoke-теста)
└── huggingface/         # HF кеш
    └── hub/
        └── models--pyannote--speaker-diarization-3.1/
```

## Фронтенд

### Structured errors в карточках задач

Функция `renderError(errorRaw)` в `app.js`:

```javascript
function renderError(errorRaw) {
    let err;
    try { err = JSON.parse(errorRaw); } catch { err = null; }
    
    if (!err || !err.message) {
        return `<div class="text-error">${escapeHtml(errorRaw)}</div>`;
    }

    let html = `<div class="text-error structured-error">`;
    html += `<div class="error-message">${escapeHtml(err.message)}</div>`;
    if (err.hint) {
        let hintHtml = escapeHtml(err.hint);
        if (err.url) {
            hintHtml += ` <a href="${escapeHtml(err.url)}" target="_blank" rel="noopener">${escapeHtml(err.url)}</a>`;
        }
        html += `<div class="error-hint">${hintHtml}</div>`;
    }
    if (err.details) {
        html += `<details><summary>${i18n.t('errors.show_details')}</summary>`;
        html += `<pre class="error-details">${escapeHtml(err.details)}</pre></details>`;
    }
    html += `</div>`;
    return html;
}
```

### GPU Diagnostics в настройках

В модалке настроек — секция внизу:

- Кнопка `i18n.t('settings.gpu_diagnostics')` → `POST /api/diagnostics/gpu` + спиннер.
- Результат — компактная карточка:
  - Строка 1: GPU name + VRAM (зелёный) или «CPU mode» (жёлтый)
  - Строка 2: Smoke test result
  - Строка 3: Cached models list
  - Строка 4: Volume status

### Device info в прогрессе задачи

`status_message` уже отображается под прогресс-баром. Изменений фронтенда не нужно — бэкенд будет писать в `status_message` строки вида «Transcribing on CUDA (RTX 3060)».

### Локализация

Новые ключи (все 15 файлов locales):

| Ключ | ru | en |
|------|----|----|
| `settings.gpu_diagnostics` | Диагностика GPU | GPU Diagnostics |
| `diagnostics.cuda_available` | CUDA доступна | CUDA available |
| `diagnostics.cuda_unavailable` | CUDA недоступна, используется CPU | CUDA unavailable, using CPU |
| `diagnostics.smoke_passed` | Smoke-тест пройден | Smoke test passed |
| `diagnostics.smoke_failed` | Smoke-тест не пройден | Smoke test failed |
| `diagnostics.smoke_skipped` | Smoke-тест пропущен (GPU занят) | Smoke test skipped (GPU busy) |
| `diagnostics.volumes_ok` | Volume смонтирован корректно | Volume mounted correctly |
| `diagnostics.volumes_warning` | Volume не смонтирован — данные будут потеряны при перезапуске | Volume not mounted — data will be lost on restart |
| `diagnostics.cached_models` | Закешированные модели | Cached models |
| `errors.show_details` | Подробности | Details |

## Тестирование

Ручная проверка end-to-end:

1. **Сборка base-образа**: `docker build -f Dockerfile.base -t whisper-subtitles-base:latest .` — validation step печатает `torch=... cuda_build=12.1, whisper OK, pyannote OK`.
2. **Сборка app-образа**: `docker compose build` — занимает 1-2 секунды.
3. **Запуск**: `docker compose up -d`.
4. **GPU Diagnostics**: в настройках нажать кнопку → CUDA available, smoke test passed, volume writable, модели закешированы.
5. **Загрузка тестового видео**: через UI загрузить `R:\output.mp4`.
6. **Транскрипция**: в прогрессе видно «Transcribing on CUDA (GPU name)».
7. **Диаризация**: после транскрипции — «Diarizing speakers on CUDA», появляется экран спикеров (файл содержит >1 спикера в начале).
8. **HF ошибка**: сбросить HF-токен → structured error с URL модели и подсказкой.
9. **Перезапуск**: `docker compose restart` → повторить 4-7, модели не перескачиваются.
10. **CUDA fallback**: `DEVICE=cpu docker compose up -d` → «Transcribing on CPU», диагностика показывает CPU mode.
