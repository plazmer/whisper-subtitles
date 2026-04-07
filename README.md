# 🎬 Whisper Subtitles

<div align="center">

![Whisper Subtitles Logo](https://img.shields.io/badge/🎬-Whisper%20Subtitles-blue?style=for-the-badge)

**AI-powered subtitle generator using OpenAI Whisper**

[![License](https://img.shields.io/badge/License-Non--Commercial-red.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](docker-compose.yml)
[![NVIDIA GPU](https://img.shields.io/badge/NVIDIA-CUDA%2012.1-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![OpenAI Whisper](https://img.shields.io/badge/OpenAI-Whisper-00A67E.svg)](https://github.com/openai/whisper)

</div>

---


<div align="center">

## 🌐 Поддерживаемые языки интерфейса / Supported Interface Languages

Интерфейс приложения доступен на **15 языках** / The interface is available in **15 languages**:

| | | | |
|:---:|:---:|:---:|:---:|
| 🇷🇺 Русский | 🇬🇧 English | 🇺🇦 Українська | 🇰🇿 Қазақша |
| 🇩🇪 Deutsch | 🇫🇷 Français | 🇪🇸 Español | 🇮🇹 Italiano |
| 🇧🇷 Português | 🇯🇵 日本語 | 🇨🇳 中文 | 🇰🇷 한국어 |
| 🇹🇷 Türkçe | 🇸🇦 العربية | 🇮🇳 हिन्दी | |

📖 **Инструкция / Documentation:** [🇷🇺 Русский](#-русский) | [🇬🇧 English](#-english)

</div>

---

# 🇷🇺 Русский

## 📋 Описание

> 👂 **Создано для людей, которые заслуживают смотреть любое видео без барьеров**

**Whisper Subtitles** — это не просто программа, это **мост между миром звука и текста**.

🎬 Миллионы фильмов, сериалов и видео не имеют субтитров. Для людей с нарушениями слуха это означает невозможность наслаждаться контентом, который доступен всем остальным.

**Whisper Subtitles меняет это:**

- 🏠 **Домашние видео** — семейные записи станут доступны всем членам семьи
- 🎥 **Фильмы без субтитров** — наконец-то можно смотреть!
- 📚 **Видеолекции** — обучение без ограничений
- 🌍 **Документальные фильмы** — познавайте мир без барьеров
- 🎙️ **Подкасты и интервью** — вся информация теперь в тексте

Приложение использует передовую нейросеть **OpenAI Whisper** с поддержкой **NVIDIA GPU** через **CUDA**, что позволяет значительно ускорить обработку видео.

### ✨ Возможности

- 🎯 **Автоматическое распознавание речи** — поддержка 99 языков
- 📁 **Множество источников** — загрузка видео, URL, магнет-ссылки, торренты
- 🎬 **Любые форматы видео** — MKV, MP4, AVI, MOV, WebM и другие
- 🎵 **Выбор аудиодорожки** — возможность выбрать нужную дорожку при обработке
- 📦 **Выборочная загрузка** — в торрентах можно выбрать только нужные серии/файлы
- ▶️ **Онлайн просмотр** — смотрите обработанные видео прямо в браузере
- 📝 **Экспорт субтитров** — скачивание в SRT формате
- 🎬 **Вшивание субтитров** — субтитры добавляются как отдельная дорожка в видеофайл
- 🎨 **Современный интерфейс** — адаптивный дизайн на 15 языках
- ⚡ **NVIDIA GPU ускорение** — CUDA обработка для высокой скорости
- 🔒 **Безопасность** — авторизация с JWT токенами

### 📸 Интерфейс

<div align="center">

![Интерфейс приложения](img/interface.png)
*Главный экран приложения*

</div>

### 🎬 Демонстрация

<details>
<summary><b>▶️ Обработка одного видео</b></summary>

https://github.com/user-attachments/assets/de31e0c5-0e7d-4589-b4dc-f57ddbaa9616

</details>

<details>
<summary><b>🔗 Загрузка торрента (полный процесс)</b></summary>

https://github.com/user-attachments/assets/69f2dcfc-6609-4928-b92a-af08f5303b90

</details>

<details>
<summary><b>📦 Выбор серий в торренте</b></summary>

![Выбор серий](img/torrent_m.png)

https://github.com/user-attachments/assets/torrent_multi.mp4

</details>

### ⚙️ Настройки

<div align="center">

| Языки интерфейса | Модели распознавания | Настройки |
|:---:|:---:|:---:|
| ![Языки](img/Interface_Language.png) | ![Модели](img/Recognition_Model.png) | ![Настройки](img/Settings.png) |

</div>

### 🖥️ Системные требования

| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| GPU | NVIDIA с 4GB VRAM | NVIDIA с 8GB+ VRAM |
| RAM | 4 GB | 8+ GB |
| Диск | 10 GB | 50+ GB (для моделей и видео) |
| ОС | Linux (Docker + NVIDIA Container Toolkit) | Ubuntu 22.04+ |

### 🚀 Быстрый старт

#### ⚠️ Предварительные требования

Для работы с NVIDIA GPU вам потребуется:

1. **NVIDIA GPU** с поддержкой CUDA и установленными драйверами
2. **NVIDIA Container Toolkit** для Docker

Установка NVIDIA Container Toolkit:

```bash
# Для Ubuntu/Debian
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### 1️⃣ Клонируйте репозиторий

```bash
git clone https://github.com/timoil/whisper-subtitles.git
cd whisper-subtitles
```

#### 2️⃣ Соберите базовый образ (один раз, ~10 минут)

Базовый образ содержит все зависимости и собирается один раз:

```bash
docker build -f Dockerfile.base -t whisper-subtitles-base:latest .
```

#### 3️⃣ Соберите и запустите приложение (~5 секунд)

Основной образ собираётся мгновенно — только копирует код:

```bash
docker compose up -d
```

Всё! Приложение запущено! 🎉

#### 3️⃣ Откройте в браузере

```
http://localhost:8000
```

**Данные для входа:**
- 👤 Логин: `admin`
- 🔑 Пароль: `admin123`

> ⚠️ **Важно!** Смените пароль в настройках после первого входа!

### 🎤 Диаризация спикеров (HuggingFace access)

Для работы диаризации используется `pyannote.audio`, а модели диаризации на HuggingFace являются **gated** (требуют принятия условий).

Перед использованием:

1. Откройте страницу модели и запросите/подтвердите доступ:  
   - [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)  
   - (опционально) [`pyannote/speaker-diarization-3.0`](https://huggingface.co/pyannote/speaker-diarization-3.0)  
   - (опционально) [`pyannote/speaker-diarization`](https://huggingface.co/pyannote/speaker-diarization)
2. Создайте токен с правами `read`:  
   [`https://huggingface.co/settings/tokens`](https://huggingface.co/settings/tokens)
3. В приложении откройте **Settings** и заполните:
   - **Diarization Model**
   - **HuggingFace Token**

Если доступ не выдан или токен не задан, диаризация не выполнится (в логах будет ошибка доступа к gated repo).

### 📊 Модели Whisper

| Модель | Размер | Скорость | Качество | Рекомендация |
|--------|--------|----------|----------|--------------|
| `tiny` | 75 MB | ~32x | ⭐ | Быстрые тесты |
| `base` | 142 MB | ~16x | ⭐⭐ | Черновики |
| `small` | 466 MB | ~10x | ⭐⭐⭐ | Баланс |
| `medium` | 1.5 GB | ~5x | ⭐⭐⭐⭐ | Хорошее качество |
| `large-v2` | 3 GB | ~3x | ⭐⭐⭐⭐⭐ | Максимальная точность |
| `large-v3` | 3 GB | ~4x (GPU) | ⭐⭐⭐⭐⭐ | **Рекомендуется для NVIDIA** |
| `large-v3-turbo` | 1.5 GB | ~8x (GPU) | ⭐⭐⭐⭐ | Быстрая обработка |

### 🔧 Расширенная настройка

#### Docker Compose с NVIDIA GPU

```yaml
services:
  whisper:
    build: .
    container_name: whisper-subtitles
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - SECRET_KEY=${SECRET_KEY}
      - DEFAULT_MODEL=${DEFAULT_MODEL:-large-v3}
      - DEVICE=${DEVICE:-auto}  # auto, cuda, cpu
    restart: unless-stopped
```

#### Настройки устройства

Переменная окружения `DEVICE` управляет использованием GPU:

- `auto` — автоматически выбирает CUDA если доступна, иначе CPU
- `cuda` — принудительно использовать NVIDIA GPU
- `cpu` — принудительно использовать CPU

#### Без Docker (локальная установка)

```bash
# Установка зависимостей
pip install -r requirements.txt

# Установка PyTorch с CUDA поддержкой (отдельно)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Запуск приложения
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> **Примечание:** PyTorch устанавливается отдельно для использования правильной версии CUDA.

#### Размещение за Nginx

```nginx
server {
    listen 80;
    server_name subtitles.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 10G;  # Для больших видео
    }
}
```

### 📁 Структура проекта

```
whisper-subtitles/
├── app/                    # Исходный код приложения
│   ├── static/            # Статические файлы (CSS, JS)
│   │   └── locales/       # Файлы локализаций
│   ├── tasks/             # Фоновые задачи
│   ├── main.py            # Точка входа FastAPI
│   └── database.py        # База данных SQLite
├── data/                   # Данные (создаётся автоматически)
│   ├── uploads/           # Загруженные файлы
│   ├── downloads/         # Скачанные файлы
│   ├── temp/              # Временные файлы
│   └── output/            # Готовые субтитры
├── models/                 # Модели Whisper (скачиваются автоматически)
├── docker-compose.yml      # Docker конфигурация
├── Dockerfile             # Сборка образа
├── requirements.txt       # Python зависимости
└── .env.example           # Пример переменных окружения
```

### ❓ Частые вопросы

<details>
<summary><b>Как изменить порт?</b></summary>

В `docker-compose.yml` измените:
```yaml
ports:
  - "8080:8000"  # Теперь доступно на порту 8080
```
</details>

<details>
<summary><b>Где хранятся субтитры?</b></summary>

В папке `./data/output/` в формате SRT.
</details>

<details>
<summary><b>Можно ли использовать без Docker?</b></summary>

Да, но потребуется вручную установить FFmpeg, aria2 и Python зависимости.
</details>

### 📄 Лицензия

Этот проект предназначен **только для некоммерческого использования**.

---

# 🇬🇧 English

## 📋 Description

> 👂 **Created for people who deserve to watch any video without barriers**

**Whisper Subtitles** is not just software, it's a **bridge between the world of sound and text**.

🎬 Millions of movies, TV shows, and videos don't have subtitles. For people with hearing impairments, this means being unable to enjoy content that's available to everyone else.

**Whisper Subtitles changes this:**

- 🏠 **Home videos** — family recordings become accessible to all family members
- 🎥 **Movies without subtitles** — finally, you can watch them!
- 📚 **Video lectures** — learning without limitations
- 🌍 **Documentaries** — explore the world without barriers
- 🎙️ **Podcasts and interviews** — all information now in text

The application uses the cutting-edge **OpenAI Whisper** neural network with **NVIDIA GPU** support via **CUDA**, significantly speeding up video processing.

### ✨ Features

- 🎯 **Automatic speech recognition** — 99 languages supported
- 📁 **Multiple input sources** — video files, URLs, magnet links, torrents
- 🎬 **Any video format** — MKV, MP4, AVI, MOV, WebM and more
- 🎵 **Audio track selection** — choose the desired audio track during processing
- 📦 **Selective download** — download only specific episodes/files from torrents
- ▶️ **Online playback** — watch processed videos directly in your browser
- 📝 **Subtitle export** — download in SRT format
- 🎬 **Subtitle embedding** — subtitles are added as a separate track in the video file
- 🎨 **Modern interface** — responsive design in 15 languages
- ⚡ **NVIDIA GPU acceleration** — CUDA processing for high speed
- 🔒 **Security** — JWT token authorization

### 📸 Interface

<div align="center">

![Application interface](img/interface.png)
*Main application screen*

</div>

### 🎬 Demo

<details>
<summary><b>▶️ Processing a single video</b></summary>

https://github.com/user-attachments/assets/de31e0c5-0e7d-4589-b4dc-f57ddbaa9616

</details>

<details>
<summary><b>🔗 Torrent download (full process)</b></summary>

https://github.com/user-attachments/assets/69f2dcfc-6609-4928-b92a-af08f5303b90

</details>

<details>
<summary><b>📦 Selecting episodes from torrent</b></summary>

![Episode selection](img/torrent_m.png)

https://github.com/user-attachments/assets/torrent_multi.mp4

</details>

### ⚙️ Settings

<div align="center">

| Interface Languages | Recognition Models | Settings |
|:---:|:---:|:---:|
| ![Languages](img/Interface_Language.png) | ![Models](img/Recognition_Model.png) | ![Settings](img/Settings.png) |

</div>

### 🖥️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA with 4GB VRAM | NVIDIA with 8GB+ VRAM |
| RAM | 4 GB | 8+ GB |
| Storage | 10 GB | 50+ GB (for models and videos) |
| OS | Linux (Docker + NVIDIA Container Toolkit) | Ubuntu 22.04+ |

### 🚀 Quick Start

#### ⚠️ Prerequisites

For NVIDIA GPU support, you need:

1. **NVIDIA GPU** with CUDA support and drivers installed
2. **NVIDIA Container Toolkit** for Docker

Install NVIDIA Container Toolkit:

```bash
# For Ubuntu/Debian
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### 1️⃣ Clone the repository

```bash
git clone https://github.com/timoil/whisper-subtitles.git
cd whisper-subtitles
```

#### 2️⃣ Build base image (one time, ~10 minutes)

The base image contains all dependencies and is built once:

```bash
docker build -f Dockerfile.base -t whisper-subtitles-base:latest .
```

#### 3️⃣ Build and run the application (~5 seconds)

The main image builds instantly — it only copies code:

```bash
docker compose up -d
```

That's it! The app is running! 🎉

#### 3️⃣ Open in browser

```
http://localhost:8000
```

**Login credentials:**
- 👤 Username: `admin`
- 🔑 Password: `admin123`

> ⚠️ **Important!** Change your password in settings after first login!

### 🎤 Speaker Diarization (HuggingFace access)

Speaker diarization uses `pyannote.audio`, and diarization models on HuggingFace are **gated** (you must accept terms first).

Before using diarization:

1. Open the model page and request/accept access:  
   - [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)  
   - (optional) [`pyannote/speaker-diarization-3.0`](https://huggingface.co/pyannote/speaker-diarization-3.0)  
   - (optional) [`pyannote/speaker-diarization`](https://huggingface.co/pyannote/speaker-diarization)
2. Create a `read` token:  
   [`https://huggingface.co/settings/tokens`](https://huggingface.co/settings/tokens)
3. In the app **Settings**, fill:
   - **Diarization Model**
   - **HuggingFace Token**

If access is not granted or token is missing, diarization will fail (logs will show gated-repo access error).

### 📊 Whisper Models

| Model | Size | Speed | Quality | Recommendation |
|-------|------|-------|---------|----------------|
| `tiny` | 75 MB | ~32x | ⭐ | Quick tests |
| `base` | 142 MB | ~16x | ⭐⭐ | Drafts |
| `small` | 466 MB | ~10x | ⭐⭐⭐ | Balance |
| `medium` | 1.5 GB | ~5x | ⭐⭐⭐⭐ | Good quality |
| `large-v2` | 3 GB | ~3x | ⭐⭐⭐⭐⭐ | Maximum accuracy |
| `large-v3` | 3 GB | ~4x (GPU) | ⭐⭐⭐⭐⭐ | **Recommended for NVIDIA** |
| `large-v3-turbo` | 1.5 GB | ~8x (GPU) | ⭐⭐⭐⭐ | Fast processing |

### 🔧 Advanced Configuration

#### Docker Compose with NVIDIA GPU

```yaml
services:
  whisper:
    build: .
    container_name: whisper-subtitles
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - SECRET_KEY=${SECRET_KEY}
      - DEFAULT_MODEL=${DEFAULT_MODEL:-large-v3}
      - DEVICE=${DEVICE:-auto}  # auto, cuda, cpu
    restart: unless-stopped
```

#### Device Settings

The `DEVICE` environment variable controls GPU usage:

- `auto` — automatically selects CUDA if available, otherwise CPU
- `cuda` — force use NVIDIA GPU
- `cpu` — force use CPU

#### Without Docker (Local Installation)

```bash
# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support (separately)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Run the application
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> **Note:** PyTorch is installed separately to use the correct CUDA version.

#### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name subtitles.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 10G;  # For large videos
    }
}
```

### 📁 Project Structure

```
whisper-subtitles/
├── app/                    # Application source code
│   ├── static/            # Static files (CSS, JS)
│   │   └── locales/       # Localization files
│   ├── tasks/             # Background tasks
│   ├── main.py            # FastAPI entry point
│   └── database.py        # SQLite database
├── data/                   # Data (created automatically)
│   ├── uploads/           # Uploaded files
│   ├── downloads/         # Downloaded files
│   ├── temp/              # Temporary files
│   └── output/            # Ready subtitles
├── models/                 # Whisper models (downloaded automatically)
├── docker-compose.yml      # Docker configuration
├── Dockerfile             # Image build
├── requirements.txt       # Python dependencies
└── .env.example           # Environment variables example
```

### ❓ FAQ

<details>
<summary><b>How to change the port?</b></summary>

In `docker-compose.yml` change:
```yaml
ports:
  - "8080:8000"  # Now available on port 8080
```
</details>

<details>
<summary><b>Where are subtitles stored?</b></summary>

In `./data/output/` folder in SRT format.
</details>

<details>
<summary><b>Can I use without Docker?</b></summary>

Yes, but you'll need to manually install FFmpeg, aria2 and Python dependencies.
</details>

### 📄 License

This project is for **non-commercial use only**.

---

<div align="center">

**Made with ❤️ using open source software**

[⬆️ Back to top](#-whisper-subtitles)

</div>
