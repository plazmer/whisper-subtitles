import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Security
    secret_key: str = "change-me-in-production"
    algorithm: str = "HS256"
    access_token_expire_hours: int = 24

    # Whisper settings
    default_model: str = "large-v3"
    device: str = "cuda"  # cuda, cpu, or auto
    default_language: str = "auto"
    default_diarization_model: str = "pyannote/speaker-diarization-3.1"
    hf_home: str = "/app/models/huggingface"

    # Paths
    data_dir: str = "/app/data"
    models_dir: str = "/app/models"

    # Available models with descriptions (OpenAI Whisper models)
    @property
    def available_models(self) -> dict:
        return {
            "tiny": {
                "name": "Tiny",
                "size": "~75 MB",
                "accuracy": 2,
                "speed": "~32x faster than realtime",
                "description": "For quick tests, low accuracy"
            },
            "base": {
                "name": "Base",
                "size": "~150 MB",
                "accuracy": 3,
                "speed": "~16x faster than realtime",
                "description": "Fast but with errors"
            },
            "small": {
                "name": "Small",
                "size": "~500 MB",
                "accuracy": 4,
                "speed": "~10x faster than realtime",
                "description": "Good balance of speed and quality"
            },
            "medium": {
                "name": "Medium",
                "size": "~1.5 GB",
                "accuracy": 4,
                "speed": "~5x faster than realtime",
                "description": "Recommended for most tasks"
            },
            "large-v2": {
                "name": "Large V2",
                "size": "~3 GB",
                "accuracy": 5,
                "speed": "~3x faster than realtime",
                "description": "Maximum accuracy, all languages"
            },
            "large-v3": {
                "name": "Large V3",
                "size": "~3 GB",
                "accuracy": 5,
                "speed": "Fast (GPU Recommended)",
                "description": "Best accuracy and speed on GPU"
            },
            "large-v3-turbo": {
                "name": "Large V3 Turbo",
                "size": "~1.5 GB",
                "accuracy": 4,
                "speed": "~2x faster than large-v3",
                "description": "Faster with good accuracy"
            }
        }

    @property
    def available_diarization_models(self) -> dict:
        return {
            "pyannote/speaker-diarization-3.1": {
                "name": "Speaker Diarization 3.1",
                "description": "Latest, highest accuracy"
            },
            "pyannote/speaker-diarization-3.0": {
                "name": "Speaker Diarization 3.0",
                "description": "Previous version, slightly lighter"
            },
            "pyannote/speaker-diarization": {
                "name": "Speaker Diarization 2.x",
                "description": "Base version, minimal resources"
            }
        }
    
    @property
    def uploads_dir(self) -> str:
        return os.path.join(self.data_dir, "uploads")
    
    @property
    def downloads_dir(self) -> str:
        return os.path.join(self.data_dir, "downloads")
    
    @property
    def temp_dir(self) -> str:
        return os.path.join(self.data_dir, "temp")
    
    @property
    def output_dir(self) -> str:
        return os.path.join(self.data_dir, "output")
    
    @property
    def db_path(self) -> str:
        return os.path.join(self.data_dir, "whisper.db")
    
    @property
    def config_path(self) -> str:
        return os.path.join(self.data_dir, "config.json")
    
    class Config:
        env_prefix = ""
        case_sensitive = False

settings = Settings()

# Ensure directories exist
for dir_path in [settings.uploads_dir, settings.downloads_dir, settings.temp_dir, settings.output_dir, settings.models_dir]:
    os.makedirs(dir_path, exist_ok=True)
