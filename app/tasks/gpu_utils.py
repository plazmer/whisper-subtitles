"""GPU / CUDA helpers: device resolution, diagnostics, volume checks."""
import os
import time
from typing import Tuple


def resolve_device(requested: str = "auto") -> Tuple[str, dict]:
    """Determine the actual compute device and collect GPU diagnostics.

    Returns ``(device, diagnostics)`` where *diagnostics* is a dict with keys
    like ``gpu_name``, ``vram_total_mb``, ``fallback``, etc.
    """
    import torch

    diag: dict = {
        "requested": requested,
        "resolved": None,
        "gpu_name": None,
        "vram_total_mb": None,
        "vram_free_mb": None,
        "cuda_version": getattr(torch.version, "cuda", None),
        "driver_version": None,
        "torch_version": torch.__version__,
        "fallback": False,
        "fallback_reason": None,
    }

    cuda_available = torch.cuda.is_available()

    if requested == "cpu":
        diag["resolved"] = "cpu"
        return "cpu", diag

    if requested == "auto":
        device = "cuda" if cuda_available else "cpu"
    else:
        device = requested

    if device == "cuda" and not cuda_available:
        diag["resolved"] = "cpu"
        diag["fallback"] = True
        diag["fallback_reason"] = "CUDA not available"
        return "cpu", diag

    if device == "cuda":
        try:
            props = torch.cuda.get_device_properties(0)
            diag["gpu_name"] = props.name
            diag["vram_total_mb"] = round(props.total_memory / (1024 * 1024))
        except Exception:
            pass
        try:
            free, total = torch.cuda.mem_get_info(0)
            diag["vram_free_mb"] = round(free / (1024 * 1024))
        except Exception:
            pass
        try:
            major, minor = torch.cuda.get_device_capability(0)
            diag["driver_version"] = f"compute {major}.{minor}"
        except Exception:
            pass

    diag["resolved"] = device
    return device, diag


def format_device_message(prefix: str, diag: dict) -> str:
    """Human-readable status message, e.g. 'Transcribing on CUDA (RTX 3060, 10.5 GB free)'."""
    device = diag.get("resolved", "cpu")
    if device == "cuda":
        parts = [prefix, "on CUDA"]
        gpu = diag.get("gpu_name")
        free = diag.get("vram_free_mb")
        extra = []
        if gpu:
            extra.append(gpu)
        if free is not None:
            extra.append(f"{free / 1024:.1f} GB free")
        if extra:
            parts.append(f"({', '.join(extra)})")
        return " ".join(parts)

    if diag.get("fallback"):
        return f"{prefix} on CPU (CUDA fallback: {diag.get('fallback_reason', 'unknown')})"
    return f"{prefix} on CPU"


# ---------------------------------------------------------------------------
# Volume checks
# ---------------------------------------------------------------------------

def check_volumes(models_path: str = "/app/models") -> dict:
    """Check that model directories are writable persistent volumes."""
    hf_path = os.path.join(models_path, "huggingface")
    os.makedirs(hf_path, exist_ok=True)

    result = {
        "models_path": models_path,
        "writable": False,
        "cached_whisper_models": [],
        "hf_cache_path": hf_path,
        "hf_cache_size_mb": 0,
    }

    probe = os.path.join(models_path, ".probe")
    try:
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        result["writable"] = True
    except OSError:
        pass

    try:
        for entry in os.listdir(models_path):
            full = os.path.join(models_path, entry)
            if entry == "huggingface" or entry.startswith("."):
                continue
            if os.path.isdir(full) or entry.endswith(".pt"):
                result["cached_whisper_models"].append(entry)
    except OSError:
        pass

    result["hf_cache_size_mb"] = _dir_size_mb(hf_path)
    return result


def _dir_size_mb(path: str) -> int:
    total = 0
    try:
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    except OSError:
        pass
    return round(total / (1024 * 1024))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test(device: str = "cuda", models_path: str = "/app/models") -> dict:
    """Load whisper tiny model and transcribe synthetic audio. Returns result dict."""
    import numpy as np

    result = {
        "status": "skipped",
        "model": "tiny",
        "duration_sec": None,
        "device_used": device,
        "error": None,
    }

    try:
        import whisper
        import tempfile
        import soundfile as sf

        sr = 16000
        duration_s = 5
        t = np.linspace(0, duration_s, sr * duration_s, dtype=np.float32)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            sf.write(tmp.name, audio, sr)
            tmp.close()

            start = time.time()
            model = whisper.load_model("tiny", device=device, download_root=models_path)
            model.transcribe(tmp.name)
            elapsed = time.time() - start

            result["status"] = "passed"
            result["duration_sec"] = round(elapsed, 2)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)

    return result
