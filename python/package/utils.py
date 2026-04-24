import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
import onnxruntime as ort
import onnx


LOGGER = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def load_session(model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = True
    options.enable_mem_pattern = True
    
    return ort.InferenceSession(
        path_or_bytes=str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"]
    )


def get_model_size(path_to_model: Path) -> float:
    model = onnx.load(path_to_model)
    return round(model.ByteSize() / (1024 * 1024), 2)


def log_session_io(name: str, session: ort.InferenceSession) -> None:
    LOGGER.info("=== %s ===", name)
    for input_ in session.get_inputs():
        LOGGER.info("input  | %s | shape=%s | type=%s", input_.name, input_.shape, input_.type)
    for output in session.get_outputs():
        LOGGER.info("output | %s | shape=%s | type=%s", output.name, output.shape, output.type)


def save_outputs(out_dir: Path, prefix: str, output_names: list[str], outputs: list[Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, value in zip(output_names, outputs):
        safe_name = name.replace("/", "_").replace(":", "_")
        path = out_dir / f"{prefix}_{safe_name}.npz"
        if isinstance(value, np.ndarray):
            np.savez_compressed(path, value=value)
        else:
            np.savez_compressed(path, value=np.array(value, dtype=object))


def summarize_outputs(prefix: str, output_names: list[str], outputs: list[Any]) -> dict[str, Any]:
    name_to_val = {}
    LOGGER.info("%s outputs:", prefix)
    for name, value in zip(output_names, outputs):
        name_to_val[name] = value
        if isinstance(value, np.ndarray):
            LOGGER.info("  %s: shape=%s dtype=%s", name, value.shape, value.dtype)
        else:
            LOGGER.info("  %s: type=%s", name, type(value))
    return name_to_val


def save_wav(path: Path, signal: np.ndarray, sample_rate: int = 24000) -> None:
    arr = signal
    if arr.ndim == 3: arr = arr[0, 0]
    elif arr.ndim == 2: arr = arr[0]
    elif arr.ndim != 1: raise ValueError(f"Unexpected waveform shape: {signal.shape}")
    sf.write(str(path), arr, sample_rate)
