from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class OnnxTTSAdapter:
    """Generic ONNX TTS adapter for Hydra quality benchmark.

    This adapter assumes the ONNX model exposes a text input and returns audio samples.
    Configure input/output names in the Hydra config if the model signature differs.
    """

    def __init__(
        self,
        onnx_model_path: str | Path,
        language_map: dict[str, str] | None = None,
        sample_rate: int = 16000,
        device: str = "cpu",
        input_text_name: str | None = None,
        language_input_name: str | None = None,
        output_audio_name: str | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "Package `onnxruntime` is required for OnnxTTSAdapter. Install it first."
            ) from exc

        self.language_map = language_map or {}
        self.sample_rate = sample_rate
        self.onnx_model_path = Path(onnx_model_path)
        self.device = device

        providers = ["CPUExecutionProvider"]
        if device.lower() in {"cuda", "gpu"}:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._session = ort.InferenceSession(str(self.onnx_model_path), providers=providers)

        self._input_text_name = input_text_name or self._find_string_input_name()
        self._language_input_name = language_input_name
        self._output_audio_name = output_audio_name or self._find_audio_output_name()

    def _find_string_input_name(self) -> str:
        for inp in self._session.get_inputs():
            if inp.type.endswith("string"):
                return inp.name
        # Fallback to first input if no explicit string type is found.
        return self._session.get_inputs()[0].name

    def _find_audio_output_name(self) -> str:
        outputs = self._session.get_outputs()
        if self._output_audio_name is not None:
            return self._output_audio_name
        return outputs[0].name

    def _normalize_audio(self, audio: Any) -> np.ndarray:
        if isinstance(audio, list):
            audio = np.asarray(audio, dtype=np.float32)
        elif isinstance(audio, np.ndarray):
            audio = audio.astype(np.float32)
        else:
            audio = np.array(audio, dtype=np.float32)
        return audio

    def generate(self, text: str, language: str) -> tuple[np.ndarray, int]:
        inputs: dict[str, Any] = {}
        if self._input_text_name is not None:
            inputs[self._input_text_name] = np.array([text], dtype=object)
        if self._language_input_name is not None:
            inputs[self._language_input_name] = np.array([language], dtype=object)

        outputs = self._session.run([self._output_audio_name], inputs)
        audio = outputs[0]
        audio = self._normalize_audio(audio)

        # If model returns [batch, time] or [[time]], unpack first batch.
        if audio.ndim > 1 and audio.shape[0] == 1:
            audio = audio[0]

        return audio, int(self.sample_rate)
