from __future__ import annotations

from typing import Any

import numpy as np
import torch


class QwenTTSAdapter:
    """Thin adapter over qwen_tts model for quality benchmark."""

    def __init__(
        self,
        model_id: str,
        language_map: dict[str, str],
        voice: str,
        gpu_device: str = "cpu",
        dtype: str = "bfloat16",
    ) -> None:
        from qwen_tts import Qwen3TTSModel

        self.model_id = model_id
        self.language_map = language_map
        self.voice = voice
        self.gpu_device = gpu_device
        self.dtype = getattr(torch, dtype, torch.bfloat16)

        use_cuda = gpu_device.isdigit() and torch.cuda.is_available()
        device = {"": int(gpu_device)} if use_cuda else "cpu"
        self._device: Any = device
        if device == "cpu":
            self.model = Qwen3TTSModel.from_pretrained(model_id, dtype=self.dtype)
        else:
            self.model = Qwen3TTSModel.from_pretrained(model_id, device_map=device, dtype=self.dtype)

    def generate(self, text: str, language: str) -> tuple[np.ndarray, int]:
        if self._device != "cpu":
            torch.cuda.synchronize(self._device)
        wavs, sr = self.model.generate_custom_voice(
            text=text,
            language=language,
            speaker=self.voice,
        )
        if self._device != "cpu":
            torch.cuda.synchronize(self._device)
        audio = wavs[0]
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        return audio.astype(np.float32), int(sr)
