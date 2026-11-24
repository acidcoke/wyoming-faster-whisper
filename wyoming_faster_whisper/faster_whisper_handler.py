"""Event handler for clients of the server."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import faster_whisper

from .const import Transcriber
from .device import ctranslate2_device, is_gpu
from .languages import whisper_language

_LOGGER = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    """Fall back to the CPU, with a warning, when CTranslate2 has no CUDA.

    The CTranslate2 wheels on PyPI are CUDA-enabled only for x86_64; on arm64
    (e.g. NVIDIA GB10 / DGX Spark machines) they are CPU-only, and asking for
    "cuda" raises at model load. Other backends in the same process may still
    use the GPU, so --device cuda should not be fatal here.
    """
    if not is_gpu(device):
        return device

    try:
        import ctranslate2  # pylint: disable=import-outside-toplevel

        if ctranslate2.get_cuda_device_count() > 0:
            return device
    except Exception:  # pylint: disable=broad-exception-caught
        return device

    _LOGGER.warning(
        "Device '%s' was requested but this CTranslate2 build has no CUDA "
        "support (or no GPU is visible); running faster-whisper on the CPU.",
        device,
    )
    return "cpu"


class FasterWhisperTranscriber(Transcriber):
    """Event handler for clients."""

    def __init__(
        self,
        model_id: str,
        cache_dir: Union[str, Path],
        local_files_only: bool = False,
        device: str = "cpu",
        compute_type: str = "default",
        cpu_threads: int = 4,
        vad_parameters: Optional[Dict[str, Any]] = None,
        task: Optional[str] = None,
    ) -> None:
        self.vad_filter = vad_parameters is not None
        self.vad_parameters = vad_parameters
        self.task = task

        # CTranslate2 takes the GPU ordinal separately, so "cuda:1" has to be
        # split into device + device_index.
        ct2_device, ct2_device_index = ctranslate2_device(_resolve_device(device))
        extra_args: Dict[str, Any] = {}
        if ct2_device_index is not None:
            extra_args["device_index"] = ct2_device_index

        self.model = faster_whisper.WhisperModel(
            model_id,
            download_root=str(cache_dir),
            local_files_only=local_files_only,
            device=ct2_device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            **extra_args,
        )

    def count_prompt_tokens(self, text: str) -> Optional[int]:
        tokenizer = getattr(self.model, "hf_tokenizer", None)
        if tokenizer is None:
            return None

        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    def transcribe(
        self,
        wav_path: Union[str, Path],
        language: Optional[str],
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> str:

        kwargs = {
            "beam_size": beam_size,
            "language": whisper_language(language),
            "initial_prompt": initial_prompt,
            "vad_filter": self.vad_filter,
            "vad_parameters": self.vad_parameters,
        }
        if self.task:
            kwargs["task"] = self.task

        segments, _info = self.model.transcribe(str(wav_path), **kwargs)
        text = " ".join(segment.text for segment in segments)
        return text
