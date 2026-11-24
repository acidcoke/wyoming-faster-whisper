"""Tests for faster-whisper's CPU fallback when CTranslate2 has no CUDA."""

import sys
import types

import pytest

from wyoming_faster_whisper.faster_whisper_handler import _resolve_device


def _fake_ctranslate2(monkeypatch: pytest.MonkeyPatch, cuda_devices: int) -> None:
    module = types.ModuleType("ctranslate2")
    module.get_cuda_device_count = lambda: cuda_devices  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ctranslate2", module)


def test_cpu_is_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_ctranslate2(monkeypatch, 0)
    assert _resolve_device("cpu") == "cpu"


def test_cuda_kept_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_ctranslate2(monkeypatch, 1)
    assert _resolve_device("cuda:1") == "cuda:1"


def test_cuda_falls_back_without_cuda_build(monkeypatch: pytest.MonkeyPatch) -> None:
    # e.g. the CPU-only CTranslate2 wheel on arm64 (NVIDIA GB10 machines)
    _fake_ctranslate2(monkeypatch, 0)
    assert _resolve_device("cuda") == "cpu"
