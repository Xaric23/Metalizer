from __future__ import annotations

import shutil
from pathlib import Path

import librosa
import numpy as np
import pytest
import soundfile as sf

from app.services.processor import AnalysisResult, MetalizerPipeline, PipelineContext, ProcessingConfig


@pytest.fixture()
def sample_audio(tmp_path: Path) -> Path:
    sr = 44100
    duration = 4
    times = np.arange(0, duration, 0.5)
    audio = librosa.clicks(times=times, sr=sr, click_duration=0.03, length=sr * duration)
    path = tmp_path / "sample.wav"
    sf.write(path, audio, sr)
    return path


@pytest.fixture()
def pipeline(tmp_path: Path) -> MetalizerPipeline:
    config = ProcessingConfig(
        output_dir=tmp_path / "outputs",
        assets_dir=tmp_path / "assets",
    )
    return MetalizerPipeline(config=config)


def test_analyze_detects_key_and_bpm(pipeline: MetalizerPipeline, sample_audio: Path) -> None:
    result = pipeline._analyze(sample_audio)
    assert isinstance(result, AnalysisResult)
    assert result.bpm > 0
    assert result.key.endswith("Major") or result.key.endswith("Minor")


def test_process_generates_export(monkeypatch: pytest.MonkeyPatch, pipeline: MetalizerPipeline, sample_audio: Path) -> None:
    def stub_separate(self: MetalizerPipeline, normalized_path: Path, ctx: PipelineContext):  # type: ignore[override]
        vocals = ctx.allocate("vocals.wav")
        accompaniment = ctx.allocate("accompaniment.wav")
        shutil.copyfile(sample_audio, vocals)
        shutil.copyfile(sample_audio, accompaniment)
        return {"vocals": vocals, "accompaniment": accompaniment}

    monkeypatch.setattr(MetalizerPipeline, "_separate", stub_separate)

    output_path = pipeline.process(sample_audio)
    assert output_path.exists()
    assert output_path.suffix == ".wav"