from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any, Dict, Literal

import librosa
import numpy as np
import soundfile as sf
from loguru import logger
from pydub import AudioSegment
from pydub.generators import Sine, WhiteNoise
from scipy import signal

from app.core.config import get_settings

try:  # Heavy dependencies are imported lazily to keep startup fast
    from spleeter.separator import Separator
except Exception:  # pragma: no cover - exercised only when spleeter missing
    Separator = None  # type: ignore


class SeparationError(RuntimeError):
    """Raised when both separation backends fail."""


@dataclass
class AnalysisResult:
    bpm: float
    beat_grid: list[float]
    key: str
    tuning_offset: float


@dataclass
class ProcessingConfig:
    output_dir: Path = get_settings().output_dir
    assets_dir: Path = get_settings().assets_dir
    drums_bank: Path = get_settings().assets_dir / "drums"
    midi_bank: Path = get_settings().assets_dir / "midi"
    guitars_bank: Path = get_settings().assets_dir / "guitars"
    sample_rate: int = 44100


@dataclass
class PipelineContext:
    session_dir: Path

    def allocate(self, filename: str) -> Path:
        path = self.session_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


class MetalizerPipeline:
    """High-level orchestration for the Metalizer audio transformation."""

    def __init__(
        self,
        config: ProcessingConfig | None = None,
        separator_engine: Literal["spleeter", "demucs"] = "spleeter",
        spleeter_model: str = "spleeter:2stems",
        demucs_model: str = "htdemucs",
        demucs_device: str | None = None,
    ) -> None:
        self.config = config or ProcessingConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.separator_engine = separator_engine
        self.spleeter_model = spleeter_model
        self.demucs_model = demucs_model
        self.demucs_device = demucs_device or self._auto_select_device()
        self.sample_rate = self.config.sample_rate
        self._spleeter_separator: Separator | None = None
        self._demucs_model = None
        self._demucs_lock = Lock()

    def process(self, audio_path: Path) -> Path:
        """Full end-to-end pipeline returning the exported metal mix."""

        logger.info("Starting Metalizer pipeline", path=audio_path)
        with TemporaryDirectory(prefix="metalizer_") as tmp:
            ctx = PipelineContext(session_dir=Path(tmp))
            normalized_path = self._ingest(audio_path, ctx)
            stems = self._separate(normalized_path, ctx)
            analysis = self._analyze(stems["accompaniment"])
            transformed = self._transform(stems, analysis, ctx)
            mix = self._mix(stems["vocals"], transformed, ctx)
            export_path = self._export(mix, source_path=audio_path)
        logger.info("Pipeline complete", output=export_path)
        return export_path

    def _ingest(self, audio_path: Path, ctx: PipelineContext) -> Path:
        """Load and normalize the uploaded audio file."""

        logger.debug("Ingesting audio", path=audio_path)
        audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=False)
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)
        normalized = librosa.util.normalize(audio, axis=1)
        normalized_path = ctx.allocate("normalized.wav")
        sf.write(normalized_path, normalized.T, self.sample_rate)
        return normalized_path

    def _separate(self, audio_path: Path, ctx: PipelineContext) -> Dict[str, Path]:
        """Run stem separation (spleeter/demucs) and return file handles."""

        logger.debug("Separating stems", path=audio_path, engine=self.separator_engine)
        separator_errors: list[str] = []
        if self.separator_engine == "demucs":
            try:
                return self._separate_with_demucs(audio_path, ctx)
            except Exception as err:  # pragma: no cover - exercised when demucs missing
                logger.warning("Demucs separation failed, falling back to Spleeter", error=str(err))
                separator_errors.append(str(err))

        try:
            return self._separate_with_spleeter(audio_path, ctx)
        except Exception as err:
            separator_errors.append(str(err))
            logger.error("Spleeter separation failed", error=str(err))
            raise SeparationError(
                "Unable to separate stems with any configured engine: " + "; ".join(separator_errors)
            ) from err

    def _separate_with_spleeter(self, audio_path: Path, ctx: PipelineContext) -> Dict[str, Path]:
        if Separator is None:
            raise ImportError("spleeter is not installed")

        if self._spleeter_separator is None:
            self._spleeter_separator = Separator(self.spleeter_model)

        logger.debug("Running Spleeter", model=self.spleeter_model)
        prediction = self._spleeter_separator.separate(str(audio_path))
        vocals_path = ctx.allocate("vocals.wav")
        accomp_path = ctx.allocate("accompaniment.wav")
        sf.write(vocals_path, prediction["vocals"], self.sample_rate)
        sf.write(accomp_path, prediction["accompaniment"], self.sample_rate)
        return {"vocals": vocals_path, "accompaniment": accomp_path}

    def _separate_with_demucs(self, audio_path: Path, ctx: PipelineContext) -> Dict[str, Path]:
        import torch
        from demucs.apply import apply_model
        from demucs.audio import AudioFile
        from demucs.pretrained import get_model

        logger.debug("Running Demucs", model=self.demucs_model, device=self.demucs_device)

        with self._demucs_lock:
            if self._demucs_model is None:
                model = get_model(self.demucs_model)
                model.to(self.demucs_device)
                model.eval()
                self._demucs_model = model

        wav = AudioFile(str(audio_path)).read(streams=0, samplerate=self.sample_rate)
        wav = wav.mean(0)
        wav = wav - wav.mean()
        tensor = torch.tensor(wav, dtype=torch.float32, device=self.demucs_device)[None]
        with torch.no_grad():
            sources = apply_model(self._demucs_model, tensor, shifts=1, overlap=0.5, progress=False)[0]
        sources = sources.cpu().numpy()
        source_map = {name: sources[idx] for idx, name in enumerate(self._demucs_model.sources)}
        vocals = source_map.get("vocals")
        if vocals is None:
            raise RuntimeError("Demucs model did not return a 'vocals' stem")
        accompaniment_sources = [source_map[name] for name in self._demucs_model.sources if name != "vocals"]
        accompaniment = np.sum(accompaniment_sources, axis=0)
        vocals_path = ctx.allocate("vocals.wav")
        accomp_path = ctx.allocate("accompaniment.wav")
        sf.write(vocals_path, vocals.T, self.sample_rate)
        sf.write(accomp_path, accompaniment.T, self.sample_rate)
        return {"vocals": vocals_path, "accompaniment": accomp_path}

    def _analyze(self, accompaniment_path: Path) -> AnalysisResult:
        """Detect BPM, beat grid, key and tuning with librosa."""

        logger.debug("Analyzing accompaniment", path=accompaniment_path)
        audio, sr = librosa.load(accompaniment_path, sr=self.sample_rate)
        tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
        beat_grid = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        key = self._estimate_key(chroma)
        tuning_offset = float(librosa.estimate_tuning(y=audio, sr=sr) * 12)
        bpm = float(np.atleast_1d(tempo)[0])
        return AnalysisResult(bpm=bpm, beat_grid=beat_grid, key=key, tuning_offset=tuning_offset)

    def _transform(
        self, stems: Dict[str, Path], analysis: AnalysisResult, ctx: PipelineContext
    ) -> Dict[str, Path]:
        """Generate metal drums, distorted guitars, and pitch-shifted accompaniment."""

        logger.debug("Transforming stems", bpm=analysis.bpm, key=analysis.key)
        accompaniment, sr = librosa.load(stems["accompaniment"], sr=self.sample_rate, mono=False)
        tuned = librosa.effects.pitch_shift(accompaniment, sr=sr, n_steps=-2)
        distorted = self._apply_distortion(tuned)
        guitars_path = ctx.allocate("guitars.wav")
        sf.write(guitars_path, distorted.T, sr)
        self._layer_guitar_samples(guitars_path)

        duration_seconds = distorted.shape[-1] / sr
        drums_segment = self._generate_metal_drums(analysis.bpm, duration_seconds)
        drums_path = ctx.allocate("drums.wav")
        drums_segment.export(drums_path, format="wav")

        return {
            "drums": drums_path,
            "guitars": guitars_path,
        }

    def _mix(self, vocals_path: Path, transformed: Dict[str, Path], ctx: PipelineContext) -> Path:
        """Blend the clean vocals with the new instrumental bed."""

        logger.debug("Mixing final arrangement", vocals=vocals_path)
        vocals = AudioSegment.from_file(vocals_path)
        guitars = AudioSegment.from_file(transformed["guitars"])
        drums = AudioSegment.from_file(transformed["drums"])

        instrumental = self._fit_segment(guitars, max(len(guitars), len(vocals)))
        drums_loop = self._fit_segment(drums, len(instrumental))
        instrumental = instrumental.overlay(drums_loop, loop=True)
        vocals_aligned = self._fit_segment(vocals, len(instrumental))
        final_mix = instrumental.overlay(vocals_aligned)

        mix_path = ctx.allocate("mix.wav")
        final_mix.export(mix_path, format="wav")
        return mix_path

    def _export(self, mix_path: Path, source_path: Path) -> Path:
        """Persist the final WAV/MP3 to the outputs directory."""

        output_path = self.config.output_dir / f"{source_path.stem}_metalized.wav"
        logger.debug("Exporting final track", destination=output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data, sr = librosa.load(mix_path, sr=None, mono=False)
        sf.write(output_path, data.T, sr)
        return output_path

    # ------------------------------------------------------------------
    # Helper utilities

    def _apply_distortion(self, audio: np.ndarray, drive: float = 6.0) -> np.ndarray:
        gain = 10 ** (drive / 20)
        processed = np.tanh(audio * gain)
        cutoff = 80 / (self.sample_rate / 2)
        b, a = signal.butter(N=2, Wn=cutoff, btype="highpass")
        if processed.ndim == 1:
            filtered = signal.lfilter(b, a, processed)
        else:
            filtered = np.vstack([signal.lfilter(b, a, ch) for ch in processed])
        return librosa.util.normalize(filtered, axis=-1)

    def _generate_metal_drums(self, bpm: float, duration_seconds: float) -> AudioSegment:
        track = AudioSegment.silent(duration=int(duration_seconds * 1000) + 1000)
        beat_ms = 60000 / max(bpm, 1.0)
        kick = self._load_drum_sample("double_kick.wav", Sine(60).to_audio_segment(duration=120))
        snare = self._load_drum_sample("snare_hit.wav", WhiteNoise().to_audio_segment(duration=100).apply_gain(-10))
        hat = self._load_drum_sample("hat_tick.wav", WhiteNoise().to_audio_segment(duration=80).apply_gain(-20))

        position = 0.0
        while position < len(track):
            track = track.overlay(kick, position=int(position))
            track = track.overlay(kick, position=int(position + beat_ms / 4))
            track = track.overlay(snare, position=int(position + beat_ms / 2))
            track = track.overlay(hat, position=int(position))
            track = track.overlay(hat, position=int(position + beat_ms / 3))
            position += beat_ms

        if track.max_dBFS > 0:
            track = track.apply_gain(-track.max_dBFS)
        return track

    def _load_drum_sample(self, filename: str, fallback: AudioSegment) -> AudioSegment:
        sample_path = self.config.drums_bank / filename
        if sample_path.exists():
            try:
                return AudioSegment.from_file(sample_path)
            except Exception as err:
                logger.warning("Failed to read drum sample, using fallback", path=sample_path, error=str(err))
        return fallback

    def _layer_guitar_samples(self, guitars_path: Path) -> None:
        sample_path = self.config.guitars_bank / "chug_riff.wav"
        if not sample_path.exists():
            return
        try:
            base = AudioSegment.from_file(guitars_path)
            layer = AudioSegment.from_file(sample_path)
        except Exception as err:
            logger.warning("Failed to layer guitar sample", error=str(err))
            return

        layer_loop = self._fit_segment(layer, len(base))
        blended = base.overlay(layer_loop - 3)
        blended.export(guitars_path, format="wav")

    def _fit_segment(self, segment: AudioSegment, target_ms: int) -> AudioSegment:
        if len(segment) == target_ms:
            return segment
        if len(segment) > target_ms:
            return segment[:target_ms]
        looped = AudioSegment.silent(duration=0)
        while len(looped) < target_ms:
            looped += segment
        return looped[:target_ms]

    def _estimate_key(self, chroma: np.ndarray) -> str:
        krumhansl_major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        krumhansl_minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        chroma_vector = chroma.mean(axis=1)
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        best_key = "C"
        best_mode = "Major"
        best_score = -np.inf
        for i, key in enumerate(keys):
            major_score = np.dot(np.roll(krumhansl_major, i), chroma_vector)
            minor_score = np.dot(np.roll(krumhansl_minor, i), chroma_vector)
            if major_score > best_score:
                best_score = major_score
                best_key = key
                best_mode = "Major"
            if minor_score > best_score:
                best_score = minor_score
                best_key = key
                best_mode = "Minor"
        return f"{best_key} {best_mode}"

    def _auto_select_device(self) -> str:
        try:  # pragma: no cover - torch availability depends on environment
            import torch
        except Exception:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
