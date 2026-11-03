from pathlib import Path
import torch
import av


_orig_av_open = av.open


def _compat_av_open(*args, **kwargs):
    kwargs.pop("metadata_errors", None)
    return _orig_av_open(*args, **kwargs)


av.open = _compat_av_open

from faster_whisper import WhisperModel

BASE_DIR = Path(__file__).resolve().parent.parent


class Audiostep2:
    def __init__(self, model_path: Path | str | None = None):
        if model_path is None:
            model_path = BASE_DIR / "models" / "fasterwhisper"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if self.device == "cuda" else "int8"
        self.model = WhisperModel(str(model_path), device=self.device, compute_type=compute_type)

    def text(self, audio_path: Path | str):
        segments, info = self.model.transcribe(str(audio_path), beam_size=5)
        wisper_res = []

        for segment in segments:
            wisper_res.append({
                "text": segment.text,
                "time": [segment.start, segment.end]
            })
        return wisper_res
