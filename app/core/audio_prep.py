import os
import subprocess
from pathlib import Path
import yaml
import torch
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.core.plda import PLDA

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

BASE_DIR = Path(__file__).resolve().parent.parent


class AudioPreprocessing:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        seg_dir = BASE_DIR / "models" / "segmentation"
        emb_dir = BASE_DIR / "models" / "embedding"
        cfg_file = BASE_DIR / "models" / "config.yaml"
        self.file = BASE_DIR / "test_sounf" / "last.wav"
        self.json = BASE_DIR / "test_sounf" / "dialogue.json"
        dummy_plda = object.__new__(PLDA)
        self.pipeline = SpeakerDiarization(
            segmentation=str(seg_dir),
            embedding=str(emb_dir),
            clustering="AgglomerativeClustering",
            plda=dummy_plda,
            legacy=True,
        )

        if cfg_file.exists():
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                if "params" in cfg:
                    self.pipeline.instantiate(cfg["params"])

        self.pipeline.to(self.device)
    def clear(self):
        if os.path.isfile(self.file):
            os.remove(self.file)
        if os.path.isfile(self.json):
            os.remove(self.json)

        

    def convert_audio(self, input_file, output_file) :
        command = [
            "ffmpeg", "-y",
            "-i", str(input_file),
            "-ar", "16000",
            "-ac", "1",
            str(output_file),
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError as e:
            return False

    def speakers(self, audio_path):
        diarization = self.pipeline(str(audio_path))
        result = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            result.append({
                "speaker":speaker,
                "time":[turn.start, turn.end]
            })
        return result 
