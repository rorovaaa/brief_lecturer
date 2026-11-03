import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PACKAGES = [
    "torch",
    "torchaudio",
    "pyannote.audio",
    "faster-whisper",
    "llama-cpp-python",
    "pyyaml",
    "python-dotenv",
    "huggingface_hub",
    "av",
]


def install_libraries():
    subprocess.check_call([sys.executable, "-m", "pip", "install", *PACKAGES])


def download_models():
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import hf_hub_download, snapshot_download

    token = os.getenv("HF_TOKEN") or os.getenv("TOKEN")

    models_dir = BASE_DIR / "models"
    whisper_dir = models_dir / "fasterwhisper"
    qwen_dir = models_dir / "qwen"
    seg_dir = models_dir / "segmentation"
    emb_dir = models_dir / "embedding"

    for d in (whisper_dir, qwen_dir, seg_dir, emb_dir):
        d.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id="Systran/faster-whisper-base",
        local_dir=str(whisper_dir),
    )

    hf_hub_download(
        repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
        filename="qwen2.5-3b-instruct-q4_k_m.gguf",
        local_dir=str(qwen_dir),
    )

    try:
        snapshot_download(
            repo_id="pyannote/segmentation-3.0",
            local_dir=str(seg_dir),
            token=token,
        )
        snapshot_download(
            repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
            local_dir=str(emb_dir),
            token=token,
        )
    except Exception:
        pass

    config_yaml = models_dir / "config.yaml"
    if not config_yaml.exists():
        content = f"""version: 3.1.0

pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    clustering: AgglomerativeClustering
    embedding: {emb_dir}/pytorch_model.bin
    embedding_batch_size: 32
    embedding_exclude_overlap: true
    segmentation: {seg_dir}/pytorch_model.bin
    segmentation_batch_size: 32

params:
  clustering:
    method: centroid
    min_cluster_size: 12
    threshold: 0.7045654963945799
  segmentation:
    min_duration_off: 0.0
"""
        config_yaml.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    install_libraries()
    download_models()
