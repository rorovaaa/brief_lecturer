#Brief Lecturer
# Audio Dialogue Extraction & Summarization

Local offline pipeline that extracts audio from video/audio files, diarizes speakers with Pyannote, transcribes speech with Faster-Whisper, aligns dialogue turns, and generates summaries or answers questions using a local Qwen GGUF model via llama.cpp.

## Pipeline Overview

1. FFmpeg extracts audio to 16 kHz mono WAV.
2. Pyannote (`segmentation-3.0` + `wespeaker-voxceleb-resnet34-LM`) determines speaker intervals.
3. Faster-Whisper (`faster-whisper-base`) transcribes speech segments.
4. Aligned turns match text midpoint against speaker intervals and export to `test_sounf/dialogue.json`.
5. `llama-cpp-python` loads `qwen2.5-3b-instruct-q4_k_m.gguf` to summarize the dialogue and answer console queries.

GPU and CPU memory are cleared between steps via `gc.collect()` and `torch.cuda.empty_cache()` so the full pipeline can run on limited VRAM.

## Prerequisites

- Linux or macOS (WSL for Windows)
- Python 3.10+
- System FFmpeg installed
- Hugging Face token with access to `pyannote/segmentation-3.0`

## Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
python3 -m venv .venv
source .venv/bin/activate
```

Export your Hugging Face token and run `install.py` to fetch dependencies and weights:

```bash
export HF_TOKEN="hf_..."
python install.py
```

`install.py` populates `models/` with `faster-whisper-base`, `segmentation-3.0`, `wespeaker`, and `qwen2.5-3b-instruct-q4_k_m.gguf`, and generates `models/config.yaml`.

## Running

Place your input recording at `test_sounf/test.mp4` (or adjust the path in `app.py`), then run:

```bash
python app.py
```

The script writes dialogue turns to `test_sounf/dialogue.json`, prints a summary, and waits for questions in the terminal. Enter `стоп` to exit.

## Repository Layout

```text
├── app.py                 # Main execution script
├── install.py             # Dependency installer and model downloader
├── LICENSE                # MIT License
├── core/
│   ├── audio_prep.py      # Audio conversion and speaker diarization
│   ├── audio_step2.py     # Speech transcription
│   └── LLM_nteretion.py   # Llama.cpp Qwen interface
├── models/                # Downloaded weights and config.yaml
└── test_sounf/            # Media files and dialogue.json
```

## License

MIT. See [LICENSE](LICENSE).
# brief_lecturer
# brief_lecturer
