import warnings
warnings.filterwarnings("ignore")

import gc
import json
from pathlib import Path
import sys
import torch

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.audio_prep import AudioPreprocessing
from core.audio_step2 import Audiostep2
from core.LLM_nteretion import LLM

input_file = BASE_DIR / "test_sounf" / "test.mp4"
output_file = BASE_DIR / "test_sounf" / "last.wav"
json_path = BASE_DIR / "test_sounf" / "dialogue.json"
res = []

c = AudioPreprocessing()
c.clear()

if c.convert_audio(input_file, output_file):
    result = c.speakers(output_file)
    del c
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    s = Audiostep2()
    wisper_res = s.text(audio_path=output_file)
    del s
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for sp in wisper_res:
        text = sp["text"]
        text_start = sp["time"][0]
        text_end = sp["time"][1]
        centre = (text_start + text_end) / 2

        for i in result:
            spk_name = i["speaker"]
            spk_start = i["time"][0]
            spk_end = i["time"][1]

            if spk_start <= centre <= spk_end:
                res.append({
                    "speaker": spk_name,
                    "text": text,
                    "start": round(text_start, 2),
                    "end": round(text_end, 2),
                })
                break
                break

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)

    l = LLM()
    l.LLM(json_path=json_path)
    l.quation()
    del l
    gc.collect()
