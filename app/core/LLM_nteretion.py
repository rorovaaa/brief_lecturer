import os
import atexit
from pathlib import Path
import torch
from llama_cpp import Llama

BASE_DIR = Path(__file__).resolve().parent.parent


class LLM:
    def __init__(self):
        self.qwen_path = BASE_DIR / "models" / "qwen" / "qwen2.5-3b-instruct-q4_k_m.gguf"
        self.file_content = ""
        is_cuda = torch.cuda.is_available()

        if is_cuda:
            self.llm = Llama(
                model_path=str(self.qwen_path),
                n_gpu_layers=36,
                n_ctx=8192,
                n_threads=os.cpu_count(),
                flash_attn=True,
                verbose=False,
            )
        else:
            self.llm = Llama(
                model_path=str(self.qwen_path),
                n_gpu_layers=0,
                n_ctx=4096,
                n_threads=os.cpu_count(),
                verbose=False,
            )

        atexit.register(self.close)

    def close(self):
        if hasattr(self, "llm") and self.llm is not None:
            try:
                self.llm.close()
            except Exception:
                pass

    def __del__(self):
        self.close()

    def summarize(self, json_path: Path | str) -> str:
        with open(json_path, "r", encoding="utf-8") as f:
            self.file_content = f.read()

        system_prompt = (
            "Твоя цель обработать Json файл лекции и дать пользователю основную информацию, "
            "которую вещают спикеры. Отвечай на языке, на котором велась лекция."
        )

        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self.file_content},
            ],
            max_tokens=2000,
            temperature=0.2,
        )

        output = response["choices"][0]["message"]["content"]
        print(output)
        return output

    def LLM(self, json_path: Path | str):
        return self.summarize(json_path)

    def quation(self):
        while True:
            prompt = input("Введите запрос (stop для выхода): ").strip()
            if prompt.lower() == "stop":
                break
            response = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "Ты ассистент по материалам лекции. Отвечай на вопросы пользователя на основе предоставленного диалога.\n\n" + self.file_content,
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=0.2,
            )
            print(response["choices"][0]["message"]["content"])
