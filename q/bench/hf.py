import os

import psutil
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.generation.streamers import BaseStreamer

from .base import BenchTextGeneration, BenchTextGenerationOutput


class HFBenchTextGeneration(BenchTextGeneration):
    """Class to represent a Hugging Face transformer model."""

    model_name: str
    model: AutoModelForCausalLM
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="mps",
        )

    def generate(self, prompt: str, max_length: int) -> BenchTextGenerationOutput:
        streamer = MemoryProfileStreamer()

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to("mps")  # type: ignore
        generated_tokens = self.model.generate(  # type: ignore
            input_ids=input_ids,
            max_length=max_length,
            temperature=1.0,
            streamer=streamer,
        )

        generated_text = self.tokenizer.decode(
            generated_tokens[0], skip_special_tokens=True
        )

        token_count = generated_tokens.shape[1] - input_ids.shape[1]
        return BenchTextGenerationOutput(
            generated_text=generated_text,
            token_count=token_count,
            peak_memory=streamer.peak_memory,
        )


class MemoryProfileStreamer(BaseStreamer):
    peak_memory: float = 0.0

    def put(self, value):
        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss / (1024 * 1024)
        self.peak_memory = max(self.peak_memory, rss)

    def end(self):
        pass
