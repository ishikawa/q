import os

import psutil

from q.common import ModelSize
from q.encoder import load_encoder
from q.generation import TokenGenerator
from q.gpt2 import GPT2Model
from q.params import load_hparams_and_params

from .base import BenchTextGeneration, BenchTextGenerationOutput

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class QBenchTextGeneration(BenchTextGeneration):

    def __init__(
        self,
        model_size: ModelSize = "124M",
        models_dir: str = "models",
    ):
        self.encoder = load_encoder(model_size, models_dir)
        hparams, params = load_hparams_and_params(
            model_size=model_size,
            models_dir=models_dir,
        )

        self.model = GPT2Model(params, hparams)
        self.generator = TokenGenerator(self.model)

    def generate(self, prompt: str, max_length: int) -> BenchTextGenerationOutput:
        peak_memory = 0.0
        output_ids = []

        input_ids = self.encoder.encode(prompt)

        for token in self.generator(input_ids, max_length=max_length):
            process = psutil.Process(os.getpid())
            rss = process.memory_info().rss / (1024 * 1024)
            peak_memory = max(peak_memory, rss)
            output_ids.append(token)

        generated_text = self.encoder.decode(output_ids)

        token_count = len(output_ids)
        return BenchTextGenerationOutput(
            generated_text=generated_text,
            token_count=token_count,
            peak_memory=peak_memory,
        )
