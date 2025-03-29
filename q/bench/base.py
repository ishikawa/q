from dataclasses import dataclass


@dataclass
class BenchTextGenerationOutput:
    generated_text: str
    token_count: int
    peak_memory: float


class BenchTextGeneration:
    def generate(self, prompt: str, max_length: int) -> BenchTextGenerationOutput:
        raise NotImplementedError()
