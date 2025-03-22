import pytest

from q.encoder import Encoder
from q.generation import TokenGenerator
from q.gpt2 import GPT2Model


@pytest.fixture
def generator(model: GPT2Model) -> TokenGenerator:
    return TokenGenerator(model)


@pytest.fixture
def sample_tokens(encoder: Encoder) -> list[int]:
    text = "Hello, world!"
    return encoder.encode(text)


def test_init(generator: TokenGenerator):
    assert generator.model is not None
    assert isinstance(generator.model, GPT2Model)


def test_generate(generator: TokenGenerator, sample_tokens: list[int]):
    generated_output = list(generator(sample_tokens))

    assert generated_output is not None
    assert len(generated_output) > 0


def test_generate_with_max_length(generator: TokenGenerator, sample_tokens: list[int]):
    max_length = 10
    generated_output = list(generator(sample_tokens, max_length=max_length))

    assert generated_output is not None
    assert len(generated_output) == max_length - len(sample_tokens)


def test_generate_with_max_new_tokens(
    generator: TokenGenerator, sample_tokens: list[int]
):
    max_new_tokens = 5
    generated_output = list(generator(sample_tokens, max_new_tokens=max_new_tokens))

    assert generated_output is not None
    assert len(generated_output) == max_new_tokens
