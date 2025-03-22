import pytest

from q.generation import TokenGenerator
from q.gpt2 import GPT2Model


@pytest.fixture(scope="session")
def generator(model):
    return TokenGenerator(model)


def test_init(generator):
    assert generator.model is not None
    assert isinstance(generator.model, GPT2Model)
