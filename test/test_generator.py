import pytest

from q.encoder import load_encoder
from q.generation import TokenGenerator
from q.gpt2 import GPT2Model
from q.params import load_hparams_and_params


@pytest.fixture(scope="session")
def model():
    hparams, params = load_hparams_and_params()
    return GPT2Model(params, hparams)


@pytest.fixture(scope="session")
def generator(model):
    return TokenGenerator(model)


def test_init(generator):
    assert generator.model is not None
    assert isinstance(generator.model, GPT2Model)
