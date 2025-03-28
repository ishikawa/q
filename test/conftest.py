import pytest

from q.common import DEFAULT_MODELS_DIR, ModelSize
from q.encoder import Encoder, load_encoder
from q.gpt2 import GPT2Model
from q.params import load_hparams_and_params


@pytest.fixture(scope="session")
def model_size() -> ModelSize:
    return "124M"


@pytest.fixture(scope="session")
def model_dir() -> str:
    return DEFAULT_MODELS_DIR


@pytest.fixture(scope="session")
def encoder(model_dir: str, model_size: ModelSize) -> Encoder:
    return load_encoder(model_size=model_size, models_dir=model_dir)


@pytest.fixture(scope="session")
def model(model_dir: str, model_size: ModelSize) -> GPT2Model:
    hparams, params = load_hparams_and_params(
        model_size=model_size, models_dir=model_dir
    )
    return GPT2Model(params, hparams)
