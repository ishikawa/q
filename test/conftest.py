import pytest

from q.common import ModelSize
from q.encoder import load_encoder


@pytest.fixture(scope="session")
def default_model_size() -> ModelSize:
    return "124M"
