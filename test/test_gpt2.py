import os

import mlx.core as mx
import pytest

from q.gpt2 import GPT2Model
from q.params import load_hparams_and_params

models_dir = os.path.join(os.path.dirname(__file__), "..", "models")


@pytest.fixture
def gpt2_hparams_and_params():
    """Fixture that loads hparams and params for the specified model size."""
    return load_hparams_and_params(model_size="124M", models_dir=models_dir)


@pytest.fixture
def gpt2_model(gpt2_hparams_and_params):
    """Fixture that creates a GPT2Model instance with the loaded parameters."""
    hparams, params = gpt2_hparams_and_params
    return GPT2Model(params, hparams)


def test_logits_shape(gpt2_model, gpt2_hparams_and_params):
    # Unpack the hparams and params
    hparams, _ = gpt2_hparams_and_params

    # Define test input
    inputs = [1, 2, 3]

    # Get logits
    logits = gpt2_model(inputs).logits

    # Assert the shape of logits
    assert logits.shape == (1, len(inputs), hparams["n_vocab"])

    print("Logits shape test passed!")


def test_compute_loss(gpt2_model, gpt2_hparams_and_params):
    # Unpack the hparams
    hparams, _ = gpt2_hparams_and_params

    # Define test input
    inputs = [15496, 2159]

    # Case 1: Test without compute_loss (default behavior)
    output_without_loss = gpt2_model(inputs)
    assert output_without_loss.loss is None

    # Case 2: Test with compute_loss=True
    output_with_loss = gpt2_model(inputs, compute_loss=True)
    assert output_with_loss.loss is not None
    assert output_with_loss.loss.shape == ()  # Scalar loss

    expected_loss = mx.array(8.682313919067383)

    # Check that our computed loss matches the expected loss
    assert mx.allclose(output_with_loss.loss, expected_loss)
