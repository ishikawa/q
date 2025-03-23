import mlx.core as mx

from q.gpt2 import GPT2Model


def test_logits_shape(model: GPT2Model):
    hparams = model.hparams
    inputs = mx.array([[1, 2, 3]])

    # Get logits
    logits = model(inputs).logits

    # Assert the shape of logits
    assert logits.shape == (1, 3, hparams["n_vocab"])


def test_compute_loss(model: GPT2Model):
    inputs = mx.array([[15496, 2159]])

    # Case 1: Test without compute_loss (default behavior)
    output_without_loss = model(inputs)
    assert output_without_loss.loss is None

    # Case 2: Test with compute_loss=True
    output_with_loss = model(inputs, compute_loss=True)
    assert output_with_loss.loss is not None
    assert output_with_loss.loss.shape == ()  # Scalar loss

    expected_loss = mx.array(8.682313919067383)

    # Check that our computed loss matches the expected loss
    assert mx.allclose(output_with_loss.loss, expected_loss)


def test_batch_inputs(model: GPT2Model):
    inputs = mx.array([[15496, 2159], [15496, 2159]])

    # Get logits
    logits = model(inputs).logits

    # Assert the shape of logits
    assert logits.shape == (2, 2, model.hparams["n_vocab"])
