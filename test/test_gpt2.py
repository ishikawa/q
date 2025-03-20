import mlx.core as mx
import mlx.nn as nn

from q.gpt2 import GPT2HyperParams, GPT2Model, GPT2Output, GPT2Params


def test_logits_shape():
    # Define dummy hyperparameters
    hparams = GPT2HyperParams(
        n_layer=1,
        n_head=1,
        n_embd=16,
        n_vocab=32,
        rotary_ndims=8,
    )

    # Define dummy parameters
    wte = mx.random.uniform(shape=(hparams["n_vocab"], hparams["n_embd"]))
    wpe = mx.random.uniform(shape=(hparams["n_vocab"], hparams["n_embd"]))

    params = GPT2Params(
        wte=wte,
        wpe=wpe,
        blocks=[
            {
                "ln_1": {
                    "g": mx.ones(hparams["n_embd"]),
                    "b": mx.zeros(hparams["n_embd"]),
                },
                "attn": {
                    "c_attn": {
                        "w": mx.random.uniform(
                            shape=(hparams["n_embd"], 3 * hparams["n_embd"])
                        ),
                        "b": mx.zeros(3 * hparams["n_embd"]),
                    },
                    "c_proj": {
                        "w": mx.random.uniform(
                            shape=(hparams["n_embd"], hparams["n_embd"])
                        ),
                        "b": mx.zeros(hparams["n_embd"]),
                    },
                },
                "ln_2": {
                    "g": mx.ones(hparams["n_embd"]),
                    "b": mx.zeros(hparams["n_embd"]),
                },
                "mlp": {
                    "c_fc": {
                        "w": mx.random.uniform(
                            shape=(hparams["n_embd"], 4 * hparams["n_embd"])
                        ),
                        "b": mx.zeros(4 * hparams["n_embd"]),
                    },
                    "c_proj": {
                        "w": mx.random.uniform(
                            shape=(4 * hparams["n_embd"], hparams["n_embd"])
                        ),
                        "b": mx.zeros(hparams["n_embd"]),
                    },
                },
            }
            for _ in range(hparams["n_layer"])
        ],
        ln_f={
            "g": mx.ones(hparams["n_embd"]),
            "b": mx.zeros(hparams["n_embd"]),
        },
    )

    # Create GPT2Model instance
    model = GPT2Model(params, hparams)

    # Define dummy input
    inputs = [1, 2, 3]

    # Get logits
    logits = model(inputs).logits

    # Assert the shape of logits
    assert logits.shape == (1, len(inputs), hparams["n_vocab"])

    print("Logits shape test passed!")


def test_compute_loss():
    # Define dummy hyperparameters
    hparams = GPT2HyperParams(
        n_layer=1,
        n_head=1,
        n_embd=16,
        n_vocab=32,
        rotary_ndims=8,
    )

    # Define dummy parameters
    wte = mx.random.uniform(shape=(hparams["n_vocab"], hparams["n_embd"]))
    wpe = mx.random.uniform(shape=(hparams["n_vocab"], hparams["n_embd"]))
    params = GPT2Params(
        wte=wte,
        wpe=wpe,
        blocks=[
            {
                "ln_1": {
                    "g": mx.ones(hparams["n_embd"]),
                    "b": mx.zeros(hparams["n_embd"]),
                },
                "attn": {
                    "c_attn": {
                        "w": mx.random.uniform(
                            shape=(hparams["n_embd"], 3 * hparams["n_embd"])
                        ),
                        "b": mx.zeros(3 * hparams["n_embd"]),
                    },
                    "c_proj": {
                        "w": mx.random.uniform(
                            shape=(hparams["n_embd"], hparams["n_embd"])
                        ),
                        "b": mx.zeros(hparams["n_embd"]),
                    },
                },
                "ln_2": {
                    "g": mx.ones(hparams["n_embd"]),
                    "b": mx.zeros(hparams["n_embd"]),
                },
                "mlp": {
                    "c_fc": {
                        "w": mx.random.uniform(
                            shape=(hparams["n_embd"], 4 * hparams["n_embd"])
                        ),
                        "b": mx.zeros(4 * hparams["n_embd"]),
                    },
                    "c_proj": {
                        "w": mx.random.uniform(
                            shape=(4 * hparams["n_embd"], hparams["n_embd"])
                        ),
                        "b": mx.zeros(hparams["n_embd"]),
                    },
                },
            }
            for _ in range(hparams["n_layer"])
        ],
        ln_f={
            "g": mx.ones(hparams["n_embd"]),
            "b": mx.zeros(hparams["n_embd"]),
        },
    )

    # Create GPT2Model instance
    model = GPT2Model(params, hparams)

    # Define dummy input
    inputs = [1, 2, 3]

    # Case 1: Test without compute_loss (default behavior)
    output_without_loss = model(inputs)
    assert isinstance(output_without_loss, GPT2Output)
    assert output_without_loss.logits.shape == (1, len(inputs), hparams["n_vocab"])
    assert output_without_loss.loss is None

    # Case 2: Test with compute_loss=True
    output_with_loss = model(inputs, compute_loss=True)
    assert isinstance(output_with_loss, GPT2Output)
    assert output_with_loss.logits.shape == (1, len(inputs), hparams["n_vocab"])
    assert output_with_loss.loss is not None
    assert output_with_loss.loss.shape == ()  # Scalar loss

    # Verify loss is computed correctly by manually calculating it
    # Targets are inputs shifted right by one position
    targets = mx.array(inputs[1:])  # No padding token needed
    logits_for_loss = output_with_loss.logits[0, :-1, :]
    logits_2d = mx.reshape(logits_for_loss, (-1, logits_for_loss.shape[-1]))
    expected_loss = mx.mean(nn.losses.cross_entropy(logits_2d, targets))

    # Check that our computed loss matches the expected loss
    assert mx.allclose(output_with_loss.loss, expected_loss)

    print("Compute loss test passed!")
