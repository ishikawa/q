import mlx.core as mx
import numpy as np

from q.gpt2 import GPT2HyperParams, GPT2Model, GPT2Params


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
    wte = mx.array(np.random.rand(hparams["n_vocab"], hparams["n_embd"]))
    wpe = mx.array(np.random.rand(hparams["n_vocab"], hparams["n_embd"]))

    params = GPT2Params(
        wte=wte,
        wpe=wpe,
        blocks=[
            {
                "ln_1": {
                    "g": mx.array(np.ones(hparams["n_embd"])),
                    "b": mx.array(np.zeros(hparams["n_embd"])),
                },
                "attn": {
                    "c_attn": {
                        "w": mx.array(
                            np.random.rand(hparams["n_embd"], 3 * hparams["n_embd"])
                        ),
                        "b": mx.array(np.zeros(3 * hparams["n_embd"])),
                    },
                    "c_proj": {
                        "w": mx.array(
                            np.random.rand(hparams["n_embd"], hparams["n_embd"])
                        ),
                        "b": mx.array(np.zeros(hparams["n_embd"])),
                    },
                },
                "ln_2": {
                    "g": mx.array(np.ones(hparams["n_embd"])),
                    "b": mx.array(np.zeros(hparams["n_embd"])),
                },
                "mlp": {
                    "c_fc": {
                        "w": mx.array(
                            np.random.rand(hparams["n_embd"], 4 * hparams["n_embd"])
                        ),
                        "b": mx.array(np.zeros(4 * hparams["n_embd"])),
                    },
                    "c_proj": {
                        "w": mx.array(
                            np.random.rand(4 * hparams["n_embd"], hparams["n_embd"])
                        ),
                        "b": mx.array(np.zeros(hparams["n_embd"])),
                    },
                },
            }
            for _ in range(hparams["n_layer"])
        ],
        ln_f={
            "g": mx.array(np.ones(hparams["n_embd"])),
            "b": mx.array(np.zeros(hparams["n_embd"])),
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
