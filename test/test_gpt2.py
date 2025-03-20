import mlx.core as mx
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
