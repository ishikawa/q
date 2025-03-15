import argparse
from dataclasses import dataclass

from q.backend.numpy import generate

from .common import ModelSize
from .encoder import load_encoder
from .params import load_hparams_and_params

# from q.backend.mlx import generate


@dataclass
class GPTResult:
    tps: float
    output_text: str


def run(
    prompt: str,
    n_tokens_to_generate: int = 40,
    model_size: ModelSize = "124M",
    models_dir: str = "models",
) -> GPTResult:
    import time

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder = load_encoder(model_size, models_dir)
    hparams, params = load_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    t = time.time()
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    sec = time.time() - t
    tps = n_tokens_to_generate / sec

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)

    return GPTResult(tps=tps, output_text=output_text)


def main():
    parser = argparse.ArgumentParser(description="Main script for text generation.")
    parser.add_argument("prompt", type=str, help="Input prompt for text generation")
    parser.add_argument(
        "--n_tokens_to_generate",
        type=int,
        default=40,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["124M", "355M", "774M", "1558M"],
        default="124M",
        help="Size of the model to use",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Directory where models are stored",
    )

    args = parser.parse_args()

    # print(f"Prompt: {args.prompt}")
    # print(f"Number of tokens to generate: {args.n_tokens_to_generate}")
    # print(f"Model size: {args.model_size}")
    # print(f"Models directory: {args.models_dir}")

    r = run(
        prompt=args.prompt,
        n_tokens_to_generate=args.n_tokens_to_generate,
        model_size=args.model_size,
        models_dir=args.models_dir,
    )

    print(f"Generated {r.tps:.2f} tokens/sec")
    print()
    print(args.prompt + r.output_text)


if __name__ == "__main__":
    main()
