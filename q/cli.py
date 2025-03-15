import argparse
import time
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from tqdm import tqdm

from q.backend.numpy import generate as generate_numpy
from q.stream import TokenStreamHandler

from .common import ModelSize
from .encoder import load_encoder
from .params import load_hparams_and_params


@dataclass
class GPTResult:
    tps: float
    output_text: str


def run(
    prompt: str,
    n_tokens_to_generate: int = 40,
    model_size: ModelSize = "124M",
    models_dir: str = "models",
    backend: Literal["mlx", "numpy"] = "numpy",
    stream: Optional[Callable[[str], None]] = None,
) -> GPTResult:

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder = load_encoder(model_size, models_dir)
    hparams, params = load_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # 選択されたバックエンドに基づいて生成関数を選択
    if backend == "mlx":
        from q.backend.mlx import generate as generate_mlx

        generate = generate_mlx
    else:
        generate = generate_numpy

    # 開始時間を記録
    t = time.time()

    if stream is not None:
        # ストリーミングモードの場合
        text_chunks = []

        def collect_text(chunk):
            text_chunks.append(chunk)
            # カスタムコールバックがあれば実行
            stream(chunk)

        stream_handler = TokenStreamHandler(encoder=encoder, callback=collect_text)

        # トークンを生成してストリームハンドラーで処理
        output_ids = generate(
            input_ids,
            params=params,
            n_head=hparams["n_head"],
            n_tokens_to_generate=n_tokens_to_generate,
            update_progress=stream_handler,
        )

        # 生成速度を計算
        sec = time.time() - t
        tps = n_tokens_to_generate / sec if sec > 0 else 0

        # 結果を返す
        return GPTResult(tps=tps, output_text="".join(text_chunks))
    else:
        # 通常モード - 一括生成
        with tqdm(total=n_tokens_to_generate) as pbar:
            pbar.set_description("Generating")
            output_ids = generate(
                input_ids,
                params=params,
                n_head=hparams["n_head"],
                n_tokens_to_generate=n_tokens_to_generate,
                update_progress=lambda tokens: pbar.update(len(tokens)),
            )

        sec = time.time() - t
        tps = n_tokens_to_generate / sec if sec > 0 else 0

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
    parser.add_argument(
        "--backend",
        type=str,
        choices=["mlx", "numpy"],
        default="numpy",
        help="Backend to use for computation (mlx or numpy)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming output token by token",
    )

    args = parser.parse_args()

    # print(f"Prompt: {args.prompt}")
    # print(f"Number of tokens to generate: {args.n_tokens_to_generate}")
    # print(f"Model size: {args.model_size}")
    # print(f"Models directory: {args.models_dir}")

    # プロンプトを表示（ストリーミングモードの場合）
    if args.stream:
        print(args.prompt, end="", flush=True)

        # ストリーミング用のコールバック関数
        def print_chunk(chunk):
            print(chunk, end="", flush=True)

        r = run(
            prompt=args.prompt,
            n_tokens_to_generate=args.n_tokens_to_generate,
            model_size=args.model_size,
            models_dir=args.models_dir,
            backend=args.backend,
            stream=print_chunk,
        )

        print()
        print(f"Generated {r.tps:.2f} tokens/sec")
    else:
        r = run(
            prompt=args.prompt,
            n_tokens_to_generate=args.n_tokens_to_generate,
            model_size=args.model_size,
            models_dir=args.models_dir,
            backend=args.backend,
        )

        print(f"Generated {r.tps:.2f} tokens/sec")
        print()
        print(args.prompt + r.output_text)


if __name__ == "__main__":
    main()
