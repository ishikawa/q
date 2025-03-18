import argparse
import time
from dataclasses import dataclass
from typing import Callable, Optional

from tqdm import tqdm

from .common import ModelSize
from .encoder import load_encoder
from .generation import generate
from .gpt2 import GPT2Model
from .params import load_hparams_and_params
from .stream import TokenStreamHandler


@dataclass
class GPTResult:
    tps: float
    output_text: str


def run(
    prompt: str,
    n_tokens_to_generate: int = 40,
    model_size: ModelSize = "124M",
    models_dir: str = "models",
    stream: Optional[Callable[[str], None]] = None,
) -> GPTResult:

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder = load_encoder(model_size, models_dir)
    hparams, params = load_hparams_and_params(
        model_size=model_size,
        models_dir=models_dir,
    )
    model = GPT2Model(params, hparams)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

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
        for token in generate(model, input_ids, max_new_tokens=n_tokens_to_generate):
            stream_handler([token])

        # 生成速度を計算
        sec = time.time() - t
        tps = n_tokens_to_generate / sec if sec > 0 else 0

        # 結果を返す
        return GPTResult(tps=tps, output_text="".join(text_chunks))
    else:
        output_ids = []
        with tqdm(total=n_tokens_to_generate) as pbar:
            pbar.set_description("Generating")

            for token in generate(
                model, input_ids, max_new_tokens=n_tokens_to_generate
            ):
                output_ids.append(token)
                pbar.update(1)

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
        )

        print(f"Generated {r.tps:.2f} tokens/sec")
        print()
        print(args.prompt + r.output_text)


if __name__ == "__main__":
    main()
