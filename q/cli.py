import argparse
import time
from dataclasses import dataclass
from typing import Callable, Optional

from tqdm import tqdm

from .common import ModelSize
from .encoder import load_encoder
from .generation import TokenGenerator
from .gpt2 import GPT2Model
from .params import load_hparams_and_params
from .stream import TokenStreamHandler


@dataclass
class GPTResult:
    tps: float
    output_text: str


def run(
    prompt: str,
    max_length: int = 40,
    max_new_tokens: Optional[int] = None,
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
    generator = TokenGenerator(model)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    n_tokens_to_generate = max_length - len(input_ids)
    if max_new_tokens is not None:
        n_tokens_to_generate = max_new_tokens

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
        for token in generator(
            input_ids, max_length=max_length, max_new_tokens=max_new_tokens
        ):
            stream_handler([token])

        # 生成速度を計算
        sec = time.time() - t
        tps = n_tokens_to_generate / sec if sec > 0 else 0

        # 結果を返す
        return GPTResult(tps=tps, output_text="".join(text_chunks))
    else:
        output_ids = []
        with tqdm(total=n_tokens_to_generate) as progress:
            progress.set_description("Generating")

            for token in generator(input_ids, max_new_tokens=n_tokens_to_generate):
                output_ids.append(token)
                progress.update(1)

        sec = time.time() - t
        tps = n_tokens_to_generate / sec if sec > 0 else 0

        # decode the ids back into a string
        output_text = encoder.decode(output_ids)

        return GPTResult(tps=tps, output_text=output_text)


def main():
    parser = argparse.ArgumentParser(description="Main script for text generation.")
    parser.add_argument("prompt", type=str, help="Input prompt for text generation")
    parser.add_argument(
        "--max-length",
        type=int,
        default=40,
        help="Maximum length of tokens to generate",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["124M", "355M", "774M", "1558M"],
        default="124M",
        help="Size of the model to use",
    )
    parser.add_argument(
        "--models-dir",
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

    # プロンプトを表示（ストリーミングモードの場合）
    if args.stream:
        print(args.prompt, end="", flush=True)

        # ストリーミング用のコールバック関数
        def print_chunk(chunk):
            print(chunk, end="", flush=True)

        r = run(
            prompt=args.prompt,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            model_size=args.model_size,
            models_dir=args.models_dir,
            stream=print_chunk,
        )

        print()
        print(f"Generated {r.tps:.2f} tokens/sec")
    else:
        r = run(
            prompt=args.prompt,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            model_size=args.model_size,
            models_dir=args.models_dir,
        )

        print(f"Generated {r.tps:.2f} tokens/sec")
        print()
        print(args.prompt + r.output_text)


if __name__ == "__main__":
    main()
