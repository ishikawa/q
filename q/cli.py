import argparse
import time
from dataclasses import dataclass
from typing import Generator, Literal, Optional, Union

from tqdm import tqdm

from q.backend.numpy import generate as generate_numpy

from .common import ModelSize
from .encoder import load_encoder
from .params import load_hparams_and_params


@dataclass
class GPTResult:
    tps: float
    output_text: str


class TokenStreamHandler:
    """
    トークンストリームを処理するハンドラークラス
    BPEトークナイザーの問題に対処するためにバッファリング方式を使用
    日本語などのマルチバイト文字に対応するための改良版
    """
    def __init__(self, encoder, callback=None):
        """
        Args:
            encoder: トークンをデコードするエンコーダー
            callback: 新しいテキストが生成されたときに呼び出されるコールバック関数
        """
        self.encoder = encoder
        self.callback = callback
        self.token_buffer = []
        self.last_text = ""
        self.pending_bytes = bytearray()  # マルチバイト文字の処理用
        
    def __call__(self, count, token_id):
        """
        新しいトークンが生成されたときに呼び出される
        
        Args:
            count: 更新されたトークン数
            token_id: 生成されたトークンID
            
        Returns:
            True: 常に続行を指示
        """
        if token_id is not None:
            self.token_buffer.append(token_id)
            
            # 全トークンをデコード
            current_text = self.encoder.decode(self.token_buffer)
            
            # 新しく追加されたテキストのみを取得
            new_text = current_text[len(self.last_text):]
            
            # 日本語などのマルチバイト文字の処理
            if new_text:
                try:
                    # 新しいテキストをバイト配列に変換
                    new_bytes = new_text.encode('utf-8')
                    self.pending_bytes.extend(new_bytes)
                    
                    # 完全なマルチバイト文字のみをデコード
                    valid_text = self.pending_bytes.decode('utf-8', errors='ignore')
                    
                    if valid_text and self.callback:
                        self.callback(valid_text)
                        
                    # バッファをクリア
                    self.pending_bytes = bytearray()
                    self.last_text = current_text
                    
                except UnicodeDecodeError:
                    # 不完全なマルチバイト文字がある場合は次のトークンを待つ
                    pass
        
        return True  # 生成を続行


def run(
    prompt: str,
    n_tokens_to_generate: int = 40,
    model_size: ModelSize = "124M",
    models_dir: str = "models",
    backend: Literal["mlx", "numpy"] = "numpy",
    stream: bool = False,
) -> GPTResult:
    import time

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
    
    if stream:
        # ストリーミングモードの場合
        # main関数で作成されたストリームハンドラーを使用する想定
        # 空のハンドラーを作成（main関数で上書きされる）
        text_chunks = []
        
        def collect_text(chunk):
            text_chunks.append(chunk)
            
        stream_handler = TokenStreamHandler(
            encoder=encoder,
            callback=collect_text
        )
        
        # トークンを生成してストリームハンドラーで処理
        output_ids = generate(
            input_ids,
            params=params,
            n_head=hparams["n_head"],
            n_tokens_to_generate=n_tokens_to_generate,
            update_progress=stream_handler
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
                update_progress=lambda count, _: pbar.update(count)
            )
        
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

    if args.stream:
        # ストリーミングモード
        print(args.prompt, end="", flush=True)
        
        # ストリーミング用のコールバック関数
        def print_chunk(chunk):
            print(chunk, end="", flush=True)
            
        # エンコーダーとTokenStreamHandlerを作成
        encoder = load_encoder(args.model_size, args.models_dir)
        stream_handler = TokenStreamHandler(encoder, callback=print_chunk)
        
        # run関数の実行前にハンドラーを上書き
        from q.backend.mlx import generate as generate_mlx
        from q.backend.numpy import generate as generate_numpy
        
        # 選択されたバックエンドに基づいて生成関数を選択
        if args.backend == "mlx":
            generate_func = generate_mlx
        else:
            generate_func = generate_numpy
        
        # エンコーダーとパラメータを読み込み
        hparams, params = load_hparams_and_params(args.model_size, args.models_dir)
        input_ids = encoder.encode(args.prompt)
        
        # 開始時間を記録
        start_time = time.time()
        
        # 直接トークン生成を実行
        generate_func(
            input_ids,
            params=params,
            n_head=hparams["n_head"],
            n_tokens_to_generate=args.n_tokens_to_generate,
            update_progress=stream_handler
        )
        
        # 生成速度を計算して表示
        end_time = time.time()
        tps = args.n_tokens_to_generate / (end_time - start_time)
        print("\n")
        print(f"Generated {tps:.2f} tokens/sec")
    else:
        # 通常モード
        r = run(
            prompt=args.prompt,
            n_tokens_to_generate=args.n_tokens_to_generate,
            model_size=args.model_size,
            models_dir=args.models_dir,
            backend=args.backend,
            stream=False,
        )

        print(f"Generated {r.tps:.2f} tokens/sec")
        print()
        print(args.prompt + r.output_text)


if __name__ == "__main__":
    main()
