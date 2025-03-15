from typing import Callable, Optional

from q.encoder import Encoder


class TokenStreamHandler:
    """
    トークンストリームを処理するハンドラークラス
    BPEトークナイザーの問題に対処するためにバッファリング方式を使用
    """

    def __init__(
        self, encoder: Encoder, callback: Optional[Callable[[str], None]] = None
    ):
        """
        Args:
            encoder: トークンをデコードするエンコーダー
            callback: 新しいテキストが生成されたときに呼び出されるコールバック関数
        """
        self.encoder = encoder
        self.callback = callback
        self.token_buffer: list[int] = []
        self.last_text = ""

    def __call__(self, tokens: list[int]) -> bool:
        """
        新しいトークンが生成されたときに呼び出される

        Args:
            tokens: 生成されたトークンのリスト

        Returns:
            bool: 常にTrueを返す
        """
        if tokens:
            # トークンを追加
            self.token_buffer.extend(tokens)

            # 全トークンをデコード
            #
            # NOTE: BPEトークナイザーは単語の部分（サブワード）を扱うため、日本語のような非ラテン文
            # 字では1つの文字が複数のトークンに分割される可能性があります。そのため、個々のトークン
            # をデコードすると不完全な文字になることがあります。
            try:
                current_text = self.encoder.decode(self.token_buffer, errors="strict")

                # 新しく追加されたテキストのみを取得
                new_text = current_text[len(self.last_text) :]

                # 新しいテキストがある場合のみコールバックを呼び出す
                if new_text and self.callback:
                    self.callback(new_text)

                # 最後のテキストを更新
                self.last_text = current_text
            except UnicodeDecodeError:
                pass

        return True  # 生成を続行
