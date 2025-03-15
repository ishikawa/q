from unittest.mock import Mock

from q.encoder import load_encoder
from q.stream import TokenStreamHandler


def test_init():
    """TokenStreamHandlerの初期化テスト"""
    encoder = load_encoder("124M", "models")
    callback = Mock()
    
    handler = TokenStreamHandler(encoder, callback)
    
    assert handler.encoder == encoder
    assert handler.callback == callback
    assert handler.token_buffer == []
    assert handler.last_text == ""


def test_call_with_tokens():
    """トークン処理の基本機能テスト"""
    encoder = load_encoder("124M", "models")
    callback = Mock()
    
    handler = TokenStreamHandler(encoder, callback)
    
    # トークンのリストを用意
    test_tokens = [15496]  # "Hello"に対応するトークン
    
    # ハンドラーを呼び出し
    result = handler(test_tokens)
    
    # 結果の検証
    assert result is True  # 常にTrueを返す
    assert handler.token_buffer == test_tokens
    assert len(handler.last_text) > 0
    callback.assert_called_once()  # コールバックが呼び出されたことを確認


def test_call_with_empty_tokens():
    """空のトークンリストでの動作テスト"""
    encoder = load_encoder("124M", "models")
    callback = Mock()
    
    handler = TokenStreamHandler(encoder, callback)
    
    # 空のトークンリストでハンドラーを呼び出し
    result = handler([])
    
    # 結果の検証
    assert result is True  # 常にTrueを返す
    assert handler.token_buffer == []
    assert handler.last_text == ""
    callback.assert_not_called()  # コールバックが呼び出されていないことを確認


def test_accumulate_tokens():
    """複数回のトークン追加テスト"""
    encoder = load_encoder("124M", "models")
    callback = Mock()
    
    handler = TokenStreamHandler(encoder, callback)
    
    # 1回目のトークン追加
    handler([15496])  # "Hello"の一部
    
    # 2回目のトークン追加
    handler([11241])  # "world"の一部
    
    # バッファに両方のトークンが含まれていることを確認
    assert handler.token_buffer == [15496, 11241]
    
    # コールバックが2回呼ばれたことを確認
    assert callback.call_count == 2


def test_no_callback():
    """コールバックなしの場合のテスト"""
    encoder = load_encoder("124M", "models")
    
    # コールバックを指定せずにハンドラーを初期化
    handler = TokenStreamHandler(encoder)
    
    # トークンを処理
    result = handler([15496])  # "Hello"に対応するトークン
    
    # 結果の検証 - エラーなく処理できることを確認
    assert result is True
    assert handler.token_buffer == [15496]
    assert len(handler.last_text) > 0


def test_unicode_decode_error_handling():
    """UnicodeDecodeErrorのハンドリングテスト"""
    encoder = load_encoder("124M", "models")
    callback = Mock()
    
    handler = TokenStreamHandler(encoder, callback)
    
    # UTF-8デコードエラーを引き起こす可能性のある不完全なトークン
    # 実際のトークンはエンコーダによって異なる可能性があるため、
    # このテストは環境によっては失敗することがあります
    
    # トークンを部分的に追加してデコードエラーを発生させる可能性
    incomplete_tokens = [16]  # 例として、不完全かもしれないトークン
    
    # UnicodeDecodeErrorが発生しても例外が発生しないことを確認
    result = handler(incomplete_tokens)
    
    # 結果の検証
    assert result is True  # エラーがあっても常にTrueを返す
