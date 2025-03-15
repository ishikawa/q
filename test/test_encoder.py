from q.encoder import load_encoder


def test_load_encoder():
    encoder = load_encoder("124M", "models")
    assert encoder is not None


def test_encode_encode():
    encoder = load_encoder("124M", "models")
    text = "Hello, world!"
    tokens = encoder.encode(text)
    assert tokens is not None
    assert tokens == encoder.encode(text)
