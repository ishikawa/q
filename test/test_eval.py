import pytest

from q.eval import QLM


@pytest.fixture
def qlm():
    return QLM()


def test_eot_token_id(qlm):
    """Test that the eot_token_id property returns None as expected."""
    assert qlm.eot_token_id is None


def test_tok_encode(qlm):
    """Test that the tok_encode method correctly tokenizes a string using the model's encoder."""
    text = "Hello, world!"

    tokens = qlm.tok_encode(text)
    assert tokens is not None
    assert tokens == qlm.encoder.encode(text)
