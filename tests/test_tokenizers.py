import pytest

from double_jig_gen.tokenizers import Tokenizer


def test_tokenizer():
    tokenizer = Tokenizer(tokens=["hi"])
    token_seq = tokenizer.untokenize(tokenizer.tokenize("hello and hi  friend "))
    expected_seq = ["<s>", "<unk>", "<unk>", "hi", "<unk>", "</s>"]
    assert token_seq == expected_seq
    token_seq = tokenizer.untokenize(tokenizer.tokenize(["hi", " and hello friend "]))
    expected_seq = ["<s>", "hi", "<unk>", "</s>"]
    assert token_seq == expected_seq
    token_seq = tokenizer.untokenize("1 3 0 2")
    expected_seq = ["<s>", "hi", "<unk>", "</s>"]


def test_tokenizer_special_tokens():
    test_data = {
        "<unk>": ValueError,
        "<s>": ValueError,
        "</s>": ValueError,
    }
    for token, expected_exception in test_data.items():
        with pytest.raises(expected_exception):
            _ = Tokenizer(tokens=["hi", token])
