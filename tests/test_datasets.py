from pathlib import Path

import pytest

from double_jig_gen.datasets import ABCDataset
from double_jig_gen.tokenizers import Tokenizer


@pytest.fixture
def valid_abc_paths():
    return [
        Path(__file__, "..", "data", "valid_abc.abc").resolve(),
        str(Path(__file__, "..", "data", "valid_abc.abc").resolve()),
    ]


def test_abcdataset(valid_abc_paths):
    for path in valid_abc_paths:
        dataset = ABCDataset(path)
        dataset[len(dataset)]
        dataset = ABCDataset(path, tokenizer=Tokenizer(tokens=["hi"]))
        dataset[len(dataset)]
        dataset = ABCDataset(path, tokens=("hi"))
        dataset[len(dataset)]
