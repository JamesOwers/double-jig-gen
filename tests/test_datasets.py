from pathlib import Path

import pytest

from double_jig_gen.data import ABCDataset
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
        assert len(dataset) == 2
        for ii in range(len(dataset)):
            dataset[ii]
