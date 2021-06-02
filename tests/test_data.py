import pytest
import torch

from double_jig_gen.data import (
    default_pad_batch,
    fix_encoding_errors,
    pad_batch,
    remove_quoted_strings,
)


@pytest.mark.parametrize(
    "decode_error_str,cleaned_str",
    [
        ("Câ\x80\x99", "C'"),
        ("CÂ\xa0", "C'"),
    ],
)
def test_fix_encoding_errors(decode_error_str, cleaned_str):
    assert fix_encoding_errors(decode_error_str) == cleaned_str


@pytest.mark.parametrize(
    "str_with_quoted_sections,cleaned_str",
    [
        ('hello "hi" friend', "hello  friend"),
        ('"hi" hello friend', " hello friend"),
        ('hello friend "hi"', "hello friend "),
        ('hello "hi" friend "', 'hello  friend "'),
        ('hello "hi""" friend "', 'hello  friend "'),
    ],
)
def test_remove_quoted_strings(str_with_quoted_sections, cleaned_str):
    assert remove_quoted_strings(str_with_quoted_sections) == cleaned_str


def test_pad_batch():
    input_tensors = [torch.tensor(item) for item in [[1], [1, 1], [1, 1, 1], [1, 2]]]

    padding_value = -1
    expected_tensor = torch.tensor(
        [
            [1, padding_value, padding_value],
            [1, 1, padding_value],
            [1, 1, 1],
            [1, 2, padding_value],
        ]
    ).T
    expected_lengths = [1, 2, 3, 2]
    output_tensor, output_lengths = pad_batch(input_tensors, pad_idx=padding_value)
    assert torch.equal(output_tensor, expected_tensor)
    assert output_lengths == expected_lengths

    padding_value = 0
    expected_tensor = torch.tensor(
        [
            [1, padding_value, padding_value],
            [1, 1, padding_value],
            [1, 1, 1],
            [1, 2, padding_value],
        ]
    ).T
    expected_lengths = [1, 2, 3, 2]
    output_tensor, output_lengths = default_pad_batch(input_tensors)
    assert torch.equal(output_tensor, expected_tensor)
    assert output_lengths == expected_lengths
