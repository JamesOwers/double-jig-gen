import music21
import numpy as np
import pytest

from double_jig_gen.tokenizers import (
    Tokenizer,
    abc_to_events,
    compress_pianoroll,
    decompress_pianoroll,
    events_to_pianoroll_array,
)


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


@pytest.fixture()
def music21_test_examples():
    examples = []
    examples.append(
        {
            "in": "L:1/16\nV:1\nCC C2\nV:2\nA2 AA\n",
            "out": [
                {
                    "start": 0.0,
                    "duration": 0.25,
                    "pitch": 60,
                    "pitch_str": "C4",
                    "pitch_class": 0,
                    "accidental": None,
                },
                {
                    "start": 0.0,
                    "duration": 0.5,
                    "pitch": 69,
                    "pitch_str": "A4",
                    "pitch_class": 9,
                    "accidental": None,
                },
                {
                    "start": 0.25,
                    "duration": 0.25,
                    "pitch": 60,
                    "pitch_str": "C4",
                    "pitch_class": 0,
                    "accidental": None,
                },
                {
                    "start": 0.5,
                    "duration": 0.5,
                    "pitch": 60,
                    "pitch_str": "C4",
                    "pitch_class": 0,
                    "accidental": None,
                },
                {
                    "start": 0.5,
                    "duration": 0.25,
                    "pitch": 69,
                    "pitch_str": "A4",
                    "pitch_class": 9,
                    "accidental": None,
                },
                {
                    "start": 0.75,
                    "duration": 0.25,
                    "pitch": 69,
                    "pitch_str": "A4",
                    "pitch_class": 9,
                    "accidental": None,
                },
            ],
        }
    )
    examples.append(
        {
            "in": "L:1/16\nK:C#maj\n[CG_e]",
            "out": [
                {
                    "start": 0.0,
                    "duration": 0.25,
                    "pitch": 61,
                    "pitch_str": "C#4",
                    "pitch_class": 1,
                    "accidental": "sharp",
                },
                {
                    "start": 0.0,
                    "duration": 0.25,
                    "pitch": 68,
                    "pitch_str": "G#4",
                    "pitch_class": 8,
                    "accidental": "sharp",
                },
                {
                    "start": 0.0,
                    "duration": 0.25,
                    "pitch": 75,
                    "pitch_str": "Eb5",
                    "pitch_class": 3,
                    "accidental": "flat",
                },
            ],
        }
    )
    examples.append(
        {
            "in": "L:1/8\nK:C\nCG_e[CG_e]",
            "out": [
                {
                    "start": 0.0,
                    "duration": 0.5,
                    "pitch": 60,
                    "pitch_str": "C4",
                    "pitch_class": 0,
                    "accidental": None,
                },
                {
                    "start": 0.5,
                    "duration": 0.5,
                    "pitch": 67,
                    "pitch_str": "G4",
                    "pitch_class": 7,
                    "accidental": None,
                },
                {
                    "start": 1.0,
                    "duration": 0.5,
                    "pitch": 75,
                    "pitch_str": "Eb5",
                    "pitch_class": 3,
                    "accidental": "flat",
                },
                {
                    "start": 1.5,
                    "duration": 0.5,
                    "pitch": 60,
                    "pitch_str": "C4",
                    "pitch_class": 0,
                    "accidental": None,
                },
                {
                    "start": 1.5,
                    "duration": 0.5,
                    "pitch": 67,
                    "pitch_str": "G4",
                    "pitch_class": 7,
                    "accidental": None,
                },
                {
                    "start": 1.5,
                    "duration": 0.5,
                    "pitch": 75,
                    "pitch_str": "Eb5",
                    "pitch_class": 3,
                    "accidental": "flat",
                },
            ],
        }
    )
    return examples


def test_abc_to_events(music21_test_examples):
    test_results = {}
    for example in music21_test_examples:
        out = abc_to_events(example["in"])
        msg = f"abc_to_events('{example['in']}') = {out}\nexpected: {example['out']}"
        test_results[msg] = out == example["out"]
    assert all(test_results.values()), (
        "abc_to_events produced the incorrect output for: "
        f"{[key for key, val in test_results.items() if val is False]}"
    )


@pytest.fixture()
def abc_to_pianoroll_test_examples():
    examples = []
    examples.append(
        {
            "in": "L:1/16\nV:1\nCC C2\nV:2\nE2 _E^D",
            "out": (
                np.array(
                    [
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        ],
                        [
                            [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ],
                    ],
                    dtype=bool,
                ),
                60,
                0,
            ),
        },
    )
    return examples


def test_abc_to_pianoroll(abc_to_pianoroll_test_examples):
    test_results = {}
    for example in abc_to_pianoroll_test_examples:
        abc_str = example["in"]
        event_list = abc_to_events(abc_str)
        out = events_to_pianoroll_array(event_list)
        msg = (
            f"events_to_pianoroll_array(abc_to_events('{example['in']}') = {out}\n"
            f"expected: {example['out']}\n"
            f"got: {out}"
        )
        pianoroll, min_pitch, min_time = out
        expected_pianoroll, expected_min_pitch, expected_min_time = example["out"]
        test_results[msg] = (
            np.array_equal(pianoroll, expected_pianoroll)
            and min_pitch == expected_min_pitch
            and min_time == expected_min_time
        )
    assert all(test_results.values()), (
        "events_to_pianoroll_array produced the incorrect output for: "
        f"{[key for key, val in test_results.items() if val is False]}"
    )


def test_compress_and_decompress_pianoroll(abc_to_pianoroll_test_examples):
    test_results = {}
    for example in abc_to_pianoroll_test_examples:
        pianoroll, _, _ = example["out"]
        decompressed_pianoroll = decompress_pianoroll(compress_pianoroll(pianoroll))
        msg = (
            f"decompress_pianoroll(compress_pianoroll({pianoroll})) = "
            f"{decompressed_pianoroll}\n"
        )
        test_results[msg] = np.array_equal(pianoroll, decompressed_pianoroll)
    assert all(test_results.values()), (
        "events_to_pianoroll_array produced the incorrect output for: "
        f"{[key for key, val in test_results.items() if val is False]}"
    )
