import music21
import pytest

from double_jig_gen.tokenizers import Tokenizer, abc_to_events


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
            "in": "L:1/16\nK:C#maj\n[CG_e]",
            "out": [
                {
                    "start": 0.0,
                    "dur": 0.25,
                    "end": 0.25,
                    "midipitch": 61,
                    "pitch_str": "C#4",
                    "accidental": "sharp",
                    "pitch_class": 1,
                },
                {
                    "start": 0.0,
                    "dur": 0.25,
                    "end": 0.25,
                    "midipitch": 68,
                    "pitch_str": "G#4",
                    "accidental": "sharp",
                    "pitch_class": 8,
                },
                {
                    "start": 0.0,
                    "dur": 0.25,
                    "end": 0.25,
                    "midipitch": 75,
                    "pitch_str": "Eb5",
                    "accidental": "flat",
                    "pitch_class": 3,
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
                    "dur": 0.5,
                    "end": 0.5,
                    "midipitch": 60,
                    "pitch_str": "C4",
                    "accidental": None,
                    "pitch_class": 0,
                },
                {
                    "start": 0.5,
                    "dur": 0.5,
                    "end": 1.0,
                    "midipitch": 67,
                    "pitch_str": "G4",
                    "accidental": None,
                    "pitch_class": 7,
                },
                {
                    "start": 1.0,
                    "dur": 0.5,
                    "end": 1.5,
                    "midipitch": 75,
                    "pitch_str": "Eb5",
                    "accidental": "flat",
                    "pitch_class": 3,
                },
                {
                    "start": 1.5,
                    "dur": 0.5,
                    "end": 2.0,
                    "midipitch": 60,
                    "pitch_str": "C4",
                    "accidental": None,
                    "pitch_class": 0,
                },
                {
                    "start": 1.5,
                    "dur": 0.5,
                    "end": 2.0,
                    "midipitch": 67,
                    "pitch_str": "G4",
                    "accidental": None,
                    "pitch_class": 7,
                },
                {
                    "start": 1.5,
                    "dur": 0.5,
                    "end": 2.0,
                    "midipitch": 75,
                    "pitch_str": "Eb5",
                    "accidental": "flat",
                    "pitch_class": 3,
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
