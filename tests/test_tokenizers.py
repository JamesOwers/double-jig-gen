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
                    "start_quartertime": 0.0,
                    "midipitch": 61.0,
                    "pitch_str": "C#4",
                    "accidental": music21.pitch.Accidental("#"),
                    "pitch_class": 1,
                    "beat_dur": 0.25,
                },
                {
                    "start_quartertime": 0.0,
                    "midipitch": 68.0,
                    "pitch_str": "G#4",
                    "accidental": music21.pitch.Accidental("#"),
                    "pitch_class": 8,
                    "beat_dur": 0.25,
                },
                {
                    "start_quartertime": 0.0,
                    "midipitch": 75.0,
                    "pitch_str": "Eb5",
                    "accidental": music21.pitch.Accidental("-"),
                    "pitch_class": 3,
                    "beat_dur": 0.25,
                },
            ],
        }
    )
    examples.append(
        {
            "in": "L:1/8\nK:C\nCG_e[CG_e]",
            "out": [
                {
                    "start_quartertime": 0.0,
                    "midipitch": 60.0,
                    "pitch_str": "C4",
                    "accidental": None,
                    "pitch_class": 0,
                    "beat_dur": 0.5,
                },
                {
                    "start_quartertime": 0.5,
                    "midipitch": 67.0,
                    "pitch_str": "G4",
                    "accidental": None,
                    "pitch_class": 7,
                    "beat_dur": 0.5,
                },
                {
                    "start_quartertime": 1.0,
                    "midipitch": 75.0,
                    "pitch_str": "Eb5",
                    "accidental": music21.pitch.Accidental("-"),
                    "pitch_class": 3,
                    "beat_dur": 0.5,
                },
                {
                    "start_quartertime": 1.5,
                    "midipitch": 60.0,
                    "pitch_str": "C4",
                    "accidental": None,
                    "pitch_class": 0,
                    "beat_dur": 0.5,
                },
                {
                    "start_quartertime": 1.5,
                    "midipitch": 67.0,
                    "pitch_str": "G4",
                    "accidental": None,
                    "pitch_class": 7,
                    "beat_dur": 0.5,
                },
                {
                    "start_quartertime": 1.5,
                    "midipitch": 75.0,
                    "pitch_str": "Eb5",
                    "accidental": music21.pitch.Accidental("-"),
                    "pitch_class": 3,
                    "beat_dur": 0.5,
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
