import copy
import json
import textwrap

import music21
import numpy as np
import pytest

from double_jig_gen.tokenizers import (
    ABCTune,
    ABCTuneError,
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
        abc_data = example["in"]
        event_list = abc_to_events(abc_data)
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


@pytest.fixture()
def abctune_examples():
    example_0_abc_data = textwrap.dedent(
        """
        T: Cooley's
        M: 4/4
        L: 1/8
        K: Edor
        |:D2|EBBA B2 EB|B2 AB dBAG|FDAD BDAD|FDAD dAFD|
        EBBA B2 EB|B2 AB defg|afec dBAF|DEFD E2:|
        |:gf|eB B2 efge|eB B2 gedB|A2 FA DAFA|A2 FA defg|
        eB B2 eBgB|eB B2 defg|afec dBAF|DEFD E2:|
        """
    ).strip()
    example_0 = {
        "init_kwargs": dict(
            abc_data=example_0_abc_data,
            pianoroll_divisions_per_quarternote=2,
            min_pitch=None,
            min_time=None,
            transpose_to_pitchclass="C",
        ),
        "expected_attributes": dict(
            abc_data=example_0_abc_data,
            min_pitch=62,
            min_time=0,
            key_guessed=False,
            original_key=music21.key.Key(tonic="E", mode="dorian"),
            transpose_semitones=-4,
            key=music21.key.Key(tonic="C", mode="dorian"),
            metadata={
                "tune title": "Cooley's",
                "meter": "4/4",
                "unit note length": "1/8",
                "key": "Edor",
            },
        ),
    }
    # Same as example 1 except for no key (i.e. added in F and C sharps manually)
    example_1_abc_data = textwrap.dedent(
        """
        T: Cooley's
        M: 4/4
        L: 1/8
        |:D2|EBBA B2 EB|B2 AB dBAG|^FDAD BDAD|^FDAD dA^FD|
        EBBA B2 EB|B2 AB de^fg|a^fe^c dBA^F|DE^FD E2:|
        |:g^f|eB B2 e^fge|eB B2 gedB|A2 ^FA DA^FA|A2 ^FA de^fg|
        eB B2 eBgB|eB B2 de^fg|a^fe^c dBA^F|DE^FD E2:|
        """
    ).strip()
    example_1 = {
        "init_kwargs": dict(
            abc_data=example_1_abc_data,
            pianoroll_divisions_per_quarternote=2,
            min_pitch=None,
            min_time=None,
            transpose_to_pitchclass="C",
        ),
        "expected_attributes": dict(
            abc_data=example_1_abc_data,
            min_pitch=62,
            min_time=0,
            key_guessed=True,
            # This is a bad guess! Misses the C sharps.
            original_key=music21.key.Key(tonic="E", mode="minor"),
            transpose_semitones=-4,
            key=music21.key.Key(tonic="C", mode="minor"),
            metadata={
                "tune title": "Cooley's",
                "meter": "4/4",
                "unit note length": "1/8",
            },
        ),
    }
    return example_0, example_1


def test_ABCTune(abctune_examples):
    test_results = {}
    for example_idx, example in enumerate(abctune_examples):
        abc_tune = ABCTune(**example["init_kwargs"])
        test_results = {}
        for attr_name in example["expected_attributes"].keys():
            attr_value = getattr(abc_tune, attr_name)
            expected_attr_value = example["expected_attributes"][attr_name]
            msg = f"{attr_name} = {attr_value} (expected: {expected_attr_value})"
            test_results[msg] = bool(attr_value == expected_attr_value)
        assert all(test_results.values()), (
            f"abctune_examples[{example_idx}] not as expected:\n"
            f"{json.dumps([k for k, v in test_results.items() if not v], indent=4)}"
        )


def test_ABCTune_pianoroll(abctune_examples):
    abc_tune_0, abc_tune_1 = [
        ABCTune(**ex["init_kwargs"]) for ex in abctune_examples[:2]
    ]
    assert np.array_equal(abc_tune_0.pianoroll, abc_tune_1.pianoroll)
    assert np.array_equal(
        abc_tune_0.compressed_pianoroll,
        abc_tune_1.compressed_pianoroll,
    )


def test_ABCTune_invalid_abc():
    with pytest.raises(ABCTuneError):
        ABCTune("")
    with pytest.raises(ABCTuneError):
        ABCTune(" ")
    with pytest.raises(ABCTuneError):
        ABCTune("A")


def test_ABCTune_invalid_metadata():
    # http://abcnotation.com/wiki/abc:standard:v2.1#outdated_syntax
    invalid_field = "E"
    abc_data = f"{invalid_field}: oh no\nL: 1/8\nA"
    abc_tune = ABCTune(abc_data)
    assert f"Unexpected field {repr(invalid_field)}" in abc_tune.metadata


def test_ABCTune_play():
    abc_data = "L:1/8\nA"
    ABCTune(abc_data).play()


def test_ABCTune_show(abctune_examples):
    abc_data = "L:1/8\nA"
    ABCTune(abc_data).show()
    ABCTune(abc_data).show("text")


def test_ABCTune_plot_pianoroll(abctune_examples):
    abc_data = "L:1/8\nA"
    ABCTune(abc_data).plot_pianoroll()
