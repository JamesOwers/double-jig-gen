"""Classes which turn strings into lists of tokens."""
import logging
import re
from typing import Collection, List, Mapping, Optional, Sequence, Tuple, Union

import music21
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class Tokenizer:
    """Stores a list of valid string tokens, and uses this to convert data to integers.

    The instantiated class is callable and will return a sequence of integers. By
    default, tokens 0, 1, and 2 are reserved as special tokens denoting unknown
    strings, the start of the sequence, and the end of the sequence, respectively.
    """

    def __init__(
        self,
        tokens: Collection[str],
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        start_token: str = "<s>",
        end_token: str = "</s>",
    ):
        self.special_tokens = [pad_token, unk_token, start_token, end_token]
        self.pad_token_index = 0
        self.unk_token_index = 1
        self.start_token_index = 2
        self.end_token_index = 3
        for special_token in self.special_tokens:
            if special_token in tokens:
                msg = f"Special token {repr(special_token)} is in the supplied tokens"
                LOGGER.error(msg)
                raise ValueError(msg)
        self.tokens = self.special_tokens + sorted(list(set(tokens)))
        self.token_to_index = dict(zip(self.tokens, range(len(self.tokens))))

    # TODO: Make __call__ which (un)tokenizes automagically

    def tokenize(
        self,
        str_or_sequence: Union[str, Sequence[str]],
        wrap: bool = True,
    ) -> List[int]:
        if isinstance(str_or_sequence, str):
            token_sequence = str_or_sequence.split()
        else:
            token_sequence = list(str_or_sequence)
        int_sequence = [
            self.token_to_index[token]
            if token in self.token_to_index
            else self.unk_token_index
            for token in token_sequence
        ]
        if wrap:
            int_sequence = (
                [self.start_token_index] + int_sequence + [self.end_token_index]
            )
        return int_sequence

    def untokenize(self, str_or_sequence: Union[str, Sequence[int]]) -> List[str]:
        if isinstance(str_or_sequence, str):
            int_sequence = [int(string) for string in str_or_sequence.split()]
        else:
            int_sequence = list(str_or_sequence)
        token_sequence = [self.tokens[index] for index in int_sequence]
        return token_sequence


# Taken from: http://abcnotation.com/wiki/abc:standard:v2.1
ABC_FIELDS = {
    "A": "area",
    "B": "book",
    "C": "composer",
    "D": "discography",
    "F": "file url",
    "G": "group",
    "H": "history",
    "I": "instruction",
    "K": "key",
    "L": "unit note length",
    "M": "meter",
    "m": "macro",
    "N": "notes",
    "O": "origin",
    "P": "parts",
    "Q": "tempo",
    "R": "rhythm",
    "r": "remark",
    "S": "source",
    "s": "symbol line",
    "T": "tune title",
    "U": "user defined",
    "V": "voice",
    "W": "words",
    "w": "words",
    "X": "reference number",
    "Z": "transcription",
}


def merge_continuation_lines(lines: Sequence[str]) -> Sequence[str]:
    r"""Merges lines which end with \ with the next line.

    Args:
        lines: a list of strings to check and merge.

    Returns:
        lines: the list of strings which have been merged.
    """
    nr_checks = len(lines)
    idx = 0
    for _ in range(nr_checks):
        line = lines[idx]
        if line.endswith("\\"):
            line = line[:-1]  # remove trailing backslash
            try:
                lines[idx] = f"{line} {lines[idx+1]}"
                del lines[idx + 1]  # del and stay on this line
            except IndexError as e:
                # don't care if last line - just remove slash
                if (idx + 1) == nr_checks:
                    lines[idx] = f"{line}"
                else:
                    print(lines, line)
                    raise e
        else:
            idx += 1
    return lines


# # TODO: keeping since could be useful to parse metadata later
# def parse_abc(abc_str: str) -> Mapping[str, str]:
#     """Get the required information from each tune.

#     Extracts the metadata as defined in [1] from the string, plus the transcription
#     which is assumed not to have a prefix. Assumes all metadata is stated before the
#     transcription is started. Then adds everything remaining to the transcription.

#     Args:
#         abc_str: the string containing an abc file to parse.

#     Returns:
#         out_dict: a dictionary containing all the extracted values.

#     See also:
#     [1] http://abcnotation.com/wiki/abc:standard:v2.1
#     """
#     lines = [line.strip() for line in abc_str.split("\n")]

#     # if a line ends with \ then merge with subsequent line
#     lines = merge_continuation_lines(lines)

#     valid_field = "|".join(ABC_FIELDS)
#     metadata_line_regex = re.compile(f"^({valid_field}):")

#     _, metadata_end_idx = min(
#         (val, idx)
#         for (idx, val) in enumerate(
#             [bool(metadata_line_regex.match(line)) for line in lines]
#         )
#     )

#     out_dict = {}
#     for line in lines[:metadata_end_idx]:
#         meta_key, value = line.split(":", 1)
#         field_name = ABC_FIELDS[meta_key]
#         if field_name in out_dict:
#             concat_fields = ("tune title", "words")
#             if field_name not in concat_fields:
#                 msg = (
#                     f"Tried to add {(field_name, value)} to {out_dict}."
#                     f"\nLines: {lines}"
#                 )
#                 raise ValueError(msg)
#             else:
#                 out_dict[field_name] = f"{out_dict[field_name]} --- {line}"
#         out_dict[field_name] = value
#     if "meter" in out_dict:
#         tune_lines = [f"M:{out_dict['meter']}"]
#     else:
#         raise ValueError("An abc tune must have a meter")
#     if "key" in out_dict:
#         tune_lines += [f"K:{out_dict['key']}"]
#     else:
#         raise ValueError("An abc tune must have a key")
#     for line in lines[metadata_end_idx:]:
#         # If the line starts with a W: we assume the whole thing is words and move on
#         if line.upper().startswith("W:"):
#             if "words" not in out_dict:
#                 out_dict["words"] = line
#             else:
#                 out_dict["words"] = f"{out_dict['words']} --- {line}"
#             continue
#         # TODO: may require manual handling of things here e.g. mid line key changes
#         tune_lines.append(line)
#     out_dict["tune_lines"] = tune_lines
#     out_dict["tune_str"] = "\n".join(out_dict["tune_lines"])
#     return out_dict


def abc_to_events(abc_str):
    """Converts abc to a list of note events.

    Times are in number of quavers (quarters). End times are not inclusive.
    """
    # TODO: get position within measure
    # TODO: get pitch spelling information e.g. position within the scale, and whether
    #       an accidental is applied *with respect to the scale*
    abc = music21.converter.parse(abc_str, format="abc")
    # N.B. calling .flat on abc makes offset times stored in not.offset relative to the
    # start of the piece, rather than relative to the containing stream.
    note_stream = abc.flat.getElementsByClass(["Note", "Chord"])
    events = []
    for element in note_stream:
        if isinstance(element, music21.note.Note):
            accidental_str = (
                element.pitch.accidental.fullName
                if element.pitch.accidental is not None
                else None
            )
            events.append(
                dict(
                    start=element.offset,
                    dur=element.duration.quarterLength,
                    end=element.offset + element.duration.quarterLength,
                    midipitch=int(element.pitch.ps),
                    pitch_str=f"{element.name.replace('-', 'b')}{element.octave}",
                    accidental=accidental_str,
                    pitch_class=element.pitch.pitchClass,
                )
            )
        if isinstance(element, music21.chord.Chord):
            for pitch in element.pitches:
                accidental_str = (
                    pitch.accidental.fullName if pitch.accidental is not None else None
                )
                events.append(
                    dict(
                        start=element.offset,
                        dur=element.duration.quarterLength,
                        end=element.offset + element.duration.quarterLength,
                        midipitch=int(pitch.ps),
                        pitch_str=f"{pitch.name.replace('-', 'b')}{pitch.octave}",
                        accidental=accidental_str,
                        pitch_class=pitch.pitchClass,
                    )
                )
    return events


def events_to_pianoroll_array(
    event_list, divisions_per_quarternote=12, min_pitch=None, min_time=None
) -> Tuple[np.array, int, int]:
    event_df = pd.DataFrame(event_list)

    if min_pitch is None:
        min_pitch = event_df["midipitch"].min()
    event_df.loc[:, "midipitch"] = event_df["midipitch"] - min_pitch

    # time_colnames = ["start", "end", "dur"]
    time_colnames = ["start", "end"]
    event_df.loc[:, time_colnames] = (
        event_df[time_colnames] * divisions_per_quarternote
    ).astype(int)
    if min_time is None:
        min_time = event_df["start"].min()
    event_df.loc[:, time_colnames] = event_df[time_colnames] - min_time

    height = event_df["midipitch"].max() + 1
    width = event_df["end"].max()  # end is not inclusive, so don't need to add 1
    sounding_pr = np.zeros((height, width), dtype=bool)
    onset_pr = sounding_pr.copy()
    for start, end, pitch in event_df[["start", "end", "midipitch"]].itertuples(
        index=False, name=None
    ):
        sounding_pr[pitch, start:end] = 1
        onset_pr[pitch, start] = 1
    return np.stack((sounding_pr, onset_pr), axis=0), min_pitch, min_time


def compress_pianoroll(pr_array, min_pitch) -> np.array:
    max_time = pr_array.shape[-1]
    output_array = np.zeros((max_time, 2), dtype=np.uint64)
    for sounding_or_onset, pitch, time in zip(*np.where(pr_array)):
        pitch_int = min_pitch + pitch
        output_array[time, sounding_or_onset] += pitch_int
    return output_array


# TODO: Write uncompressor and tests


class ABCParser:
    """Takes a string containing a single tune and codifies the information."""

    def __init__(
        self,
        abc_str,
        pianoroll_divisions_per_quarternote=12,
        min_pitch=0,
        min_time=0,
    ):
        self.abc_str = abc_str
        self.abc_handler = music21.abcFormat.ABCHandler()
        self.abc_handler.tokenize(self.abc_str)
        self._tokens_music21 = self.abc_handler.tokens
        self.metadata = [
            token.src
            for token in self._tokens_music21
            if isinstance(token, music21.abcFormat.ABCMetadata)
        ]
        self.events = abc_to_events(abc_str)
        self.pianoroll_divisions_per_quarternote = pianoroll_divisions_per_quarternote
        self.min_pitch = min_pitch
        self.min_time = min_time
        self._pianoroll = None
        self._compressed_pianoroll = None

    @property
    def compressed_pianoroll(self):
        if self._compressed_pianoroll is None:
            self.make_compressed_pianoroll()
        return self._compressed_pianoroll

    @property
    def pianoroll(self):
        if self._pianoroll is None:
            self.make_pianoroll()
        return self._pianoroll

    def make_pianoroll(self):
        self._pianoroll, self.min_pitch, self.min_time = events_to_pianoroll_array(
            self.events,
            divisions_per_quarternote=self.pianoroll_divisions_per_quarternote,
            min_pitch=self.min_pitch,
            min_time=self.min_time,
        )

    def make_compressed_pianoroll(self):
        self._compressed_pianoroll = compress_pianoroll(
            self.pianoroll,
            min_pitch=self.min_pitch,
        )
