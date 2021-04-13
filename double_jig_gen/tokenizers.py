"""Classes which turn strings into lists of tokens."""
import logging
import re
from typing import (
    Any,
    Collection,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import music21
import numpy as np
import pandas as pd

from .utils import human_round

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


def abc_to_events(abc_str: str) -> List[Dict[str, Any]]:
    """Converts string of abc data to a list of note events.

    A note event describes information about a note, such as its start time and pitch.
    Note event information is stored in a dictionary. The resulting list of events can
    be read as a pd.DataFrame by wrapping the function output in a pd.DataFrame call
    e.g. pd.DataFrame(abc_to_events(abc_str)). All times are measured in number of
    quavers (quarters). End times are not inclusive.

    The parsing of the abc data is handled by music21.converter.parse, therefore the
    input abc_str must be valid according to this method. See [1] for information about
    valid abc data.

    For example, an abc tune consisting of a two middle C semiquavers followed by a
    middle C quaver played at the same time as a concert A quavers followed by two
    concert A semiquavers could be represented in the following way, and produce the
    following output:

    >>> abc_str = "L:1/16\nV:1\nCC C2\nV:2\nA2 AA\n"
    >>> abc_to_events(abc_str)
    [
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
    ]

    Args:
        abc_str: a string containing valid abc data

    Returns:
        events: a list of dictionaries. Each dictionary contains information about the
            note event.

    References:
    [1]: http://abcnotation.com/wiki/abc:standard:v2.1
    """
    # TODO: get position within measure
    # TODO: get pitch spelling information e.g. position within the scale, and whether
    # an accidental is applied *with respect to the scale*
    abc = music21.converter.parse(abc_str, format="abc")
    # N.B. getting flat attr on abc makes offset times stored in a given element.offset
    # relative to the start of the piece, rather than relative to the containing stream.
    note_stream = abc.flat.getElementsByClass(["Note", "Chord"])
    events = []
    for element in note_stream:
        if isinstance(element, music21.note.Note):
            pitches = [element.pitch]
        elif isinstance(element, music21.chord.Chord):
            pitches = element.pitches
        for pitch in pitches:
            accidental_str = (
                pitch.accidental.fullName if pitch.accidental is not None else None
            )
            events.append(
                dict(
                    start=element.offset,
                    duration=element.duration.quarterLength,
                    pitch=int(pitch.ps),
                    pitch_str=f"{pitch.name.replace('-', 'b')}{pitch.octave}",
                    pitch_class=pitch.pitchClass,
                    accidental=accidental_str,
                )
            )
    return events


def events_to_pianoroll_array(
    event_list: List[Dict[str, Any]],
    divisions_per_quarternote: int = 12,
    min_pitch: Optional[int] = None,
    min_time: Optional[float] = None,
) -> Tuple[np.array, int, int]:
    """Converts list of note events into a pianoroll.

    The list of note events must contain start times (in number of quarters)
    The resulting pianoroll is of shape: (2, nr_pitches, nr_timepoints) i.e. a stack of
    2 matrices which are each of shape (nr_pitches, nr_timepoints) the first matrix at
    index 0 indicates where a pitch should be sounding (1 for sounding, 0 for silent),
    the second matrix at index 1 indicates where a pitch begins. For example:

    >>> event_list = abc_to_event_list("L:1/16\nV:1\nCC C2\nV:2\nE2 _E^D")
    >>> pianoroll, min_pitch, min_time = events_to_pianoroll_array(event_list)
    >>> pianoroll.astype(int)
    array([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]],

           [[1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])
    >>> min_pitch
    60
    >>> min_time
    0

    Args:
        event_list: list of note events. Each note event should be a dictionary with
            keys "start", "duration", and "pitch", determining the start time of the
            note, the duration that the note should be held, and the pitch the note
            should sound at respectively. Times are expected to be in numbers of
            quarternotes.
        divisions_per_quarternote: the number of divisions to make per quarternote in
            the resulting pianoroll i.e. this defines the resolution of the pianoroll.
            For example, if divisions_per_quarternote is 12, each index in the time axis
            (the last axis) of the pianoroll will amount to 1/12 quarternotes of time.
        min_pitch: the integer midinote pitch number of the first index of the pitch
            axis of the resulting pianoroll. If left the default of None, the minimum
            pitch found will be used.
        min_time: the floating point time, in number of quarternotes, of the first index
            of the time axis of the resulting pianoroll. If left the default of None,
            the minimum time found will be used.

    Returns:
        pianoroll: the resulting pianoroll
        min_pitch: the integer midinote pitch number of the first index of the pitch
            axis of the resulting pianoroll.
        min_time: the floating point time, in number of quarternotes, of the first index
            of the time axis of the resulting pianoroll.
    """
    event_df = pd.DataFrame(event_list)
    event_df["end"] = event_df["start"] + event_df["duration"]
    if min_pitch is None:
        min_pitch = event_df["pitch"].min()
    event_df.loc[:, "pitch"] = event_df["pitch"] - min_pitch

    time_colnames = ["start", "end"]
    # TODO: check if divisions_per_quarternote is satisfactory (i.e. large enough) and,
    # if not, throw a warning
    event_df.loc[:, time_colnames] = (
        (event_df[time_colnames] * divisions_per_quarternote)
        .applymap(lambda x: int(human_round(x, 0)))
        .astype(int)
    )
    if min_time is None:
        min_time = event_df["start"].min()
    event_df.loc[:, time_colnames] = event_df[time_colnames] - min_time

    height = event_df["pitch"].max() + 1
    width = event_df["end"].max()  # end is not inclusive, so don't need to add 1
    sounding_pr = np.zeros((height, width), dtype=bool)
    onset_pr = sounding_pr.copy()
    for start, end, pitch in event_df[["start", "end", "pitch"]].itertuples(
        index=False, name=None
    ):
        sounding_pr[pitch, start:end] = 1
        onset_pr[pitch, start] = 1
    return np.stack((sounding_pr, onset_pr), axis=0), min_pitch, min_time


def compress_pianoroll(pianoroll: np.array) -> np.array:
    """Compresses a pianoroll into a sequence of integers.

    Compresses the pitch dimension of the pianoroll by encoding it as an integer. The
    integer is the decimal representation of the bianary number represented by the
    column. For example:

    >>> abc_str = "L:1/16\nV:1\nCC C2\nV:2\nE2 _E^D\n"
    >>> pianoroll, min_pitch, min_time = events_to_pianoroll_array(event_list, 8)
    >>> pianoroll.astype(int)
    array([[[1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0]],

           [[1, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 0]]])
    >>> compressed_pianoroll = compress_pianoroll(pianoroll)
    >>> compressed_pianoroll
    array([[17, 17, 17, 17,  9,  9,  9,  9],
           [17,  0,  1,  0,  9,  0,  8,  0]], dtype=uint64)

    At the second to last time point, i.e. compressed_pianoroll[:, -2], we have two
    numbers: (9, 8). The first represents the pitches sounding: 9 = 2**0 (the pitch at
    index 0) + 2**3 (the pitch at index 3), which is, in binary, 01001 = 00001 + 01000.
    Similarly, for the pitches beginning, we have 8 = 2**3, or 01000.

    Args:
        pianoroll: of shape (2, nr_pitches, nr_timepoints) i.e. a stack of 2 matrices
            which are each of shape (nr_pitches, nr_timepoints) the first matrix at
            index 0 indicates where a pitch should be sounding (1 for sounding, 0 for
            silent), the second matrix at index 1 indicates where a pitch begins.
        min_pitch: the integer midinote pitch number of the first index of the pitch
            axis of the input pianoroll.

    Returns:
        output_array: of shape (2, nr_timepoints)
    """
    max_time = pianoroll.shape[-1]
    output_array = np.zeros((2, max_time), dtype=np.uint64)
    for sounding_or_onset, pitch, time in zip(*np.where(pianoroll)):
        pitch_int = 2 ** pitch
        output_array[sounding_or_onset, time] += pitch_int
    return output_array


def decompress_pianoroll(compressed_pianoroll: np.array) -> np.array:
    """Reverses the compression process described in compress_pianoroll."""
    max_int = np.max(compressed_pianoroll)
    # int truncates => this is largest power of two <= max_int
    max_power_of_2 = int(np.log2(max_int))
    pitch_dimension_size = max_power_of_2 + 1

    def binarise_element(array_element: int) -> np.array:
        binary_string = np.binary_repr(array_element, width=pitch_dimension_size)
        integer_data = np.frombuffer(bytes(binary_string, "utf8"), dtype="uint8")
        big_endian_binary_data = integer_data - ord("0")
        little_endian_binary_data = big_endian_binary_data[::-1]
        return little_endian_binary_data

    list_of_binary_arrays = [
        binarise_element(element) for element in compressed_pianoroll.ravel()
    ]
    time_dimension_size = compressed_pianoroll.shape[-1]
    pianoroll = (
        np.array(list_of_binary_arrays, dtype=bool)
        .reshape((2, time_dimension_size, pitch_dimension_size))
        .swapaxes(1, 2)
    )
    return pianoroll


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
