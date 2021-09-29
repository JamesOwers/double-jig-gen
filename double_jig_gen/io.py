"""Functions for reading and writing data."""
import logging
import os
import sys
from functools import wraps
from io import StringIO
from pathlib import Path
from typing import List

import music21
from joblib import Parallel, delayed
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
# print(music21.sites.WEAKREF_ACTIVE = False)
# music21.sites.WEAKREF_ACTIVE = False


def get_abs_path_str(path: os.PathLike) -> str:
    """Resolve path and return a string."""
    path = Path(path)
    return str(path.expanduser().resolve())


def read_and_rstrip_file(filepath: os.PathLike) -> str:
    """Reads file line-by-line and strips the right-hand-side."""
    with open(filepath, "r") as filehandle:
        lines = [line.rstrip() for line in filehandle.readlines()]
    return "\n".join(lines)


class CapturingStderr(list):
    def __enter__(self):
        self._stderr = sys.stderr
        sys.stderr = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio
        sys.stderr = self._stderr


def exception_handler(func):
    @wraps(func)
    def wrapper_function(*args, error_handling: str = "warn", **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exception:
            if error_handling == "warn":
                LOGGER.warning(exception)
                return None
            else:
                raise exception

    return wrapper_function


@exception_handler
def abc_to_music_21_score(abc_str: str) -> music21.stream.Score:
    if abc_str.strip() == "":
        raise ValueError("abc_data can't be an empty string")
    try:
        with CapturingStderr() as music21_warnings:
            music21_score = music21.converter.parse(abc_str, format="abc")
    except Exception as e:
        raise ValueError(
            f"music21.converter.parse({repr(abc_str)}, format='abc') failed:\n{repr(e)}"
        )
    if music21_warnings:
        raise ValueError(
            f"music21.converter.parse({repr(abc_str)}, format='abc') raised "
            f"warnings, this usually leads to incorrect data:\n{music21_warnings}"
        )
    return music21_score


def multi_abc_file_to_music21_scores(
    filepath: os.path,
    tune_sep: str = "\n\n",
    error_handling: str = "warn",
    n_jobs: int = -1,
    no_progress_bar: bool = False,
) -> List[music21.stream.Score]:
    with open(filepath, "r") as file_object:
        abc_string_list = file_object.read().strip().split(tune_sep)
    # Threading backend required for now:
    # https://stackoverflow.com/questions/69327363/python-joblib-returning-typeerror-cannot-pickle-weakref-object-with-the-sam  # noqa
    return Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(abc_to_music_21_score)(abc_string)
        for abc_string in tqdm(
            abc_string_list,
            desc="Reading abc tunes",
            disable=no_progress_bar,
        )
    )
