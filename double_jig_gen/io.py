"""Functions for reading and writing data."""
from pathlib import Path
from typing import Callable, Optional, Union

PATHLIKE = Union[str, Path]


def get_abs_path_str(path: PATHLIKE) -> str:
    """Resolve path and return a string."""
    path = Path(path)
    return str(path.expanduser().resolve())


def read_and_rstrip_file(filepath: PATHLIKE) -> str:
    """Reads file line-by-line and strips the right-hand-side."""
    with open(filepath, "r") as filehandle:
        lines = [line.rstrip() for line in filehandle.readlines()]
    return "\n".join(lines)
