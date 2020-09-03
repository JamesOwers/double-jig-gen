"""Dataset preprocessing and item getting classes."""
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

from double_jig_gen.io import PATHLIKE, read_and_rstrip_file
from double_jig_gen.tokenizers import Tokenizer

DEFAULT_TOKENS = ()


# TODO: enable adding them together trivially
class ABCDataset:
    """Parser for an ABC dataset contained in a single file.

    Assumes that each data item is on contiguous lines i.e. data items are separated
    by at least one blank line.

    Attributes:

    """

    def __init__(
        self,
        filepath: PATHLIKE,
        tokens: Optional[Sequence[str]] = None,
        tokenizer: Optional[Tokenizer] = None,
    ):
        """Initialises class.

        Args:
            filepath: path to the file containing the ABC data.
        """
        self._filepath = filepath
        self.data = read_and_rstrip_file(filepath)
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif tokens is not None:
            self.tokenizer = Tokenizer(tokens=tokens)
        else:
            self.tokenizer = Tokenizer(tokens=DEFAULT_TOKENS)

    def __getitem__(self, index):
        pass

    def __len__(self):
        return 0
