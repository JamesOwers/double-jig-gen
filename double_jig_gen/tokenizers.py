"""Classes which turn strings into lists of tokens."""
import logging
from typing import Collection, List, Optional, Sequence, Union

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
        self, str_or_sequence: Union[str, Sequence[str]], wrap: bool = True,
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
