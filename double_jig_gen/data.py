"""Dataset preprocessing and item getting classes."""
from pathlib import Path
from typing import Callable, Collection, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from double_jig_gen.io import PATHLIKE, read_and_rstrip_file
from double_jig_gen.tokenizers import Tokenizer

DEFAULT_TOKENS = ()


# TODO: enable adding them together trivially
# TODO: separate out pytorch datasets and those that dont need to be
class ABCDataset:
    """Parser for an ABC dataset contained in a single file.

    Assumes that each data item is on contiguous lines i.e. data items are separated
    by at least one blank line.

    Attributes:

    """

    def __init__(
        self,
        filepath: Optional[PATHLIKE] = None,
        tunes: Optional[List[str]] = None,
        tokens: Optional[Collection[str]] = None,
    ) -> None:
        """Initialises class.

        Args:
            filepath: path to the file containing the ABC data.
        """
        if tunes is None:
            self._filepath = filepath
            self.data = read_and_rstrip_file(filepath)
            self.tunes = [tune.split() for tune in self.data.split("\n\n")]
        else:
            self.tunes = tunes

        if tokens is None:
            all_tokens = [token for tune in self.tunes for token in tune]
            self.tokens = set(all_tokens)
        else:
            self.tokens = set(tokens)
        self.tokenizer = Tokenizer(tokens=self.tokens)
        self.tokenized_tunes = [self.tokenizer.tokenize(tune) for tune in self.tunes]
        self.nr_tunes = len(self.tunes)
        self.tune_lengths = [len(tune) for tune in self.tokenized_tunes]
        self.mean_tune_len = np.mean(self.tune_lengths)
        self.median_tune_len = np.median(self.tune_lengths)
        self.max_tune_len = np.max(self.tune_lengths)
        self.min_tune_len = np.min(self.tune_lengths)
        self.vocabulary_size = len(self.tokenizer.tokens)
    
    def __str__(self):
        tokens = self.tokenizer.tokens  # this is self.tokens plus special tokens
        msg = (
            f"vocabulary size: {self.vocabulary_size}\n"
            f"vocabulary (each token separated by a space): \n{' '.join(tokens)}\n"
            f"dataset_size: {len(self)}\n"
            f"tune length stats:\n\t* max {self.max_tune_len}"
            f"\n\t* mean {self.mean_tune_len}"
            f"\n\t* median {self.median_tune_len}"
            f"\n\t* min {self.min_tune_len}"
        )
        return msg

    def __getitem__(self, idx):
        tune = self.tokenized_tunes[idx]
        return torch.Tensor(tune).long()
    
    def __len__(self):
        return self.nr_tunes


class FolkRNNDataset:
    def __init__(self, subset):
        pass


def get_folkrnn_dataloaders(
    batch_size: int, num_workers: int, pin_memory: bool,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Returns training, validation, and test dataloaders (respectively) for folkrnn.

    Args:
        batch_size: the number of items within the batch to return.
        num_workers: the number of workers for the dataloaders.
        pin_memory: whether to pin memory.

    Returns:
        train_dataloader, validation_dataloader, test_dataloader: Dataloader classes.
    """
    train_dataset = FolkRNNDataset(subset="train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dataset = FolkRNNDataset(subset="valid")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_dataset = FolkRNNDataset(subset="test")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_dataloader, val_dataloader, test_dataloader


def get_oneills_dataloaders():
    pass
