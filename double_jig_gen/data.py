"""Dataset preprocessing and item getting classes."""
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

from torch.utils.data import DataLoader

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

    def __init__(self, filepath: PATHLIKE,) -> None:
        """Initialises class.

        Args:
            filepath: path to the file containing the ABC data.
        """
        self._filepath = filepath
        self.data = read_and_rstrip_file(filepath)
        self.tokens = set(self.data.split())
        self.tokenizer = Tokenizer(tokens=self.tokens)
        self.tunes = [self.tokenizer.tokenize(tune) for tune in self.data.split("\n\n")]

    def __str__(self):
        msg = (
            f"vocabulary size: {len(self.tokens)}\n"
            f"vocabulary (each token separated by a space): \n{' '.join(self.tokens)}\n"
            f"dataset_size: {len(self.tunes)}"
        )
        return msg

    def __getitem__(self, index):
        pass

    def __len__(self):
        return 0


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
