"""Dataset preprocessing and item getting classes."""
import logging
from pathlib import Path
from typing import Callable, Collection, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from double_jig_gen.io import PATHLIKE, read_and_rstrip_file
from double_jig_gen.tokenizers import Tokenizer
from double_jig_gen.utils import round_to_nearest_batch_size

LOGGER = logging.getLogger(__name__)


def pad_batch(batch, pad_idx):
    "Function which adds padding to each batch up to the longest sequence."
    lengths = [seq.shape[0] for seq in batch]
    data = pad_sequence(batch, batch_first=False, padding_value=pad_idx)
    return data, lengths


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
        subset: Optional[Collection[int]] = None,
        wrap_tunes: bool = True,
    ) -> None:
        """Initialises class.

        Args:
            filepath: path to the file containing the ABC data.
        """
        self._wrap_tunes = wrap_tunes
        if tunes is None:
            self._filepath = filepath
            self.data = read_and_rstrip_file(filepath)
            self.tunes = [tune.split() for tune in self.data.split("\n\n")]
        else:
            self.tunes = tunes
        if subset is not None:
            self.tunes = np.array(self.tunes, dtype="object")[subset].tolist()
        if tokens is None:
            all_tokens = [token for tune in self.tunes for token in tune]
            self.tokens = set(all_tokens)
        else:
            self.tokens = set(tokens)
        self.tokenizer = Tokenizer(tokens=self.tokens)
        self.tokenized_tunes = [
            self.tokenizer.tokenize(tune, wrap=wrap_tunes) for tune in self.tunes
        ]
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


class FolkRNNDataset(ABCDataset):
    """Expects vocab and splits files to have been made."""

    def __init__(self, filepath: PATHLIKE, subset_name: Optional[str] = None,) -> None:
        filepath = Path(filepath).resolve()
        with open(f"{str(filepath)}_vocabulary.txt", "r") as file_handle:
            tokens = file_handle.read().splitlines()
        if subset_name is not None:
            with open(f"{str(filepath)}_{subset_name}_split.txt", "r") as file_handle:
                subset_indices = [int(idx) for idx in file_handle.read().splitlines()]
        else:
            subset_indices = None

        super().__init__(
            filepath=filepath, tokens=tokens, subset=subset_indices,
        )


def get_folkrnn_dataloaders(
    filepath: PATHLIKE, batch_size: int, num_workers: int, pin_memory: bool,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Returns training, validation, and test dataloaders (respectively) for folkrnn.

    Args:
        batch_size: the number of items within the batch to return.
        num_workers: the number of workers for the dataloaders.
        pin_memory: whether to pin memory.

    Returns:
        train_dataloader, validation_dataloader, test_dataloader: Dataloader classes.
    """
    LOGGER.info("Loading folkrnn train dataset")
    train_dataset = FolkRNNDataset(filepath=filepath, subset_name="train")
    pad_idx = train_dataset.tokenizer.pad_token_index
    LOGGER.info(f"Padding token index read as {pad_idx}")

    def pad_folkrnn_batch(batch):
        return pad_batch(batch, pad_idx=pad_idx)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=pad_folkrnn_batch,
    )
    LOGGER.info("Loading folkrnn validation dataset")
    val_dataset = FolkRNNDataset(filepath=filepath, subset_name="valid")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=pad_folkrnn_batch,
    )
    LOGGER.info("Loading folkrnn test dataset")
    test_dataset = FolkRNNDataset(filepath=filepath, subset_name="test")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=pad_folkrnn_batch,
    )
    return train_dataloader, val_dataloader, test_dataloader


def get_oneills_dataloaders(
    filepath: PATHLIKE,
    folkrnn_vocab_filepath: PATHLIKE,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    val_prop: float = 0.05,
    val_seed: Optional[int] = None,
    val_shuffle: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Returns training, validation, and test dataloaders (respectively) for folkrnn.

    Args:
        batch_size: the number of items within the batch to return.
        num_workers: the number of workers for the dataloaders.
        pin_memory: whether to pin memory.
        val_prop:
        val_seed: seed for doing the validation split. If not set, chosen randomly.
        val_shuffle:
    Returns:
        train_dataloader, validation_dataloader, test_dataloader: Dataloader classes.
    """
    LOGGER.info(f"Reading folkrnn vocabulary from {folkrnn_vocab_filepath}.")
    with open(folkrnn_vocab_filepath, "r") as file_handle:
        tokens = file_handle.read().splitlines()
    LOGGER.info("Loading oneills dataset and creating train/test split")
    oneills_size = 361
    test_prop = 0.1
    test_idx, train_idx = split_array(list(range(oneills_size)), test_prop, batch_size)
    test_dataset = ABCDataset(filepath=filepath, tokens=tokens, subset=test_idx)
    LOGGER.info(f"Test dataset:\n{test_dataset}")
    if val_prop == 1:
        LOGGER.info(
            "Using the full training dataset for both training and validation. "
            "In this context we are training and validating on the whole dataset. "
            "val_shuffle will be set to true such that the whole dataset is used. "
            "TIP: use pytorch lightning limit_train_batches and limit_val_batches."
        )
        val_idx = train_idx
        train_dataset = ABCDataset(filepath, tokens=tokens, subset=train_idx)
        val_dataset = ABCDataset(filepath, tokens=tokens, subset=val_idx)
        val_shuffle = True
    else:
        if val_seed is None:
            val_seed = np.random.randint(0, 2 ** 32 - 1)
        val_idx, train_idx = split_array(train_idx, val_prop, batch_size, val_seed)
        LOGGER.info(
            f"Splitting training set into a train val set of {len(train_idx)} and "
            f"{len(val_idx)} respectively. Used seed {val_seed}"
        )
        train_dataset = ABCDataset(filepath, tokens=tokens, subset=train_idx)
        val_dataset = ABCDataset(filepath, tokens=tokens, subset=val_idx)

    pad_idx = test_dataset.tokenizer.pad_token_index
    LOGGER.info(f"Padding token index read as {pad_idx}")

    def _pad_batch(batch):
        return pad_batch(batch, pad_idx=pad_idx)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_pad_batch,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=val_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_pad_batch,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_pad_batch,
    )
    return train_dataloader, val_dataloader, test_dataloader


def split_array(array, prop, batch_size, seed=None):
    nr_items = len(array)
    nr_train = round_to_nearest_batch_size(nr_items, prop, batch_size)
    rng = np.random.RandomState(seed)
    rng.shuffle(array)
    train_idx, test_idx = np.split(array, [nr_train])
    return train_idx, test_idx
