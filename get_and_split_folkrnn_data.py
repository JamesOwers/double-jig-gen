#!/usr/bin/env python
"""Script to download folkrnn data, extract the vocab, and define splits."""
import fire
import logging
from pathlib import Path

import numpy as np

from double_jig_gen.utils import download_file

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel("DEBUG")

FOLKRNN_V3_URL = (
    "https://raw.githubusercontent.com/IraKorshunova/folk-rnn/master/data/data_v3"
)


def round_to_nearest_batch_size(
    total: int,
    prop: float,
    batch_size: int,
):
    float_count = total * prop
    # round to a multiple of batch_size
    rounded_count = batch_size * max(
        1,
        int(np.rint(float_count / batch_size))
    )
    return rounded_count


def download_get_tokens_and_make_splits(
    data_path: str = './data_v3',
    url: str = FOLKRNN_V3_URL,
    seed: int = 42,
    length_percentile: float = .99,
    test_prop: float = .05,
    valid_prop: float = .05,
    train_prop: float = .9,
    batch_size: int = 64,
) -> None:
    """"""
    assert test_prop + valid_prop + train_prop == 1, (
        "train, valid, and test prop must sum to 1"
    )
    
    data_path = Path(data_path).resolve()
    data_path.parent.mkdir(parents=True, exist_ok=True)
    download_file(url, data_path)
    
    with open(data_path, 'r') as fh:
        data = fh.read()

    tunes = [tune.split() for tune in data.split('\n\n')]
    # Save vocabulary prior to subsetting
    LOGGER.info("Extracting vocabulary")
    all_tokens = [token for tune in tunes for token in tune]
    all_tokens = sorted(list(set(all_tokens)))
    LOGGER.info(f"vocabulary size: {len(all_tokens)}")
    LOGGER.info(
        f"vocabulary (each token separated by a space): \n{' '.join(all_tokens)}"
    )
    filepath = str(data_path) + f"_vocabulary.txt" 
    with open(filepath, "w") as file_handle:
        file_handle.write("\n".join(all_tokens))
    
    tune_lens = np.array([len(t) for t in tunes])

    nr_kept = int(np.rint(len(tune_lens)*length_percentile))
    max_tune_len = sorted(tune_lens)[nr_kept - 1]
    LOGGER.info(
        f"Choosing the shortest {length_percentile*100}% of tunes for train/valid/test"
    )
    LOGGER.info(f"They are all {max_tune_len} tokens long or less")
    tune_indexes = np.array(
        [idx for idx, tune in enumerate(tunes) if len(tune) <= max_tune_len]
    )
    
    nr_tunes = len(tune_indexes)
    LOGGER.info(f"Selecting from a total of {nr_tunes} tunes")
    LOGGER.info(
        f"Rounding test and validation splits to nearest batch size of {batch_size}"
    )
    nr_test = round_to_nearest_batch_size(nr_tunes, test_prop, batch_size)
    nr_valid = round_to_nearest_batch_size(nr_tunes, valid_prop, batch_size)
    LOGGER.info(f"Seeding rng with value {seed} and shuffling indexes prior ot split")
    rng = np.random.RandomState(seed)
    rng.shuffle(tune_indexes)
    test_idx, valid_idx, train_idx = np.split(
        tune_indexes,
        [nr_test, nr_test+nr_valid],
    )
    split_name_to_idx = {
        'test': test_idx,
        'valid': valid_idx,
        'train': train_idx,
    }
    split_sizes = {name: len(idx) for name, idx in split_name_to_idx.items()}
    split_proportions = {
        name: f"{len(idx)/nr_tunes:.3f}" for name, idx in split_name_to_idx.items()
    }
    LOGGER.info(
        f"Split sizes: {split_sizes}"
    )
    LOGGER.info(
        f"Split proportions: {split_proportions}"
    )
    for split_name, idx in split_name_to_idx.items():
        filepath = str(data_path) + f"_{split_name}_split.txt" 
        with open(filepath, "w") as file_handle:
            file_handle.write("\n".join([str(ii) for ii in idx]))

            
if __name__ == "__main__":
    fire.Fire(download_get_tokens_and_make_splits)
    