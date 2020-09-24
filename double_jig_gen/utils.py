"""Utility functions."""
import json
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pytorch_lightning as pl
import requests

LOGGER = logging.getLogger(__name__)


def save_args(
    filepath: Union[str, Path], args: Union[dict, Namespace], **json_kwargs
) -> None:
    """Saves all key value pairs in a dict or an argparse.Namespace to a json file.

    Args:
        filepath: the path for the file to save the args to.
        args: either a dict or argparse Namespace containing args to save.
        **json_kwargs: all further keyword arguments are passed to the json.dump() call.
    """
    filepath = Path(filepath).resolve()
    if not filepath.parent.exists():
        raise ValueError(f"{filepath.parent} doesn't exist.")
    elif not filepath.parent.is_dir():
        raise ValueError(f"{filepath.parent} is not a directory.")

    if isinstance(args, Namespace):
        args = vars(args)

    # TODO: handle endoding invalid data for json
    with open(filepath, "w") as fp:
        json.dump(args, fp, **json_kwargs)


def load_key_value_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Returns a dictionary of a json file of keys and values.

    Args:
        filepath: path to the file to load.
    """
    with open(filepath) as fp:
        args = json.load(fp)
    # TODO: handle decoding invalid data for json
    return args


def load_args(filepath: Union[str, Path]) -> Namespace:
    """Returns argparse Namespace of a json file.

    Args:
        filepath: path to the file to load.
    """
    args_dict = load_key_value_json(filepath)
    args = Namespace()
    args_vars = vars(args)
    args_vars.update(args_dict)  # this updates the underlying args Namespace
    return args


def get_trainer_from_checkpoint(
    ckpt_path: Union[str, Path], **trainer_kwargs,
) -> pl.Trainer:
    """Returns a lightning trainer object loaded with parameters from checkpoint.

    Note that, when called with .fit(), this trainer will load the parameters
    to the model contained within the checkpoint too.

    Args:
        ckpt_path: the path to the checkpoint file to load.
        trainer_kwargs: keyword arguments to additionally provide to the trainer. For
            instance, the gpu configuration is not automatically reloaded, so this
            should be reloaded with gpus=nr_gpus.
    Returns:
        lightning_trainer: the trainer object.
    """
    LOGGER.info(
        "Restoring training from pytorch_lightning trainer using checkpoint %s.",
        ckpt_path,
    )
    if not Path(ckpt_path).exists():
        raise ValueError(f"Checkpoint file {ckpt_path} doesn't exist.")
    lightning_trainer = pl.Trainer(
        resume_from_checkpoint=str(ckpt_path), **trainer_kwargs
    )
    return lightning_trainer


def get_model_from_checkpoint(
    ckpt_path: Union[str, Path], ModelClass: pl.LightningModule
) -> pl.LightningModule:
    """Instantiates a lightning LightningModule model from a checkpoint.

    Args:
        ckpt_path: the path to the checkpoint file to load.
        ModelClass: the model class to load the checkpoint file to.

    Returns:
        model: the instantiated model.
    """
    LOGGER.info(
        "Restoring model %s using checkpoint %s.", ModelClass.__name__, ckpt_path,
    )
    if not Path(ckpt_path).exists():
        raise ValueError(f"Checkpoint file {ckpt_path} doesn't exist.")
    model = ModelClass.load_from_checkpoint(checkpoint_path=str(ckpt_path))
    return model


def download_file(url, local_filename=None, min_size_bytes=100):
    """Downloads file at url."""
    if local_filename is None:
        local_filename = url.rsplit("/")[-1]
    file_exists = Path(local_filename).exists()
    file_corrupt = False
    if file_exists:
        LOGGER.info("file already exists at %s, not downloading", local_filename)
        file_size = Path(local_filename).stat().st_size
        if file_size < min_size_bytes:
            LOGGER.warning(
                "file less than %d bytes, downloading again.", min_size_bytes
            )
            file_corrupt = True
    if (not file_exists) or file_corrupt:
        req_result = requests.get(url, allow_redirects=True)
        with open(f"{local_filename}", "wb") as filehandle:
            filehandle.write(req_result.content)


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
