#!/usr/bin/env python
"""Script to train models for sequence models.

Boilerplate code for training a model (selected with `--model`) defined as a
pytorch_lightning.LightningModule with a pytorch_lightning.Trainer. Also allows for
the loading of different datasets (selected with `--dataset`).

# Usage
Arguments passed to the model, dataloaders, and pytorch_lightning.Trainer objects are
exposed and available to set on the command line. These are all saved as part of the
logging process. See # Logs below for more information.

See the pytorch lightning documentation for descriptions of the arguments available to
use 'out-of-the-box':

    https://pytorch-lightning.readthedocs.io/en/stable/trainer.html#trainer-flags

Example call:

    # See man page
    python train.py --help

    # Rule of thumb is to set --num_workers to the number of cpu cores you have
    python training/source_separation/train.py \
        --num_workers 4 \
        --max_epochs 10 \
        --model tranformer \
        --dataset folkrnn

# Logs
Logs will now be available in ./lightning_logs, and viewable with tensorboard at
localhost:6006 (or whichever port tensorboard started on). You may start tensorboard
by executing:

    tensorboard --logdir lightning_logs/

This script also saves all the args the experiment was called with to the
trainer's logging directory. They can be manually loaded using:

    from double_jig_gen.utils import load_args
    expt_args_path = "lightning_logs/version_0/experiment_args.json"
    args = load_args(expt_args_path)

# Checkpoints
The checkpoints automatically saved by pytorch lightning contain the following keys:

    ['epoch', 'global_step', 'pytorch-ligthning_version' (SIC),
    'checkpoint_callback_best_model_score', 'checkpoint_callback_best_model_path',
    'early_stop_callback_wait', 'early_stop_callback_patience', 'optimizer_states',
    'lr_schedulers', 'state_dict', 'hparams_name', 'hyper_parameters']

They can be loaded manually using:

    from pytorch_lightning.utilities.cloud_io import load as pl_load
    ckpt_path = "lightning_logs/version_0/checkpoints/epoch=1.ckpt"
    ckpt = pl_load(ckpt_path)

Keys within checkpoint of greatest interest:

    state_dict: this contains all the learned parameters to load to the pytorch model.
    hyper_parameters: this is a dict which can be used to instantiate the pytorch model
        with kwargs exactly as it was initially instantiated, i.e.
        Model(**ckpt["hyper_parameters"]). This is created from model.hparams (which is
        made with the self.save_hyperparameters() in the model class).
"""
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Tuple, Type, Union

import numpy as np
import pytorch_lightning as pl
import torch

from double_jig_gen.data import get_folkrnn_dataloaders, get_oneills_dataloaders
from double_jig_gen.models import SimpleRNN, Transformer
from double_jig_gen.utils import (
    get_model_from_checkpoint,
    get_trainer_from_checkpoint,
    save_args,
)

LOGGER = logging.getLogger(__name__)
MODELS: Dict[str, Union[Type[SimpleRNN], Type[Transformer]]] = {
    "rnn": SimpleRNN,
    "transformer": Transformer,
}
DATASETS = ["folkrnn", "oneills"]


def add_user_args(parent_parser: ArgumentParser) -> ArgumentParser:
    """Adds arguments to the given parser for training.

    These should not conflict with the args provided with pl.Trainer.add_argparse_args.
    If this is done by accident, an ArgumentError will be raised in the main call.
    """
    new_parser = ArgumentParser(parents=[parent_parser], add_help=False)
    new_parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of data items in the batch for the training dataloader.",
    )
    new_parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers to use for the dataloader.",
    )
    new_parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=50,
        help=(
            "maximum number of epochs to train after no improvement in validation "
            "loss."
        ),
    )
    new_parser.add_argument("--pin_memory", default=None, action="store_true", help="")
    new_parser.add_argument(
        "--no_progress_bars",
        default=False,
        action="store_true",
        help="Flag to supress tqdm progress bars.",
    )
    new_parser.add_argument(
        "--model",
        type=str,
        default=list(MODELS.keys())[0],
        help="Name of the model to fit.",
        choices=MODELS.keys(),
    )
    new_parser.add_argument(
        "--dataset",
        type=str,
        default=DATASETS[0],
        help="Name of the dataset to use.",
        choices=DATASETS,
    )
    new_parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Whether to evaluate on test set.",
    )
    new_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="[0,2^32-1]",
        help=(
            "Seed to use for random initialisation for reproducibility. If not set "
            "a randomly selected value will be used and recorded."
        ),
    )
    new_parser.add_argument(
        "--model_load_from_checkpoint",
        default=None,
        type=str,
        help=(
            "Path to checkpoint file to instantiate model but *not* trainer. This "
            "begins training from epoch zero, but initialises the model with the "
            "learned parameters from the checkpoint first."
        ),
    )
    new_parser.add_argument(
        "--trainer_resume_from_checkpoint",
        default=None,
        type=str,
        help=(
            "Path to checkpoint file to instantiate trainer and model. Automatically "
            "restores model, epoch, step, LR schedulers, apex, etc. This is used, for "
            "example, for continuing training which has prematurely ended."
        ),
    )
    new_parser.add_argument(
        "--log_level",
        default="INFO",
        type=str,
        help="Logging level to set for the logger.",
    )
    
    new_parser.add_argument(
        "--folkrnn_data_path",
        type=str,
        help="Location of the folkrnn data and adjacent splits and vocab files.",
    )
        
    new_parser.add_argument(
        "--oneills_data_path",
        type=str,
        help="Location of the oneills data.",
    )
        
    new_parser.add_argument(
        "--val_prop",
        default=0.05,
        type=float,
        help=(
            "Oneills data - proportion of the training set to use for validation, "
            "set to 1 to use the full dataset for both training and validaiton."
        ),
    )
    new_parser.add_argument(
        "--val_shuffle",
        default=False,
        action="store_true",
        help=(
            "Oneills data - whether to shuffle the validation set. Makes sense if "
            "using the full set for both test and train."
        ),
    )
    return new_parser


def parse_and_validate_args() -> Namespace:
    """Creates parser, and parses and checks args provided.

    Adds all args for pytorch_lightning.Trainer, for the model selected, and other user
    defined args.
    """
    parser = ArgumentParser(description="Model trainer.")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = add_user_args(parser)

    # Select model to fit and add args for use on command line defined within the
    # model's classmethod .add_model_specific_args()
    temp_args, _ = parser.parse_known_args()
    if temp_args.model not in MODELS.keys():
        raise NotImplementedError(
            f"Model {repr(temp_args.model)} is not known. Only {MODELS.keys()} have "
            "been configured thus far."
        )
    else:
        ModelClass = MODELS[temp_args.model]
        parser = ModelClass.add_model_specific_args(parser)

    # This overrides default pytorch_lightning behaviour and uses all gpus if they exist
    gpus_default = -1 if torch.cuda.is_available() else None
    parser.set_defaults(gpus=gpus_default)
    args = parser.parse_args()

    # args.gpus is None if using cpu:
    #     Set pin_memory to True if using gpu and False if using cpu by default unless
    #     user has set pin_memory themselves.
    if (args.gpus is not None) and (args.pin_memory is None):
        LOGGER.info("Automatically setting pin_memory to True.")
        args.pin_memory = True
    else:
        args.pin_memory = False

    # Progress bars are on by default. Also remove pytorch_lightning if user requests.
    if args.no_progress_bars:
        args.progress_bar_refresh_rate = 0

    # Initialise modules with RNGs with random seed provided
    if args.seed is None:
        args.seed = np.random.randint(0, 2 ** 32 - 1)
        LOGGER.info("No random seed was specified. Selecting one at random.")
    LOGGER.info("The random seed %d was used to seed modules with RNGs.", args.seed)
    pl.seed_everything(seed=args.seed)

    if (
        args.model_load_from_checkpoint is not None
        and args.trainer_resume_from_checkpoint is not None
    ):
        raise ValueError(
            "Either --model_load_from_checkpoint or "
            "--trainer_resume_from_checkpoint, not both."
        )

    return args


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    args = parse_and_validate_args()

    LOGGER.info("Setting logging level to: %s", args.log_level)
    logging.basicConfig(level=args.log_level)
    LOGGER.setLevel(args.log_level)

    text_args = [f"{name} = {val}" for name, val in vars(args).items()]
    LOGGER.info(
        "Running experiment with the following config:\n %s", "\n".join(text_args)
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=args.early_stopping_patience,
        verbose=True,
        mode="min",
    )
    if args.trainer_resume_from_checkpoint is not None:
        ckpt_path = Path(args.trainer_resume_from_checkpoint).resolve()
        lightning_trainer = get_trainer_from_checkpoint(
            ckpt_path,
            gpus=args.gpus,
            max_epochs=args.max_epochs,
            deterministic=True,
            early_stop_callback=early_stop_callback,
        )
    else:
        lightning_trainer = pl.Trainer.from_argparse_args(
            args, deterministic=True, early_stop_callback=early_stop_callback
        )
    
    LOGGER.info(f"Loading '{args.dataset}' dataset and getting dataloaders")
    if args.dataset == "folkrnn":
        dataloaders = get_folkrnn_dataloaders(
            filepath=args.folkrnn_data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        train_dataloader, val_dataloader, test_dataloader = dataloaders
    if args.dataset == "oneills":
        dataloaders = get_oneills_dataloaders(
            filepath=args.oneills_data_path,
            folkrnn_vocab_filepath=str(args.folkrnn_data_path) + "_vocabulary.txt",
            val_prop=args.val_prop,
            val_seed=args.seed,
            val_shuffle=args.val_shuffle,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        train_dataloader, val_dataloader, test_dataloader = dataloaders
    else:
        raise NotImplementedError(f"{args.dataset} is not a configured dataset.")
    
    # modelling args to read from the dataset - same in all datasets
    if args.embedding_padding_idx is None:
        args.embedding_padding_idx = train_dataloader.dataset.tokenizer.pad_token_index
    if args.ntoken is None:
        args.ntoken = train_dataloader.dataset.vocabulary_size
    if args.model_batch_size is None:
        args.model_batch_size = args.batch_size
    
    # All the keyword arguments for each model are defined within its classmethod
    # .add_model_specific_args() and added to args in parse_and_validate_args() above.
    ModelClass = MODELS[args.model]
    if args.model_load_from_checkpoint is not None:
        ckpt_path = Path(args.model_load_from_checkpoint).expanduser().resolve()
        model = get_model_from_checkpoint(ckpt_path, ModelClass)
        model.model_batch_size = args.model_batch_size  # could change between runs
    else:
        # TODO: There's totally a more transparent way of doing this. Look into
        # using inspect.signature to avoid the creation of instantiate_from_namespace.
        # TODO: currently, we must state ntoken, and embedding_padding_idx from command line
        #       this should be shifted to being read from dataset
        model = ModelClass.instantiate_from_namespace(args)

    experiment_args_path = Path(
        lightning_trainer.logger.log_dir, "experiment_args.yaml"
    )
    experiment_args_path.parent.mkdir(parents=True)
    LOGGER.info("Saving experiment call args to %s", experiment_args_path)
    pl.core.saving.save_hparams_to_yaml(experiment_args_path, args)

    if args.test:
        LOGGER.info("%s Testing (not_training) %s", 30 * "=", 30 * "=")
        lightning_trainer.test(
            model,
            test_dataloaders=test_dataloader,
#             ckpt_path="/disk/scratch_fast/s0816700/logs/lightning_logs/version_23/checkpoints/epoch=95.ckpt",
#             ckpt_path=str(args.model_load_from_checkpoint),
            ckpt_path=None,
        )
    else:
        LOGGER.info("%s Training %s", 30 * "=", 30 * "=")
        lightning_trainer.fit(
            model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader
        )