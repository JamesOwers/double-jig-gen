"""Model classes for fitting.

https://gist.github.com/williamFalcon/f27c7b90e34b4ba88ced042d9ef33edd#file-pytorch_lstm_variable_mini_batches-py

and

https://github.com/pytorch/examples/blob/master/word_language_model/main.py
"""
from argparse import ArgumentParser
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleRNN(pl.LightningModule):
    """Container module with an encoder, a recurrent module, and a decoder.
    
    Taken from https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    """
    # TODO: Make it such that these are populated in the init call - it will
    #       cause bugs to write kwargs in two places.
    # Argparse settings
    _hparam_defaults = {
        "rnn_type": {"type": str, "choices": ("LSTM", "GRU", "RNN_TANH", "RNN_RELU")},
        "ntoken": {"type": int},
        "model_batch_size": {"type": int},
        "embedding_padding_idx": {"type": int},
        "ninp": {"default": 256, "type": int},
        "nhid": {"default": 512, "type": int},
        "nlayers": {"default": 3, "type": int},
        "dropout": {"default": 0.5, "type": float},
        "tie_weights": {"default": False, "action": "store_true"},
        "learning_rate": {"default": 3e-3, "type": float},
        "weight_decay": {"default": 1e-5, "type": float},
        "lr_decay_gamma": {"default": 0.5, "type": float},
        "lr_decay_patience": {"default": 10, "type": int},
        "optimizer": {"default": "Adam", "type": str},
        "scheduler": {"default": "ReduceLROnPlateau", "type": str},
    }
    
    @classmethod
    def instantiate_from_namespace(cls, args):
        kwargs = {kk: vv for kk, vv in vars(args).items() if kk in cls._hparam_defaults}
        return cls(**kwargs)

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        """Adds args to argparse parent_parser."""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        for arg_name, arg_kwargs in cls._hparam_defaults.items():
            parser.add_argument(f"--{arg_name}", **arg_kwargs)
        return parser
    
    def __init__(
        self,
        rnn_type,
        # TODO: refactor: make better arg names
        ntoken,
        ninp,
        nhid,
        nlayers,
        embedding_padding_idx,
        model_batch_size,
        dropout=0.5,
        tie_weights=False,
        learning_rate: float = 3e-3,
        weight_decay: float = 1e-5,
        lr_decay_gamma: float = 0.5,
        lr_decay_patience: int = 10,
        optimizer: str = "Adam",
        scheduler: Optional[str] = None,
    ):
        """Initialises the model.
        
        Args:
            rnn_type: 'LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU'
            ntoken: number of tokens in the input vocabulary, size of the embedding
                layer input dimension
            ninp: size of the embedding layer output dimension
            nhid: size of the rucurrent layer output dimensions
            nlayers: number of recurrent layers to use
            dropout: proportion of dropout to use between layers
            tie_weights: whether to tie weights
            learning_rate: (for pytorch_lightning configure_optimizers method)
                the value for the learning rate to initialise the optimizer with.
            weight_decay: (for pytorch_lightning configure_optimizers method)
                the value for the weight decay to initialise the optimizer with.
            lr_decay_gamma: (for pytorch_lightning configure_optimizers method)
                the value for the learning rate scheduler's `factor` arg.
            lr_decay_patience: (for pytorch_lightning configure_optimizers method)
                the value for the learning rate scheduler's `patience` arg
            optimizer: string name for the optimizer class to load from torch.optim.
            scheduler: string name for the learning rate scheduler class to load from
                torch.
        """
        super().__init__()

        # This creates the attribute `hparams` for pytorch_lightning. This is used in
        # saving parameters to the logs and for reloading from checkpoints
        self.save_hyperparameters()
        
        self.model_batch_size = model_batch_size
        self.embedding_padding_idx = embedding_padding_idx
        self.ntoken = ntoken
        self.dropout_layer = nn.Dropout(dropout)
        self.encoder_layer = nn.Embedding(
            num_embeddings=ntoken,
            embedding_dim=ninp,
            padding_idx=embedding_padding_idx,
        )
        # TODO: think about how to use a bi-directional lstm
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn_layer = getattr(nn, rnn_type)(
                input_size=ninp,
                hidden_size=nhid,
                num_layers=nlayers,
                dropout=dropout,
                batch_first=False,
            )
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn_layer = nn.RNN(input_size=ninp,
                hidden_size=nhid,
                num_layers=nlayers,
                dropout=dropout,
                nonlinearity=nonlinearity,
                batch_first=False,
            )
            
        self.decoder_layer = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder_layer.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        
        # Pytorch Lightning args
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_decay_gamma = lr_decay_gamma
        self.lr_decay_patience = lr_decay_patience
        self.OptimizerClass = getattr(torch.optim, optimizer)
        self.optimizer = self.OptimizerClass(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        if scheduler is not None:
            self.SchedulerClass = getattr(torch.optim.lr_scheduler, scheduler)
            self.scheduler = self.SchedulerClass(
                self.optimizer,
                factor=self.lr_decay_gamma,
                patience=self.lr_decay_patience,
                cooldown=0,
            )
        else:
            self.scheduler = None

    def init_weights(self, initrange=0.1):
        nn.init.uniform_(self.encoder_layer.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder_layer.weight)
        nn.init.uniform_(self.decoder_layer.weight, -initrange, initrange)

    def forward(self, padded_batch, seq_lengths):
        """padded batch of shape seq_len, batch_size"""
        self.hidden = self.init_hidden()
        
        seq_len, batch_size = padded_batch.shape
        
        # seq_len, batch_size, embedding_size 
        outputs = self.dropout_layer(self.encoder_layer(padded_batch))
        
        # https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        outputs = torch.nn.utils.rnn.pack_padded_sequence(
            outputs,
            seq_lengths,
            batch_first=False,
            enforce_sorted=False,
        )
        outputs, self.hidden = self.rnn_layer(outputs, self.hidden)
        # undo the packing operation
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
            outputs,
            batch_first=False,
        )
                
        outputs = self.dropout_layer(outputs)
        outputs = self.decoder_layer(outputs)
        outputs = outputs.view(-1, self.ntoken)
        outputs = F.log_softmax(outputs, dim=1)
        return outputs.view(seq_len, batch_size, self.ntoken)

    def init_hidden(self):        
        hidden_a = torch.randn(self.nlayers, self.model_batch_size, self.nhid)
        hidden_b = torch.randn(self.nlayers, self.model_batch_size, self.nhid)
        
#         if self.hparams.on_gpu:
        # TODO: fix this hack to make work on CPU too. Need to work out how to get
        # device info from pytorch_lightning when .forward() is called.
        if True:
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()
            
        if self.rnn_type == 'LSTM':
            return hidden_a, hidden_b
        else:
            return hidden_a
    
    def loss(self, outputs, padded_batch):
        """padded batch of shape seq_len, batch_size. outputs are next step preds shape
        seq_len, batch_size, vocab_size. seq_lengths include both start and end tokens."""
        targets = padded_batch[1:].view(-1)  # next steps being predicted
        outputs = outputs[:-1].view(-1, self.ntoken)  # never predict final token

        # create a mask by filtering out all tokens that ARE NOT the padding token
        mask = (targets != self.embedding_padding_idx).float()

        # count how many tokens we have
        nr_tokens = int(torch.sum(mask))

        # pick the values for the label and zero out the rest with the mask
        outputs = outputs[range(outputs.shape[0]), targets] * mask

        # compute cross entropy loss which ignores all <PAD> tokens
        ce_loss = -torch.sum(outputs) / nr_tokens

        return ce_loss
    
    def _step(self, batch):
        """Takes a forward pass step on the batch and returns the loss."""
        padded_data, seq_lengths = batch
        outputs = self(padded_data, seq_lengths)
        loss = self.loss(outputs, padded_data)
        return loss

    def training_step(self, batch, batch_idx):
        """Calculates the loss for a training batch and logs for Tensorboard."""
        loss = self._step(batch)
        logs = {"training_loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        """Calculates the loss for a validation batch."""
        loss = self._step(batch)
        logs = {"validation_loss": loss}
        return {"val_loss": loss, "log": logs}

    def validation_epoch_end(self, outputs):
        """Calculates the average loss for the validation at the end of the epoch."""
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"validation_loss_mean": val_loss_mean}
        return {"val_loss": val_loss_mean, "log": logs, "progress_bar": logs}

    def test_step(self, batch, batch_idx):
        """Calculates the loss for the test batch."""
        loss = self._step(batch)
        logs = {"test_loss": loss}
        return {"test_loss": loss, "log": logs}

    def test_epoch_end(self, outputs):
        """Calculates the average loss for the test at the end of the epoch."""
        test_loss_mean = torch.stack([x["test_loss"] for x in outputs]).mean()
        logs = {"test_loss_mean": test_loss_mean}
        return {"test_loss": test_loss_mean, "log": logs}

    def configure_optimizers(self):
        """Initialises the optimisers and learning rate schedulers for training."""
        if self.scheduler is None:
            return self.optimizer
        else:
            return [self.optimizer], [self.scheduler]
        

class Transformer(pl.LightningModule):
    def __init__(self):
        pass

    @classmethod
    def instantiate_from_namespace(cls, args):
        pass

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        pass

    
    