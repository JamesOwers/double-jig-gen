"""Model classes for fitting."""
import pytorch_lightning as pl


class LSTM(pl.LightningModule):
    def __init__(self):
        pass

    @classmethod
    def instantiate_from_namespace(cls, args):
        pass

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        pass


class Transformer(pl.LightningModule):
    def __init__(self):
        pass

    @classmethod
    def instantiate_from_namespace(cls, args):
        pass

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        pass
