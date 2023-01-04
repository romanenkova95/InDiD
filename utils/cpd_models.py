"""Methods and modules for experiments with seq2seq modeld ('indid', 'bce' and 'combided')"""
from . import datasets, loss

from typing import List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import numpy as np
import random

from sklearn.base import BaseEstimator

def fix_seeds(seed: int) -> None:
    """Fix random seeds for experiments reproducibility.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

class ClassicBaseline(nn.Module):
    """Class for classic (from ruptures) Baseline models."""

    def __init__(
        self,
        model: BaseEstimator,
        pen: Optional[float] = None,
        n_pred: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.pen = pen
        self.n_pred = n_pred

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """CPD for Baseline models (from ruptures package).

        :param inputs: input signal
        :return: tensor with change point predictions
        """
        all_predictions = []
        for i, seq in enumerate(inputs):
            # signal should have dimensions (n_samples, n_dims)
            # TODO: check for 1D signals
            signal = seq.flatten(1).detach().cpu().numpy()
            algo = self.model.fit(signal)

            cp_pred = []
            if self.pen is not None:
                cp_pred = self.model.predict(pen=self.pen)
            elif self.n_pred is not None:
                cp_pred = self.model.predict(self.n_pred)
            else:
                cp_pred = self.model.predict()

            # We need only first change point (our assumption)
            cp_pred = cp_pred[0]
            baselines_pred = np.zeros(inputs.shape[1])
            baselines_pred[cp_pred:] = np.ones(inputs.shape[1] - cp_pred)
            all_predictions.append(baselines_pred)

        # TODO: check
        # out = torch.from_numpy(np.array(all_predictions))
        out = torch.cat(all_predictions)
        return out


class CPDModel(pl.LightningModule):
    """Pytorch Lightning wrapper for change point detection models."""

    def __init__(
        self,
        loss_type: str,
        args: dict,
        model: nn.Module,
        train_dataset: Dataset,
        test_dataset: Dataset,
        #extractor: Optional[nn.Module] = None #, Use extractor by default as in the paper
    ) -> None:
        """Initialize CPD model.

        :param experiment_type: type of data used for training
        :param loss_type: type of loss function for training special CPD or common BCE loss
        :param args: dict with supplementary argumemnts
        :param model: base model
        :param train_dataset: train data
        :param test_dataset: test data
        """
        super().__init__()

        self.experiments_name = args["experiments_name"]
        self.model = model
        
        if self.experiments_name in ["explosion", "road_accidents"]:
            print("Loading extractor...")
            self.extractor = torch.hub.load(
                "facebookresearch/pytorchvideo:main", "x3d_m", pretrained=True
            )
            self.extractor = nn.Sequential(*list(self.extractor.blocks[:5]))

            # freeze extractor parameters
            for param in self.extractor.parameters():
                param.requires_grad = False
        else:
            self.extractor = None

        self.learning_rate = args["learning"]["lr"]
        self.batch_size = args["learning"]["batch_size"]
        self.num_workers = args["num_workers"]

        self.T = args["loss"]["T"]

        if loss_type == "indid":
            self.loss = loss.CPDLoss(len_segment=self.T)
        elif loss_type == "bce":
            self.loss = nn.BCELoss()
        else:
            raise ValueError(
                "Wrong loss_type {}. Please, choose 'indid' or 'bce' loss_type.".format(
                    loss_type
                )
            )
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def __preprocess(self, input: torch.Tensor) -> torch.Tensor:
        """Preprocess batch before forwarding (i.e. apply extractor for video input).

        :param input: input torch.Tensor
        :return: processed input tensor to be fed into .forward method 
        """
        if self.experiments_name in ["explosion", "road_accidents"]:
            input = self.extractor(input.float()) 
            input = input.transpose(1, 2).flatten(2) # shape is (batch_size,  C*H*W, seq_len)

        # do nothing for non-video experiments
        return input

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward step for CPD model.

        :param inputs: batch of data
        :return: predictions
        """
        return self.model(self.__preprocess(inputs))

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Train CPD model.

        :param batch: data for training
        :param batch_idx: index of batch (special for pytorch-lightning)
        :return: loss function value
        """
        inputs, labels = batch
        pred = self.forward(inputs.float())

        train_loss = self.loss(pred.squeeze(), labels.float().squeeze())
        train_accuracy = (
            ((pred.squeeze() > 0.5).long() == labels.squeeze()).float().mean()
        )

        self.log("train_loss", train_loss, prog_bar=True)
        self.log("train_acc", train_accuracy, prog_bar=True)

        return train_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Test CPD model.

        :param batch: data for validation
        :param batch_idx: index of batch (special for pytorch-lightning)
        :return: loss function value
        """
        inputs, labels = batch
        pred = self.forward(inputs.float())

        val_loss = self.loss(pred.squeeze(), labels.float().squeeze())
        val_accuracy = (
            ((pred.squeeze() > 0.5).long() == labels.squeeze()).float().mean()
        )

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", val_accuracy, prog_bar=True)

        return val_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Initialize optimizer.

        :return: optimizer for training CPD model
        """
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return opt

    def train_dataloader(self) -> DataLoader:
        """Initialize dataloader for training.

        :return: dataloader for training
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Initialize dataloader for validation.

        :return: dataloader for validation
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

class Baseline_model(pl.LightningModule):
    """Pytorch Lightning wrapper for change point detection models."""

    def __init__(
        self,
        model: nn.Module,
        experiment_type: str = "simple",
        experiment_data_type: str = "mnist",
        lr: float = 1e-3,
        batch_size: int = 64,
        num_workers: int = 4,
        subseq_len=None
    ) -> None:
        """Initialize Baseline model.

        :param model: baseline model
        :param experiment_type: type of the baseline model
        :param experiment_data_type: dataset name
        :param lr: learning rate for training
        :param batch_size: batch size for training
        :param num_workers: number of CPUs
        :param subseq_len: length of the subsequence (for 'weak_labels' baseline)
        """ 
        super().__init__()
        self.model = model

        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.experiment_type = experiment_type
        self.experiment_data_type = experiment_data_type

        self.train_dataset, self.test_dataset = datasets.CPDDatasets(
            experiments_name=self.experiment_data_type
        ).get_dataset_()

        # create datasets
        self.train_dataset = datasets.BaselineDataset(
            self.train_dataset,
            baseline_type=self.experiment_type,
            subseq_len=subseq_len,
        )
        self.test_dataset = datasets.BaselineDataset(
            self.test_dataset, baseline_type=self.experiment_type, subseq_len=subseq_len
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Perform forward pass."""
        
        return self.model(inputs)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Train Baseline model.

        :param batch: data for training
        :param batch_idx: index of batch (special for pytorch-lightning)
        :return: loss function value
        """
        inputs, labels = batch
        pred = self.forward(inputs.float())

        train_loss = nn.BCELoss()(pred.squeeze(), labels.float().squeeze())
        train_accuracy = (
            ((pred.squeeze() > 0.5).long() == labels.squeeze()).float().mean()
        )

        self.log("train_loss", train_loss, prog_bar=True)
        self.log("train_acc", train_accuracy, prog_bar=True)

        return train_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Test Baseline model.

        :param batch: data for validation
        :param batch_idx: index of batch (special for pytorch-lightning)
        :return: loss function value
        """
        inputs, labels = batch
        pred = self.forward(inputs.float())

        val_loss = nn.BCELoss()(pred.squeeze(), labels.float().squeeze())
        val_accuracy = (
            ((pred.squeeze() > 0.5).long() == labels.squeeze()).float().mean()
        )

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", val_accuracy, prog_bar=True)

        return val_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Initialize optimizer.

        :return: optimizer for training CPD model
        """
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return opt

    def train_dataloader(self) -> DataLoader:
        """Initialize dataloader for training.

        :return: dataloader for training
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Initialize dataloader for validation.

        :return: dataloader for validation
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)