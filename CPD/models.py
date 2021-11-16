"""Module with methods for network models"""
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from CPD import datasets, loss
import ruptures as rpt
import numpy as np


class MnistRNN(nn.Module):
    """Initialize class with recurrent network for MNIST sequences."""

    def __init__(
        self,
        input_size: int,
        hidden_rnn: int,
        rnn_n_layers: int,
        linear_dims: List[int],
        rnn_dropout: float = 0.0,
        dropout: float = 0.5,
        rnn_type: str = "LSTM",
    ) -> None:
        """Initialize model's parameters.

        :param input_size: number of input features
        :param hidden_rnn: size of recurrent model's hidden layer
        :param rnn_n_layers: number of recurrent layers
        :param linear_dims: list of dimensions for linear layers
        :param rnn_dropout: dropout in recurrent block
        :param dropout: dropout in fully-connected block
        :param rnn_type: type of recurrent block (LSTM, GRU, RNN)
        """
        super().__init__()

        # initialize rnn layer
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size,
                hidden_rnn,
                rnn_n_layers,
                dropout=rnn_dropout,
                batch_first=True,
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size,
                hidden_rnn,
                rnn_n_layers,
                dropout=rnn_dropout,
                batch_first=True,
            )
        elif rnn_type == "RNN":
            self.rnn = nn.RNN(
                input_size,
                hidden_rnn,
                rnn_n_layers,
                dropout=rnn_dropout,
                batch_first=True,
            )

        # initialize linear layers
        if linear_dims[0] != hidden_rnn:
            linear_dims = [hidden_rnn] + linear_dims

        self.linears = nn.ModuleList(
            [
                nn.Linear(linear_dims[i], linear_dims[i + 1])
                for i in range(len(linear_dims) - 1)
            ]
        )
        self.output_layer = nn.Linear(linear_dims[-1], 1)

        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward propagation through model.

        :param inputs: batch of sequences with MNIST image
        :return: probabilities of changes for each sequence
        """
        batch_size, seq_len = inputs.size()[:2]

        if inputs.type() != "torch.FloatTensor":
            inputs = inputs.float()

        out = inputs.flatten(2, -1)  # batch_size, seq_len, input_size
        out, _ = self.rnn(out)  # batch_size, seq_len, hidden_dim
        out = out.flatten(0, 1)  # batch_size * seq_len, hidden_dim

        for layer in self.linears:
            out = layer(out)
            out = self.relu(self.dropout(out))

        out = self.output_layer(out)
        out = self.sigmoid(out)
        out = out.reshape(batch_size, seq_len, 1)
        return out
    

class CPD_model(pl.LightningModule):
    """Pytorch Lightning wrapper for change point detection models."""

    def __init__(
        self,
        model: nn.Module,
        T: int,
        loss_type: str = "CPD",
        experiment_type: str = "mnist",
        lr: float = 1e-3,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        """Initialize CPD model.

        :param model: base model
        :param T: parameter restricted the size of a considered segment in delay loss (T in the paper)
        :param loss_type: type of loss function for training special CPD or common BCE loss
        :param experiment_type: type of data used for training (only mnist is available now)
        :param lr: learning rate
        :param batch_size: size of batch
        :param num_workers: num of kernels used for evaluation         
        """
        super().__init__()
        self.model = model

        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.T = T

        self.experiment_type = experiment_type

        if loss_type == "CPD":
            self.loss = loss.CPDLoss(len_segment=self.T)
        elif loss_type == "BCE":
            self.loss = nn.BCELoss()
        else:
            raise ValueError(
                "Wrong loss_type {}. Please, choose CPD or BCE loss_type.".format(
                    loss_type
                )
            )

        self.train_dataset, self.test_dataset = datasets.CPDDatasets(
            experiments_name=self.experiment_type
        ).get_dataset_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward step for CPD model.

        :param inputs: batch of data
        :return: predictions
        """
        return self.model(inputs)

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
        
        ############################################
        #if self.T:
        #    cpd_loss = loss.CPDLoss(len_segment=self.T)
        #else:
        #    cpd_loss = loss.CPDLoss(len_segment=len(pred.squeeze()[0]))
        #cpd_loss = cpd_loss(pred.squeeze(), labels.float().squeeze())
        #
        #bce_loss = nn.BCELoss()(pred.squeeze(), labels.float().squeeze())
        #
        #self.log("cpd_loss", cpd_loss, prog_bar=False)
        #self.log("bce_loss", bce_loss, prog_bar=False)

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
        
        ############################################
        #if self.T:
        #    cpd_loss = loss.CPDLoss(len_segment=self.T)
        #else:
        #    cpd_loss = loss.CPDLoss(len_segment=len(pred.squeeze()[0]))
        #cpd_loss = cpd_loss(pred.squeeze(), labels.float().squeeze())        
        #bce_loss = nn.BCELoss()(pred.squeeze(), labels.float().squeeze())
        
        #self.log("cpd_loss", cpd_loss, prog_bar=False)
        #self.log("bce_loss", bce_loss, prog_bar=False)        

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
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        """Initialize dataloader for validation.

        :return: dataloader for validation
        """
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        """Initialize dataloader for test (same as for validation).

        :return: dataloader for test
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    
class L2Baseline(nn.Module):
    def __init__(self, l2_type, extractor=None, device='cuda'):
        super().__init__()
        self.device = device
        self.type = l2_type
        self.extractor = extractor

    def forward(self, inputs):
        batch_size, seq_len = inputs.size()[:2]
        l2_dist = []
        
        if self.extractor is not None:
            inputs = extractor(inputs)
        for seq in inputs:
            seq = seq.float().to(self.device)
            if self.type == "one_by_one":
                curr_l2_dist = [0] + [((x - y)**2).sum().item() for x, y in zip(seq[1:], seq[:-1])]   
            elif self.type == "vs_first":
                curr_l2_dist = [0] + [((x - seq[0])**2).sum().item() for x in seq[1:]]
            elif self.type == "vs_mean":
                mean_seq = torch.mean(seq, 0)
                curr_l2_dist = [0] + [((x - mean_seq)**2).sum().item() for x in seq[1:]]
            curr_l2_dist = np.array(curr_l2_dist) / max(curr_l2_dist)
            l2_dist.append(curr_l2_dist)
        l2_dist = torch.from_numpy(np.array(l2_dist))
        return l2_dist
    
    
class ZeroBaseline(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, inputs):
        batch_size, seq_len = inputs.size()[:2]
        out = torch.zeros((batch_size, seq_len, 1))
        return out
    
    
class ClassicBaseline(nn.Module):

    def __init__(self, model, pen=None, n_pred=None, device='cuda'):
        super().__init__()
        self.device=device
        self.model = model
        self.pen = pen
        self.n_pred = n_pred

    def forward(self, inputs):
        all_predictions = []
        for i, seq in enumerate(inputs):
            # (n_samples, n_dims)
            try:
                signal = seq.flatten(1, 2).detach().cpu().numpy()
            except:
                signal = seq.detach().cpu().numpy()
            algo = self.model.fit(signal)
            cp_pred = []
            if self.pen is not None:
                cp_pred = self.model.predict(pen=self.pen)
            elif self.n_pred is not None:
                cp_pred = self.model.predict(self.n_pred) 
            else:
                cp_pred = self.model.predict()                 
            cp_pred = cp_pred[0]
            baselines_pred = np.zeros(inputs.shape[1])
            baselines_pred[cp_pred:] = np.ones(inputs.shape[1] - cp_pred)        
            all_predictions.append(baselines_pred)
        out = torch.from_numpy(np.array(all_predictions))
        return out    