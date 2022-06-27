"""Module with methods for network models"""
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from CPD_explosion import datasets, loss

import ruptures as rpt
import numpy as np

import random

def fix_seeds(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

""" Modules for baselines """

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

    
""" Base models for our experiments """

class LSTM(nn.Module):
    """Initialize LSTM class for experiments with Synthetic Normal data and Human Activity."""
    
    def __init__(
        self, 
        input_size: int,
        hidden_dim: int,
        n_layers: int,
        drop_prob: float,
    ) -> None:
        """Initialize model's parameters.

        :param input_size: size of elements in input sequence
        :param output_size: length of the generated sequence
        :param hidden_dim: size of the hidden layer(-s)
        :param n_layers: number of recurrent layers
        :param drop_prob: dropout probability
        """
        super(LSTM, self).__init__()
        
        self.input_size = input_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.activation = nn.Sigmoid()
        
    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Forward propagation through model.

        :param input_seq: batch of generated sunthetic normal sequences
        :return: probabilities of changes for each sequence
        """

        batch_size = input_seq.size(0)
        lstm_out, hidden = self.lstm(input_seq)  
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)      
        out = self.linear(lstm_out)
        out = self.activation(out)
        out = out.view(batch_size, -1)

        return out

class MnistRNN(nn.Module):
    """Initialize class with recurrent network for MNIST experimrnts."""

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
    
    
#TODO: base models for Exposions & Car Accidents
    

class CPD_model(pl.LightningModule):
    """Pytorch Lightning wrapper for change point detection models."""

    def __init__(
        self,
        experiments_name: str,
        loss_type: str,
        T: int = None,
        model: nn.Module = None,
        lr: float = 1e-3,
        batch_size: int = 64,
        num_workers: int = 2,
    ) -> None:
        """Initialize CPD model.
        :param experiment_type: type of data used for training
        :param loss_type: type of loss function for training special CPD or common BCE loss
        :param T: parameter restricted the size of a considered segment in delay loss (T in the paper)
        :param model: custom base model (or None if need a default one)
        :param lr: learning rate
        :param batch_size: size of batch
        :param num_workers: num of kernels used for evaluation
        """
        super().__init__()
        
        if experiments_name.startswith('synthetic'):
            # format is "synthetic_nD"
            D = int(experiments_name.split('_')[1].split('D')[0])
        
            if D not in [1, 100]:
                raise ValueError("Wrong dimension D. We have experiments with 1D and 100D synthetic normal data")
        
            if model is None:
                if D == 1:
                    hidden_dim = 4
                else:
                    hidden_dim = 8
            
                # initialize default base model for SyntheticNormal experiment
                model = LSTM(input_size=D, hidden_dim=hidden_dim, n_layers=1, drop_prob=0.5)
                
            if loss_type == 'CPD' and T is None:
                # choose default segment length T for 'CPD' loss
                T = 32
 
        elif experiments_name == 'mnist':
        
            if model is None:
                # initialize default base model for MNIST experiment
                model = MnistRNN(input_size=28*28, hidden_rnn=32, rnn_n_layers=1, 
                                 linear_dims=[32], rnn_dropout=0.25, dropout=0.5, rnn_type='LSTM')
        
            if loss_type == 'CPD' and T is None:
                T = 32

        elif experiments_name == 'human_activity':
            if model is None:
                # initialize default base model for Human Activity experiment
                model = LSTM(input_size=561, n_layers=1, hidden_dim=8, drop_prob=0.25)
            if loss_type == 'CPD' and T is None:
                T = 5
                
        elif experiments_name == 'explosion':
            # TODO
            pass
    
        elif experiments_name == 'road_accidents':
            # TODO
            pass
    
        else:
            raise ValueError("Wrong experiment_name {}.".format(experiments_name))
            
        self.model = model

        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.T = T

        self.experiments_name = experiments_name

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
            experiments_name=self.experiments_name
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
        subseq_len = None
    ) -> None:
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
        
        self.train_dataset = datasets.BaselineDataset(self.train_dataset, 
                                                     baseline_type=self.experiment_type,
                                                     subseq_len=subseq_len)
        self.test_dataset = datasets.BaselineDataset(self.test_dataset, 
                                                     baseline_type=self.experiment_type,
                                                     subseq_len=subseq_len)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
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
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return opt

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
