"""Module with methods for network models"""
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from CPD import datasets, loss

import scipy.sparse

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
        self.experiment_name = 'rnn'

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

######### with/without mask ##################################
class MnistTransformer(nn.Module):
    def __init__(self, input_size, linear_dims, name_mask,
                 src_mask = torch.ones((64, 64), dtype=torch.bool),
                 # flag mask = True -- with mask
                 flag_mask=True,
                 dim_feedforward = 256,
                 dropout=0.5,
                 nhead = 4,
                 device='cuda'
                ):
        super().__init__()
        # initialize transformer
        self.TEL = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward = dim_feedforward,
                                           dropout = dropout, activation = 'relu', layer_norm_eps = 1e-5,
                                           batch_first=True)
        self.output_layer = nn.Linear(input_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.flag_mask = flag_mask
        self.device = device
        if self.flag_mask == True:
            self.src_mask = src_mask.to(self.device)
            self.name_mask = name_mask 
        else:
            self.name_mask = 'without'
        self.experiment_name = 'transformer'
        
    def forward(self, inputs):
        batch_size, seq_len = inputs.size()[:2]
        if inputs.type() != 'torch.FloatTensor':
            inputs = inputs.float() 
        out = inputs.float()
        out = out.flatten(2, -1) # batch_size, seq_len, input_size
        if self.flag_mask == True:
            out = self.TEL(out, self.src_mask) 
        else:
            out = self.TEL(out)
        out = out.flatten(0, 1) # batch_size * seq_len, hidden_dim
        out = self.output_layer(out)
        out = self.sigmoid(out)
        out = out.reshape(batch_size, seq_len, 1)
        return out

######### with 3 masks ##################################
def get_hessenberg_mask1(seq_len, low_n, up_n):
    if low_n + 1 <= seq_len:
        position = [-seq_len + i for i in range(low_n + 1)]+[ 0] +  [seq_len - i for i in range(up_n + 1)]
        mask = scipy.sparse.diags([1]*len(position), position, shape=(seq_len, seq_len), dtype = bool).toarray()
        return torch.as_tensor(mask)
    else:
        print('Error: low_n is too large, that is why offset array contains duplicate values. Try low_n <= seq_len - 1')
        
class MnistTransformer_3masks(nn.Module):
    def __init__(self, input_size, linear_dims, seq_len,
                 dim_feedforward = 256, 
                 dropout=0.5,
                 nhead = 4,
                 device = 'cuda' 
                ):
        super().__init__()
        # initialize transformer
        self.TEL = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward = dim_feedforward,
                                           dropout = dropout, activation = 'relu', layer_norm_eps = 1e-5,
                                           batch_first=True)
        self.output_layer = nn.Linear(input_size, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.device = device
        self.experiment_name = 'transformer'
        self.name_mask = '3'
        
        self.masks = []
        # Mask 1 - triangular
        self.masks.append( ~torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool)).to(self.device )
                         )
        #Mask 2 - triangular + 8-diagonal
        self.masks.append( ~get_hessenberg_mask1(seq_len, 8, 0).to(self.device )
                         )
        # Mask 3 - 8-diagonal 
        self.masks.append( ~torch.as_tensor(scipy.sparse.diags([1]*8, [ -i for i in range(8)],
                                                       shape=(seq_len, seq_len), dtype=bool).toarray()).to(self.device )
                         )
        
    def forward(self, inputs):
        batch_size, seq_len = inputs.size()[:2]
        if inputs.type() != 'torch.FloatTensor':
            inputs = inputs.float() 
        out = inputs.float()
        out = out.flatten(2, -1) # batch_size, seq_len, input_size
        
        for mask in self.masks:
            out = self.TEL(out, mask) 
        
        out = out.flatten(0, 1) # batch_size * seq_len, hidden_dim
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
    ) -> None:
        """Initialize CPD model.

        :param model: base model
        :param T: parameter restricted the size of a considered segment in delay loss (T in the paper)
        :param loss_type: type of loss function for training special CPD or common BCE loss
        :param experiment_type: type of data used for training (only mnist is available now)
        :param lr: learning rate
        :param batch_size: size of batch
        """
        super().__init__()
        self.model = model

        self.lr = lr
        self.batch_size = batch_size

        self.T = T

        self.experiment_type = experiment_type
        self.loss_type = loss_type
        if loss_type == "CPD":
            if self.model.experiment_name == 'rnn':
                alpha = 2.
            if self.model.experiment_name == 'transformer':
                alpha = 1.
            self.loss = loss.CPDLoss(len_segment=self.T, alpha=alpha)
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
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self) -> DataLoader:
        """Initialize dataloader for validation.

        :return: dataloader for validation
        """
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

    def test_dataloader(self) -> DataLoader:
        """Initialize dataloader for test (same as for validation).

        :return: dataloader for test
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
