"""Module with methods for network models"""
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import Tensor
import math
from CPD import datasets, loss

import scipy.sparse
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
    
#-------------------------------------------------------------------------------------------------------------------------#
#Работает Для 64, 128, 256, 512 : nhead=2,  d_hid=1024//2,  nlayers=2, dropout=0.5,

#    Для 2changes, 4changes, 128:  nhead=8, d_hid=1024//4,   nlayers=2,  dropout=0.5,

# Informer
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x
    
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model,dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        
        return self.dropout(x)

#---------------------------------------------------------------------------------------------#
# Работает на 128, 256, 512, 2,4-10changes
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1,  d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        #pe[:, 0, 0, 0::2] = torch.sin(position * div_term)
        #pe[:, 0, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        #print(x.shape, pe.shape )
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
#-----------------------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnablePositionEmbedding(nn.Module):

    MODE_EXPAND = 'MODE_EXPAND'
    MODE_ADD = 'MODE_ADD'
    MODE_CONCAT = 'MODE_CONCAT'

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 mode=MODE_ADD):
        super(LearnablePositionEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        if self.mode == self.MODE_EXPAND:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings * 2 + 1, embedding_dim))
        else:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        if self.mode == self.MODE_EXPAND:
            indices = torch.clamp(x, -self.num_embeddings, self.num_embeddings) + self.num_embeddings
            return F.embedding(indices.type(torch.LongTensor), self.weight)
        batch_size, seq_len = x.size()[:2]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.embedding_dim)
        if self.mode == self.MODE_ADD:
            return x + embeddings
        if self.mode == self.MODE_CONCAT:
            return torch.cat((x, embeddings.repeat(batch_size, 1, 1)), dim=-1)
        raise NotImplementedError('Unknown mode: %s' % self.mode)
        
#-----------------------------------------------------------------------------------------------#
        
class Transformer(nn.Module):
    def __init__(self,  d_model: int = 28*28, nhead: int = 4, d_hid: int = 1024,
                 nlayers: int = 3, dropout: float = 0.1, src_mask = None, 
                 experiment_name='transformer', name_mask='wo_mask',
                 pos_enc: str = 'usual', device: str = 'cuda'):
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'
        self.pos_enc = pos_enc
        self.d_model = d_model
        if pos_enc == 'informer':
            self.pos_encoder = DataEmbedding(c_in = d_model, d_model=d_model, dropout=dropout) #PositionalEncoding(d_model=d_model, dropout=dropout)
        if pos_enc == 'usual':
            self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout) 
        if pos_enc == 'learnable':
            self.pos_encoder = LearnablePositionEmbedding(d_model, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_hid,
                                                        dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)  
        self.decoder = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
        
        self.experiment_name = experiment_name
        self.name_mask = name_mask
        self.device=device
        self.src_mask = None
        if src_mask != None:
            self.src_mask = src_mask.to(device)
        
        
        
    def init_weights(self) -> None:
        initrange = 0.1
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
          #  src: Tensor, shape  [batch_size, seq_len, input_dim=784], but need [seq_len, batch_size]
          #  src_mask: Tensor, shape [seq_len, seq_len]
          #  output Tensor of shape [seq_len, batch_size, ntoken]
        
        batch_size, seq_len = src.size()[:2]
        src = src.reshape(batch_size, seq_len, self.d_model).to(torch.float)
        src = src.permute(1, 0, 2) # [ seq_len, batch_size]
        ####src = self.encoder(src.long()) * math.sqrt(self.d_model) # [ seq_len, batch_size, d_model]
        src = self.pos_encoder(src)
        #if without_mask == False:
        #    src_mask = generate_square_subsequent_mask(seq_len).to(self.device)
        src = src.permute(1, 0, 2) # [  batch_size, seq_len]
        if self.src_mask != None:
            out = self.transformer_encoder(src, self.src_mask)
        else:
            out = self.transformer_encoder(src)
            
        out = self.decoder(out)
        out = self.sigmoid(out)
        return out

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        
    
class Transformer2(nn.Module):
    def __init__(self,  d_model: int = 28*28, nhead: int = 4, d_hid: int = 1024,
                 nlayers: int = 3, dropout: float = 0.1, src_mask = None, pos_enc: str = 'usual', device: str = 'cuda'):
        super(Transformer2, self).__init__()
        self.model_type = 'Transformer'
        self.pos_enc = pos_enc
        self.d_model = d_model
        if pos_enc == 'informer':
            self.pos_encoder = DataEmbedding(c_in = d_model, d_model=d_model, dropout=dropout) #PositionalEncoding(d_model=d_model, dropout=dropout)
        if pos_enc == 'usual':
            self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout) 
        if pos_enc == 'learnable':
            self.pos_encoder = LearnablePositionEmbedding(d_model, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_hid,
                                                        dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)  
        self.decoder1 = nn.Linear(d_model, d_model//2)
        self.relu = nn.ReLU()
        self.decoder2 = nn.Linear( d_model//2, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
        self.device=device
        if src_mask != None:
            self.experiment_name = 'transformer'
            self.name_mask = 'triang'
            self.src_mask = src_mask.to(device)
        else:
            self.experiment_name = 'transformer'
            self.name_mask = 'wo_mask'
            self.src_mask = None
        
        
    def init_weights(self) -> None:
        initrange = 0.1
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder1.bias.data.zero_()
        self.decoder1.weight.data.uniform_(-initrange, initrange)
        self.decoder2.bias.data.zero_()
        self.decoder2.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
          #  src: Tensor, shape  [batch_size, seq_len, input_dim=784], but need [seq_len, batch_size]
          #  src_mask: Tensor, shape [seq_len, seq_len]
          #  output Tensor of shape [seq_len, batch_size, ntoken]
        
        batch_size, seq_len = src.size()[:2]
        src = src.reshape(batch_size, seq_len, self.d_model).to(torch.float)
        src = src.permute(1, 0, 2) # [ seq_len, batch_size]
        ####src = self.encoder(src.long()) * math.sqrt(self.d_model) # [ seq_len, batch_size, d_model]
        src = self.pos_encoder(src)
        #if without_mask == False:
        #    src_mask = generate_square_subsequent_mask(seq_len).to(self.device)
        src = src.permute(1, 0, 2) # [  batch_size, seq_len]
        out = self.transformer_encoder(src, self.src_mask)
        out = self.decoder1(out)
        out = self.relu(out)
        out = self.decoder2(out)
        out = self.sigmoid(out)
        return out


#---------------------------------------------------------------------------------------------#


    

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
                alpha = 1. ##########################################################################################################
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
        #############################################################################################################################
       
        out = self.model(inputs)
        ################################################################################################################################
        return out

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
