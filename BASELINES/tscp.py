from tsai.models.TCN import TCN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import numpy as np

import torch.nn.functional as F

class ResidualBlock(nn.Module):
    
    def __init__(self,
                 in_channels,
                 dilation_rate: int,
                 nb_filters: int,
                 kernel_size: int,
                 dropout_rate: float = 0, 
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 use_weight_norm: bool = False, 
                 training: bool = True):
        """Defines the residual block for the WaveNet TCN
        Args:
            dilation_rate: The dilation power of 2 we are using for this residual block
            nb_filters: The number of convolutional filters to use in this block
            kernel_size: The size of the convolutional kernel
            activation: The final activation used in o = Activation(x + F(x))
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
        """

        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        # for causal padding
        self.padding = (self.kernel_size - 1) * self.dilation_rate
        
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        
        self.training = training

        super().__init__()
        
        self.conv_1 = nn.Conv1d(in_channels, self.nb_filters, self.kernel_size, 
                                padding=0, dilation=self.dilation_rate)        
        if self.use_weight_norm:
            weight_norm(self.conv_1) 
        self.bn_1 = nn.BatchNorm1d(self.nb_filters)
        self.ln_1 = nn.LayerNorm(self.nb_filters)              
        self.relu_1 = nn.ReLU()

        self.conv_2 = nn.Conv1d(self.nb_filters, self.nb_filters, self.kernel_size, 
                                padding=0, dilation=self.dilation_rate)        
        if self.use_weight_norm:
            weight_norm(self.conv_1)    
        self.bn_2 = nn.BatchNorm1d(self.nb_filters)
        self.ln_2 = nn.LayerNorm(self.nb_filters)              
        self.relu_2 = nn.ReLU()        
        
        self.conv_block = nn.Sequential()
        self.downsample = nn.Conv1d(in_channels, self.nb_filters, kernel_size=1) if in_channels != self.nb_filters else nn.Identity()
        
        self.relu = nn.ReLU()  
                
        self.init_weights()
        
        
    def init_weights(self):
        # in the realization, they use random normal initialization
        torch.nn.init.normal_(self.conv_1.weight, mean=0, std=0.05)
        torch.nn.init.zeros_(self.conv_1.bias)            
        
        torch.nn.init.normal_(self.conv_2.weight, mean=0, std=0.05)
        torch.nn.init.zeros_(self.conv_2.bias)            
        
        if isinstance(self.downsample, nn.Conv1d):         
            torch.nn.init.normal_(self.downsample.weight, mean=0, std=0.05)
            torch.nn.init.zeros_(self.downsample.bias)                    
            
    def forward(self, inp):
        # inp batch, channels, time
        ######################
        # do causal padding        
        out = F.pad(inp, (self.padding, 0))
        out = self.conv_1(out)
        
        if self.use_batch_norm:
            out = self.bn_1(out)
        elif self.use_layer_norm:
            out = self.ln_1(out)        
        out = self.relu_1(out)
        
        # spatial dropout
        out = out.permute(0, 2, 1)   # convert to [batch, time, channels]
        out = F.dropout2d(out, self.dropout_rate, training=self.training)        
        out = out.permute(0, 2, 1)   # back to [batch, channels, time]    
        
        #######################
        # do causal padding
        out = F.pad(out, (self.padding, 0))
        out = self.conv_2(out)
        if self.use_batch_norm:
            out = self.bn_2(out)
        elif self.use_layer_norm:
            out = self.ln_2(out)
        out = self.relu_2(out)            
        out = self.relu_2(out)    
        # spatial dropout
        # out batch, channels, time 
        
        out = out.permute(0, 2, 1)   # convert to [batch, time, channels]
        out = F.dropout2d(out, self.dropout_rate, training=self.training)
        out = out.permute(0, 2, 1)   # back to [batch, channels, time]            
        
        #######################        
        skip_out = self.downsample(inp)
        #######################
        res = self.relu(out + skip_out)
        return res, skip_out
    
# only causal padding
# only return sequence = True
    
class TCN(nn.Module):        
    def __init__(self,
                 in_channels=1,
                 nb_filters=64,
                 kernel_size=3,
                 nb_stacks=1,
                 dilations=(1, 2, 4, 8, 16, 32),
                 use_skip_connections=True,
                 dropout_rate=0.0, 
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False, 
                 use_weight_norm: bool = False):

        super().__init__()
        
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.in_channels = in_channels
        
        if self.use_batch_norm + self.use_layer_norm + self.use_weight_norm > 1:
            raise ValueError('Only one normalization can be specified at once.')        
        
        self.residual_blocks = []        
        res_block_filters = 0
        for s in range(self.nb_stacks):
            for i, d in enumerate(self.dilations):
                in_channels = self.in_channels if i + s == 0 else res_block_filters                
                res_block_filters = self.nb_filters[i] if isinstance(self.nb_filters, list) else self.nb_filters
                self.residual_blocks.append(ResidualBlock(in_channels=in_channels, 
                                                          dilation_rate=d,
                                                          nb_filters=res_block_filters,
                                                          kernel_size=self.kernel_size,
                                                          dropout_rate=self.dropout_rate, 
                                                          use_batch_norm=self.use_batch_norm,
                                                          use_layer_norm=self.use_layer_norm,
                                                          use_weight_norm=self.use_weight_norm))

        
        self.residual_blocks = nn.ModuleList(self.residual_blocks)
                                            
    def forward(self, inp):
        out = inp
        for layer in self.residual_blocks:
            out, skip_out = layer(out)
        if self.use_skip_connections:
            out = out + skip_out
        return out

########################### model #########################################
class Encoder(nn.Module):
    def __init__(self, c_in=1, nb_filters=64, kernel_size=4, 
                 dilations=[1,2,4,8], nb_stacks=2, n_steps=50, code_size=10):       
        super(Encoder, self).__init__()        
        
        self.tcn_layer = TCN(in_channels=c_in, nb_filters=nb_filters, 
                             nb_stacks=nb_stacks, dilations=dilations, use_skip_connections=True, dropout_rate=0)
        
        self.fc1 = nn.Linear(nb_filters, 2 * n_steps)  
        self.fc2 = nn.Linear(2 * n_steps, n_steps)    
        self.output_layer = nn.Linear(n_steps, code_size)           
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = x
        out = self.tcn_layer(out) # (None, nb_filters, seq_len)
        out = out.transpose(1, 2) # (None, seq_len, nb_filters)
        out = self.relu(self.fc1(out)) # (None, seq_len, 2*n_steps)
        out = self.relu(self.fc2(out)) # (None, seq_len, n_steps)
        out = out.flatten(0, 1) # (None * seq_len, n_steps)
        out = self.output_layer(out)
        return out
    
########################### loss #########################################
def _cosine_simililarity_dim2(x, y):
    # x shape: (N, 1, C)
    # y shape: (1, 2N, C)
    # v shape: (N, 2N)
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    v = cos(x.unsqueeze(1), y.unsqueeze(0))
    return v    

def nce_loss_fn(history, future, similarity, temperature=0.1):
    try:
        device = history.device
    except:
        device = 'cpu'
        
    criterion = torch.nn.BCEWithLogitsLoss()
    N = history.shape[0]
    sim = similarity(history, future)
    pos_sim = torch.exp(torch.diag(sim) / temperature)

    tri_mask = torch.ones((N, N), dtype=bool)
    tri_mask[np.diag_indices(N)] = False
    
    neg = sim[tri_mask].reshape(N, N - 1)    
    all_sim = torch.exp(sim / temperature)
    
    logits = torch.divide(torch.sum(pos_sim), torch.sum(all_sim, axis=1))
        
    lbl = torch.ones(history.shape[0]).to(device)
    # categorical cross entropy
    loss = criterion(logits, lbl)    
    # loss = K.sum(logits)
    # divide by the size of batch
    #loss = loss / lbl.shape[0]
    # similarity of positive pairs (only for debug)
    mean_sim = torch.mean(torch.diag(sim))
    mean_neg = torch.mean(neg)
    return loss, mean_sim, mean_neg


######################### preprocessing ###########################

def _history_future_separation(data, window):    
    
    if len(data.shape) <= 4:
        history = data[:, :window]
        future = data[:, window:2*window]    
    elif len(data.shape) == 5:
        history = data[:, :, :window]
        future = data[:, :, window:2*window]    
        
    return history, future


################## PL wrapper ###############################################
class TSCP_model(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,     
        train_dataset: Dataset, 
        test_dataset: Dataset, 
        batch_size: int = 64,        
        num_workers: int = 2,        
        temperature: float = 0.1, 
        lr: float = 1e-4,
        decay_steps: int = 1000, 
        window_1: int = 100,
        window_2: int = 100
    ) -> None:
        super().__init__()
                    
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        
        self.batch_size = batch_size
        self.num_workers = num_workers        
        
        self.temperature = temperature
        
        self.lr = lr
        self.decay_steps = decay_steps   
        
        self.window = window_1
        self.window_1 = window_1
        self.window_2 = window_2

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)
    
    def training_step(self, batch, batch_idx):

        history, future = _history_future_separation(batch[0], self.window)        
        history_emb = self.forward(history.float())
        future_emb = self.forward(future.float()) 

        history_emb = nn.functional.normalize(history_emb, p=2, dim=1)
        future_emb = nn.functional.normalize(future_emb, p=2, dim=1)

        train_loss, pos_sim, neg_sim = nce_loss_fn(history_emb, future_emb, similarity=_cosine_simililarity_dim2, 
                                                   temperature=self.temperature)

        self.log("train_loss", train_loss, prog_bar=True, on_epoch=True)     
        self.log("pos_sim", pos_sim, prog_bar=True, on_epoch=True)        
        self.log("neg_sim", neg_sim, prog_bar=True, on_epoch=True)        

        return train_loss
        
    def validation_step(self, batch, batch_idx):

        history, future = _history_future_separation(batch[0], self.window)        
                
        history_emb = self.forward(history.float())
        future_emb = self.forward(future.float()) 
                
        history_emb = nn.functional.normalize(history_emb, p=2, dim=1)
        future_emb = nn.functional.normalize(future_emb, p=2, dim=1)
        

        val_loss, pos_sim, neg_sim = nce_loss_fn(history_emb, future_emb, similarity=_cosine_simililarity_dim2, 
                                                 temperature=self.temperature)

        self.log("val_loss", val_loss, prog_bar=True)     
        #self.log("val_pos_sim", pos_sim, prog_bar=False)        
        #self.log("val_neg_sim", neg_sim, prog_bar=False)        

        return val_loss


    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Initialize optimizer.

        :return: optimizer for training CPD model
        """
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt, T_max=self.decay_steps)
        return opt

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
    

def _history_future_separation_test(data, window_1, window_2, step=1):    
    
    if len(data.shape) <= 4:
        history = data[:, :window_1]
        future = data[:, window_1:]
        seq_len = future.shape[1]
    elif len(data.shape) == 5:
        history = data[:, :, :window_1]
        future = data[:, :, window_1:]
        seq_len = future.shape[2]
    
    future_slices = []
    for i in range(0, (seq_len - window_2) // step + 1):
        start_ind = i*step
        end_ind = window_2 + step * i
        if len(data.shape) <= 4:
            future_slices.append(future[:, start_ind: end_ind])
        elif len(data.shape) == 5:
            future_slices.append(future[:, :, start_ind: end_ind])   
    future_slices = torch.cat(future_slices)
    
    if len(data.shape) == 4:
        future_slices = future_slices.reshape(future.shape[0], -1, future_slices.shape[1], 
                                              future.shape[2], future.shape[3])

    elif len(data.shape) == 3:
        future_slices = future_slices.reshape(future.shape[0], -1, future_slices.shape[1], 
                                              future.shape[2])
    elif len(data.shape) == 5:
        future_slices = future_slices.reshape(future.shape[0], -1, future.shape[1], future_slices.shape[2], 
                                              future.shape[3], future.shape[4])        
    return history, future_slices

def _history_future_separation_test_2(data, window, step=1):    
    
    future_slices = []
    history_slices = []
    
    if len(data.shape) > 4:
        data = data.transpose(1, 2)
    
    seq_len = data.shape[1]    
    for i in range(0, (seq_len - 2 * window) // step + 1):
        start_ind = i * step
        end_ind = 2 * window + step * i
        slice_2w = data[:, start_ind: end_ind]
        history_slices.append(slice_2w[:, :window].unsqueeze(0))        
        future_slices.append(slice_2w[:, window:].unsqueeze(0))
    
    future_slices = torch.cat(future_slices).transpose(0, 1)
    history_slices = torch.cat(history_slices).transpose(0, 1)
    
    if len(data.shape) > 4:
        history_slices = history_slices.transpose(2, 3)
        future_slices = future_slices.transpose(2, 3)

    return history_slices, future_slices

def _cosine_simililarity_dim1(x, y):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    v = cos(x, y)
    return v    

def get_tscp_output(tscp_model, batch, window_1, window_2):
    if len(batch.shape) <= 4:
        seq_len = batch.shape[1]
    else:
        seq_len = batch.shape[2]

    batch_history, batch_future_slices = _history_future_separation_test(batch, window_1, window_2)
    batch_history = tscp_model(batch_history)
    
    pred_out = []
    for i, history in enumerate(batch_history):
        zeros = torch.zeros(1, seq_len)
        curr_future = tscp_model(batch_future_slices[i]) 
        curr_history = history.repeat(curr_future.shape[0], len(history.shape))
        rep_sim = _cosine_simililarity_dim1(curr_history, curr_future).data
        
        zeros[:, window_1: seq_len - window_2 + 1] = rep_sim
        pred_out.append(zeros)
    pred_out = torch.cat(pred_out).to(batch.device)
    pred_out = torch.sigmoid(-pred_out / 0.1)    
    #TODOOOOOOOOOOOOOOOOOOOOOOOOO
    pred_out[pred_out == 0.5] = 0
    return pred_out    


def get_tscp_output_2(tscp_model, batch, window):
    if len(batch.shape) <= 4:
        seq_len = batch.shape[1]
    else:
        seq_len = batch.shape[2]

    batch_history_slices, batch_future_slices = _history_future_separation_test_2(batch, window)
    
    pred_out = []
    for i in range(len(batch_history_slices)):
        zeros = torch.ones(1, seq_len)
        curr_history = tscp_model(batch_history_slices[i]) 
        curr_future = tscp_model(batch_future_slices[i]) 
        rep_sim = _cosine_simililarity_dim1(curr_history, curr_future).data
        
        zeros[:, 2 * window - 1:] = rep_sim
        pred_out.append(zeros)
    pred_out = torch.cat(pred_out).to(batch.device)
    pred_out = torch.sigmoid(-pred_out)
    #pred_out[pred_out == 0.5] = 0
    #print(pred_out)
    return pred_out    