"""Functions & models for KL-CPD baseline training and testing."""
from typing import List, Optional, Tuple
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

# --------------------------------------------------------------------------------------#
#                                          Loss                                         #
# --------------------------------------------------------------------------------------#
def median_heuristic(med_sqdist: float, beta: float = 0.5) -> List[float]:
    """Initialize kernel's sigma with median heuristic.

    See https://arxiv.org/pdf/1707.07269.pdf

    :param med_sqdist: scale parameter
    :param beta: target sigma
    :return: list of possible sigmas
    """
    beta_list = [beta ** 2, beta ** 1, 1, (1.0 / beta) ** 1, (1.0 / beta) ** 2]
    return [med_sqdist * b for b in beta_list]


def batch_mmd2_loss(
    enc_past: torch.Tensor, enc_future: torch.Tensor, sigma_var: torch.Tensor
) -> object:
    """Calculate MMD loss for batch.

    :param enc_past: encoded past sequence
    :param enc_future: encoded future
    :param sigma_var: tensor with considered kernel parameters
    :return: batch MMD loss
    """
    device = enc_past.device

    n_basis = 1024
    gumbel_lmd = 1e6
    norm_cnst = math.sqrt(1.0 / n_basis)
    n_mixtures = sigma_var.size(0)
    n_samples = n_basis * n_mixtures
    batch_size, _, n_latent = enc_past.size()

    weights = (
        torch.FloatTensor(batch_size * n_samples, n_latent).normal_(0, 1).to(device)
    )
    weights.requires_grad = False

    # gumbel trick to get masking matrix to uniformly sample sigma
    uniform_mask = (
        torch.FloatTensor(batch_size * n_samples, n_mixtures).uniform_().to(device)
    )
    sigma_samples = F.softmax(uniform_mask * gumbel_lmd, dim=1).matmul(sigma_var)
    weights_gmm = weights.mul(1.0 / sigma_samples.unsqueeze(1))
    weights_gmm = weights_gmm.reshape(
        batch_size, n_samples, n_latent
    )  # batch_size x n_samples x nz
    weights_gmm = torch.transpose(
        weights_gmm, 1, 2
    ).contiguous()  # batch_size x nz x n_samples

    _kernel_enc_past = torch.bmm(
        enc_past, weights_gmm
    )  # batch_size x seq_len x n_samples
    _kernel_enc_future = torch.bmm(
        enc_future, weights_gmm
    )  # batch_size x seq_len x n_samples

    # approximate kernel with cos and sin
    kernel_enc_past = norm_cnst * torch.cat(
        (torch.cos(_kernel_enc_past), torch.sin(_kernel_enc_past)), 2
    )
    kernel_enc_future = norm_cnst * torch.cat(
        (torch.cos(_kernel_enc_future), torch.sin(_kernel_enc_future)), 2
    )
    batch_mmd2_rff = torch.sum(
        (kernel_enc_past.mean(1) - kernel_enc_future.mean(1)) ** 2, 1
    )
    return batch_mmd2_rff


def mmd_loss_disc(
    input_future: torch.Tensor,
    fake_future: torch.Tensor,
    enc_future: torch.Tensor,
    enc_fake_future: torch.Tensor,
    enc_past: torch.Tensor,
    dec_future: torch.Tensor,
    dec_fake_future: torch.Tensor,
    lambda_ae: float,
    lambda_real: float,
    sigma_var: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate loss for discriminator in KL-CPD model.

    :param input_future: real input subsequence corresponding to the future
    :param fake_future: fake subsequence obtained from generator
    :param enc_future: net_discriminator(input_future)
    :param enc_fake_future: net_discriminator(fake_future)
    :param enc_past: net_discriminator(input_past)
    :param dec_future: last hidden from net_discriminator(input_future)
    :param dec_fake_future: last hidden from net_discriminator(fake_future)
    :param lambda_ae: coefficient before reconstruction loss
    :param lambda_real: coefficient before MMD between past and future
    :param sigma_var: list of sigmas for MMD calculation
    :return: discriminator loss, MMD between real subsequences
    """
    # batch-wise MMD2 loss between real and fake
    mmd2_fake = batch_mmd2_loss(enc_future, enc_fake_future, sigma_var)

    # batch-wise MMD2 loss between past and future
    mmd2_real = batch_mmd2_loss(enc_past, enc_future, sigma_var)

    # reconstruction loss
    real_l2_loss = torch.mean((input_future - dec_future) ** 2)
    fake_l2_loss = torch.mean((fake_future - dec_fake_future) ** 2)

    loss_disc = (
        mmd2_fake.mean()
        - lambda_ae * (real_l2_loss + fake_l2_loss)
        - lambda_real * mmd2_real.mean()
    )
    return loss_disc.mean(), mmd2_real.mean()


# --------------------------------------------------------------------------------------#
#                             Data preprocessing                                        #
# --------------------------------------------------------------------------------------#

# separation for training
def history_future_separation(
    data: torch.Tensor,
    window: int
 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split sequences in batch on two equal slices.

    :param data: input sequences
    :param window: slice size
    :return: set of "past" subsequences and corresponded "future" subsequences
    """
    # for all the datasets, except video data
    if len(data.shape) <= 4:
        history = data[:, :window]
        future = data[:, window:2*window]

    # for video data 
    elif len(data.shape) == 5:
        history = data[:, :, :window]
        future = data[:, :, window:2*window]    
        
    return history, future


# separation for test
def history_future_separation_test(
    data: torch.Tensor,
    window_1: int,
    window_2: Optional[int] = None,
    step: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare data for testing. Separate it in set of "past"-"future" slices.

    :param data: input sequence
    :param window_1: "past" subsequence size
    :param window_2: "future" subsequence size (default None), if None set equal to window_1
    :param step: step size
    :return: set of "past" subsequences and corresponded "future" subsequences
    """
    future_slices = []
    history_slices = []

    if window_2 is None:
        window_2 = window_1

    if len(data.shape) > 4:
        data = data.transpose(1, 2)

    seq_len = data.shape[1]
    for i in range(0, (seq_len - window_1 - window_2) // step + 1):
        start_ind = i * step
        end_ind = window_1 + window_2 + step * i
        slice_2w = data[:, start_ind:end_ind]
        history_slices.append(slice_2w[:, :window_1].unsqueeze(0))
        future_slices.append(slice_2w[:, window_1:].unsqueeze(0))

    future_slices = torch.cat(future_slices).transpose(0, 1)
    history_slices = torch.cat(history_slices).transpose(0, 1)

    # in case of video data
    if len(data.shape) > 4:
        history_slices = history_slices.transpose(2, 3)
        future_slices = future_slices.transpose(2, 3)

    return history_slices, future_slices

# --------------------------------------------------------------------------------------#
#                                     Predictions                                       #
# --------------------------------------------------------------------------------------#
def get_klcpd_output_scaled(
    kl_cpd_model: nn.Module,
    batch: torch.Tensor,
    window_1: int,
    window_2: Optional[int] = None,
    scale: float = 1.,
) -> List[torch.Tensor]:
    """Get KL-CPD predictions scaled to [0, 1].

    :param kl_cpd_model: pre-trained KL-CPD model
    :param batch: input data
    :param window_1: "past" subsequence size
    :param window_2: "future" subsequence size (default None), if None set equal to window_1
    :param scales: scale factor
    :return: scaled prediction of MMD score
    """
    device = kl_cpd_model.device
    batch = batch.to(device).float()
    sigma_var = kl_cpd_model.sigma_var.to(device)

    if len(batch.shape) <= 4:
        seq_len = batch.shape[1]
    else:
        seq_len = batch.shape[2]

    batch_history_slices, batch_future_slices = history_future_separation_test(
        batch, window_1, window_2
    )

    pred_out = []
    for i in range(len(batch_history_slices)):
        zeros = torch.zeros(1, seq_len)
        mmd_scores = kl_cpd_model.get_disc_embeddings(batch_history_slices[i], batch_future_slices[i])
        zeros[:, window_1 + window_2 - 1 :] = mmd_scores
        pred_out.append(zeros)

    pred_out = torch.cat(pred_out).to(device)
    pred_out = torch.tanh(pred_out * scale)
    return pred_out


# --------------------------------------------------------------------------------------#
#                                     Models                                            #
# --------------------------------------------------------------------------------------#
class NetG(nn.Module):
    def __init__(self, args: dict) -> None:
        """Initialize generator model.

        :param args: dict with all the parameters
        """
        super(NetG, self).__init__()
        self.input_dim = args["model"]["input_dim"]
        self.rnn_hid_dim = args["model"]["rnn_hid_dim"]
        self.num_layers = args["model"]["num_layers"]

        self.rnn_enc_layer = nn.GRU(self.input_dim, self.rnn_hid_dim, num_layers=self.num_layers, batch_first=True)
        self.rnn_dec_layer = nn.GRU(self.input_dim, self.rnn_hid_dim, num_layers=self.num_layers, batch_first=True)
        self.fc_layer = nn.Linear(self.rnn_hid_dim, self.input_dim)

        # X_p:   batch_size x wnd_dim x input_dim (Encoder input)
        # X_f:   batch_size x wnd_dim x input_dim (Decoder input)
        # h_t:   1 x batch_size x RNN_hid_dim
        # noise: 1 x batch_size x RNN_hid_dim
    
    def forward(self, X_p: torch.Tensor, X_f: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Do forwars pass through the model.

        :param X_p: past window slice
        :param X_f: future window slice
        :param noice: standard Normal noice
        :return: output of the model
        """ 
        X_p_enc, h_t = self.rnn_enc_layer(X_p.float())
        X_f_shft = self.shft_right_one(X_f.float())
        hidden = h_t + noise
        Y_f, _ = self.rnn_dec_layer(X_f_shft, hidden)
        output = self.fc_layer(Y_f)
        return output

    def shft_right_one(self, X: torch.Tensor) -> torch.Tensor:
        """Shift input tensor to the right and fill the 1st element with zero.

        :param X: input tensor
        returns X_shft: shifted tensor 
        """
        X_shft = X.clone()
        X_shft[:, 0, :].data.fill_(0)
        X_shft[:, 1:, :] = X[:, :-1, :]
        return X_shft
    
class NetD(nn.Module):
    def __init__(self, args: dict) -> None:
        """Initialize discriminator model.
        
        :param args: dict with all the parameters
        """
        super(NetD, self).__init__()
        self.input_dim = args["model"]["input_dim"]
        self.rnn_hid_dim = args["model"]['rnn_hid_dim']
        self.num_layers = args["model"]["num_layers"]

        self.rnn_enc_layer = nn.GRU(self.input_dim, self.rnn_hid_dim, num_layers=self.num_layers, batch_first=True)
        self.rnn_dec_layer = nn.GRU(self.rnn_hid_dim, self.input_dim, num_layers=self.num_layers, batch_first=True)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Do forward pass through the model.""" 
        X_enc, _ = self.rnn_enc_layer(X.float())
        X_dec, _ = self.rnn_dec_layer(X_enc)
        return X_enc, X_dec


class VideoNetG(nn.Module):
    def __init__(self, args: dict) -> None:
        """Initialize generator model for experiments with videos.

        :param args: dict with all the parameters
        """
        super(VideoNetG, self).__init__()
        self.input_dim = args["model"]["input_dim"]
        self.lin_emb_dim = args["model"]["lin_emb_dim"]

        self.rnn_hid_dim = args["model"]["rnn_hid_dim"]
        self.num_layers = args["model"]["num_layers"]

        self.rnn_enc_layer = nn.GRU(self.lin_emb_dim, self.rnn_hid_dim, num_layers=self.num_layers, batch_first=True)
        self.rnn_dec_layer = nn.GRU(self.lin_emb_dim, self.rnn_hid_dim, num_layers=self.num_layers, batch_first=True)
        self.fc_layer = nn.Linear(self.rnn_hid_dim, self.input_dim)

        # extra linear layer to reduce number of parameters in GRUs
        self.linear_encoder = nn.Linear(self.input_dim, self.lin_emb_dim)
        self.relu = nn.ReLU()
    
    def forward(self, X_p: torch.Tensor, X_f: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Do forwars pass through the model.

        :param X_p: past window slice
        :param X_f: future window slice
        :param noice: standard Normal noice
        :return: output of the model
        """
        X_p = self.relu(self.linear_encoder(X_p))
        X_f = self.relu(self.linear_encoder(X_f))

        X_p_enc, h_t = self.rnn_enc_layer(X_p)
        X_f_shft = self.shft_right_one(X_f)
        hidden = h_t + noise
        Y_f, _ = self.rnn_dec_layer(X_f_shft, hidden)
        output = self.fc_layer(Y_f)
        return output

    def shft_right_one(self, X: torch.Tensor) -> torch.Tensor:
        """Shift input tensor to the right and fill the 1st element with zero.

        :param X: input tensor
        returns X_shft: shifted tensor 
        """
        X_shft = X.clone()
        X_shft[:, 0, :].data.fill_(0)
        X_shft[:, 1:, :] = X[:, :-1, :]
        return X_shft
    
class VideoNetD(nn.Module):
    def __init__(self, args: dict) -> None:
        """Initialize discriminator model for experiments with videos.
        
        :param args: dict with all the parameters
        """
        super(VideoNetD, self).__init__()
        self.input_dim = args["model"]["input_dim"]
        self.lin_emb_dim = args["model"]["lin_emb_dim"]
        self.rnn_hid_dim = args["model"]['rnn_hid_dim']
        self.num_layers = args["model"]["num_layers"]

        self.rnn_enc_layer = nn.GRU(self.lin_emb_dim, self.rnn_hid_dim, num_layers=self.num_layers, batch_first=True)
        self.rnn_dec_layer = nn.GRU(self.rnn_hid_dim, self.lin_emb_dim, num_layers=self.num_layers, batch_first=True)

        # extra linear layers to reduce number of parameters in GRUs
        self.linear_encoder = nn.Linear(self.input_dim, self.lin_emb_dim)
        self.linear_decoder = nn.Linear(self.lin_emb_dim, self.input_dim)
        self.relu = nn.ReLU()

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Do forward pass through the model."""
        X = self.relu(self.linear_encoder(X))
        X_enc, _ = self.rnn_enc_layer(X)
        X_dec, _ = self.rnn_dec_layer(X_enc)
        X_dec = self.relu(self.linear_decoder(X_dec))
        return X_enc, X_dec
    

class KLCPD(pl.LightningModule):
    """Class for implementation KL-CPD model."""
    def __init__(
        self,
        args: dict,
        net_generator: nn.Module,
        net_discriminator: nn.Module,
        train_dataset: Dataset,
        test_dataset: Dataset,
    ) -> None:
        """Initialize KL-CPD model.

        :param args: dictionary with models' parameters
        :param net_generator: generator model
        :param net_discriminator: discriminator model
        :param train_dataset: train dataset
        :param test_dataset: test dataset
        """
        super().__init__()
        self.args = args
        self.net_generator = net_generator
        self.net_discriminator = net_discriminator

        # Feature extractor for video datasets
        if args["experiments_name"] in ["explosion", "road_accidents"]:
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

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        
        self.batch_size = args["learning"]["batch_size"]

        sigma_list = median_heuristic(self.args["loss"]["sqdist"], beta=0.5)
        self.sigma_var = torch.FloatTensor(sigma_list)

        # to get predictions
        self.window_1 = self.args["model"]["window_1"]
        self.window_2 = self.args["model"]["window_2"]

        self.num_workers = args["num_workers"]

    def __preprocess(self, input: torch.Tensor) -> torch.Tensor:
        """Preprocess batch before forwarding (i.e. apply extractor for video input).

        :param input: input torch.Tensor
        :return: processed input tensor to be fed into .forward method 
        """
        if self.args["experiments_name"] in ["explosion", "road_accidents"]:
            input = self.extractor(input.float())  # batch_size, C, seq_len, H, W
            input = input.transpose(1, 2).flatten(2)
        else:
            input = input.reshape(-1, self.args["model"]['wnd_dim'], self.args["model"]['input_dim'])
        return input

    def __initialize_noise(self, batch_size: int) -> torch.Tensor:
        """Initialize standard normal noise tensor for generator.

        :param batch_size: batch size (noise has shape 1 x batch_size x rnn_hid_dim)
        :return: noise
        """
        if np.isscalar(self.args["model"]["rnn_hid_dim"]):
            noise = torch.FloatTensor(1, batch_size, self.args["model"]["rnn_hid_dim"]).normal_(
                0, 1
            )
        else:
            noise = torch.FloatTensor(batch_size, *self.args["model"]["rnn_hid_dim"]).normal_(
                0, 1
            )
        noise.requires_grad = False
        noise = noise.to(self.device)
        return noise

    def get_disc_embeddings(
        self,
        input_past: torch.Tensor,
        input_future: torch.Tensor
        ) -> torch.Tensor:
        """Pass input through the discriminator network and compute MMD score.

        :param input_past: tensor with past slice (len is window_1)
        :param inoput_future: tensor with future slice (len is window_2)
        :return: predicted MMD scores
        """
        # prepare input
        input_past = self.__preprocess(input_past)
        input_future = self.__preprocess(input_future)

        # pass it through the discriminator
        enc_past, _ = self.net_discriminator(input_past)
        enc_future, _ = self.net_discriminator(input_future)

        enc_past, enc_future = [
            enc_i.reshape(*enc_i.shape[:2], -1) for enc_i in [enc_past, enc_future]
        ]
        # get predicted MMD scores
        predicted_mmd_score = batch_mmd2_loss(
            enc_past, enc_future, self.sigma_var.to(self.device)
        )
        return predicted_mmd_score

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward step for KL-CPD model.

        :param inputs: input data
        :return: embedded data
        """
        input_past, input_future = history_future_separation(
            inputs[0].to(torch.float32), self.args["model"]["wnd_dim"]
        )

        predicted_mmd_score = self.get_disc_embeddings(input_past, input_future)
        return predicted_mmd_score

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_idx: int,
        optimizer_closure,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ):
        """Define optimization step for KL-CPD model.

        :param epoch: number of current epoch
        :param batch_idx: number of used batch index
        :param optimizer: used optimizers (for generator and discriminator)
        :param optimizer_idx: if 0 - update generator, if 1 - update discriminator
        :param optimizer_closure: closure
        :param on_tpu: if True, calculate on TPU
        :param using_native_amp: some parameters
        :param using_lbfgs: some parameters
        """
        # update generator every CRITIC_ITERS steps
        if optimizer_idx == 0:
            if (batch_idx + 1) % self.args["learning"]["critic_iters"] == 0:
                # the closure (which includes the `training_step`) will be
                # executed by `optimizer.step`
                optimizer.step(closure=optimizer_closure)
            else:
                # call the closure by itself to run `training_step` + `backward`
                # without an optimizer step
                optimizer_closure()

        # update discriminator every step
        if optimizer_idx == 1:
            for param in self.net_discriminator.rnn_enc_layer.parameters():
                param.data.clamp_(-self.args["learning"]["weight_clip"], self.args["learning"]["weight_clip"])
            optimizer.step(closure=optimizer_closure)

    def training_step(
        self, batch: torch.Tensor, batch_idx: int, optimizer_idx: int
    ) -> torch.Tensor:
        """Training step for KL-CPD model.

        :param batch: input data
        :param batch_idx: index of batch
        :param optimizer_idx: index of optimizer (0 for generator, 1 for discriminator)
        :return: train loss
        """
        batch_size = batch[0].size(0)
        input_past, input_future = history_future_separation(
            batch[0].to(torch.float32), self.args["model"]["wnd_dim"]
        )
        input_past = self.__preprocess(input_past)
        input_future = self.__preprocess(input_future)

        enc_past, _ = self.net_discriminator(input_past)
        enc_future, hidden_future = self.net_discriminator(input_future)
        noise = self.__initialize_noise(batch_size)

        fake_future = self.net_generator(input_past, input_future, noise)
        enc_fake_future, hidden_fake_future = self.net_discriminator(fake_future)
        # optimize discriminator
        if optimizer_idx == 1:
            all_data = [
                input_future,
                fake_future,
                enc_future,
                enc_fake_future,
                enc_past,
                hidden_future,
                hidden_fake_future,
            ]
            all_data = [data_i.reshape(*data_i.shape[:2], -1) for data_i in all_data]

            loss_disc, mmd2_real = mmd_loss_disc(
                *all_data,
                self.args["loss"]["lambda_ae"],
                self.args["loss"]["lambda_real"],
                self.sigma_var.to(self.device)
            )
            loss_disc = (-1) * loss_disc
            self.log("tlD", loss_disc, prog_bar=True)
            self.log("train_mmd2_real_D", mmd2_real, prog_bar=True)

            return loss_disc

        # optimize generator
        if optimizer_idx == 0:
            all_future_enc = [enc_future, enc_fake_future]
            all_future_enc = [
                enc_i.reshape(*enc_i.shape[:2], -1) for enc_i in all_future_enc
            ]

            # batch-wise MMD2 loss between input_future and fake_future
            gen_mmd2 = batch_mmd2_loss(
                *all_future_enc, self.sigma_var.to(self.device)
            )
            loss_gen = gen_mmd2.mean()
            self.log("tlG", loss_gen, prog_bar=True)

            return loss_gen

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Validation step for KL-CPD model.

        :param batch: input data
        :param batch_idx: index of batch
        :return: MMD score
        """
        val_mmd2_real = self.forward(batch).mean()
        self.log("val_mmd2_real_D", val_mmd2_real, prog_bar=True)

        return val_mmd2_real

    def configure_optimizers(
        self,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """Configure optimizers.

        :return: optimizers
        """
        optimizer_gen = torch.optim.Adam(
            self.net_generator.parameters(),
            lr=self.args["learning"]["lr"],
            weight_decay=self.args["learning"]["weight_decay"],
        )

        optimizer_disc = torch.optim.Adam(
            self.net_discriminator.parameters(),
            lr=self.args["learning"]["lr"],
            weight_decay=self.args["learning"]["weight_decay"],
        )

        return optimizer_gen, optimizer_disc

    def train_dataloader(self) -> DataLoader:
        """Set train dataloader.

        :return: dataloader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Set val dataloader.

        :return: dataloader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )