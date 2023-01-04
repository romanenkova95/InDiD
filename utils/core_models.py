""" Core models for our experiments."""
import torch
import torch.nn as nn

from typing import List

class BaseRnn(nn.Module):
    """LSTM-based network for experiments with Synthetic Normal data and Human Activity."""
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
        super().__init__()

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
        lstm_out, hidden = self.lstm(input_seq.float())
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.linear(lstm_out)
        out = self.activation(out)
        out = out.view(batch_size, -1)
        return out

class MnistRNN(nn.Module):
    """Recurrent network for MNIST experiments."""
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
    
class CombinedVideoRNN(nn.Module):
        """LSTM-based network for experiments with videos."""
        def __init__(
            self,
            input_dim: int,
            rnn_hidden_dim: int,
            num_layers: int,
            rnn_dropout: float,
            dropout: float
            ) -> None:
            """ Initialize combined LSTM model for video datasets.

            :param input_dim: dimension of the input data (after feature extraction)
            :param rnn_hidden_dim: hidden dimension for LSTM block
            :param rnn_dropuot: dropout probability in LSTM block
            :param dropout: dropout probability in Dropout layer
            """
            super(CombinedVideoRNN, self).__init__()
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=rnn_hidden_dim, 
                num_layers=num_layers,
                batch_first=True, 
                dropout=rnn_dropout
            )

            self.fc = nn.Linear(rnn_hidden_dim, 1)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()  
            self.activation = nn.Sigmoid()        

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through the model.

            :param x: input torch tensor
            :return: out of the model
            """
            r_out, _ = self.rnn(x)
            r_out = self.dropout(self.fc(r_out))
            out = torch.sigmoid(r_out)
            return out 