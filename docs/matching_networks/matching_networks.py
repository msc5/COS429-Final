import torch
import torch.nn as nn

import numpy as np


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# Generates a convolution block which performs:
# 1. 3x3 Convolution (in_channels, out_channels, 3)
# 2. Batch Normalization
# 3. ReLU
# 4. 2x2 Max Pooling
def conv_block(
    in_channels: int,
    out_channels: int,
):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


def get_few_shot_encoder(num_input_channels=1):
    return nn.Sequential(
        conv_block(num_input_channels, 64),
        conv_block(64, 64),
        conv_block(64, 64),
        conv_block(64, 64),
        Flatten(),
    )


class BidirectionalLSTM(nn.Module):
    def __init__(self, size: int, layers: int):
        """Bidirectional LSTM used to generate fully conditional embeddings (FCE) of the support set as described
        in the Matching Networks paper.
        # Arguments
            size: Size of input and hidden layers. These are constrained to be the same in order to implement the skip
                connection described in Appendix A.2
            layers: Number of LSTM layers
        """
        super(BidirectionalLSTM, self).__init__()
        self.num_layers = layers
        self.batch_size = 1
        # Force input size and hidden size to be the same in order to implement
        # the skip connection as described in Appendix A.1 and A.2 of Matching Networks
        self.lstm = nn.LSTM(input_size=size,
                            num_layers=layers,
                            hidden_size=size,
                            bidirectional=True)

    def forward(self, inputs):
        # Give None as initial state and Pytorch LSTM creates initial hidden states
        output, (hn, cn) = self.lstm(inputs, None)

        forward_output = output[:, :, :self.lstm.hidden_size]
        backward_output = output[:, :, self.lstm.hidden_size:]

        # g(x_i, S) = h_forward_i + h_backward_i + g'(x_i) as written in Appendix A.2
        # AKA A skip connection between inputs and outputs is used
        output = forward_output + backward_output + inputs
        return output, hn, cn


class AttentionLSTM(nn.Module):
    def __init__(self, size: int, unrolling_steps: int):
        """Attentional LSTM used to generate fully conditional embeddings (FCE) of the query set as described
        in the Matching Networks paper.
        # Arguments
            size: Size of input and hidden layers. These are constrained to be the same in order to implement the skip
                connection described in Appendix A.2
            unrolling_steps: Number of steps of attention over the support set to compute. Analogous to number of
                layers in a regular LSTM
        """
        super(AttentionLSTM, self).__init__()
        self.unrolling_steps = unrolling_steps
        self.lstm_cell = nn.LSTMCell(input_size=size,
                                     hidden_size=size)

    def forward(self, support, queries):
        # Get embedding dimension, d
        if support.shape[-1] != queries.shape[-1]:
            raise(ValueError("Support and query set have different embedding dimension!"))

        batch_size = queries.shape[0]
        embedding_dim = queries.shape[1]

        h_hat = torch.zeros_like(queries).cuda().double()
        c = torch.zeros(batch_size, embedding_dim).cuda().double()

        for k in range(self.unrolling_steps):
            # Calculate hidden state cf. equation (4) of appendix A.2
            h = h_hat + queries

            # Calculate softmax attentions between hidden states and support set embeddings
            # cf. equation (6) of appendix A.2
            attentions = torch.mm(h, support.t())
            attentions = attentions.softmax(dim=1)

            # Calculate readouts from support set embeddings cf. equation (5)
            readout = torch.mm(attentions, support)

            # Run LSTM cell cf. equation (3)
            # h_hat, c = self.lstm_cell(queries, (torch.cat([h, readout], dim=1), c))
            h_hat, c = self.lstm_cell(queries, (h + readout, c))

        h = h_hat + queries

        return h


class MatchingNetwork(nn.Module):

    def __init__(
        self,
        n: int,
        k: int,
        q: int,
        fce: bool,
        num_input_channels: int,
        lstm_layers: int,
        lstm_input_size: int,
        unrolling_steps: int,
        device: torch.device
    ):
        super(MatchingNetwork, self).__init__()

        self.n, self.k, self.q = n, k, q
        self.fce, self.num_input_channels = fce, num_input_channels

        self.encoder = get_few_shot_encoder(self.num_input_channels)

        if self.fce:
            self.g = BidirectionalLSTM(lstm_input_size, lstm_layers)
            self.f = AttentionLSTM(
                lstm_input_size, unrolling_steps=unrolling_steps)
