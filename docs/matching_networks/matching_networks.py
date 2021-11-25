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
            self.f = AttentionLSTM(lstm_input_size, unrolling_steps=unrolling_steps)
