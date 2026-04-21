from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RNNConfig:
    input_size: int = 1
    hidden_size: int = 64
    num_layers: int = 1
    num_classes: int = 15


class WaveformRNNClassifier(nn.Module):
    """
    Simple RNN classifier for 1D waveforms.
    Input shape: (B, T) or (B, T, 1)
    Output shape: (B, num_classes)
    """

    def __init__(self, cfg: RNNConfig):
        super().__init__()
        self.cfg = cfg
        self.rnn = nn.RNN(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(cfg.hidden_size, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        h0 = torch.zeros(self.cfg.num_layers, x.size(0), self.cfg.hidden_size, device=x.device, dtype=x.dtype)
        out, _ = self.rnn(x, h0)
        last = out[:, -1, :]
        return self.fc(last)


def build_mlp(input_size: int, num_classes: int, hidden: Tuple[int, ...] = (128, 64)) -> nn.Module:
    layers = []
    in_features = input_size
    for h in hidden:
        layers.append(nn.Linear(in_features, h))
        layers.append(nn.ReLU())
        in_features = h
    layers.append(nn.Linear(in_features, num_classes))
    return nn.Sequential(*layers)

