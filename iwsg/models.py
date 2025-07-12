import torch.nn as nn
from model.utils import build_sequential_nn


class DiscNN(nn.Module):
    def __init__(
            self,
            n_features: int,
            output_size: int,
            hidden_size: list,
            norm: bool,
            activation: str,
            dropout: float,
    ):
        super(DiscNN, self).__init__()
        self.in_features = n_features
        self.output_size = output_size

        self.hidden_size = hidden_size
        self.norm = norm
        self.activation = activation
        self.dropout = dropout

        self.model = self._build_model()

    def _build_model(self):
        if not self.hidden_size:
            layers = [nn.Linear(self.in_features, self.output_size)]
        else:
            layers = build_sequential_nn(self.in_features, self.hidden_size, self.norm, self.activation, self.dropout)
            layers.append(nn.Linear(self.hidden_size[-1], self.output_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

