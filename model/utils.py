from torch import nn as nn


def build_sequential_nn(
        in_features: int,
        hidden_dims: list,
        batch_norm: bool,
        activation: str='ReLU',
        dropout: float = None
):
    """Build a sequential neural network, except last layer."""
    assert len(hidden_dims) > 0, "hidden_dims should have at least one element"

    layers = []
    in_dim = in_features
    out_dim = hidden_dims[0]
    for i in range(len(hidden_dims)):
        layers.append(nn.Linear(in_dim, out_dim))

        if batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))

        layers.append(getattr(nn, activation)())

        if dropout is not None and dropout > 0:
            layers.append(nn.Dropout(dropout))

        in_dim = out_dim
        out_dim = hidden_dims[i + 1] if i + 1 < len(hidden_dims) else None
    return layers
