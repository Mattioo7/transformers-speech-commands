import math
from typing import Any

import torch
from torch import nn

from .config import FeatureFixedParams


class LSTMBaseline(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        dropout: float,
        num_classes: int,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        direction_multiplier = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * direction_multiplier, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(inputs)
        pooled = output.mean(dim=1)
        return self.classifier(pooled)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.encoding[:, : inputs.size(1)]


class TransformerBaseline(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        heads: int,
        layers: int,
        ff_dim: int,
        dropout: float,
        num_classes: int,
    ):
        super().__init__()
        self.projection = nn.Linear(input_size, d_model)
        self.position = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.position(self.projection(inputs)))
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)


def build_model(model_params: dict[str, Any], feature_params: FeatureFixedParams, num_classes: int) -> nn.Module:
    if model_params["model_type"] == "lstm":
        return LSTMBaseline(
            input_size=feature_params.n_mels,
            hidden_size=model_params["lstm_hidden_size"],
            num_layers=model_params["lstm_layers"],
            bidirectional=model_params["lstm_bidirectional"],
            dropout=model_params["dropout"],
            num_classes=num_classes,
        )

    return TransformerBaseline(
        input_size=feature_params.n_mels,
        d_model=model_params["transformer_d_model"],
        heads=model_params["transformer_heads"],
        layers=model_params["transformer_layers"],
        ff_dim=model_params["transformer_ff_dim"],
        dropout=model_params["dropout"],
        num_classes=num_classes,
    )
