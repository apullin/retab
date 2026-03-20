from __future__ import annotations

import torch
import torch.nn as nn

from lib.registry import ModelSpec, register_model


class DecomposedElmanRNN(nn.Module):
    def __init__(self, input_size: int = 16, hidden_size: int = 24, num_classes: int = 5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.in_proj = nn.Linear(input_size, hidden_size)
        self.h_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        h = x.new_zeros(batch, self.hidden_size)
        for x_t in x.unbind(dim=1):
            h = torch.tanh(self.in_proj(x_t) + self.h_proj(h))
        return self.out_proj(h)


class DecomposedGRU(nn.Module):
    def __init__(self, input_size: int = 16, hidden_size: int = 24, num_classes: int = 5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.x_r = nn.Linear(input_size, hidden_size)
        self.h_r = nn.Linear(hidden_size, hidden_size)
        self.x_z = nn.Linear(input_size, hidden_size)
        self.h_z = nn.Linear(hidden_size, hidden_size)
        self.x_n = nn.Linear(input_size, hidden_size)
        self.h_n = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        h = x.new_zeros(batch, self.hidden_size)
        one = torch.ones_like(h)
        for x_t in x.unbind(dim=1):
            r = torch.sigmoid(self.x_r(x_t) + self.h_r(h))
            z = torch.sigmoid(self.x_z(x_t) + self.h_z(h))
            n = torch.tanh(self.x_n(x_t) + r * self.h_n(h))
            h = (one - z) * n + z * h
        return self.out_proj(h)


class DecomposedLSTM(nn.Module):
    def __init__(self, input_size: int = 16, hidden_size: int = 24, num_classes: int = 5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.x_i = nn.Linear(input_size, hidden_size)
        self.h_i = nn.Linear(hidden_size, hidden_size)
        self.x_f = nn.Linear(input_size, hidden_size)
        self.h_f = nn.Linear(hidden_size, hidden_size)
        self.x_g = nn.Linear(input_size, hidden_size)
        self.h_g = nn.Linear(hidden_size, hidden_size)
        self.x_o = nn.Linear(input_size, hidden_size)
        self.h_o = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        h = x.new_zeros(batch, self.hidden_size)
        c = x.new_zeros(batch, self.hidden_size)
        for x_t in x.unbind(dim=1):
            i = torch.sigmoid(self.x_i(x_t) + self.h_i(h))
            f = torch.sigmoid(self.x_f(x_t) + self.h_f(h))
            g = torch.tanh(self.x_g(x_t) + self.h_g(h))
            o = torch.sigmoid(self.x_o(x_t) + self.h_o(h))
            c = f * c + i * g
            h = o * torch.tanh(c)
        return self.out_proj(h)


register_model(
    ModelSpec(
        name="elman_rnn_decomposed",
        factory=DecomposedElmanRNN,
        input_shape=(1, 8, 16),
        family="recurrent",
        notes="Decomposed Elman cell unrolled over time",
    )
)

register_model(
    ModelSpec(
        name="gru_rnn_decomposed",
        factory=DecomposedGRU,
        input_shape=(1, 8, 16),
        family="recurrent",
        notes="Decomposed GRU cell with explicit gates",
    )
)

register_model(
    ModelSpec(
        name="lstm_rnn_decomposed",
        factory=DecomposedLSTM,
        input_shape=(1, 8, 16),
        family="recurrent",
        notes="Decomposed LSTM cell with explicit gates",
    )
)
