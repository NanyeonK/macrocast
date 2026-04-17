"""TCN (Temporal Convolutional Network) adapter for Phase 5.

Minimal single-stack causal dilated-conv implementation. Kept in-tree to
avoid pulling an external pytorch-tcn dep into the [deep] extra.
"""
from __future__ import annotations

from ._base import _BaseDeepModel, _make_last_timestep_module
from ._import_guard import require_torch


def _make_temporal_block(c_in: int, c_out: int, *, kernel_size: int, dilation: int, dropout: float):
    torch = require_torch("tcn")

    class _Chomp(torch.nn.Module):
        def __init__(self, chomp_size: int):
            super().__init__()
            self.chomp_size = chomp_size

        def forward(self, x):
            return x[..., : -self.chomp_size] if self.chomp_size > 0 else x

    class _TemporalBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            padding = (kernel_size - 1) * dilation
            self.conv1 = torch.nn.Conv1d(
                c_in, c_out, kernel_size, padding=padding, dilation=dilation
            )
            self.chomp1 = _Chomp(padding)
            self.relu1 = torch.nn.ReLU()
            self.drop1 = torch.nn.Dropout(dropout)

            self.conv2 = torch.nn.Conv1d(
                c_out, c_out, kernel_size, padding=padding, dilation=dilation
            )
            self.chomp2 = _Chomp(padding)
            self.relu2 = torch.nn.ReLU()
            self.drop2 = torch.nn.Dropout(dropout)

            self.downsample = (
                torch.nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
            )
            self.relu_final = torch.nn.ReLU()

        def forward(self, x):
            out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
            out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
            res = x if self.downsample is None else self.downsample(x)
            return self.relu_final(out + res)

    return _TemporalBlock()


def _make_channels_first():
    torch = require_torch("tcn")

    class _Tx(torch.nn.Module):
        def forward(self, x):
            return x.transpose(1, 2)

    return _Tx()


class TCNModel(_BaseDeepModel):
    model_family = "tcn"

    def _build_net(self, n_features: int):
        torch = require_torch(self.model_family)
        channels = [self.config.hidden_size] * self.config.n_layers
        blocks = []
        for i, c_out in enumerate(channels):
            c_in = n_features if i == 0 else channels[i - 1]
            blocks.append(
                _make_temporal_block(
                    c_in,
                    c_out,
                    kernel_size=3,
                    dilation=2 ** i,
                    dropout=self.config.dropout,
                )
            )
        return torch.nn.Sequential(
            _make_channels_first(),
            *blocks,
            _make_last_timestep_module(),
            torch.nn.Linear(channels[-1], 1),
        )


__all__ = ["TCNModel"]
