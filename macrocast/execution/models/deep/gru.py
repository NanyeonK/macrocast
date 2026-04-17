"""GRU deep-model adapter for Phase 5."""
from __future__ import annotations

from ._base import _BaseDeepModel, _make_packer_module
from ._import_guard import require_torch


class GRUModel(_BaseDeepModel):
    model_family = "gru"

    def _build_net(self, n_features: int):
        torch = require_torch(self.model_family)
        rnn = torch.nn.GRU(
            input_size=n_features,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.n_layers,
            dropout=self.config.dropout if self.config.n_layers > 1 else 0.0,
            batch_first=True,
        )
        return torch.nn.Sequential(
            _make_packer_module("gru", rnn),
            torch.nn.Linear(self.config.hidden_size, 1),
        )


__all__ = ["GRUModel"]
