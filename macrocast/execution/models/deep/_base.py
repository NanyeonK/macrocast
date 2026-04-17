"""Shared infrastructure for Phase 5 deep-model executors.

Every family (LSTM, GRU, TCN) inherits ``_BaseDeepModel`` and implements a
single ``_build_net(n_features)`` factory; the fit loop, early stopping,
best-weights restoration, mini-batch determinism, and seeding live here so
the per-family files stay short.

The config is intentionally immutable with fixed defaults — Phase 5 ships
these three families with no tuning axis. HP tuning is a Phase 10 deliverable.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass

import numpy as np

from ._import_guard import require_torch


@dataclass(frozen=True)
class DeepModelConfig:
    """Fixed defaults for Phase 5 deep families. No tuning axis in v0.7."""

    lookback: int = 12
    hidden_size: int = 64
    n_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    max_epochs: int = 50
    batch_size: int = 32
    early_stopping_patience: int = 10
    validation_fraction: float = 0.2
    seed: int = 0


def _seed_everything(seed: int) -> None:
    torch = require_torch("deep")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class _LastTimeStep:
    """Sentinel used by _Packer — selects the final timestep of an RNN output."""


class _Packer:
    """Build-time helper that wraps an nn.LSTM/nn.GRU to return only the last
    timestep's hidden state as a flat ``(batch, hidden_size)`` tensor.
    """

    def __init__(self, kind: str, rnn):
        self.kind = kind
        self.rnn = rnn


def _make_packer_module(kind: str, rnn):
    """Build the ``_Packer`` wrapper as a torch.nn.Module subclass.

    Factory (rather than a module-level class) so this file has no
    import-time torch dependency — the class is assembled only when a deep
    executor runs.
    """
    torch = require_torch("deep")

    class _PackerModule(torch.nn.Module):
        def __init__(self, rnn):
            super().__init__()
            self.rnn = rnn

        def forward(self, x):  # x: (batch, time, features)
            out, _ = self.rnn(x)
            return out[:, -1, :]  # last timestep → (batch, hidden_size)

    return _PackerModule(rnn)


def _make_last_timestep_module():
    torch = require_torch("deep")

    class _LastStep(torch.nn.Module):
        def forward(self, x):  # x: (batch, channels, time)
            return x[..., -1]

    return _LastStep()


class _BaseDeepModel:
    """Shared fit / predict machinery for Phase 5 deep families.

    Subclasses set ``model_family`` and implement ``_build_net(n_features)``
    returning a ready-to-train ``torch.nn.Module`` that maps
    ``(batch, time, features)`` → ``(batch, 1)`` (or ``(batch,)`` — the loop
    handles both via ``.squeeze(-1)``).
    """

    model_family: str = "base"

    def __init__(self, *, config: DeepModelConfig):
        self.config = config
        _seed_everything(config.seed)
        self.net = None  # built lazily in fit() once n_features is known
        self._device = "cpu"

    def _build_net(self, n_features: int):
        raise NotImplementedError

    def fit(self, X_seq: np.ndarray, y_seq: np.ndarray) -> "_BaseDeepModel":
        torch = require_torch(self.model_family)
        if X_seq.ndim != 3:
            raise ValueError(
                f"{self.model_family}: X_seq must be 3-D (n_windows, lookback, features), "
                f"got shape {X_seq.shape}"
            )
        n_windows, lookback, n_features = X_seq.shape
        if n_windows < 1:
            raise ValueError(f"{self.model_family}: no training windows available")
        if lookback != self.config.lookback:
            raise ValueError(
                f"{self.model_family}: X_seq lookback {lookback} != config.lookback "
                f"{self.config.lookback}"
            )

        self.net = self._build_net(n_features).to(self._device)

        # Chronological validation split — tail of the windowed array.
        n_val = int(math.floor(n_windows * self.config.validation_fraction))
        use_val = n_val >= 5 and (n_windows - n_val) >= 5
        if use_val:
            X_train = X_seq[: n_windows - n_val]
            y_train = y_seq[: n_windows - n_val]
            X_val = X_seq[n_windows - n_val :]
            y_val = y_seq[n_windows - n_val :]
        else:
            X_train, y_train = X_seq, y_seq
            X_val = y_val = None

        X_train_t = torch.as_tensor(X_train, dtype=torch.float32, device=self._device)
        y_train_t = torch.as_tensor(y_train, dtype=torch.float32, device=self._device)
        if use_val:
            X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=self._device)
            y_val_t = torch.as_tensor(y_val, dtype=torch.float32, device=self._device)

        # Deterministic minibatch shuffling tied to config.seed.
        gen = torch.Generator().manual_seed(int(self.config.seed))
        dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(self.config.batch_size, len(X_train_t)),
            shuffle=True,
            generator=gen,
        )

        loss_fn = torch.nn.MSELoss()
        optim = torch.optim.Adam(self.net.parameters(), lr=self.config.learning_rate)

        best_val = math.inf
        best_state = None
        since_improve = 0

        for _epoch in range(self.config.max_epochs):
            self.net.train()
            for xb, yb in loader:
                optim.zero_grad()
                pred = self.net(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                loss.backward()
                optim.step()

            if use_val:
                self.net.eval()
                with torch.no_grad():
                    val_pred = self.net(X_val_t).squeeze(-1)
                    val_loss = float(loss_fn(val_pred, y_val_t).item())
                if val_loss < best_val - 1e-8:
                    best_val = val_loss
                    best_state = copy.deepcopy(self.net.state_dict())
                    since_improve = 0
                else:
                    since_improve += 1
                    if since_improve >= self.config.early_stopping_patience:
                        break

        if use_val and best_state is not None:
            self.net.load_state_dict(best_state)

        self.net.eval()
        return self

    def predict_next(self, history: np.ndarray) -> float:
        torch = require_torch(self.model_family)
        if self.net is None:
            raise RuntimeError(f"{self.model_family} model not fitted")
        if history.ndim != 1 or len(history) < self.config.lookback:
            raise ValueError(
                f"{self.model_family}: history must be 1-D with length >= lookback "
                f"({self.config.lookback}), got shape {history.shape}"
            )
        window = np.asarray(history[-self.config.lookback :], dtype=float)
        x = torch.as_tensor(
            window.reshape(1, self.config.lookback, 1),
            dtype=torch.float32,
            device=self._device,
        )
        with torch.no_grad():
            y = self.net(x).squeeze(-1)
        return float(y.item())


__all__ = [
    "DeepModelConfig",
    "_BaseDeepModel",
    "_make_packer_module",
    "_make_last_timestep_module",
]
