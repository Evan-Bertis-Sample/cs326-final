# analysis/models/torch_lstm.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from analysis.predict import ModelInputs, ModelOutput, PredictorModel
from analysis.models.base import BasePredictorModel


class _LSTMHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        num_layers: int,
        out_dim: int,
        meta_dim: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # final representation = last hidden (hidden,) + optional meta (meta_dim,)
        self.fc = nn.Linear(hidden + meta_dim, out_dim)

    def forward(self, seq: torch.Tensor, meta: Optional[torch.Tensor] = None):
        # seq: (B, W, F)
        out, (h_n, _) = self.lstm(seq)  # h_n: (num_layers, B, hidden)
        h_last = h_n[-1]  # (B, hidden)
        if meta is not None:
            h_last = torch.cat([h_last, meta], dim=-1)  # (B, hidden+meta_dim)
        y = self.fc(h_last)  # (B, out_dim)
        return y


class TorchLSTMRegressor(BasePredictorModel):
    def __init__(self, **params: Any):
        super().__init__()
        # required
        self.hidden: int = 128
        self.num_layers: int = 1
        self.dropout: float = 0.0
        self.lr: float = 1e-3
        self.weight_decay: float = 0.0
        self.epochs: int = 10
        self.batch_size: int = 128
        self.use_meta: bool = True
        self.device: str = "cpu"  # set "cuda" if available and desired

        super().set_hyperparameters(
            hidden=self.hidden,
            num_layers=self.num_layers,
            dropout=self.dropout,
            lr=self.lr,
            weight_decay=self.weight_decay,
            epochs=self.epochs,
            batch_size=self.batch_size,
            use_meta=self.use_meta,
            device=self.device,
        )
        if params:
            self.set_hyperparameters(**params)

        self._model: Optional[_LSTMHead] = None
        self._out_dim: Optional[int] = None
        self._seq_feat_dim: Optional[int] = None
        self._meta_dim: int = 0

    def name(self) -> str:
        return "torch_lstm"

    def _seq_features(self, x: ModelInputs) -> np.ndarray:
        # per-step features: concat outcome_t | policy_t  -> (W, O+P)
        oh = np.asarray(x.outcome_history, dtype=float)
        ph = np.asarray(x.policy_history, dtype=float)
        return np.concatenate([oh, ph], axis=1)  # (W, F)

    def _meta_features(self, x: ModelInputs) -> Optional[np.ndarray]:
        if not self.use_meta:
            return None
        m = np.asarray(x.meta, dtype=float).ravel()
        return m[None, :]  # (1, M) for batching convenience

    def _stack_batch(self, batch: List[Tuple[ModelInputs, ModelOutput]]):
        seqs = [self._seq_features(x) for (x, _) in batch]  # list of (W, F)
        metas = [self._meta_features(x) for (x, _) in batch]  # list of (1, M) or None
        ys = [y.outcomes for (_, y) in batch]  # list of (O,)

        X_seq = np.stack(seqs, axis=0)  # (N, W, F)
        Y = np.stack(ys, axis=0).astype(float)  # (N, O)

        if self.use_meta:
            # assume same meta per sample
            X_meta = np.concatenate(metas, axis=0)  # (N, M)
        else:
            X_meta = None

        return X_seq, X_meta, Y

    def fit_batch(self, batch: List[Tuple[ModelInputs, ModelOutput]]) -> None:
        if not batch:
            return

        X_seq_np, X_meta_np, Y_np = self._stack_batch(batch)
        N, W, F = X_seq_np.shape
        O = Y_np.shape[1]
        M = 0 if X_meta_np is None else X_meta_np.shape[1]

        self._out_dim = O
        self._seq_feat_dim = F
        self._meta_dim = M

        self._model = _LSTMHead(
            in_dim=F,
            hidden=self.hidden,
            num_layers=self.num_layers,
            out_dim=O,
            meta_dim=M,
            dropout=self.dropout,
        ).to(self.device)

        opt = optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()

        # to torch
        X_seq = torch.tensor(X_seq_np, dtype=torch.float32, device=self.device)
        Y = torch.tensor(Y_np, dtype=torch.float32, device=self.device)
        X_meta = (
            None
            if X_meta_np is None
            else torch.tensor(X_meta_np, dtype=torch.float32, device=self.device)
        )

        # simple mini-batch loop
        idxs = np.arange(N)
        for _ in range(self.epochs):
            np.random.shuffle(idxs)
            for s in range(0, N, self.batch_size):
                bidx = idxs[s : s + self.batch_size]
                xb_seq = X_seq[bidx]
                yb = Y[bidx]
                xb_meta = None if X_meta is None else X_meta[bidx]

                opt.zero_grad()
                pred = self._model(xb_seq, xb_meta)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

    def predict(self, x: ModelInputs) -> ModelOutput:
        if self._model is None:
            # persistence fallback
            y = x.outcome_history[-1, :].astype(float, copy=False)
            return ModelOutput(x.end_date + np.timedelta64(x.horizon, "D"), y)

        self._model.eval()
        with torch.no_grad():
            seq = torch.tensor(
                self._seq_features(x)[None, ...],
                dtype=torch.float32,
                device=self.device,
            )
            meta_np = self._meta_features(x)
            meta_t = (
                None
                if meta_np is None
                else torch.tensor(meta_np, dtype=torch.float32, device=self.device)
            )
            pred = self._model(seq, meta_t).cpu().numpy().reshape(-1)
        return ModelOutput(x.end_date + np.timedelta64(x.horizon, "D"), pred)

    def set_hyperparameters(self, **params: Any) -> None:
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        super().set_hyperparameters(**params)

    def get_hyperparameters(self) -> Dict[str, List[Any]]:
        return {
            "hidden": [64, 128, 256],
            "num_layers": [1, 2],
            "dropout": [0.0, 0.2],
            "lr": [1e-3, 3e-4],
            "weight_decay": [0.0, 1e-4],
            "epochs": [5, 10, 20],
            "batch_size": [64, 128, 256],
            "use_meta": [True, False],
        }
