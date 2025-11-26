# analysis/models/linear_decay_ridge.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from scipy import linalg, signal

from analysis.predict import ModelInputs, ModelOutput
from analysis.models.base import BasePredictorModel


class _Scaler:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, X: np.ndarray) -> "_Scaler":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0.0] = 1.0
        return cls(mean, std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


class LinearDecayRidge(BasePredictorModel):
    def __init__(self, **params: Any):
        super().__init__()
        # core hyperparams
        self.l2: float = 1e-3
        self.tau_days: float = 30.0
        self.use_meta: bool = True  # optional

        # filter hyperparams (APPLIED TO TARGETS / OUTPUTS)
        # filter_type: "none" | "ema" | "boxcar"
        self.filter_type: str = "none"
        self.filter_alpha: float = 0.3   # EMA smoothing factor
        self.filter_window: int = 7      # boxcar window size

        super().set_hyperparameters(
            l2=self.l2,
            tau_days=self.tau_days,
            use_meta=self.use_meta,
            filter_type=self.filter_type,
            filter_alpha=self.filter_alpha,
            filter_window=self.filter_window,
        )
        if params:
            self.set_hyperparameters(**params)

        self._scaler: Optional[_Scaler] = None
        self._W: Optional[np.ndarray] = None
        self._out_dim: Optional[int] = None

    def name(self) -> str:
        return "linear_decay_ridge"


    def _filter_targets_1geo(self, Y_geo: np.ndarray) -> np.ndarray:
        """
        Apply 1D temporal filter along time axis (axis=0) to the per-geo
        target matrix Y_geo of shape (T, O).
        """
        if Y_geo.size == 0:
            return Y_geo

        ftype = self.filter_type.lower()
        if ftype == "none":
            return Y_geo

        if ftype == "ema":
            alpha = float(self.filter_alpha)
            alpha = max(1e-4, min(alpha, 1.0))
            # y[t] = alpha * x[t] + (1 - alpha) * y[t-1]
            b = [alpha]
            a = [1.0, -(1.0 - alpha)]
            return signal.lfilter(b, a, Y_geo, axis=0)

        if ftype == "boxcar":
            win = max(int(self.filter_window), 1)
            win = min(win, Y_geo.shape[0])  # don't exceed length
            b = np.ones(win, dtype=float) / float(win)
            a = [1.0]
            return signal.lfilter(b, a, Y_geo, axis=0)

        # unknown filter type -> no-op
        return Y_geo


    def _featurize_one(self, x: ModelInputs) -> np.ndarray:
        feats: List[np.ndarray] = []

        if self.use_meta:
            feats.append(np.asarray(x.meta, dtype=float).ravel())

        feats.append(np.asarray(x.outcome_history, dtype=float).ravel())
        feats.append(np.asarray(x.policy_history, dtype=float).ravel())

        return np.concatenate(feats, axis=0)

    def _stack_XY_dates(
        self, batch: List[Tuple[ModelInputs, ModelOutput]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        X = np.stack([self._featurize_one(x) for (x, _) in batch], axis=0)
        Y = np.stack([y.outcomes for (_, y) in batch], axis=0).astype(float)
        dates = np.array([x.end_date.toordinal() for (x, _) in batch], dtype=float)
        geo_ids = [x.geo_id for (x, _) in batch]
        return X, Y, dates, geo_ids

    def fit_batch(self, batch: List[Tuple[ModelInputs, ModelOutput]]) -> None:
        if not batch:
            return

        # stack
        X, Y, dates_ord, geo_ids = self._stack_XY_dates(batch)
        self._out_dim = Y.shape[1]

        # temporal filtering on geos
        if self.filter_type.lower() != "none":
            Y_filt = np.empty_like(Y)
            # group indices by geo_id
            idx_by_geo: Dict[str, List[int]] = {}
            for i, gid in enumerate(geo_ids):
                idx_by_geo.setdefault(str(gid), []).append(i)

            for gid, idxs in idx_by_geo.items():
                idxs_arr = np.array(idxs, dtype=int)
                # sort by time within this geo
                order = np.argsort(dates_ord[idxs_arr])
                rev_order = np.empty_like(order)
                rev_order[order] = np.arange(len(order))

                y_geo = Y[idxs_arr][order]            # (T, O)
                y_geo_f = self._filter_targets_1geo(y_geo)
                y_geo_f = y_geo_f[rev_order]          # back to original order
                Y_filt[idxs_arr] = y_geo_f

            Y = Y_filt

        self._scaler = _Scaler.fit(X)
        Z = self._scaler.transform(X)

        ones = np.ones((Z.shape[0], 1), dtype=Z.dtype)
        Zb = np.concatenate([ones, Z], axis=1)

        t_ref = np.max(dates_ord)
        deltas = t_ref - dates_ord
        tau = max(float(self.tau_days), 1e-6)
        w = np.exp(-deltas / tau)  # (N,)

        Wdiag = np.diag(w)
        R = np.eye(Zb.shape[1], dtype=Zb.dtype)
        R[0, 0] = 0.0  # don't regularize bias

        XtWX = Zb.T @ Wdiag @ Zb + float(self.l2) * R
        XtWY = Zb.T @ Wdiag @ Y

        self._W = linalg.solve(XtWX, XtWY, assume_a="sym")

    def predict(self, x: ModelInputs) -> ModelOutput:
        if self._W is None or self._scaler is None or self._out_dim is None:
            # fall back to persistence if not trained
            y = x.outcome_history[-1, :].astype(float, copy=False)
            return ModelOutput(
                x.end_date + np.timedelta64(x.horizon, "D"),
                y,
            )

        f = self._featurize_one(x)[None, :]
        z = self._scaler.transform(f)
        zb = np.concatenate([np.ones((1, 1), dtype=z.dtype), z], axis=1)
        y = (zb @ self._W).ravel()
        return ModelOutput(
            x.end_date + np.timedelta64(x.horizon, "D"),
            y,
        )

    def set_hyperparameters(self, **params: Any) -> None:
        if "l2" in params:
            self.l2 = float(params["l2"])
        if "tau_days" in params:
            self.tau_days = float(params["tau_days"])
        if "use_meta" in params:
            self.use_meta = bool(params["use_meta"])
        if "filter_type" in params:
            self.filter_type = str(params["filter_type"])
        if "filter_alpha" in params:
            self.filter_alpha = float(params["filter_alpha"])
        if "filter_window" in params:
            self.filter_window = int(params["filter_window"])

        super().set_hyperparameters(**params)

    def get_hyperparameters(self) -> Dict[str, List[Any]]:
        return {
            "l2": [1e-5, 1e-4, 1e-3, 1e-2],
            "tau_days": [7.0, 14.0, 30.0, 60.0],
            "use_meta": [True, False],
            "filter_type": ["none", "ema", "boxcar"],
            "filter_alpha": [0.2, 0.5, 0.8],  # EMA
            "filter_window": [3, 7, 14],      # boxcar
        }
