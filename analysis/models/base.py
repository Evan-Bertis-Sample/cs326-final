from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from analysis.predict import PredictorModel

class BasePredictorModel(PredictorModel):
    def __init__(self):
        self._hyperparams: Dict[str, Any] = {}

    def set_hyperparameters(self, **params: Any) -> None:
        for k, v in params.items():
            self._hyperparams[k] = v

    def get_hyperparameters(self) -> Dict[str, List[Any]]:
        """Override in subclass to define search space."""
        return {}

    def __repr__(self) -> str:
        name = getattr(self, "name", lambda: self.__class__.__name__)()
        if not getattr(self, "_hyperparams", None):
            return f"<{name}>"
        pairs = ", ".join(f"{k}={v}" for k, v in self._hyperparams.items())
        return f"<{name}({pairs})>"