from __future__ import annotations
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load

def _bind_args(func: Any, *args, **kwargs) -> Dict[str, Any]:
    try:
        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except Exception:
        d: Dict[str, Any] = {f"arg{i}": a for i, a in enumerate(args)}
        d.update(kwargs)
        return d

def _value_fingerprint(v: Any) -> str:
    """Readable, stable-ish name for folders/files (joblib handles the heavy lifting for content)."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        s = str(v)
        s = s.strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
        return s[:64] or "none"
    if isinstance(v, np.ndarray):
        return f"nd_{v.shape}_{str(v.dtype)}"
    if isinstance(v, (pd.Series, pd.DataFrame)):
        cols = getattr(v, "columns", None)
        return f"pd_{len(v)}x{len(cols) if cols is not None else 1}"
    return (type(v).__name__ + "_obj")

def _seg(name: str, val: Any) -> str:
    return f"{name}__{_value_fingerprint(val)}"

@dataclass
class CacheConfig:
    root: Path
    compress: int = 3        # joblib compression (0..9)
    default_verbose: bool = True

    def __post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

class Cache:
    _instance: Optional["Cache"] = None
    _stack: List[Tuple[str, List[str], bool]]  # (func_name, dir_arg_order, verbose)

    def __init__(self, cfg: CacheConfig):
        self.cfg = cfg
        self._stack = []

    @classmethod
    def init(cls, cfg: CacheConfig) -> "Cache":
        cls._instance = Cache(cfg)
        return cls._instance

    @classmethod
    def instance(cls) -> "Cache":
        if cls._instance is None:
            raise RuntimeError("Cache not initialized. Call Cache.init(CacheConfig(...)) first.")
        return cls._instance

    # Begin/End
    @classmethod
    def Begin(cls, function_name: str, dir_arg_order: Optional[List[str]] = None, verbose: Optional[bool] = None) -> None:
        self = cls.instance()
        order = list(dir_arg_order or [])
        v = self.cfg.default_verbose if verbose is None else bool(verbose)
        self._stack.append((function_name, order, v))
        if v:
            print(f"[Cache] Begin: {function_name} (order={order or 'auto'})")

    @classmethod
    def End(cls) -> None:
        self = cls.instance()
        if not self._stack:
            return
        name, _, v = self._stack.pop()
        if v:
            print(f"[Cache] End: {name}")

    # internal
    def _context(self) -> Tuple[str, List[str], bool]:
        if not self._stack:
            return ("_default", [], self.cfg.default_verbose)
        return self._stack[-1]

    def _entry_path(self, func: Any, bound: Dict[str, Any]) -> Tuple[Path, Path]:
        func_ctx_name, dir_order_ctx, _ = self._context()
        func_name = getattr(func, "__name__", func_ctx_name) or func_ctx_name

        # arg order: context override first, then remaining in signature order
        ordered: List[str] = []
        for n in dir_order_ctx:
            if n in bound:
                ordered.append(n)
        for n in bound.keys():
            if n not in ordered:
                ordered.append(n)

        base = self.cfg.root / func_name
        segments = [_seg(n, bound[n]) for n in ordered]

        if not segments:
            dir_path = base
            file_path = dir_path / "result.joblib"
            dir_path.mkdir(parents=True, exist_ok=True)
            return dir_path, file_path

        *dir_segs, last = segments
        dir_path = base
        for seg in dir_segs:
            dir_path = dir_path / seg
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / f"{last}.joblib"
        return dir_path, file_path

    # public API
    @classmethod
    def call(cls, func: Any, *args, **kwargs) -> Any:
        self = cls.instance()
        bound = _bind_args(func, *args, **kwargs)
        _, fp = self._entry_path(func, bound)

        if fp.exists():
            out = load(fp)
            if self._context()[2]:
                print(f"[Cache] Hit: {fp.relative_to(self.cfg.root)}")
            return out

        out = func(*args, **kwargs)
        dump(out, fp, compress=self.cfg.compress)
        if self._context()[2]:
            print(f"[Cache] Save: {fp.relative_to(self.cfg.root)}")
        return out

    @classmethod
    def exists(cls, func: Any, *args, **kwargs) -> bool:
        self = cls.instance()
        bound = _bind_args(func, *args, **kwargs)
        _, fp = self._entry_path(func, bound)
        return fp.exists()
