# analysis/cache_joblib.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import inspect
import shutil

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
    if isinstance(v, (str, int, float, bool)) or v is None:
        s = str(v).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
        return s[:64] or "none"
    if isinstance(v, np.ndarray):
        return f"nd_{v.shape}_{str(v.dtype)}"
    if isinstance(v, (pd.Series, pd.DataFrame)):
        cols = getattr(v, "columns", None)
        return f"pd_{len(v)}x{len(cols) if cols is not None else 1}"
    return type(v).__name__ + "_obj"

def _seg(name: str, val: Any) -> str:
    return f"{name}__{_value_fingerprint(val)}"


@dataclass
class CacheConfig:
    root: Path
    compress: int = 3
    default_verbose: bool = True

    def __post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)


class Cache:
    _instance: Optional["Cache"] = None

    def __init__(self, cfg: CacheConfig):
        self.cfg = cfg
        # stack of active block directories (absolute Paths) + verbosity per block
        self._stack: List[Tuple[str, Path, bool]] = []  # (block_name, dir_path, verbose)

    # lifecycle
    @classmethod
    def init(cls, cfg: CacheConfig) -> "Cache":
        cls._instance = Cache(cfg)
        return cls._instance

    @classmethod
    def instance(cls) -> "Cache":
        if cls._instance is None:
            raise RuntimeError("Cache not initialized. Call Cache.init(CacheConfig(...)) first.")
        return cls._instance

    # Begin/End blocks
    @classmethod
    def Begin(cls, block_name: str, *, verbose: Optional[bool] = None) -> None:
        """
        Open a new block directory nested under the current block (or root if none).
        Example nesting: <root>/training/build_pairs/...
        """
        self = cls.instance()
        parent_dir = self._stack[-1][1] if self._stack else self.cfg.root
        block_dir = parent_dir / block_name
        block_dir.mkdir(parents=True, exist_ok=True)
        v = self.cfg.default_verbose if verbose is None else bool(verbose)
        self._stack.append((block_name, block_dir, v))
        if v:
            print(f"[Cache] Begin: {block_name}  -> {block_dir.relative_to(self.cfg.root)}")

    @classmethod
    def End(cls) -> None:
        self = cls.instance()
        if not self._stack:
            return
        name, p, v = self._stack.pop()
        if v:
            print(f"[Cache] End:   {name}  <- {p.relative_to(self.cfg.root)}")

    # path building for a function call within the current block
    def _current_block_dir(self) -> Path:
        if not self._stack:
            return self.cfg.root
        return self._stack[-1][1]

    def _entry_path(self, func: Any, bound: Dict[str, Any]) -> Tuple[Path, Path]:
        """
        Layout:
          <root>/<block1>/<block2>/.../<func_name>/<arg1__v>/<arg2__v>/.../<last_arg__v>.joblib
        If no args: <func_name>/result.joblib
        """
        base = self._current_block_dir()
        func_name = getattr(func, "__name__", "func")
        func_dir = base / func_name
        func_dir.mkdir(parents=True, exist_ok=True)

        ordered = list(bound.keys()) 
        segments = [_seg(n, bound[n]) for n in ordered]

        if not segments:
            dir_path = func_dir
            file_path = dir_path / "result.joblib"
            return dir_path, file_path

        *dir_segs, last = segments
        dir_path = func_dir
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

        verbose = self._stack[-1][2] if self._stack else self.cfg.default_verbose
        if fp.exists():
            out = load(fp)
            if verbose:
                print(f"[Cache] Hit:  {fp.relative_to(self.cfg.root)}")
            return out

        out = func(*args, **kwargs)
        dump(out, fp, compress=self.cfg.compress)
        if verbose:
            print(f"[Cache] Save: {fp.relative_to(self.cfg.root)}")
        return out

    @classmethod
    def exists(cls, func: Any, *args, **kwargs) -> bool:
        self = cls.instance()
        bound = _bind_args(func, *args, **kwargs)
        _, fp = self._entry_path(func, bound)
        return fp.exists()

    # invalidation via directory structure
    @classmethod
    def invalidate_block(cls, block_name: str, *, cascade_up: bool = True) -> int:
        """
        Delete all directories named `block_name` anywhere under the cache root.
        If cascade_up=True, also delete all ancestor directories up to the root for each match.
        Returns the number of directories removed.
        """
        self = cls.instance()
        removed = 0
        # find any path ".../<block_name>"
        matches = list(self.cfg.root.rglob(block_name))
        # Keep only directories, and only those living under root (avoid partial string collisions)
        matches = [p for p in matches if p.is_dir() and self.cfg.root in p.parents]

        # Sort deeper paths first to delete children before parents safely
        matches.sort(key=lambda p: len(p.relative_to(self.cfg.root).parts), reverse=True)

        seen: set[Path] = set()
        for p in matches:
            # delete matched dir
            if p not in seen and p.exists():
                shutil.rmtree(p, ignore_errors=True)
                seen.add(p)
                removed += 1
                if self.cfg.default_verbose:
                    print(f"[Cache] Invalidate: removed {p.relative_to(self.cfg.root)}")

            if cascade_up:
                # walk up to root, delete each ancestor dir (but not the root)
                for ancestor in reversed(p.parents):
                    if ancestor == self.cfg.root:
                        break
                    if ancestor not in seen and ancestor.exists():
                        shutil.rmtree(ancestor, ignore_errors=True)
                        seen.add(ancestor)
                        removed += 1
                        if self.cfg.default_verbose:
                            print(f"[Cache] Invalidate (cascade): removed {ancestor.relative_to(self.cfg.root)}")

        return removed
