# analysis/cache_joblib.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Mapping
import inspect
import shutil
import enum
import datetime as dt
import json
import os
import hashlib

import numpy as np
import pandas as pd
from joblib import dump, load


def _bind_args(func: Any, *args, **kwargs) -> Dict[str, Any]:
    """Best-effort binding of args/kwargs to a name->value mapping."""
    try:
        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except Exception:
        d: Dict[str, Any] = {f"arg{i}": a for i, a in enumerate(args)}
        d.update(kwargs)
        return d

def _sanitize(s: str) -> str:
    """Make string filesystem-friendly."""
    s = str(s).strip().replace("\\", "/").replace(" ", "_")
    for ch in ["/", ":", "*", "?", "\"", "<", ">", "|"]:
        s = s.replace(ch, "_")
    return s or "none"

def _sha10(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def _shorten(seg: str, maxlen: int = 40) -> str:
    seg = _sanitize(seg)
    if len(seg) <= maxlen:
        return seg
    return f"{seg[: maxlen - 11]}__{_sha10(seg)}"

def _canonicalize_value(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (Path, os.PathLike)):
        return str(Path(v).resolve())
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    if isinstance(v, enum.Enum):
        return v.name
    if isinstance(v, np.ndarray):
        return {"__nd__": True, "shape": list(v.shape), "dtype": str(v.dtype)}
    if isinstance(v, pd.Series):
        return {"__pd_series__": True, "len": int(len(v))}
    if isinstance(v, pd.DataFrame):
        return {"__pd_df__": True, "shape": [int(v.shape[0]), int(v.shape[1])]}
    if isinstance(v, (list, tuple)):
        return [_canonicalize_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _canonicalize_value(v[k]) for k in sorted(v.keys(), key=str)}
    return {"__obj__": type(v).__name__}

def _fingerprint_args(bound_args: Mapping[str, Any]) -> tuple[str, dict]:
    canon = {k: _canonicalize_value(v) for k, v in sorted(bound_args.items())}
    ser = json.dumps(canon, sort_keys=True, separators=(",", ":"))
    return _sha10(ser), canon


@dataclass
class CacheConfig:
    root: Path
    compress: int = 3
    default_verbose: bool = True

    def __post_init__(self):
        self.root = Path(self.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)


class Cache:
    _instance: Optional["Cache"] = None

    def __init__(self, cfg: CacheConfig):
        self.cfg = cfg
        # stack: (block_name, dir_path, verbose)
        self._stack: List[Tuple[str, Path, bool]] = []

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

    # blocks
    @classmethod
    def Begin(cls, block_name: str, *, verbose: Optional[bool] = None) -> None:
        self = cls.instance()
        parent_dir = self._stack[-1][1] if self._stack else self.cfg.root
        # keep block segment short
        block_dir = parent_dir / _shorten(block_name)
        block_dir.mkdir(parents=True, exist_ok=True)
        v = self.cfg.default_verbose if verbose is None else bool(verbose)
        self._stack.append((block_name, block_dir, v))
        if v:
            rel = block_dir.relative_to(self.cfg.root)
            print(f"[Cache] Begin: {block_name}  -> {rel}")

    @classmethod
    def End(cls) -> None:
        self = cls.instance()
        if not self._stack:
            return
        name, p, v = self._stack.pop()
        if v:
            rel = p.relative_to(self.cfg.root)
            print(f"[Cache] End:   {name}  <- {rel}")

    # internals
    def _current_block_dir(self) -> Path:
        return self._stack[-1][1] if self._stack else self.cfg.root

    def _entry_path(self, func: Any, bound: Dict[str, Any]) -> Tuple[Path, Path]:
        base = self._current_block_dir()
        func_name = getattr(func, "__name__", "func")
        func_dir = base / _shorten(func_name)
        func_dir.mkdir(parents=True, exist_ok=True)

        h, canon = _fingerprint_args(bound)
        args_dir = func_dir / f"args__{h}"
        args_dir.mkdir(parents=True, exist_ok=True)

        # sidecar
        args_json = args_dir / "args.json"
        if not args_json.exists():
            try:
                args_json.write_text(json.dumps(canon, indent=2, sort_keys=True), encoding="utf-8")
            except Exception:
                # sidecar is best-effort
                pass

        file_path = args_dir / "result.joblib"
        return args_dir, file_path

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

    @classmethod
    def invalidate_block(
        cls,
        block_name: str,
        *,
        cascade_up: bool = False,
        force: bool = False,
        print_only: bool = False,
    ) -> int:
        self = cls.instance()
        root = self.cfg.root.resolve()
        verbose = self.cfg.default_verbose

        # find matches strictly under root
        matches: List[Path] = []
        for p in root.rglob(block_name):
            if p.is_dir():
                try:
                    _ = p.resolve().relative_to(root)
                except ValueError:
                    continue
                matches.append(p.resolve())

        if not matches:
            if verbose:
                print(f"[Cache] Invalidate: no '{block_name}' directories under {root}")
            return 0

        def _chain_to_top(path: Path) -> List[Path]:
            chain = [path]
            cur = path
            while True:
                parent = cur.parent
                if parent == root:
                    break
                try:
                    _ = parent.relative_to(root)
                except ValueError:
                    break
                chain.append(parent)
                cur = parent
            return chain  # leaf -> ... -> top-under-root

        chains = [_chain_to_top(p) for p in matches]

        # print chains
        print(f"[Cache] Invalidate '{block_name}' — {len(chains)} match(es) under {root}:")
        for ch in chains:
            rels = [c.relative_to(root) for c in ch]
            print("  - " + "  ->  ".join(str(x) for x in rels))

        # determine targets
        if cascade_up:
            targets = {ch[-1] for ch in chains}  # top-most under root
            mode = "cascade-up (delete top-most roots)"
        else:
            targets = {ch[0] for ch in chains}   # matched directories only
            mode = "non-cascading (delete matches only)"

        targets_sorted = sorted(
            targets, key=lambda p: len(p.relative_to(root).parts), reverse=True
        )

        print("\n[Cache] Deletion mode:", mode)
        print("[Cache] Targets:")
        for t in targets_sorted:
            print("   •", t.relative_to(root))

        if print_only:
            print("\n[Cache] print_only=True — no deletion performed.")
            return 0

        if not force:
            resp = input("\nDelete the cache? (y/n) ").strip().lower()
            if resp not in ("y", "yes"):
                print("[Cache] Aborted. No deletion performed.")
                return 0

        removed = 0
        for t in targets_sorted:
            if t.exists():
                shutil.rmtree(t, ignore_errors=True)
                removed += 1
                if verbose:
                    print(f"[Cache] Removed: {t.relative_to(root)}")

        return removed
