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
    require_subdir: bool = True  # safety: root must be inside project, not project root

    def __post_init__(self):
        self.root = Path(self.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

        # SAFETY 1: refuse to use repository root as cache root
        if self.require_subdir:
            # Heuristic: project root has a .git folder and we shouldn't equal it
            git_dir = None
            cur = self.root
            for parent in [cur] + list(cur.parents):
                if (parent / ".git").exists():
                    git_dir = parent
                    break
            if git_dir is not None and self.root == git_dir:
                raise RuntimeError(f"Refusing to use repository root as cache root: {self.root}")

        # SAFETY 2: write a sentinel file; invalidator will never delete above this
        sentinel = self.root / SENTINEL_NAME
        if not sentinel.exists():
            sentinel.write_text("cache root sentinel\n", encoding="utf-8")


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

        # 1) find block directories strictly under cache root
        matches = []
        for p in root.rglob(block_name):
            if p.is_dir():
                try:
                    _ = p.resolve().relative_to(root)
                except ValueError:
                    continue  # skip anything outside root (paranoia)
                matches.append(p.resolve())

        if not matches:
            if verbose:
                print(f"[Cache] Invalidate: no '{block_name}' directories under {root}")
            return 0

        # build chain from each match up to the top-most child of cache root
        def _chain_to_top_under_root(path: Path) -> list[Path]:
            chain = [path]
            cur = path
            while True:
                parent = cur.parent
                # stop when parent is root; 'cur' is then the top-most child under root
                if parent == root:
                    break
                # sanity: if we somehow climbed above root, stop
                try:
                    _ = parent.relative_to(root)
                except ValueError:
                    break
                chain.append(parent)
                cur = parent
            return chain  # [match, ..., top-under-root]

        chains = [_chain_to_top_under_root(p) for p in matches]

        # print chains
        print(f"[Cache] Invalidate '{block_name}' — found {len(chains)} match(es) under {root}:")
        for ch in chains:
            rel = [c.relative_to(root) for c in ch]
            # show from leaf to top, e.g.: a/b/build_pairs -> a/b -> a
            print("  - " + "  ->  ".join(str(x) for x in rel))

        # choose deletion targets
        if cascade_up:
            # delete only the top-most directory for each chain
            targets = {ch[-1] for ch in chains}  # set of Paths
            mode = "cascade-up (delete top-most roots)"
        else:
            # delete each matched directory only
            targets = {ch[0] for ch in chains}
            mode = "non-cascading (delete matches only)"

        # sort deepest-first (safer when overlapping paths exist)
        targets_sorted = sorted(targets, key=lambda p: len(p.relative_to(root).parts), reverse=True)

        print("\n[Cache] Deletion mode:", mode)
        print("[Cache] Targets:")
        for t in targets_sorted:
            print("   •", t.relative_to(root))

        if print_only:
            print("\n[Cache] print_only=True — no deletion performed.")
            return 0

        if not force:
            resp = input("\nDelete the cache? (y/n) ").strip().lower()
            if not (resp == "y" or resp == "yes"):
                print("[Cache] Aborted. No deletion performed.")
                return 0

        # delete
        removed = 0
        for t in targets_sorted:
            if t.exists():
                shutil.rmtree(t, ignore_errors=True)
                removed += 1
                if verbose:
                    print(f"[Cache] Removed: {t.relative_to(root)}")

        return removed
