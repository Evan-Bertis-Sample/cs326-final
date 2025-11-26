# analysis.py

from __future__ import annotations
from pathlib import Path
import argparse
import json
import hashlib
from typing import Dict, Any, Tuple
from joblib import dump
import os

from analysis.config import AnalysisConfig
from analysis.cache import Cache, CacheConfig
import analysis.procs as procs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--invalidate",
        nargs="*",
        default=[],
        help="Block names to invalidate inside the cache.",
    )
    p.add_argument(
        "--no-cascade-up",
        action="store_true",
        help="Do NOT cascade upward when invalidating blocks.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt when invalidating.",
    )
    p.add_argument(
        "--print-only",
        action="store_true",
        help="Show chains/targets without deleting.",
    )
    p.add_argument(
        "--clusters-dir",
        default="data/clusters",
        help="Directory containing cluster files (one GeoID per line).",
    )
    p.add_argument(
        "--verbose-cache", action="store_true", help="Cache logging enabled."
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Run only a small subset (window_size=14, geo_max=100) for quick debugging.",
    )
    return p.parse_args()


def init_cache(args):
    cache_root = AnalysisConfig.paths.cache
    Cache.init(
        CacheConfig(root=cache_root, compress=3, default_verbose=args.verbose_cache)
    )
    for blk in args.invalidate:
        Cache.invalidate_block(
            blk,
            cascade_up=not args.no_cascade_up,
            force=args.force,
            print_only=args.print_only,
        )
    print(f"Cache initialized at: {cache_root}\n")


def _discover_cluster_files(clusters_dir: Path) -> list[Path]:
    if not clusters_dir.exists():
        return []
    files = [
        p for p in clusters_dir.iterdir() if p.is_file() and not p.name.startswith(".")
    ]
    return sorted(files, key=lambda p: p.name.lower())


def _hp_hash(params: Dict[str, Any]) -> str:
    s = json.dumps(params or {}, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def _serialize_model_once(model, model_dir: Path, params: Dict[str, Any]) -> Path:
    model_name = model.name()
    h = _hp_hash(params)
    out_dir = Path("models") / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"{model_name}_{h}.joblib"
    if not model_path.exists():
        dump(model, model_path, compress=3)

        # save hyperparameters next to it
        (out_dir / f"hyperparameters_{h}.json").write_text(
            json.dumps(params or {}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return model_path


def main():
    args = parse_args()
    init_cache(args)

    clusters_dir = Path(args.clusters_dir)
    cluster_files = _discover_cluster_files(clusters_dir)

    if not cluster_files:
        print(f"No cluster files found in: {clusters_dir}")
        return

    print(
        f"Found {len(cluster_files)} cluster file(s) in {clusters_dir}:\n"
        + "\n".join(f"  - {p.name}" for p in cluster_files)
        + "\n"
    )

    if args.debug:
        window_sizes = [14]
        geo_max_values = [100]
    else:
        window_sizes = range(3, 21, 5)
        geo_max_values = range(100, 599, 100)

    # Best-per-Geo accumulator across all (cluster, window, geo_max) runs
    # geo -> (rmse, mae, model, params)
    best_for_geo: Dict[str, Tuple[float, float, Any, Dict[str, Any]]] = {}

    Cache.Begin("training")
    try:
        for cfile in cluster_files:
            for window_size in window_sizes:
                for geo_max in geo_max_values:
                    cluster_name = cfile.stem
                    block_name = f"train_{cluster_name}_window_size_{window_size}_geo_max_{geo_max}"

                    Cache.Begin(block_name)
                    try:
                        best_model, best_params, best_metrics, geos = (
                            procs.handle_models(
                                cluster_file=cfile,
                                window=window_size,
                                horizon=1,
                                max_per_geo=geo_max,
                            )
                        )
                        # Assign this run's best model to all geos in the cluster if it's an improvement
                        for gid in geos:
                            prev = best_for_geo.get(gid)
                            key = (best_metrics.rmse, best_metrics.mae)
                            if (prev is None) or (key < (prev[0], prev[1])):
                                best_for_geo[gid] = (
                                    best_metrics.rmse,
                                    best_metrics.mae,
                                    best_model,
                                    best_params,
                                )
                    finally:
                        Cache.End()
    finally:
        Cache.End()

    model_file_cache: Dict[Tuple[str, str], Path] = {}  # (model_name, hp_hash) -> path
    final_map: Dict[str, Dict[str, str]] = {}  # GeoID -> {model_name, model_file}

    for gid, (rmse, mae, model, params) in best_for_geo.items():
        model_name = model.name()
        h = _hp_hash(params)
        key = (model_name, h)
        if key not in model_file_cache:
            model_path = _serialize_model_once(
                model, Path("models") / model_name, params
            )
            model_file_cache[key] = model_path
        else:
            model_path = model_file_cache[key]

        final_map[gid] = {
            "model_name": model_name,
            "model_file": str(model_path.as_posix()),
        }

    # Save mapping
    mapping_path = Path("models") / "model_map.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(
        json.dumps(final_map, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Print compact summary
    print("\nFinal GeoID → Model mapping (first 50 entries):")
    for i, (gid, rec) in enumerate(final_map.items()):
        print(f"  {gid:20s} -> {rec['model_name']}  ({rec['model_file']})")

    print(f"\nSaved mapping to {mapping_path}")


if __name__ == "__main__":
    main()
