from __future__ import annotations
from pathlib import Path
import argparse

from analysis.config import AnalysisConfig
from analysis.cache import Cache, CacheConfig
import analysis.procs as procs

from analysis.oxcgrt_data import OxCGRTData, GeoID


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--invalidate", nargs="*", default=[],
                   help="Block names to invalidate inside the cache.")
    p.add_argument("--no-cascade-up", action="store_true",
                   help="Do NOT cascade upward when invalidating blocks.")
    p.add_argument("--force", action="store_true",
                   help="Skip confirmation prompt when invalidating.")
    p.add_argument("--print-only", action="store_true",
                   help="Show chains/targets without deleting.")
    p.add_argument("--clusters-dir", default="figures/clusters",
                   help="Directory containing cluster files (one GeoID per line).")
    p.add_argument("--verbose-cache", action="store_true",
                   help="Cache logging enabled.")
    return p.parse_args()


def init_cache(args):
    cache_root = Path(AnalysisConfig.paths.output) / ".cache"
    Cache.init(CacheConfig(root=cache_root, compress=3, default_verbose=args.verbose_cache))

    # Apply invalidations before any work
    for blk in args.invalidate:
        Cache.invalidate_block(
            blk,
            cascade_up=not args.no_cascade_up,
            force=args.force,
            print_only=args.print_only,
        )

    print(f"Cache initialized at: {cache_root}\n")


def _discover_cluster_files(clusters_dir: Path) -> list[Path]:
    # accept any file (txt, csv, etc.); skip dirs and hidden files
    if not clusters_dir.exists():
        return []
    files = [p for p in clusters_dir.iterdir()
             if p.is_file() and not p.name.startswith(".")]
    # sort for stable ordering
    return sorted(files, key=lambda p: p.name.lower())


def main():
    args = parse_args()
    init_cache(args)

    # Load dataset and convert GeoIDs to strings
    data = OxCGRTData(AnalysisConfig.paths.data)
    ids = data.geo_ids(unique=True)
    geo_strings = GeoID.to_strings(ids)

    # Choose output path (you can change this as needed)
    out_path = Path("figures/clusters/all.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write one ID per line
    with out_path.open("w", encoding="utf-8") as f:
        for geo in geo_strings:
            f.write(f"{geo}\n")

    print(f"Wrote {len(geo_strings)} GeoIDs to {out_path}")

    clusters_dir = Path(args.clusters_dir)
    cluster_files = _discover_cluster_files(clusters_dir)

    if not cluster_files:
        print(f"No cluster files found in: {clusters_dir}")
        return

    print(f"Found {len(cluster_files)} cluster file(s) in {clusters_dir}:\n"
          + "\n".join(f"  - {p.name}" for p in cluster_files) + "\n")

    # High-level training block for the run
    Cache.Begin("training")
    try:
        for cfile in cluster_files:
            cluster_name = cfile.stem  # file name w/o extension
            block_name = f"train_{cluster_name}"

            Cache.Begin(block_name)
            try:
                # Orchestrated training/eval (internally uses cache for sub-steps)
                Cache.call(procs.handle_models, cluster_file=cfile, window=14, horizon=1, max_per_geo=20)
            finally:
                Cache.End()

    finally:
        Cache.End()

    print("\nAll clusters processed.")


if __name__ == "__main__":
    main()
