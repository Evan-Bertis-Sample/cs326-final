from analysis.config import AnalysisConfig
from analysis.oxcgrt_data import OxCGRTData, GeoID

from analysis.predict import OutcomePredictor, ModelIOPairBuilder
from analysis.models.persistence import PersistenceModel
from analysis.cache import Cache, CacheConfig

from pathlib import Path
import argparse

import analysis.procs as procs

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--invalidate", nargs="*", default=[],
                   help="Block names to invalidate (e.g., build_pairs training)")
    p.add_argument("--no-cascade", action="store_true",
                   help="Do not cascade invalidation to parent directories")
    return p.parse_args()

def init_cache(args):
    # init cache
    cache_root = Path(AnalysisConfig.paths.output) / ".cache"
    Cache.init(CacheConfig(root=cache_root, compress=3, default_verbose=True))

    # apply invalidations before any work
    for blk in args.invalidate:
        Cache.invalidate_block(blk, cascade_up=(not args.no_cascade))

    print("Cache initialized!\n")
    

def main():
    args = parse_args()
    init_cache(args)
    # Load dataset
    Cache.Begin("training")

    Cache.Begin("build_pairs_all")
    train_pairs = Cache.call(procs.build_training_pairs, window=14, horizon=1, max_per_geo=1)
    have_same = Cache.exists(procs.build_training_pairs, window=14, horizon=1, cluster="ALL")
    Cache.End()

    Cache.Begin("train")

    Cache.End()


if __name__ == "__main__":
    main()
