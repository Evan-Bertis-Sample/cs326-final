from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Union, Tuple, Dict, Any, Iterable
import itertools
import hashlib
import numpy as np
import pandas as pd

from analysis.config import AnalysisConfig
from analysis.oxcgrt_data import OxCGRTData, GeoID
from analysis.predict import (
    OutcomePredictor,
    ModelIOPairBuilder,
    PredictorModel,
    ModelPerformanceMetrics,
    ModelInputs,
    ModelOutput,
)
from analysis.cache import Cache
from analysis.io import banner

# Models
from analysis.models.persistence import PersistenceBaseline
from analysis.models.linear_ridge import LinearWindowRegressor


Pair = Tuple[ModelInputs, ModelOutput]

@dataclass(frozen=True)
class ModelTrainingPairSet:
    training: List[Pair]
    testing: List[Pair]
    validation: Optional[List[Pair]]


def _param_grid(space: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    if not space:
        yield {}
        return
    keys = list(space.keys())
    for vals in itertools.product(*[space[k] for k in keys]):
        yield dict(zip(keys, vals))


def _get_cluster_from_path(cluster_file: Union[str, Path]) -> List[GeoID]:
    p = Path(cluster_file)
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    names = [ln for ln in lines if ln and not ln.startswith("#")]
    return GeoID.from_strings(names, unique=True)

@banner
def build_training_pairs(
    cluster_file: Union[str, Path],
    window: int,
    horizon: int,
    max_per_geo: Optional[int],
) -> ModelTrainingPairSet:
    all_data = OxCGRTData(AnalysisConfig.paths.data)

    # Filter to the cluster
    geos = _get_cluster_from_path(cluster_file)
    filtered = all_data.filter(geo_ids=[str(g) for g in geos])

    # Split by region so each split retains complete time-series per geo
    splits = filtered.split_by_region(ratio=(0.8, 0.1, 0.1), seed=1)

    builder = ModelIOPairBuilder(
        window_size=window,
        horizon=horizon,
        max_per_geo=max_per_geo,
        policy_missing="ffill_then_zero",
        outcome_missing="ffill",
        verbose=True,
    )

    train_pairs = builder.get_pairs(OxCGRTData(splits.training))
    test_pairs = builder.get_pairs(OxCGRTData(splits.testing))
    val_pairs = (
        builder.get_pairs(OxCGRTData(splits.validation))
        if splits.validation is not None
        else None
    )

    return ModelTrainingPairSet(
        training=train_pairs, testing=test_pairs, validation=val_pairs
    )


def _train_model(
    pairset: ModelTrainingPairSet,
    model: PredictorModel,
    *,
    hyperparams: Optional[Dict[str, Any]] = None,
    use_val: bool = False,
) -> PredictorModel:

    if hyperparams:
        model.set_hyperparameters(**hyperparams)

    train_pairs = (
        pairset.training
        if not use_val or pairset.validation is None
        else (pairset.training + pairset.validation)
    )

    # Actually train
    model.fit_batch(train_pairs)
    return model


def _evaluate_model(
    pairset: ModelTrainingPairSet,
    model: PredictorModel,
    *,
    split: str = "validation",  # "validation" | "testing"
) -> ModelPerformanceMetrics:
    if split == "validation":
        eval_pairs = (
            pairset.validation if pairset.validation is not None else pairset.testing
        )
    elif split == "testing":
        eval_pairs = pairset.testing
    else:
        raise ValueError("split must be 'validation' or 'testing'")

    return OutcomePredictor(model).evaluate(eval_pairs)


@banner
def _search_model_hyperspace(
    pairset: ModelTrainingPairSet,
    model: PredictorModel,
) -> Tuple[Dict[str, Any], ModelPerformanceMetrics]:
    space = model.get_hyperparameters()  # Dict[str, List[Any]]

    # no space -> single eval
    if not space or all(len(v) == 1 for v in space.values()):
        params = {k: v[0] for k, v in (space.items() if space else [])}
        trained = Cache.call(
            _train_model,
            pairset,
            model,
            hyperparams=params,
            use_val=False,
        )  # extra kw to key the cache
        metrics = _evaluate_model(pairset, trained, split="validation")
        return params, metrics

    best_params: Dict[str, Any] = {}
    best_metrics: Optional[ModelPerformanceMetrics] = None

    for params in _param_grid(space):
        trained = Cache.call(
            _train_model,
            pairset,
            model,
            hyperparams=params,
            use_val=False,
        )
        metrics = _evaluate_model(pairset, trained, split="validation")

        if best_metrics is None:
            best_params, best_metrics = params, metrics
        else:
            if metrics.rmse < best_metrics.rmse or (
                np.isfinite(metrics.rmse)
                and np.isclose(metrics.rmse, best_metrics.rmse)
                and metrics.mae < best_metrics.mae
            ):
                best_params, best_metrics = params, metrics

    assert best_metrics is not None
    return best_params, best_metrics

@banner
def handle_models(cluster_file: Union[str, Path], window : int, horizon : int, max_per_geo : int) -> None:
    pairs = Cache.call(build_training_pairs,
        cluster_file=cluster_file,
        window=window,
        horizon=horizon,
        max_per_geo=max_per_geo,
    )

    # Models to compare
    models_to_train: List[PredictorModel] = [
        PersistenceBaseline(),
        LinearWindowRegressor(),
    ]

    results: List[Tuple[str, Dict[str, Any], ModelPerformanceMetrics]] = []

    for m in models_to_train:
        block = f"train_{m.name()}"
        Cache.Begin(block)
        try:
            # Cache-aware search+train per model
            params, metrics = _search_model_hyperspace(pairs, m)
            results.append((m.name(), params, metrics))

            # Final train on train+val, cached under a distinct key
            trained_final = Cache.call(
                _train_model,
                pairs,
                m,
                hyperparams=params,
                use_val=True,
            )
            final_test = _evaluate_model(pairs, trained_final, split="testing")
            print(
                f"[{m.name()}] final test: rmse={final_test.rmse:.4f}  mae={final_test.mae:.4f}  r2={final_test.r2:.4f}"
            )
        finally:
            Cache.End()

    # Quick summary table
    print("\nModel results (validation selection):")
    for name, params, met in results:
        print(
            f"- {name:22s} rmse={met.rmse:.4f}  mae={met.mae:.4f}  r2={met.r2:.4f}  params={params}"
        )

    _graph_model_performance(
        model_names=[n for (n, _, _) in results],
        model_params=[p for (_, p, _) in results],
        model_metrics=[m for (_, _, m) in results],
    )


def _graph_model_performance(
    model_names: List[str],
    model_params: List[Dict[str, Any]],
    model_metrics: List[ModelPerformanceMetrics],
) -> None:
    # TODO
    return
