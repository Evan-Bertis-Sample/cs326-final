import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


# Type aliases
MetricPoint = Tuple[float, float, float]  # (window_size, geo_max, value)


def find_results_files(root: str) -> List[str]:
    results_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "results.json" in filenames:
            parts = dirpath.split(os.sep)
            if any(p.startswith("window_") for p in parts) and any(
                p.startswith("geo_max_") for p in parts
            ):
                results_files.append(os.path.join(dirpath, "results.json"))
    return results_files


def parse_config_from_path(path: str, root: str) -> Tuple[str, str]:
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    if len(parts) < 5:
        raise ValueError(f"Unexpected path layout for {path}")
    model_type = parts[0]
    cluster_name = parts[1]
    return model_type, cluster_name


def collect_metrics(
    root: str,
    split_name: str = "testing",
) -> Dict[str, Dict[str, Dict[str, List[MetricPoint]]]]:
    metrics_by_model: Dict[str, Dict[str, Dict[str, List[MetricPoint]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    results_files = find_results_files(root)
    if not results_files:
        print(f"No results.json files found under {root}")
        return metrics_by_model

    print(f"Found {len(results_files)} results.json files.")

    for fp in results_files:
        try:
            model_type, cluster_name = parse_config_from_path(
                os.path.dirname(fp), root
            )
        except ValueError as e:
            print(f"Skipping {fp}: {e}")
            continue

        try:
            with open(fp, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load {fp}: {e}")
            continue

        # Prefer top-level window_size / geo_max, fall back to hyperparameters
        window_size = data.get("window_size")
        geo_max = data.get("geo_max")

        if window_size is None or geo_max is None:
            hp = data.get("hyperparameters", {})
            if window_size is None:
                window_size = hp.get("window_size")
            if geo_max is None:
                geo_max = hp.get("geo_max")

        if window_size is None or geo_max is None:
            print(f"Skipping {fp}: missing window_size or geo_max")
            continue

        splits = data.get("splits", {})
        if split_name not in splits:
            print(f"Skipping {fp}: split '{split_name}' not found")
            continue

        split_metrics = splits[split_name]
        for metric_name in ["mae", "mse", "rmse", "mape"]:
            if metric_name not in split_metrics:
                print(
                    f"Warning: metric '{metric_name}' not found in {fp} for split {split_name}"
                )
                continue

            value = split_metrics[metric_name]
            metrics_by_model[model_type][cluster_name][metric_name].append(
                (float(window_size), float(geo_max), float(value))
            )

    return metrics_by_model


def plot_surface_with_min(
    points: List[MetricPoint],
    model_type: str,
    cluster_name: str,
    metric_name: str,
    split_name: str,
    output_path: str,
) -> None:
    if not points:
        print(
            f"No points to plot for {model_type} / {cluster_name} / {metric_name}, skipping."
        )
        return

    if len(points) < 3:
        # Need at least 3 points for a triangulated surface; fall back to scatter
        print(
            f"Not enough points for surface for {model_type} / {cluster_name} / {metric_name}, using scatter."
        )

    xs = np.array([p[0] for p in points], dtype=float)  # window_size
    ys = np.array([p[1] for p in points], dtype=float)  # geo_max
    zs = np.array([p[2] for p in points], dtype=float)  # metric value

    # Find minimum configuration
    min_idx = np.argmin(zs)
    x_min, y_min, z_min = xs[min_idx], ys[min_idx], zs[min_idx]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    if len(points) >= 3:
        # Triangulated surface
        surf = ax.plot_trisurf(xs, ys, zs, cmap="viridis", alpha=0.8)
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label=metric_name)
    else:
        # Fallback scatter
        sc = ax.scatter(xs, ys, zs, c=zs)
        fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label=metric_name)

    # Highlight minimum configuration
    ax.scatter(
        [x_min],
        [y_min],
        [z_min],
        color="red",
        s=60,
        marker="o",
        depthshade=False,
        label="Minimum",
    )

    # Optional: project minimum point down to x-y plane for easier reading
    ax.scatter(
        [x_min],
        [y_min],
        [ax.get_zlim()[0]],
        color="red",
        s=30,
        marker="x",
        depthshade=False,
    )

    ax.text(
        x_min,
        y_min,
        z_min,
        f"min\nw={x_min:g}\ngeo={y_min:g}\n{metric_name}={z_min:.3g}",
        fontsize=8,
        color="red",
    )

    ax.set_xlabel("Window Size")
    ax.set_ylabel("Geo Max")
    ax.set_zlabel(f"{metric_name.upper()} ({split_name})")

    ax.set_title(f"{model_type} | {cluster_name} | {metric_name.upper()} ({split_name})")

    # Legend for the min marker
    ax.legend(loc="upper right")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

    print(
        f"Saved: {output_path} (min at window={x_min:g}, geo_max={y_min:g}, {metric_name}={z_min:.4g})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot 3D surface metric graphs over window_size and geo_max."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory containing model_type folders (default: current directory).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="testing",
        choices=["training", "testing", "validation"],
        help="Which split to use for metrics (default: testing).",
    )
    args = parser.parse_args()

    metrics_by_model = collect_metrics(args.root, split_name=args.split)

    if not metrics_by_model:
        print("No metrics collected. Exiting.")
        return

    for model_type, clusters in metrics_by_model.items():
        for cluster_name, metric_dict in clusters.items():
            for metric_name, points in metric_dict.items():
                fname = f"{model_type}_{metric_name}.png"
                output_path = os.path.join(args.root, model_type, cluster_name, fname)
                plot_surface_with_min(
                    points,
                    model_type=model_type,
                    cluster_name=cluster_name,
                    metric_name=metric_name,
                    split_name=args.split,
                    output_path=output_path,
                )


if __name__ == "__main__":
    main()
