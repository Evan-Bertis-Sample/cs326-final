import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for saving to files
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (required for 3D)


# Type alias: (window_size, geo_max, metric_value)
MetricPoint = Tuple[float, float, float]


def find_results_files(root: str) -> List[str]:
    results_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "results.json" in filenames:
            results_files.append(os.path.join(dirpath, "results.json"))
    return results_files


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
            with open(fp, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load {fp}: {e}")
            continue

        model = data.get("model")
        cluster_name = data.get("cluster_name")

        if model is None or cluster_name is None:
            print(f"Skipping {fp}: missing 'model' or 'cluster_name' in JSON.")
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
            print(f"Skipping {fp}: missing window_size or geo_max in JSON.")
            continue

        splits = data.get("splits", {})
        if split_name not in splits:
            print(f"Skipping {fp}: split '{split_name}' not found in JSON.")
            continue

        split_metrics = splits[split_name]
        for metric_name in ["mae", "mse", "rmse", "mape", "r2"]:
            if metric_name not in split_metrics:
                print(
                    f"Warning: metric '{metric_name}' not found in {fp} for split {split_name}"
                )
                continue

            value = split_metrics[metric_name]
            metrics_by_model[model][cluster_name][metric_name].append(
                (float(window_size), float(geo_max), float(value))
            )

    return metrics_by_model


def plot_surface(
    points: List[MetricPoint],
    model: str,
    cluster_name: str,
    metric_name: str,
    split_name: str,
    output_path: str,
) -> None:
    if not points:
        print(
            f"No points to plot for {model} / {cluster_name} / {metric_name}, skipping."
        )
        return

    xs = np.array([p[0] for p in points], dtype=float)  # window_size
    ys = np.array([p[1] for p in points], dtype=float)  # geo_max
    zs = np.array([p[2] for p in points], dtype=float)  # metric value (true scale)

    # Choose critical index: min for normal metrics, max for r2
    if metric_name == "r2":
        crit_idx = np.argmax(zs)
        crit_label = "Maximum"
    else:
        crit_idx = np.argmin(zs)
        crit_label = "Minimum"

    x_crit, y_crit, z_crit = xs[crit_idx], ys[crit_idx], zs[crit_idx]

    # For r2, transform the plot
    zs_plot = zs.copy()
    if metric_name == "r2":
        gamma = 0.2
        zs_plot = zs ** gamma
        z_crit_plot = z_crit ** gamma
    else:
        z_crit_plot = z_crit

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    if len(points) >= 3:
        # Triangulated surface using transformed Z
        surf = ax.plot_trisurf(xs, ys, zs_plot, cmap="viridis", alpha=0.8)
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label=metric_name)
    else:
        # Not enough points for a surface – fallback to scatter
        print(
            f"Not enough points for surface for {model} / {cluster_name} / {metric_name}, using scatter."
        )
        sc = ax.scatter(xs, ys, zs_plot, c=zs_plot)
        fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label=metric_name)


    ax.scatter(
        [x_crit],
        [y_crit],
        [z_crit_plot],
        color="red",
        s=60,
        marker="o",
        depthshade=False,
        label=crit_label,
    )

    # Project down to the base of the Z axis for readability
    z_low = ax.get_zlim()[0]
    ax.scatter(
        [x_crit],
        [y_crit],
        [z_low],
        color="red",
        s=30,
        marker="x",
        depthshade=False,
    )

    ax.text(
        x_crit,
        y_crit,
        z_crit_plot,
        f"{'max' if metric_name == 'r2' else 'min'}\n"
        f"w={x_crit:g}\ngeo={y_crit:g}\n{metric_name}={z_crit:.3g}",
        fontsize=8,
        color="red",
    )

    ax.set_xlabel("Window Size")
    ax.set_ylabel("Geo Max")
    ax.set_zlabel(
        f"{metric_name.upper()} ({split_name})"
        + (" (γ=0.2)" if metric_name == "r2" else "")
    )

    ax.set_title(f"{model} | {cluster_name} | {metric_name.upper()} ({split_name})")
    ax.legend(loc="upper right")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

    print(
        f"Saved: {output_path} ({crit_label.lower()} at window={x_crit:g}, "
        f"geo_max={y_crit:g}, {metric_name}={z_crit:.4g})"
    )



def main():
    parser = argparse.ArgumentParser(
        description="Plot 3D surface metric graphs over window_size and geo_max, using only data from results.json."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory where results.json files live (searched recursively).",
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

    base_dir = os.path.join(args.root, "results")

    for model, clusters in metrics_by_model.items():
        for cluster_name, metric_dict in clusters.items():
            for metric_name, points in metric_dict.items():
                fname = f"{model}_{cluster_name}_{metric_name}.png"
                output_path = os.path.join(base_dir, fname)
                plot_surface(
                    points,
                    model=model,
                    cluster_name=cluster_name,
                    metric_name=metric_name,
                    split_name=args.split,
                    output_path=output_path,
                )


if __name__ == "__main__":
    main()
