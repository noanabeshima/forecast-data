import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def analyze_market_data(
    input_file: str = "raw_markets.json",
    output_dir: str = "analysis_output",
    bins: int = 50,
    log_scale: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Analyze Manifold market data and generate histograms for volume and unique bettors.

    Args:
        input_file: Path to JSON file with market data
        output_dir: Directory to save histogram images
        bins: Number of bins for histograms
        log_scale: Whether to use log scale for x-axis

    Returns:
        Tuple of statistics for volume and unique bettors
    """
    # Load market data
    with open(input_file, "r") as f:
        markets = json.load(f)

    print(f"Loaded {len(markets)} markets from {input_file}")

    # Extract volume and uniqueBettorCount
    volumes = [market.get("volume", 0) for market in markets]
    unique_bettors = [market.get("uniqueBettorCount", 0) for market in markets]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate histograms
    volume_stats = generate_histogram(
        data=volumes,
        title="Market Volume Distribution",
        xlabel="Volume",
        ylabel="Number of Markets",
        output_path=os.path.join(output_dir, "volume_histogram.png"),
        bins=bins,
        log_scale=log_scale,
    )

    bettor_stats = generate_histogram(
        data=unique_bettors,
        title="Unique Bettors Distribution",
        xlabel="Number of Unique Bettors",
        ylabel="Number of Markets",
        output_path=os.path.join(output_dir, "unique_bettors_histogram.png"),
        bins=bins,
        log_scale=log_scale,
    )

    # Output summary statistics to console
    print_stats("Volume", volume_stats)
    print_stats("Unique Bettors", bettor_stats)

    # Output recommendation for filter thresholds
    print("\nRecommended filter thresholds:")
    print(f"--min-volume {int(volume_stats['median'])}")
    print(f"--min-unique-bettors {int(bettor_stats['median'])}")

    return volume_stats, bettor_stats


def generate_histogram(
    data: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str,
    bins: int = 50,
    log_scale: bool = True,
) -> Dict[str, float]:
    """
    Generate and save a histogram, and return statistics.

    Args:
        data: List of values to plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        output_path: Path to save the histogram image
        bins: Number of bins
        log_scale: Whether to use log scale for x-axis

    Returns:
        Dictionary of statistics (min, max, mean, median, etc.)
    """
    # Convert to numpy array for analysis
    data_array = np.array(data)

    # Calculate statistics
    stats = {
        "min": float(np.min(data_array)),
        "max": float(np.max(data_array)),
        "mean": float(np.mean(data_array)),
        "median": float(np.median(data_array)),
        "std": float(np.std(data_array)),
        "25th_percentile": float(np.percentile(data_array, 25)),
        "75th_percentile": float(np.percentile(data_array, 75)),
        "90th_percentile": float(np.percentile(data_array, 90)),
        "95th_percentile": float(np.percentile(data_array, 95)),
        "99th_percentile": float(np.percentile(data_array, 99)),
    }

    # Create figure
    plt.figure(figsize=(10, 6))

    # Generate histogram
    plt.hist(data_array, bins=bins, alpha=0.75, color="skyblue")

    # Use log scale if specified
    if log_scale and np.min(data_array[data_array > 0]) > 0:
        plt.xscale("log")
        plt.xlabel(f"{xlabel} (log scale)")
    else:
        plt.xlabel(xlabel)

    plt.ylabel(ylabel)
    plt.title(title)

    # Add vertical lines for key statistics
    plt.axvline(
        stats["median"],
        color="r",
        linestyle="--",
        label=f'Median: {stats["median"]:.2f}',
    )
    plt.axvline(
        stats["mean"], color="g", linestyle="-.", label=f'Mean: {stats["mean"]:.2f}'
    )
    plt.axvline(
        stats["75th_percentile"],
        color="orange",
        linestyle=":",
        label=f'75th %: {stats["75th_percentile"]:.2f}',
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path)
    print(f"Saved histogram to {output_path}")

    # Generate additional quantile histogram
    generate_quantile_histogram(
        data=data_array,
        title=f"{title} by Quantile",
        xlabel=xlabel,
        ylabel="Count",
        output_path=output_path.replace(".png", "_quantiles.png"),
    )

    return stats


def generate_quantile_histogram(
    data: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str,
    num_quantiles: int = 10,
) -> None:
    """
    Generate a histogram showing distribution by quantiles.
    This helps visualize the full distribution including outliers.
    """
    plt.figure(figsize=(12, 6))

    # Create quantile-based bins
    quantiles = [np.percentile(data, q) for q in np.linspace(0, 100, num_quantiles + 1)]

    # Count items in each quantile
    counts = []
    labels = []

    for i in range(len(quantiles) - 1):
        start, end = quantiles[i], quantiles[i + 1]
        count = np.sum((data >= start) & (data <= end))
        counts.append(count)

        # Create label showing range
        if i == len(quantiles) - 2:
            label = f"{start:.1f}-{end:.1f}"
        else:
            label = f"{start:.1f}-{end:.1f}"

        labels.append(label)

    # Plot bar chart
    plt.bar(range(len(counts)), counts, align="center")
    plt.xticks(range(len(counts)), labels, rotation=45, ha="right")
    plt.xlabel(f"{xlabel} Quantiles")
    plt.ylabel(ylabel)
    plt.title(title)

    # Add value annotations
    for i, count in enumerate(counts):
        plt.text(i, count + (max(counts) * 0.02), str(count), ha="center")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved quantile histogram to {output_path}")


def print_stats(name: str, stats: Dict[str, float]) -> None:
    """Print statistics in a formatted way."""
    print(f"\n{name} Statistics:")
    print(f"  Min: {stats['min']:.2f}")
    print(f"  Max: {stats['max']:.2f}")
    print(f"  Mean: {stats['mean']:.2f}")
    print(f"  Median: {stats['median']:.2f}")
    print(f"  Standard Deviation: {stats['std']:.2f}")
    print(f"  25th Percentile: {stats['25th_percentile']:.2f}")
    print(f"  75th Percentile: {stats['75th_percentile']:.2f}")
    print(f"  90th Percentile: {stats['90th_percentile']:.2f}")
    print(f"  95th Percentile: {stats['95th_percentile']:.2f}")
    print(f"  99th Percentile: {stats['99th_percentile']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Manifold markets data")
    parser.add_argument("--input", default="raw_markets.json", help="Input JSON file")
    parser.add_argument(
        "--output-dir",
        default="analysis_output",
        help="Output directory for visualizations",
    )
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins")
    parser.add_argument("--no-log-scale", action="store_true", help="Disable log scale")

    args = parser.parse_args()

    analyze_market_data(
        input_file=args.input,
        output_dir=args.output_dir,
        bins=args.bins,
        log_scale=not args.no_log_scale,
    )
