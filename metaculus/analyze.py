import argparse
import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def analyze_question_data(
    input_file: str = "raw_questions.json",
    output_dir: str = "analysis_output",
    bins: int = 50,
    log_scale: bool = True,
) -> Dict[str, Any]:
    """
    Analyze Metaculus question data and generate visualizations.

    Args:
        input_file: Path to JSON file with question data
        output_dir: Directory to save visualization images
        bins: Number of bins for histograms
        log_scale: Whether to use log scale for x-axis

    Returns:
        Dictionary of statistics and analysis results
    """
    # Load question data
    with open(input_file, "r") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions from {input_file}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract number of forecasters
    forecaster_counts = []

    for q in questions:
        if "nr_forecasters" in q:
            try:
                forecaster_count = int(q["nr_forecasters"])
                forecaster_counts.append(forecaster_count)
            except (ValueError, TypeError):
                # Skip questions with invalid forecaster count
                continue

    print(f"Found {len(forecaster_counts)} questions with valid forecaster counts")

    # Generate histogram for number of forecasters
    forecaster_stats = generate_histogram(
        data=forecaster_counts,
        title="Number of Forecasters Distribution",
        xlabel="Number of Forecasters",
        ylabel="Number of Questions",
        output_path=os.path.join(output_dir, "forecaster_count_histogram.png"),
        bins=bins,
        log_scale=log_scale,
    )

    # Output summary statistics
    print_stats("Number of Forecasters", forecaster_stats)

    # Return statistics
    return {
        "forecaster_count_stats": forecaster_stats,
        "total_questions": len(questions),
        "questions_with_forecaster_counts": len(forecaster_counts),
    }


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
    parser = argparse.ArgumentParser(description="Analyze Metaculus question data")
    parser.add_argument(
        "--input",
        default="raw_questions.json",
        help="Input JSON file with filtered questions",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_output",
        help="Directory to save analysis outputs",
    )
    parser.add_argument(
        "--bins", type=int, default=50, help="Number of bins for histograms"
    )
    parser.add_argument(
        "--no-log-scale", action="store_true", help="Disable log scale for histograms"
    )

    args = parser.parse_args()

    analyze_question_data(
        input_file=args.input,
        output_dir=args.output_dir,
        bins=args.bins,
        log_scale=not args.no_log_scale,
    )
