import argparse
import datetime
import json
from typing import Any, Dict, List, Optional, Set


def filter_markets(
    input_file: str = "raw_markets.json",
    output_file: str = "filtered_markets.json",
    min_volume: float = 0,
    min_unique_bettors: int = 0,
    only_resolved: bool = False,
    allowed_outcome_types: Optional[Set[str]] = None,
    resolved_after: Optional[datetime.datetime] = None,
    closed_after: Optional[datetime.datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Filter Manifold markets based on specified criteria.

    Args:
        input_file: Path to JSON file with raw market data
        output_file: Path to save filtered market data
        min_volume: Minimum trading volume required
        min_unique_bettors: Minimum number of unique bettors required
        only_resolved: If True, only include markets that have been resolved
        allowed_outcome_types: Set of allowed outcome types (e.g., {"BINARY", "MULTIPLE_CHOICE"})
        resolved_after: Only include markets resolved after this datetime
        closed_after: Only include markets that closed after this datetime

    Returns:
        List of filtered market objects
    """
    # Load raw market data
    with open(input_file, "r") as f:
        markets = json.load(f)

    print(f"Loaded {len(markets)} markets from {input_file}")

    # Initialize filtered markets list
    filtered_markets = []

    # Filter markets based on criteria
    for market in markets:
        # Skip if volume is below threshold
        if market.get("volume", 0) < min_volume:
            continue

        # Skip if uniqueBettorCount is below threshold
        if market.get("uniqueBettorCount", 0) < min_unique_bettors:
            continue

        # Skip if we only want resolved markets and this one isn't resolved
        if only_resolved and not market.get("isResolved", False):
            continue

        # Skip if outcomeType is not in allowed set
        if (
            allowed_outcome_types
            and market.get("outcomeType") not in allowed_outcome_types
        ):
            continue

        # Skip if market was resolved before the specified date
        if resolved_after and market.get("resolutionTime"):
            # Convert epoch milliseconds to datetime
            resolution_time = datetime.datetime.fromtimestamp(
                market["resolutionTime"] / 1000.0
            )
            if resolution_time < resolved_after:
                continue

        # Skip if market was closed before the specified date
        if closed_after and market.get("closeTime"):
            # Convert epoch milliseconds to datetime
            close_time = datetime.datetime.fromtimestamp(market["closeTime"] / 1000.0)
            if close_time < closed_after:
                continue

        # Market passed all filters
        filtered_markets.append(market)

    # Save filtered markets to output file
    with open(output_file, "w") as f:
        json.dump(filtered_markets, f, indent=2)

    print(f"Filtered to {len(filtered_markets)} markets. Saved to {output_file}")

    return filtered_markets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter Manifold markets")
    parser.add_argument(
        "--input",
        default="raw_markets.json",
        help="Input JSON file, default is raw_markets.json",
    )
    parser.add_argument(
        "--output",
        default="filtered_markets.json",
        help="Output JSON file, default is filtered_markets.json",
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=0,
        help="Minimum trading volume, default is 0",
    )
    parser.add_argument(
        "--min-unique-bettors",
        type=int,
        default=0,
        help="Minimum number of unique bettors, default is 0",
    )
    parser.add_argument(
        "--only-resolved",
        action="store_true",
        help="Only include markets that have been resolved",
    )
    parser.add_argument(
        "--allowed-outcome-types",
        nargs="+",
        default=["BINARY"],
        help="Allowed outcome types (e.g., BINARY MULTIPLE_CHOICE NUMERIC), default is BINARY",
    )
    parser.add_argument(
        "--resolved-after",
        default=None,
        type=str,
        help="Only include markets resolved after this date (format: YYYY-MM-DD), default is None",
    )
    parser.add_argument(
        "--closed-after",
        default=None,
        type=str,
        help="Only include markets that closed after this date (format: YYYY-MM-DD), default is None",
    )

    args = parser.parse_args()

    # Convert to sets for faster lookups
    allowed_outcome_types = (
        set(args.allowed_outcome_types) if args.allowed_outcome_types else None
    )

    # Parse the resolved_after date if provided
    resolved_after = None
    if args.resolved_after:
        try:
            resolved_after = datetime.datetime.fromisoformat(args.resolved_after)
            print(f"Resolved after: {resolved_after}")
        except ValueError:
            print(f"Invalid date format for --resolved-after: {args.resolved_after}")
            print("Please use YYYY-MM-DD format")
            exit(1)

    # Parse the closed_after date if provided
    closed_after = None
    if args.closed_after:
        try:
            closed_after = datetime.datetime.fromisoformat(args.closed_after)
            print(f"Closed after: {closed_after}")
        except ValueError:
            print(f"Invalid date format for --closed-after: {args.closed_after}")
            print("Please use YYYY-MM-DD format")
            exit(1)

    filter_markets(
        input_file=args.input,
        output_file=args.output,
        min_volume=args.min_volume,
        min_unique_bettors=args.min_unique_bettors,
        only_resolved=args.only_resolved,
        allowed_outcome_types=allowed_outcome_types,
        resolved_after=resolved_after,
        closed_after=closed_after,
    )
