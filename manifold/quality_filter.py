import json
import argparse
import requests
import re
import time
from typing import Dict, List, Any, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch_market_details(
    market_id: str, base_url: str = "https://api.manifold.markets/v0/market/"
) -> Dict[str, Any]:
    """
    Fetch detailed market information including topics (groupSlugs).

    Args:
        market_id: The ID of the market to fetch
        base_url: The base API endpoint

    Returns:
        Market details as a dictionary
    """
    url = f"{base_url}{market_id}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error fetching market {market_id}: Status {response.status_code}")
        return {}

    return response.json()


def quality_filter(
    input_file: str = "raw_markets.json",
    output_file: str = "quality_filtered_markets.json",
    excluded_users: Optional[List[str]] = None,
    excluded_title_patterns: Optional[List[str]] = None,
    excluded_topics: Optional[List[str]] = None,
    included_topics: Optional[List[str]] = None,
    max_workers: int = 10,
    request_delay: float = 0.2,
    batch_size: int = 100,
) -> List[Dict[str, Any]]:
    """
    Filter Manifold markets based on quality criteria.

    Args:
        input_file: Path to JSON file with market data
        output_file: Path to save filtered market data
        excluded_users: List of usernames to exclude
        excluded_title_patterns: List of regex patterns to exclude from titles
        excluded_topics: List of topic slugs to exclude
        included_topics: List of topic slugs to require (at least one must match)
        max_workers: Maximum number of concurrent API requests
        request_delay: Delay between API requests per worker
        batch_size: Process in batches to show progress

    Returns:
        List of filtered market objects
    """
    # Load market data
    with open(input_file, "r") as f:
        markets = json.load(f)

    print(f"Loaded {len(markets)} markets from {input_file}")

    # Convert excluded users to lowercase set for faster lookups
    excluded_users_set = {user.lower() for user in (excluded_users or [])}

    # Compile regex patterns for title filtering
    title_patterns = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (excluded_title_patterns or [])
    ]

    # Convert topics to sets for faster lookups
    excluded_topics_set = set(excluded_topics or [])
    included_topics_set = set(included_topics or [])

    # Initial filtering (without API calls)
    initial_filtered = []
    for market in markets:
        # Skip if creator is in excluded users list
        if market.get("creatorUsername", "").lower() in excluded_users_set:
            continue

        # Skip if title matches any excluded pattern
        question = market.get("question", "")
        if any(pattern.search(question) for pattern in title_patterns):
            continue

        # Keep for further processing
        initial_filtered.append(market)

    print(f"After initial filtering: {len(initial_filtered)} markets")

    # No need for API calls if no topic filtering is required
    if not (excluded_topics_set or included_topics_set):
        with open(output_file, "w") as f:
            json.dump(initial_filtered, f, indent=2)
        print(
            f"No topic filtering needed. Saved {len(initial_filtered)} markets to {output_file}"
        )
        return initial_filtered

    # Process remaining markets in batches with concurrent API calls
    final_filtered = []
    total_batches = (len(initial_filtered) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(initial_filtered))
        batch = initial_filtered[start_idx:end_idx]

        print(
            f"Processing batch {batch_idx + 1}/{total_batches} ({start_idx} to {end_idx})"
        )

        # Fetch market details concurrently
        market_details = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_market = {
                executor.submit(fetch_market_details, market["id"]): market["id"]
                for market in batch
            }

            # Process results as they complete
            for i, future in enumerate(as_completed(future_to_market)):
                market_id = future_to_market[future]
                try:
                    result = future.result()
                    if result:
                        market_details[market_id] = result

                    # Add delay to avoid rate limiting
                    time.sleep(request_delay)

                    # Show progress
                    if (i + 1) % 10 == 0 or i + 1 == len(batch):
                        print(f"  Fetched {i + 1}/{len(batch)} market details")

                except Exception as e:
                    print(f"Error processing market {market_id}: {str(e)}")

        # Apply topic filtering
        for market in batch:
            market_id = market["id"]
            details = market_details.get(market_id, {})

            # Skip if market details couldn't be fetched
            if not details:
                continue

            # Get topics (groupSlugs)
            topics = set(details.get("groupSlugs", []))

            # Skip if any excluded topic is present
            if excluded_topics_set and topics.intersection(excluded_topics_set):
                continue

            # Skip if no included topic is present (when included_topics is specified)
            if included_topics_set and not topics.intersection(included_topics_set):
                continue

            # Market passed all filters - add full details to final list
            final_filtered.append(details)

    # Save filtered markets to output file
    with open(output_file, "w") as f:
        json.dump(final_filtered, f, indent=2)

    print(f"Filtered to {len(final_filtered)} markets. Saved to {output_file}")

    return final_filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter Manifold markets by quality criteria"
    )
    parser.add_argument(
        "--input", default="filtered_markets.json", help="Input JSON file"
    )
    parser.add_argument(
        "--output", default="quality_filtered_markets.json", help="Output JSON file"
    )
    parser.add_argument("--excluded-users", nargs="+", help="Usernames to exclude")
    parser.add_argument(
        "--excluded-title-patterns",
        nargs="+",
        help="Regex patterns to exclude from titles",
    )
    parser.add_argument("--excluded-topics", nargs="+", help="Topic slugs to exclude")
    parser.add_argument(
        "--included-topics",
        nargs="+",
        help="Topic slugs to require (at least one must match)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=10, help="Maximum concurrent API requests"
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.2,
        help="Delay between API requests (seconds)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Process in batches of this size"
    )

    args = parser.parse_args()

    quality_filter(
        input_file=args.input,
        output_file=args.output,
        excluded_users=args.excluded_users,
        excluded_title_patterns=args.excluded_title_patterns,
        excluded_topics=args.excluded_topics,
        included_topics=args.included_topics,
        max_workers=args.max_workers,
        request_delay=args.request_delay,
        batch_size=args.batch_size,
    )
