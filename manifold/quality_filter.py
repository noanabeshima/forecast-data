import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm


class RateLimiter:
    """Simple rate limiter using token bucket algorithm"""

    def __init__(self, max_requests_per_minute: int):
        self.max_requests_per_minute = max_requests_per_minute
        self.tokens = max_requests_per_minute
        self.last_refill_time = time.time()
        self.lock = Lock()

    def acquire(self):
        """Acquire a token, blocking if necessary"""
        with self.lock:
            self._refill_tokens()

            if self.tokens < 1:
                # Calculate sleep time needed for at least one token
                sleep_time = 60 / self.max_requests_per_minute
                time.sleep(sleep_time)
                self._refill_tokens()

            self.tokens -= 1

    def _refill_tokens(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill_time

        # Calculate how many tokens to add based on elapsed time
        new_tokens = (elapsed / 60) * self.max_requests_per_minute

        if new_tokens > 0:
            self.tokens = min(self.tokens + new_tokens, self.max_requests_per_minute)
            self.last_refill_time = now


def fetch_market_details(
    market_id: str,
    rate_limiter: Optional[RateLimiter] = None,
    base_url: str = "https://api.manifold.markets/v0/market/",
) -> Dict[str, Any]:
    """
    Fetch detailed market information including topics (groupSlugs).

    Args:
        market_id: The ID of the market to fetch
        rate_limiter: Optional rate limiter to control API request rate
        base_url: The base API endpoint

    Returns:
        Market details as a dictionary
    """
    # Acquire token from rate limiter if provided
    if rate_limiter:
        rate_limiter.acquire()

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
    max_requests_per_minute: int = 300,  # Default to 300 requests per minute
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
        max_requests_per_minute: Maximum API requests per minute to avoid rate limiting

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
    print("Performing initial filtering...")
    for market in tqdm(markets, desc="Initial filtering"):
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

    # Process all markets with concurrent API calls
    final_filtered = []

    # Create rate limiter for API requests
    rate_limiter = RateLimiter(max_requests_per_minute)

    # Fetch market details concurrently
    market_details = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_market = {
            executor.submit(fetch_market_details, market["id"], rate_limiter): market[
                "id"
            ]
            for market in initial_filtered
        }

        # Process results as they complete
        api_progress = tqdm(
            as_completed(future_to_market),
            total=len(future_to_market),
            desc="Fetching market details",
        )
        for future in api_progress:
            market_id = future_to_market[future]
            try:
                result = future.result()
                if result:
                    market_details[market_id] = result
            except Exception as e:
                api_progress.write(f"Error processing market {market_id}: {str(e)}")

    # Apply topic filtering
    topic_progress = tqdm(initial_filtered, desc="Applying topic filters")
    for market in topic_progress:
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
        "--max-requests-per-minute",
        type=int,
        default=400,
        help="Maximum API requests per minute to avoid rate limiting",
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
        max_requests_per_minute=args.max_requests_per_minute,
    )
