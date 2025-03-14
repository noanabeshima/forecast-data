import argparse
import json
import time
from typing import Any, Dict, List, Optional

import requests


def scrape_all_markets(
    base_url: str = "https://api.manifold.markets/v0/markets",
    sort: str = "created-time",
    order: str = "desc",
    user_id: Optional[str] = None,
    group_id: Optional[str] = None,
    batch_size: int = 1000,  # Maximum allowed
    rate_limit_delay: float = 0.5,  # Seconds between requests
) -> List[Dict[str, Any]]:
    """
    Scrape all markets from Manifold Markets API with pagination.

    Args:
        base_url: The API endpoint
        sort: Sorting field ('created-time', 'updated-time', 'last-bet-time', 'last-comment-time')
        order: Sort order ('asc' or 'desc')
        user_id: Filter by creator user ID
        group_id: Filter by group/topic ID
        batch_size: Number of markets to fetch per request (max 1000)
        rate_limit_delay: Delay between API requests to avoid rate limiting

    Returns:
        List of all market objects
    """
    all_markets = []
    params = {"limit": batch_size, "sort": sort, "order": order}

    if user_id:
        params["userId"] = user_id
    if group_id:
        params["groupId"] = group_id

    while True:
        response = requests.get(base_url, params=params)

        if response.status_code != 200:
            raise Exception(
                f"API request failed with status {response.status_code}: {response.text}"
            )

        markets = response.json()

        if not markets:
            break  # No more markets to fetch

        all_markets.extend(markets)
        print(f"Fetched {len(markets)} markets. Total so far: {len(all_markets)}")

        # If we got fewer markets than the batch size, we've reached the end
        if len(markets) < batch_size:
            break

        # Set the 'before' parameter for the next page
        params["before"] = markets[-1]["id"]

        # Add delay to avoid hitting rate limits
        time.sleep(rate_limit_delay)

    return all_markets


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Scrape markets from Manifold Markets API"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="raw_markets.json",
        help="Output JSON file path (default: raw_markets.json)",
    )
    return parser.parse_args()


# Example usage
if __name__ == "__main__":
    args = parse_args()

    print("Scraping markets from Manifold Markets API...")
    markets = scrape_all_markets()
    print(f"Total markets scraped: {len(markets)}")

    # Save to JSON file
    print(f"Saving results to {args.output}")
    with open(args.output, "w") as f:
        json.dump(markets, f, indent=2)
    print(f"Done! Results saved to {args.output}")
