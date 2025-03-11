import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()


# Configure your API key as an environment variable
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise EnvironmentError("Please set the ANTHROPIC_API_KEY environment variable")


class RateLimiter:
    """Simple rate limiter using token bucket algorithm"""

    def __init__(self, max_requests_per_minute: int):
        self.max_requests_per_minute = max_requests_per_minute
        self.tokens = 0
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


SYSTEM_PROMPT = """You are classifying prediction market questions for quality. Only high-quality markets should be allowed.

EXCLUDE the market if ANY of these conditions are true:
1. It's clearly a meme, joke, or spam market with no substantive information value
2. It's meta-level (referring to the prediction platform "Manifold" itself rather than external events)
3. It references non-public figures (e.g., "Will I (@JohnDoe) be in a car crash this year")
   - Public figures are notable politicians, celebrities, business leaders, or other people with significant public presence
   - Non-public figures would be regular users or people without notable public profiles

KEEP all other markets.

Answer with a single word: KEEP or EXCLUDE
"""


def call_claude_api(
    market_title: str,
    rate_limiter: Optional[RateLimiter] = None,
    model: str = "claude-3-7-sonnet-latest",
) -> bool:
    """
    Call Claude API to classify whether a market is good quality.

    Args:
        market_title: The title (question) of the market
        rate_limiter: Optional rate limiter to control API request rate
        model: The Claude model to use

    Returns:
        Boolean indicating if the market should be kept
    """
    # Acquire token from rate limiter if provided
    if rate_limiter:
        rate_limiter.acquire()

    try:
        # Initialize the Anthropic client
        client = Anthropic(api_key=ANTHROPIC_API_KEY)

        # Call the API
        response = client.messages.create(
            model=model,
            max_tokens=150,
            temperature=0.1,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Market question: {market_title}"}],
        )

        content = response.content[0].text

        # Check for "keep" or "exclude" in the response
        content_upper = content.lower()
        has_keep = "keep" in content_upper
        has_exclude = "exclude" in content_upper

        # Validate response format
        if (has_keep and has_exclude) or (not has_keep and not has_exclude):
            raise ValueError(
                f"Invalid classification response: {content}. Expected single word KEEP or EXCLUDE."
            )

        # Determine if market should be kept
        is_good_market = has_keep

        return is_good_market

    except Exception as e:
        print(f"Error calling Claude API: {str(e)}")
        return False


def llm_filter(
    input_file: str = "quality_filtered_markets.json",
    output_file: str = "llm_filtered_markets.json",
    max_workers: int = 5,
    max_requests_per_minute: int = 120,  # Default to 120 requests per minute (2 per second)
    claude_model: str = "claude-3-7-sonnet-latest",
) -> List[Dict[str, Any]]:
    """
    Filter Manifold markets using Claude to classify quality.

    Args:
        input_file: Path to JSON file with market data
        output_file: Path to save filtered market data
        max_workers: Maximum number of concurrent API requests
        max_requests_per_minute: Maximum API requests per minute to avoid rate limiting
        claude_model: Claude model to use

    Returns:
        List of filtered market objects
    """
    # Load market data
    with open(input_file, "r") as f:
        markets = json.load(f)

    print(f"Loaded {len(markets)} markets from {input_file}")

    # Create rate limiter for API requests
    rate_limiter = RateLimiter(max_requests_per_minute)

    total_processed = 0
    total_kept = 0
    total_excluded = 0
    filtered_markets = []

    # Process all markets with a single thread pool
    market_results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_market = {
            executor.submit(
                call_claude_api, market["question"], rate_limiter, claude_model
            ): (
                market["id"],
                market["question"],
            )
            for market in markets
        }

        # Process results as they complete
        progress = tqdm(total=len(markets), desc="Processing markets", unit="market")
        for i, future in enumerate(as_completed(future_to_market)):
            market_id, market_question = future_to_market[future]
            try:
                is_good_market = future.result()
                market_results[market_id] = is_good_market
            except Exception as e:
                print(f"Error processing market {market_id}: {str(e)}")
                market_results[market_id] = False

            progress.update(1)

        progress.close()

    # Apply filtering based on Claude results
    for market in markets:
        market_id = market["id"]
        is_good_market = market_results.get(market_id, False)

        # Add to filtered list if it's a good market
        if is_good_market:
            filtered_markets.append(market)
            total_kept += 1
        else:
            total_excluded += 1

        total_processed += 1

    # Save filtered markets to output file
    with open(output_file, "w") as f:
        json.dump(filtered_markets, f, indent=2)

    print("\nFiltering complete!")
    print(f"Total markets processed: {total_processed}")
    print(f"Markets kept: {total_kept} ({total_kept/total_processed*100:.1f}%)")
    print(
        f"Markets excluded: {total_excluded} ({total_excluded/total_processed*100:.1f}%)"
    )
    print(f"Filtered markets saved to: {output_file}")

    return filtered_markets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter Manifold markets using Claude LLM"
    )
    parser.add_argument(
        "--input", default="quality_filtered_markets.json", help="Input JSON file"
    )
    parser.add_argument(
        "--output", default="llm_filtered_markets.json", help="Output JSON file"
    )
    parser.add_argument(
        "--max-workers", type=int, default=5, help="Maximum concurrent API requests"
    )
    parser.add_argument(
        "--max-requests-per-minute",
        type=int,
        default=120,
        help="Maximum API requests per minute to avoid rate limiting",
    )
    parser.add_argument(
        "--model",
        default="claude-3-7-sonnet-latest",
        help="Claude model to use",
    )

    args = parser.parse_args()

    llm_filter(
        input_file=args.input,
        output_file=args.output,
        max_workers=args.max_workers,
        max_requests_per_minute=args.max_requests_per_minute,
        claude_model=args.model,
    )
