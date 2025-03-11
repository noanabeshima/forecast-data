import json
import os
import time
from typing import Any, Dict, List, Optional

import dotenv
import ray
import requests
from tqdm import tqdm

# Load environment variables
dotenv.load_dotenv()
API_KEY = os.getenv("METACULUS_API_KEY")


def scrape_metaculus_posts(
    output_file: str = "raw_questions.json",
    batch_size: int = 500,
    base_url: str = "https://metaculus.com/api/posts/",
    api_key: Optional[str] = None,
    max_batches: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Scrape posts from Metaculus API with parallel requests.

    Args:
        output_file: Path to save the raw data
        batch_size: Number of posts to fetch per request
        base_url: The API endpoint
        api_key: Metaculus API key (optional)
        max_batches: Maximum number of batches to fetch (optional)

    Returns:
        List of all posts
    """
    # Initialize Ray for parallel processing
    ray.init(ignore_reinit_error=True, num_cpus=10)

    @ray.remote
    def fetch_posts(offset, limit=batch_size):
        """Remote function to fetch posts with a specific offset"""
        headers = {}
        if api_key:
            headers["Authorization"] = f"Token {api_key}"

        return requests.get(
            f"{base_url}?limit={limit}&offset={offset}", headers=headers
        )

    results = []
    base_offset = 0
    batch_count = 0

    while True:
        should_stop = False

        # Start multiple parallel requests with different offsets
        offsets = [i for i in range(base_offset, base_offset + 10000, batch_size)]
        request_ids = [fetch_posts.remote(offset) for offset in offsets]

        # Create progress bar
        pbar = tqdm(total=len(request_ids), desc="Fetching posts")

        # Get results as they complete without blocking
        batch_results = []
        remaining_ids = request_ids.copy()

        while remaining_ids:
            # Check which ones are ready without blocking
            ready_ids, remaining_ids = ray.wait(
                remaining_ids, timeout=0.1, num_returns=1
            )

            # If any are ready, get them and update the progress bar
            if ready_ids:
                for ready_id in ready_ids:
                    batch_results.append(ray.get(ready_id))
                    pbar.update(1)

        results.extend(batch_results)
        pbar.close()

        # Check if we've reached the end of the data
        if any(r.status_code != 200 for r in batch_results):
            print("Stopping: Received non-200 status code")
            should_stop = True

        # Also break if we've processed all available data
        if any(
            r.status_code == 200 and len(r.json().get("results", [])) < batch_size
            for r in batch_results
        ):
            print("Stopping: Received fewer results than requested limit")
            should_stop = True

        # Check if we've reached the maximum number of batches
        batch_count += 1
        if max_batches and batch_count >= max_batches:
            print(f"Stopping: Reached maximum number of batches ({max_batches})")
            should_stop = True

        if should_stop:
            break

        # Continue to the next batch
        base_offset += 10000
        time.sleep(1)  # Add a small delay to avoid overwhelming the API

    # Extract posts from results
    posts = []
    for res in results:
        if res.status_code == 200:
            posts.extend(res.json().get("results", []))

    print(f"Total posts scraped: {len(posts)}")

    # Save to JSON file
    with open(output_file, "w") as f:
        json.dump(posts, f)

    print(f"Saved {len(posts)} posts to {output_file}")

    # Shutdown Ray
    ray.shutdown()

    return posts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape posts from Metaculus API")
    parser.add_argument(
        "--output", default="raw_questions.json", help="Output JSON file"
    )
    parser.add_argument(
        "--batch-size", type=int, default=500, help="Batch size for API requests"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of batches to fetch",
    )

    args = parser.parse_args()

    scrape_metaculus_posts(
        output_file=args.output,
        batch_size=args.batch_size,
        api_key=API_KEY,
        max_batches=args.max_batches,
    )
