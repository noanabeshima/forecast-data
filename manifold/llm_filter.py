import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Configure your API key as an environment variable
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise EnvironmentError("Please set the ANTHROPIC_API_KEY environment variable")


def call_claude_api(
    market_title: str, model: str = "claude-3-5-sonnet-20240307"
) -> Tuple[bool, str]:
    """
    Call Claude API to classify whether a market is good quality.

    Args:
        market_title: The title (question) of the market
        model: The Claude model to use

    Returns:
        Tuple of (is_good_market, explanation)
    """
    prompt = f"""You are classifying prediction market questions for quality. Only high-quality markets should be allowed.

You need to classify the following prediction market question as either KEEP or EXCLUDE.

EXCLUDE the market if ANY of these conditions are true:
1. It's clearly a meme or joke market with no substantive information value
2. It's meta-level (referring to the prediction platform "Manifold" itself rather than external events)
3. It references non-public figures (e.g., "Will I (@JohnDoe) be in a car crash this year")
   - Public figures are notable politicians, celebrities, business leaders, or other people with significant public presence
   - Non-public figures would be regular users or people without notable public profiles
4. It contains obvious spam or nonsensical content

KEEP all other markets.

Market question: "{market_title}"

First, analyze the market question carefully.
Then provide your classification as either KEEP or EXCLUDE.
Finally, provide a brief explanation of your reasoning.

Format your response as:
CLASSIFICATION: [KEEP or EXCLUDE]
EXPLANATION: [Your explanation]

Keep your explanation concise, 1-2 sentences.
"""

    try:
        # Initialize the Anthropic client
        client = Anthropic(api_key=ANTHROPIC_API_KEY)

        # Call the API
        response = client.messages.create(
            model=model,
            max_tokens=150,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text

        # Parse the response
        classification_line = [
            line for line in content.split("\n") if line.startswith("CLASSIFICATION:")
        ][0]
        explanation_line = [
            line for line in content.split("\n") if line.startswith("EXPLANATION:")
        ][0]

        classification = classification_line.split("CLASSIFICATION:")[1].strip()
        explanation = explanation_line.split("EXPLANATION:")[1].strip()

        is_good_market = classification == "KEEP"

        return is_good_market, explanation

    except Exception as e:
        print(f"Error calling Claude API: {str(e)}")
        return False, f"Error: {str(e)}"


def llm_filter(
    input_file: str = "quality_filtered_markets.json",
    output_file: str = "llm_filtered_markets.json",
    output_report_file: str = "llm_filter_report.json",
    max_workers: int = 5,
    request_delay: float = 0.5,
    batch_size: int = 50,
    claude_model: str = "claude-haiku-3-5-latest",
) -> List[Dict[str, Any]]:
    """
    Filter Manifold markets using Claude to classify quality.

    Args:
        input_file: Path to JSON file with market data
        output_file: Path to save filtered market data
        output_report_file: Path to save filtering report with explanations
        max_workers: Maximum number of concurrent API requests
        request_delay: Delay between API requests per worker
        batch_size: Process in batches to show progress
        claude_model: Claude model to use

    Returns:
        List of filtered market objects
    """
    # Load market data
    with open(input_file, "r") as f:
        markets = json.load(f)[:100]

    print(f"Loaded {len(markets)} markets from {input_file}")

    # Process markets in batches with concurrent API calls
    filtered_markets = []
    filtering_report = []
    total_batches = (len(markets) + batch_size - 1) // batch_size

    total_processed = 0
    total_kept = 0
    total_excluded = 0

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(markets))
        batch = markets[start_idx:end_idx]

        print(
            f"Processing batch {batch_idx + 1}/{total_batches} ({start_idx} to {end_idx})"
        )

        # Classify markets concurrently
        batch_results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_market = {
                executor.submit(call_claude_api, market["question"], claude_model): (
                    market["id"],
                    market["question"],
                )
                for market in batch
            }

            # Process results as they complete
            for i, future in enumerate(as_completed(future_to_market)):
                market_id, market_question = future_to_market[future]
                try:
                    is_good_market, explanation = future.result()
                    batch_results[market_id] = (is_good_market, explanation)

                    status = "KEPT" if is_good_market else "EXCLUDED"
                    print(f"  [{i+1}/{len(batch)}] {status}: {market_question[:50]}...")

                    # Add delay to avoid rate limiting
                    time.sleep(request_delay)

                except Exception as e:
                    print(f"Error processing market {market_id}: {str(e)}")
                    batch_results[market_id] = (False, f"Error: {str(e)}")

        # Apply filtering based on Claude results
        batch_filtered = []
        for market in batch:
            market_id = market["id"]
            is_good_market, explanation = batch_results.get(
                market_id, (False, "Processing error")
            )

            # Add to the report
            filtering_report.append(
                {
                    "id": market_id,
                    "question": market["question"],
                    "kept": is_good_market,
                    "explanation": explanation,
                }
            )

            # Add to filtered list if it's a good market
            if is_good_market:
                batch_filtered.append(market)
                total_kept += 1
            else:
                total_excluded += 1

            total_processed += 1

        # Add batch results to overall filtered list
        filtered_markets.extend(batch_filtered)

        # Report progress
        print(f"Batch complete: {len(batch_filtered)}/{len(batch)} markets kept")
        print(
            f"Running total: {total_kept}/{total_processed} markets kept ({total_excluded} excluded)"
        )

    # Save filtered markets to output file
    with open(output_file, "w") as f:
        json.dump(filtered_markets, f, indent=2)

    # Save filtering report to report file
    with open(output_report_file, "w") as f:
        json.dump(filtering_report, f, indent=2)

    print("\nFiltering complete!")
    print(f"Total markets processed: {total_processed}")
    print(f"Markets kept: {total_kept} ({total_kept/total_processed*100:.1f}%)")
    print(
        f"Markets excluded: {total_excluded} ({total_excluded/total_processed*100:.1f}%)"
    )
    print(f"Filtered markets saved to: {output_file}")
    print(f"Filtering report saved to: {output_report_file}")

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
        "--report", default="llm_filter_report.json", help="Filtering report JSON file"
    )
    parser.add_argument(
        "--max-workers", type=int, default=5, help="Maximum concurrent API requests"
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.5,
        help="Delay between API requests (seconds)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Process in batches of this size"
    )
    parser.add_argument(
        "--model", default="claude-3-5-haiku-latest", help="Claude model to use"
    )

    args = parser.parse_args()

    llm_filter(
        input_file=args.input,
        output_file=args.output,
        output_report_file=args.report,
        max_workers=args.max_workers,
        request_delay=args.request_delay,
        batch_size=args.batch_size,
        claude_model=args.model,
    )
