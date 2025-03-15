#!/usr/bin/env python3
import json
from datetime import datetime
from typing import Any, Dict, List

import datasets


def load_json_file(filename: str) -> List[Dict[str, Any]]:
    """Load a JSON file into a list of dictionaries."""
    with open(filename, "r") as f:
        content = f.read()
        try:
            # Try parsing as a JSON array first
            if content.strip().startswith("["):
                return json.loads(content)
            # Try parsing as a single JSON object
            else:
                return [json.loads(content)]
        except json.JSONDecodeError:
            # Try parsing as newline-delimited JSON
            return [
                json.loads(line) for line in content.strip().split("\n") if line.strip()
            ]


def parse_metaculus_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse Metaculus data into the unified format."""
    processed_data = []

    for item in data:
        # Extract the necessary fields
        question_data = item.get("question", {})

        # Combine description, resolution criteria, and fine print
        description_parts = []
        if question_data.get("description"):
            description_parts.append(question_data.get("description"))
        if question_data.get("resolution_criteria"):
            description_parts.append(
                f"Resolution Criteria: {question_data.get('resolution_criteria')}"
            )
        if question_data.get("fine_print"):
            description_parts.append(f"Fine Print: {question_data.get('fine_print')}")

        combined_description = "\n\n".join(description_parts)

        processed_item = {
            "id": f"meta-{item.get('id')}",
            "question": item.get("title", ""),
            "description": combined_description,
            "open_date": question_data.get("open_time", ""),
            "close_date": question_data.get("scheduled_close_time", ""),
            "resolve_date": question_data.get("actual_resolve_time", ""),
            "resolution": question_data.get("resolution", "").lower(),
            "source": "METACULUS",
        }
        processed_data.append(processed_item)

    return processed_data


def parse_manifold_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse Manifold data into the unified format."""
    processed_data = []

    for item in data:
        # Extract text description from structured content if needed
        description = item.get("textDescription", "")
        if not description and "description" in item:
            # This is a simplified approach - in a real scenario you'd need to properly parse
            # the structured content in item["description"]
            description = str(item.get("description", ""))

        processed_item = {
            "id": f"mani-{item.get('id')}",
            "question": item.get("question", ""),
            "description": description,
            "open_date": datetime.fromtimestamp(
                int(item.get("createdTime", 0) / 1000)
            ).isoformat()
            if "createdTime" in item
            else "",
            "close_date": datetime.fromtimestamp(
                int(item.get("closeTime", 0) / 1000)
            ).isoformat()
            if "closeTime" in item
            else "",
            "resolve_date": datetime.fromtimestamp(
                int(item.get("resolutionTime", 0) / 1000)
            ).isoformat()
            if "resolutionTime" in item
            else "",
            "resolution": item.get("resolution", "").lower(),
            "source": "MANIFOLD",
        }
        processed_data.append(processed_item)

    return processed_data


def main():
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a HuggingFace dataset from Metaculus and Manifold Markets data"
    )
    parser.add_argument(
        "--metaculus", required=True, help="Path to the Metaculus JSON file"
    )
    parser.add_argument(
        "--manifold", required=True, help="Path to the Manifold Markets JSON file"
    )
    parser.add_argument(
        "--output", default="prediction_markets", help="Name of the output dataset"
    )
    args = parser.parse_args()

    # Load and process data
    metaculus_data = load_json_file(args.metaculus)
    manifold_data = load_json_file(args.manifold)

    processed_metaculus = parse_metaculus_data(metaculus_data)
    processed_manifold = parse_manifold_data(manifold_data)

    # Combine data
    combined_data = processed_metaculus + processed_manifold

    # Remove all items where resolution is not "yes" or "no"
    combined_data = [
        item for item in combined_data if item["resolution"] in ["yes", "no"]
    ]

    # Create HuggingFace dataset
    dataset = datasets.Dataset.from_dict(
        {
            "id": [item["id"] for item in combined_data],
            "question": [item["question"] for item in combined_data],
            "description": [item["description"] for item in combined_data],
            "open_date": [item["open_date"] for item in combined_data],
            "close_date": [item["close_date"] for item in combined_data],
            "resolve_date": [item["resolve_date"] for item in combined_data],
            "resolution": [item["resolution"] for item in combined_data],
            "source": [item["source"] for item in combined_data],
        }
    )

    # Save the dataset
    dataset.save_to_disk(args.output)
    print(f"Dataset saved to {args.output}")

    # Optional: Push to HuggingFace Hub
    # If you want to push to HuggingFace Hub, uncomment this code and add authentication
    # dataset.push_to_hub(f"your-username/{args.output}")


if __name__ == "__main__":
    main()
