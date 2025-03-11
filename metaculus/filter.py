import argparse
import datetime
import json
from typing import Any, Dict, List, Optional


def filter_questions(
    input_file: str = "raw_questions.json",
    output_file: str = "filtered_questions.json",
    question_type: Optional[str] = "binary",
    min_resolve_date: Optional[str] = "2023-03-01",
    max_resolve_date: Optional[str] = None,
    only_resolved: bool = True,
    exclude_deleted: bool = True,
) -> List[Dict[str, Any]]:
    """
    Filter Metaculus questions based on specified criteria.

    Args:
        input_file: Path to JSON file with raw question data
        output_file: Path to save filtered question data
        question_type: Type of questions to include (e.g., "binary")
        min_resolve_date: Only include questions resolved after this date (ISO format)
        max_resolve_date: Only include questions resolved before this date (ISO format)
        only_resolved: If True, only include questions that have been resolved
        exclude_deleted: If True, exclude questions marked as deleted

    Returns:
        List of filtered question objects
    """
    # Load raw question data
    with open(input_file, "r") as f:
        posts = json.load(f)

    print(f"Loaded {len(posts)} posts from {input_file}")

    # Convert dates to datetime objects if provided
    min_date = None
    if min_resolve_date:
        min_date = datetime.datetime.fromisoformat(min_resolve_date + "T00:00:00Z")

    max_date = None
    if max_resolve_date:
        max_date = datetime.datetime.fromisoformat(max_resolve_date + "T00:00:00Z")
    else:
        max_date = datetime.datetime.now(datetime.timezone.utc)

    # Filter out non-questions
    questions = [post for post in posts if "question" in post]
    print(f"Found {len(questions)} questions out of {len(posts)} posts")

    # Filter by question type if specified
    if question_type:
        questions = [
            q for q in questions if q.get("question", {}).get("type") == question_type
        ]
        print(f"Found {len(questions)} {question_type} questions")

    # Filter by resolve date
    if min_date or max_date:
        filtered_questions = []
        for q in questions:
            if "scheduled_resolve_time" not in q:
                continue

            try:
                resolve_time = datetime.datetime.fromisoformat(
                    q["scheduled_resolve_time"]
                )

                # Check min date
                if min_date and resolve_time < min_date:
                    continue

                # Check max date
                if max_date and resolve_time > max_date:
                    continue

                filtered_questions.append(q)
            except (ValueError, TypeError):
                # Skip questions with invalid date format
                continue

        questions = filtered_questions
        print(f"Found {len(questions)} questions within the specified date range")

    # Filter by resolution status
    if only_resolved:
        questions = [q for q in questions if q.get("status") == "resolved"]
        print(f"Found {len(questions)} resolved questions")

    # Filter out deleted questions
    if exclude_deleted:
        questions = [q for q in questions if q.get("title") != "[DELETED QUESTION]"]
        print(f"Found {len(questions)} non-deleted questions")

    # Save filtered questions to output file
    with open(output_file, "w") as f:
        json.dump(questions, f, indent=2)

    print(f"Saved {len(questions)} filtered questions to {output_file}")

    return questions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter Metaculus questions")
    parser.add_argument(
        "--input",
        default="raw_questions.json",
        help="Input JSON file with raw questions",
    )
    parser.add_argument(
        "--output",
        default="filtered_questions.json",
        help="Output JSON file for filtered questions",
    )
    parser.add_argument(
        "--question-type",
        default="binary",
        help="Type of questions to include (e.g., 'binary')",
    )
    parser.add_argument(
        "--min-resolve-date",
        default="2023-03-01",
        help="Only include questions resolved after this date (ISO format)",
    )
    parser.add_argument(
        "--max-resolve-date",
        default=None,
        help="Only include questions resolved before this date (ISO format)",
    )
    parser.add_argument(
        "--include-unresolved", action="store_true", help="Include unresolved questions"
    )
    parser.add_argument(
        "--include-deleted", action="store_true", help="Include deleted questions"
    )

    args = parser.parse_args()

    filter_questions(
        input_file=args.input,
        output_file=args.output,
        question_type=args.question_type,
        min_resolve_date=args.min_resolve_date,
        max_resolve_date=args.max_resolve_date,
        only_resolved=not args.include_unresolved,
        exclude_deleted=not args.include_deleted,
    )
