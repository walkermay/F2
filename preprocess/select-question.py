import json
import os
from pathlib import Path


def load_valid_query_ids(test_tsv_file):
    """Load valid query_id set from test.tsv file"""
    valid_query_ids = set()

    with open(test_tsv_file, 'r', encoding='utf-8') as f:
        next(f, None)  # Skip header
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    query_id = parts[0].strip()
                    if query_id:
                        valid_query_ids.add(query_id)

    return valid_query_ids


def filter_who_questions(input_file, output_file, test_tsv_file, max_questions=100):
    """Filter questions starting with 'who' that exist in test.tsv"""

    # Create output directory
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load valid query_ids
    valid_query_ids = load_valid_query_ids(test_tsv_file)

    who_questions = {}

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line.strip())
            question_text = data.get('text', '').strip()

            if question_text.lower().startswith('who '):
                query_id = data.get('_id')

                if query_id in valid_query_ids:
                    who_questions[query_id] = {
                        'id': query_id,
                        'question': data.get('text')
                    }

                    if len(who_questions) >= max_questions:
                        break

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(who_questions, f, ensure_ascii=False, indent=2)

    print(f"Success! Found {len(who_questions)} questions and saved to: {output_file}")


def main():
    """
        It should be noted that due to the fact that
        there are less than 100 questions in the test.tsv dataset of msmarco,
        train.tsv with a larger number of questions was chosen.
    """
    # All three paths need to be modified accordingly.
    # nq/queries.jsonl   filter/NQ.json   nq/qrels/test.tsv
    # msmarco/queries.jsonl   filter/MS-MARCO.json   msmarco/qrels/train.tsv
    # hotpotqa/queries.jsonl   filter/HotpotQA.json   hotpotqa/qrels/test.tsv
    input_file = "../datasets/nq/queries.jsonl"
    output_file = "../datasets/filter/NQ.json"
    test_tsv_file = "../datasets/nq/qrels/test.tsv"

    filter_who_questions(input_file, output_file, test_tsv_file, max_questions=100)

if __name__ == "__main__":
    main()