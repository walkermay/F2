import json
import os
from collections import defaultdict


def preprocess_attribution_results(input_file, output_file):
    """
    Preprocess attribution results by grouping documents by single_doc_answer
    and sorting by confidence scores
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = {}

    for test_id, test_data in data.items():
        processed_item = test_data.copy()

        if 'document_attributions' in test_data:
            # Group by single_doc_answer
            answer_groups = defaultdict(list)
            for doc_attr in test_data['document_attributions']:
                answer = doc_attr.get('single_doc_answer', 'unknown')
                answer_groups[answer].append(doc_attr)

            # Sort documents within each group by confidence (descending)
            for answer, docs in answer_groups.items():
                docs.sort(key=lambda x: x.get('single_doc_confidence', 0), reverse=True)

            # Create group statistics
            answer_group_stats = {}
            for answer, docs in answer_groups.items():
                answer_group_stats[answer] = {
                    'count': len(docs),
                    'highest_confidence': max(doc.get('single_doc_confidence', 0) for doc in docs),
                    'docs_list': docs
                }

            # Sort answer groups by count (desc), then by highest confidence (desc)
            sorted_answers = sorted(
                answer_group_stats.keys(),
                key=lambda answer: (
                    -answer_group_stats[answer]['count'],
                    -answer_group_stats[answer]['highest_confidence']
                )
            )

            # Rebuild document_attributions in sorted order
            new_document_attributions = []
            for answer in sorted_answers:
                docs = answer_group_stats[answer]['docs_list']
                new_document_attributions.extend(docs)

            processed_item['document_attributions'] = new_document_attributions

        processed_data[test_id] = processed_item

    # Save processed data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)


def main():
    """Main function"""
    datasets = ['nq', 'hotpotqa', 'msmarco']

    for dataset in datasets:
        input_file = f"results/attribution/{dataset}-gpt3-70.json"
        output_file = f"results/sort/{dataset}-gpt3-70.json"
        preprocess_attribution_results(input_file, output_file)
        print(f"Processed {dataset}")

    print("All datasets processed!")


if __name__ == "__main__":
    main()