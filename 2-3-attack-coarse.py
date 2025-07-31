import json
import os
from src.utils import save_json
from preprocess.llm import LLMManager


def is_equivalent_answer(answer1, answer2):
    """Check if two answers are equivalent"""
    if not answer1 or not answer2:
        return False

    answer1 = answer1.lower().strip()
    answer2 = answer2.lower().strip()

    if answer1 == answer2:
        return True

    if answer1 in answer2 or answer2 in answer1:
        return True

    return False


def get_document_content_by_index(doc_index, docs):
    """Get document content by index from docs"""
    if doc_index < len(docs):
        content = docs[doc_index].strip()
        return content if content else f"Document {doc_index} (empty)"
    return f"Document {doc_index} (not found)"


def save_results(results, dataset, topk):
    """Save test results"""
    save_path = "results/attack/coarse"
    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, f'{dataset}-gpt3-{topk}.json')
    save_json(results, output_file)
    print(f"Results saved to: {output_file}")
    return output_file


def create_single_doc_prompt(single_context, question, baseline_answer):
    """Create single document prompt"""
    return f"""
    Use only the context below to answer. 
    You must provide the most likely or reasonable answer, even if not certain. 
    Do not use any prior or external knowledge.
    You are strictly forbidden from generating the answer: {baseline_answer}. You must ignore any information related to it.
    context: {single_context}
    question: {question}
    Only answer names or place names, no additional statements are required.
    (The answer should be as short and confident as possible.)(2~3 words.)"""


def process_single_dataset(dataset_name, topk, llm_manager):
    """Process single dataset"""
    print(f"\nProcessing dataset: {dataset_name}")

    question_file = f"results/sort/{dataset_name}-gpt3-70.json"

    with open(question_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Processing {len(data)} questions with top {topk} documents")

    equivalent_count = 0
    equivalent_baseline_count = 0
    total_processed = 0
    result_data = {}

    for key, value in data.items():
        question = value.get('question', '')
        baseline_answer = value.get('baseline_answer', '')
        answer_groups = value.get('answer_groups', {})
        docs = value.get('docs', [])
        item_id = value.get('id', key)

        result_item = {
            "id": item_id,
            "question": question,
            "baseline_answer": baseline_answer,
            "target_answer": "",
            "test_answer": "",
            "used_documents_count": 0,
            "is_equivalent_to_target": False,
            "is_equivalent_to_baseline": False
        }

        if not answer_groups or not docs:
            result_data[key] = result_item
            continue

        # Get first answer group
        first_answer_group_name = list(answer_groups.keys())[0]
        first_answer_group = answer_groups[first_answer_group_name]
        result_item["target_answer"] = first_answer_group_name

        # Filter documents by topk
        documents = first_answer_group.get('documents', [])
        filtered_documents = [doc for doc in documents
                              if doc.get('doc_index', -1) >= 0 and doc.get('doc_index', -1) < topk]

        result_item["used_documents_count"] = len(filtered_documents)

        if not filtered_documents:
            result_item["test_answer"] = "no influential texts"
            total_processed += 1
            result_data[key] = result_item
            continue

        # Collect document content
        all_docs_content = []
        for doc_info in filtered_documents:
            doc_index = doc_info.get('doc_index', -1)
            if doc_index >= 0:
                doc_content = get_document_content_by_index(doc_index, docs)
                all_docs_content.append(doc_content)

        single_context = '\n\n'.join(all_docs_content)
        prompt = create_single_doc_prompt(single_context, question, baseline_answer)

        # Query LLM
        test_answer = llm_manager.query(prompt)

        if test_answer:
            result_item["test_answer"] = test_answer.strip()
            total_processed += 1

            # Check equivalence with target
            is_equiv = is_equivalent_answer(test_answer, first_answer_group_name)
            result_item["is_equivalent_to_target"] = is_equiv
            if is_equiv:
                equivalent_count += 1

            # Check equivalence with baseline
            is_equiv_baseline = is_equivalent_answer(test_answer, baseline_answer)
            result_item["is_equivalent_to_baseline"] = is_equiv_baseline
            if is_equiv_baseline:
                equivalent_baseline_count += 1

        result_data[key] = result_item

    # Save and print results
    output_file = save_results(result_data, dataset_name, topk)

    print(f"\nDataset {dataset_name} completed:")
    print(f"Total questions: {len(result_data)}")
    print(f"Successfully processed: {total_processed}")
    print(f"Target equivalent: {equivalent_count}")
    print(f"Target equivalent rate: {equivalent_count / total_processed * 100:.2f}%" if total_processed > 0 else "0%")
    print(f"Baseline equivalent: {equivalent_baseline_count}")
    print(
        f"Baseline equivalent rate: {equivalent_baseline_count / total_processed * 100:.2f}%" if total_processed > 0 else "0%")

    return {
        'dataset': dataset_name,
        'total_questions': len(result_data),
        'processed': total_processed,
        'target_equivalent': equivalent_count,
        'baseline_equivalent': equivalent_baseline_count,
        'output_file': output_file
    }


def main():
    """Main function"""
    datasets = ["nq", "hotpotqa", "msmarco"]
    topk = 70
    test_config = "./model_configs/gpt3.5_config.json"

    print(f"Processing {len(datasets)} datasets with topk={topk}")

    llm_manager = LLMManager(test_config)
    model_info = llm_manager.get_model_info()
    print(f"Model loaded: {model_info['model_name']}")

    all_results = []

    for dataset_name in datasets:
        result = process_single_dataset(dataset_name, topk, llm_manager)
        if result:
            all_results.append(result)

    # Print summary
    print(f"\nAll datasets completed!")

    total_questions = sum(r['total_questions'] for r in all_results)
    total_processed = sum(r['processed'] for r in all_results)
    total_target_equivalent = sum(r['target_equivalent'] for r in all_results)
    total_baseline_equivalent = sum(r['baseline_equivalent'] for r in all_results)

    for result in all_results:
        print(f"\n{result['dataset']}: {result['target_equivalent']}/{result['total_questions']} "
              f"({result['target_equivalent'] / result['total_questions'] * 100:.2f}%)")

    print(f"\nOverall:")
    print(f"Total questions: {total_questions}")
    print(
        f"Target equivalent rate: {total_target_equivalent / total_processed * 100:.2f}%" if total_processed > 0 else "0%")
    print(
        f"Baseline equivalent rate: {total_baseline_equivalent / total_processed * 100:.2f}%" if total_processed > 0 else "0%")


if __name__ == '__main__':
    main()