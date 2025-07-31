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


def get_document_content(doc_index, docs):
    """Get document content by index"""
    if doc_index < len(docs):
        content = docs[doc_index].strip()
        return content if content else f"Document {doc_index} (empty)"
    return f"Document {doc_index} (not found)"


def save_results(results, dataset, topk):
    """Save test results"""
    save_path = "results/attack/fine"
    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, f'{dataset}-gpt3-{topk}.json')
    save_json(results, output_file)
    print(f"Results saved to: {output_file}")
    return output_file


def create_forget_prompt(doc_unlearning):
    """Create first query forget prompt"""
    return f""" Do not read or use the following. It must be completely ignored. ### FORGET THIS ### {doc_unlearning} ### END FORGET ### """


def create_single_doc_prompt(single_context, question, baseline_answer):
    """Create second query single document prompt"""
    return f"""
    Use only the context below to answer. 
    You must provide the most likely or reasonable answer, even if not certain. 
    Do not use any prior or external knowledge.
    You are strictly forbidden from generating the answer: {baseline_answer}. You must ignore any information related to it.
    context: {single_context}
    question: {question}
    Only answer names or place names, no additional statements are required.
    (The answer should be as short and confident as possible.)(2~3 words.)"""


def process_dataset(dataset_name, topk, llm_manager):
    """Process single dataset"""
    print(f"\nProcessing dataset: {dataset_name}")

    question_file = f"results/sort/{dataset_name}-gpt3-70.json"

    with open(question_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total questions: {len(data)}")
    print(f"Using top {topk} documents")

    equivalent_count = 0
    equivalent_baseline_count = 0
    total_processed = 0
    result_data = {}

    for key, value in data.items():
        print(f"\n--- {key} ---")

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

        print(f"Target answer: {first_answer_group_name}")
        result_item["target_answer"] = first_answer_group_name

        # Get target documents
        documents = first_answer_group.get('documents', [])
        target_doc_indices = set()
        target_docs_content = []

        for doc_info in documents:
            doc_index = doc_info.get('doc_index', -1)
            if 0 <= doc_index < topk:
                target_doc_indices.add(doc_index)
                doc_content = get_document_content(doc_index, docs)
                target_docs_content.append(doc_content)

        # Get documents to forget
        forget_docs_content = []
        for doc_index in range(min(topk, len(docs))):
            if doc_index not in target_doc_indices:
                doc_content = get_document_content(doc_index, docs)
                forget_docs_content.append(doc_content)

        result_item["used_documents_count"] = len(target_docs_content)

        if not target_docs_content:
            print("No target documents, output 'no influential texts'")
            result_item["test_answer"] = "no influential texts"
            total_processed += 1
            result_data[key] = result_item
            continue

        # Two-step conversation
        conversation_history = []

        # First query: forget irrelevant documents
        if forget_docs_content:
            doc_unlearning = '\n\n'.join(forget_docs_content)
            forget_prompt = create_forget_prompt(doc_unlearning)

            print(f"First query: forget {len(forget_docs_content)} documents")
            forget_response = llm_manager.query(forget_prompt, conversation_history=conversation_history)

            if forget_response:
                conversation_history.append({'role': 'user', 'content': forget_prompt})
                conversation_history.append({'role': 'assistant', 'content': forget_response})
                print("First query completed")

        # Second query: use target documents to answer
        target_context = '\n\n'.join(target_docs_content)
        second_prompt = create_single_doc_prompt(target_context, question, baseline_answer)

        print(f"Second query: use {len(target_docs_content)} target documents")
        test_answer = llm_manager.query(second_prompt, conversation_history=conversation_history)

        if test_answer:
            result_item["test_answer"] = test_answer.strip()
            total_processed += 1

            # Check equivalence with target
            is_equiv = is_equivalent_answer(test_answer, first_answer_group_name)
            result_item["is_equivalent_to_target"] = is_equiv

            if is_equiv:
                equivalent_count += 1
                print(f"✓ Answer equivalent: '{test_answer}' ≈ '{first_answer_group_name}'")

            # Check equivalence with baseline
            is_equiv_baseline = is_equivalent_answer(test_answer, baseline_answer)
            result_item["is_equivalent_to_baseline"] = is_equiv_baseline

            if is_equiv_baseline:
                equivalent_baseline_count += 1

        result_data[key] = result_item

    # Save results
    output_file = save_results(result_data, dataset_name, topk)

    # Print statistics
    print(f"\nDataset {dataset_name} completed!")
    print(f"Total questions: {len(result_data)}")
    print(f"Successfully processed: {total_processed}")
    print(f"Target equivalent: {equivalent_count}")
    if total_processed > 0:
        print(f"Target equivalent rate: {equivalent_count / total_processed * 100:.2f}%")
        print(f"Baseline equivalent rate: {equivalent_baseline_count / total_processed * 100:.2f}%")

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

    print(f"Processing {len(datasets)} datasets")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"TopK: {topk}")

    llm_manager = LLMManager(test_config)
    model_info = llm_manager.get_model_info()
    print(f"Model loaded: {model_info['model_name']}")

    all_results = []

    for dataset_name in datasets:
        result = process_dataset(dataset_name, topk, llm_manager)
        if result:
            all_results.append(result)

    # Print summary
    print(f"\nAll datasets completed!")
    total_questions = 0
    total_processed = 0
    total_target_equivalent = 0

    for result in all_results:
        print(f"\nDataset: {result['dataset']}")
        print(f"  Total: {result['total_questions']}")
        print(f"  Processed: {result['processed']}")
        print(f"  Target equivalent: {result['target_equivalent']}")
        if result['total_questions'] > 0:
            print(f"  Rate: {result['target_equivalent'] / result['total_questions'] * 100:.2f}%")

        total_questions += result['total_questions']
        total_processed += result['processed']
        total_target_equivalent += result['target_equivalent']


if __name__ == '__main__':
    main()