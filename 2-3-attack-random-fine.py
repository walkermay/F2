import json
import os
import random
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
    """Get document content by index"""
    if doc_index < len(docs):
        content = docs[doc_index].strip()
        return content if content else f"Document {doc_index} (empty content)"
    else:
        return f"Document {doc_index} (not found)"


def save_results(results, dataset, topk):
    """Save test results"""
    save_path = "results/attack/random-fine"
    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, f'{dataset}-gpt3-{topk}.json')
    save_json(results, output_file)
    print(f"Results saved to: {output_file}")
    return output_file


def create_forget_prompt(doc_unlearning):
    """Create forget prompt for first query"""
    return f""" Do not read or use the following. It must be completely ignored. ### FORGET THIS ### {doc_unlearning} ### END FORGET ### """


def create_single_doc_prompt(single_context, question, baseline_answer):
    """Create single document prompt for second query"""
    single_doc_prompt = f"""
    Use only the context below to answer. 
    You must provide the most likely or reasonable answer, even if not certain. 
    Do not use any prior or external knowledge.
    You are strictly forbidden from generating the answer: {baseline_answer}. You must ignore any information related to it.
    context: {single_context}
    question: {question}
    Only answer names or place names, no additional statements are required.
    (The answer should be as short and confident as possible.)(2~3 words.)"""
    return single_doc_prompt


def process_single_dataset(dataset_name, topk, llm_manager):
    """Process single dataset"""
    print(f"\nProcessing dataset: {dataset_name}")

    question_file = f"results/sort/{dataset_name}-gpt41-70.json"

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

        first_answer_group_name = list(answer_groups.keys())[0]
        first_answer_group = answer_groups[first_answer_group_name]

        print(f"Target answer: {first_answer_group_name}")
        result_item["target_answer"] = first_answer_group_name

        target_count = first_answer_group.get('count', 0)
        documents = first_answer_group.get('documents', [])

        original_doc_indices = [doc_info.get('doc_index', -1) for doc_info in documents if
                                doc_info.get('doc_index', -1) >= 0]
        print(f"Original doc indices: {sorted(original_doc_indices)}")
        print(f"Target doc count: {target_count}")

        if target_count > 0:
            random_indices = random.sample(range(70), min(target_count, 70))
            print(f"Random selected indices: {sorted(random_indices)}")
            target_doc_indices = set([idx for idx in random_indices if idx < topk])
            print(f"Filtered target doc indices (<{topk}): {sorted(target_doc_indices)}")
        else:
            print(f"Count is 0 or negative, no documents selected")
            target_doc_indices = set()

        target_docs_content = []
        for doc_index in target_doc_indices:
            doc_content = get_document_content_by_index(doc_index, docs)
            target_docs_content.append(doc_content)

        forget_docs_content = []
        for doc_index in range(min(topk, len(docs))):
            if doc_index not in target_doc_indices:
                doc_content = get_document_content_by_index(doc_index, docs)
                forget_docs_content.append(doc_content)

        result_item["used_documents_count"] = len(target_docs_content)

        if not target_docs_content:
            print(f"No target documents (topk={topk}), output 'no influential texts'")
            result_item["test_answer"] = "no influential texts"
            total_processed += 1
            result_data[key] = result_item
            continue

        conversation_history = []

        # First query: forget irrelevant documents
        if forget_docs_content:
            doc_unlearning = '\n\n'.join(forget_docs_content)
            forget_prompt = create_forget_prompt(doc_unlearning)

            print(f"First query: forget {len(forget_docs_content)} documents")
            forget_response = llm_manager.query(forget_prompt, conversation_history=conversation_history)

            if forget_response is not None:
                conversation_history.append({'role': 'user', 'content': forget_prompt})
                conversation_history.append({'role': 'assistant', 'content': forget_response})
                print(f"First query completed")
            else:
                print(f"First query failed")
        else:
            print("No documents to forget, skip first query")

        # Second query: use target documents to answer question
        target_context = '\n\n'.join(target_docs_content)
        second_prompt = create_single_doc_prompt(target_context, question, baseline_answer)

        print(f"Second query: use {len(target_docs_content)} target documents")
        test_answer = llm_manager.query(second_prompt, conversation_history=conversation_history)

        if test_answer is not None:
            result_item["test_answer"] = test_answer.strip()
            total_processed += 1

            is_equiv = is_equivalent_answer(test_answer, first_answer_group_name)
            result_item["is_equivalent_to_target"] = is_equiv

            if is_equiv:
                equivalent_count += 1
                print(f"✓ Answer equivalent: '{test_answer}' ≈ '{first_answer_group_name}'")
            else:
                print(f"✗ Answer not equivalent: '{test_answer}' ≠ '{first_answer_group_name}'")

            is_equiv_baseline = is_equivalent_answer(test_answer, baseline_answer)
            result_item["is_equivalent_to_baseline"] = is_equiv_baseline

            if is_equiv_baseline:
                equivalent_baseline_count += 1
                print(f"  (Equivalent to baseline: '{test_answer}' ≈ '{baseline_answer}')")
            else:
                print(f"  (Not equivalent to baseline: '{test_answer}' ≠ '{baseline_answer}')")
        else:
            print(f"✗ API call failed")

        result_data[key] = result_item

    output_file = save_results(result_data, dataset_name, topk)

    print(f"\nDataset {dataset_name} processing completed! Statistics:")
    print(f"Total questions: {len(result_data)}")
    print(f"Successfully processed: {total_processed}")
    print(f"Target answer equivalent: {equivalent_count}")
    print(
        f"Target answer equivalent rate: {equivalent_count / total_processed * 100:.2f}%" if total_processed > 0 else "Target answer equivalent rate: 0%")
    print(f"Baseline answer equivalent: {equivalent_baseline_count}")
    print(
        f"Baseline answer equivalent rate: {equivalent_baseline_count / total_processed * 100:.2f}%" if total_processed > 0 else "Baseline answer equivalent rate: 0%")
    print(f"Results saved to: {output_file}")

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
    datasets = ["nq", "msmarco", "hotpotqa"]
    topk = 10
    test_config = "./model_configs/gpt3.5_config.json"

    print(f"Start batch processing {len(datasets)} datasets")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"TopK: {topk}")
    print(f"Model config: {test_config}")

    llm_manager = LLMManager(test_config)
    model_info = llm_manager.get_model_info()
    print(f"Successfully loaded model config: {model_info['model_name']}")

    all_results = []

    for dataset_name in datasets:
        result = process_single_dataset(dataset_name, topk, llm_manager)
        if result:
            all_results.append(result)

    print(f"\nAll datasets processing completed! Summary:")

    total_questions = 0
    total_processed = 0
    total_target_equivalent = 0
    total_baseline_equivalent = 0

    for result in all_results:
        print(f"\nDataset: {result['dataset']}")
        print(f"  Total questions: {result['total_questions']}")
        print(f"  Successfully processed: {result['processed']}")
        print(f"  Target answer equivalent: {result['target_equivalent']}")
        print(f"  Target answer equivalent rate: {result['target_equivalent'] / result['total_questions'] * 100:.2f}%")
        print(f"  Baseline answer equivalent: {result['baseline_equivalent']}")
        print(f"  Output file: {result['output_file']}")

        total_questions += result['total_questions']
        total_processed += result['processed']
        total_target_equivalent += result['target_equivalent']
        total_baseline_equivalent += result['baseline_equivalent']


if __name__ == '__main__':
    main()