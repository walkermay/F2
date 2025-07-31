import json
import os
import re
from difflib import SequenceMatcher
from preprocess.llm import get_llm_provider
from src.utils import load_json


def create_prompt(question, doc_unlearning, doc_no_unlearning):
    doc_unlearning_text = "\n".join(doc_unlearning)
    doc_no_unlearning_text = "\n".join(doc_no_unlearning)
    prompt = f"""
    ### FORGET THIS ###
    Do not read or use the following. It must be completely ignored.
    {doc_unlearning_text}
    ### END FORGET ###

    ### QUESTION ###
    {question}

    If you don't find the answer above, reply: I don't know.
    Keep your answer short (2-3 words only).
    """
    return prompt


def normalize_answer(answer):
    if not answer:
        return ""

    invalid_responses = ["unclear", "cannot determine", "uncertain", "not mentioned", "not specified", "unknown"]
    answer_lower = answer.lower().strip()

    for invalid in invalid_responses:
        if invalid in answer_lower:
            return ""

    answer = re.sub(r'^["\'"]*|["\'"]*$', '', answer.strip())
    answer = re.sub(r'[.。!！?？]*$', '', answer)
    answer = answer.strip()

    return answer


def answers_are_equivalent(answer1, answer2, similarity_threshold=0.8):
    if not answer1 or not answer2:
        return False

    if answer1.lower().strip() == answer2.lower().strip():
        return True

    similarity = SequenceMatcher(None, answer1.lower(), answer2.lower()).ratio()
    return similarity >= similarity_threshold


def process_dataset(dataset, llm_provider, topk=70, unlearning_num=65):
    input_file = f"results/baseline/{dataset}-gpt3-70.json"
    output_file = f"results/unlearning/fine/{dataset}-gpt3-{topk}-5.json"

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Processing {dataset}: {len(data)} questions")

    results = {}
    equivalent_count = 0
    total_count = 0

    for query_id, query_info in data.items():
        question = query_info['question']
        docs = query_info.get('docs', [])
        baseline_answer = query_info.get('baseline_answer', '')

        if len(docs) < topk:
            continue

        # Use unlearning prompt method
        doc_unlearning = docs[:unlearning_num]
        doc_no_unlearning = docs[unlearning_num:unlearning_num + 5]

        prompt = create_prompt(question, doc_unlearning, doc_no_unlearning)
        delete_answer = llm_provider.query(prompt, return_json=False)

        # Check equivalence
        normalized_baseline = normalize_answer(baseline_answer)
        normalized_delete = normalize_answer(delete_answer)
        is_equivalent = answers_are_equivalent(normalized_baseline, normalized_delete)

        total_count += 1
        if is_equivalent:
            equivalent_count += 1

        results[query_id] = {
            "question": question,
            "baseline_answer": baseline_answer,
            "delete_answer": delete_answer,
            "is_equivalent": is_equivalent
        }

    # Calculate non-equivalent rate
    non_equivalent_rate = (total_count - equivalent_count) / total_count if total_count > 0 else 0

    print(f"Total: {total_count}")
    print(f"Unlearning rate: {non_equivalent_rate:.2%}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    datasets = ["hotpotqa", "msmarco", "nq"]
    config_file = "model_configs/gpt3.5_config.json"
    llm_provider = get_llm_provider(config_file)

    topk = 70
    unlearning_num = topk - 5

    for dataset in datasets:
        print(f"\nProcessing {dataset} with topk={topk}")
        process_dataset(dataset, llm_provider, topk, unlearning_num)


if __name__ == "__main__":
    main()