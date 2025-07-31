import json
import os
import re
from difflib import SequenceMatcher
from preprocess.llm import get_llm_provider
from src.utils import load_json


def create_prompt(question, context):
    combined_context = "\n".join(context)
    prompt = f"""
    Question: {question}
    Here are the documents:{combined_context}
    Based on all the above documents, please provide a concise answer to the question.
    If the answer cannot be found in any of the documents, respond with 'I don't know'.
    Limit your answer to 2-4 words maximum.
    Answer:"""
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


def process_dataset(dataset, llm_provider, topk=70):
    input_file = f"results/baseline/{dataset}-gpt3-70.json"
    output_file = f"results/unlearning/coarse/{dataset}-gpt3-{topk}-5.json"

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

        # Use last 5 documents from top topk documents
        topk_docs = docs[:topk]
        last_five_docs = topk_docs[-5:]
        prompt = create_prompt(question, last_five_docs)
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
    print(f"Non-equivalent rate: {non_equivalent_rate:.2%}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    datasets = ["hotpotqa", "msmarco", "nq"]
    config_file = "model_configs/gpt3.5_config.json"
    llm_provider = get_llm_provider(config_file)

    topk = 70  # Can be changed as needed

    for dataset in datasets:
        print(f"\nProcessing {dataset} with topk={topk}")
        process_dataset(dataset, llm_provider, topk)


if __name__ == "__main__":
    main()