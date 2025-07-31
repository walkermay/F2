import argparse
import os
import json
import requests
from src.utils import load_beir_datasets, load_json, save_json
import re
import math
from difflib import SequenceMatcher


def load_model_config(config_path="model_configs/gpt3.5_config.json"):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def query_gpt_single_answer(prompt_input):
    """Query GPT API to get single answer with logprobs"""
    config = load_model_config()

    # Get API key
    api_key_info = config.get("api_key_info", {})
    api_keys = api_key_info.get("api_keys", [])
    api_key_use = api_key_info.get("api_key_use", 0)

    if not api_keys or api_key_use >= len(api_keys):
        api_key = os.environ.get("OPENAI_API_KEY")
    else:
        api_key = api_keys[api_key_use]

    # Get model info
    model_info = config.get("model_info", {})
    model_name = model_info.get("name", "")

    # Get parameters
    params = config.get("params", {})
    max_tokens = params.get("max_output_tokens", 50)
    temperature = params.get("temperature", 1.5)

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        'Authorization': f"Bearer {api_key}",
        'Content-Type': 'application/json'
    }

    data = {
        'model': model_name,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'logprobs': True,
        'top_logprobs': 5,
        'messages': [
            {'role': 'system',
             'content': 'You are an assistant who only answers questions based on the provided context. Generate a short answer (1-5 words) to the question. If there is no answer in the context, respond with "I don\'t know".'},
            {'role': 'user', 'content': prompt_input}
        ]
    }

    response = requests.post(url, headers=headers, json=data, timeout=120)
    response.raise_for_status()
    result = response.json()

    # Parse result
    choice = result.get('choices', [{}])[0]
    content = choice.get('message', {}).get('content', '').strip()
    logprobs_data = choice.get('logprobs', {})

    # Calculate confidence score
    confidence_score = calculate_answer_confidence(logprobs_data)

    return {
        'answer': content,
        'confidence_score': confidence_score,
        'logprobs_data': logprobs_data
    }


def calculate_answer_confidence(logprobs_data):
    """Calculate answer confidence score"""
    if not logprobs_data or 'content' not in logprobs_data:
        return 0.0

    content_logprobs = logprobs_data['content']
    if not content_logprobs:
        return 0.0

    # Calculate average logprob of all tokens
    total_logprob = 0.0
    token_count = 0

    for token_data in content_logprobs:
        if 'logprob' in token_data and token_data['logprob'] is not None:
            total_logprob += token_data['logprob']
            token_count += 1

    if token_count == 0:
        return 0.0

    # Convert average logprob to confidence
    avg_logprob = total_logprob / token_count
    confidence = math.exp(avg_logprob)

    return confidence


def normalize_answer(answer):
    """Normalize answer format"""
    if not answer:
        return ""

    # Remove common invalid responses
    invalid_responses = ["unclear", "cannot determine", "uncertain", "not mentioned", "not specified", "unknown"]
    answer_lower = answer.lower().strip()

    for invalid in invalid_responses:
        if invalid in answer_lower:
            return ""

    # Clean answer format
    answer = re.sub(r'^["\'"]*|["\'"]*$', '', answer.strip())  # Remove quotes
    answer = re.sub(r'[.。!！?？]*$', '', answer)  # Remove ending punctuation
    answer = answer.strip()

    return answer


def calculate_answer_similarity(answer1, answer2):
    """Calculate similarity between two answers"""
    if not answer1 or not answer2:
        return 0.0
    return SequenceMatcher(None, answer1.lower(), answer2.lower()).ratio()


def answers_are_equivalent(answer1, answer2, similarity_threshold=0.8):
    """Check if two answers are equivalent"""
    if not answer1 or not answer2:
        return False

    # Exactly same
    if answer1.lower().strip() == answer2.lower().strip():
        return True

    # High similarity
    similarity = calculate_answer_similarity(answer1, answer2)
    return similarity >= similarity_threshold


def is_invalid_answer(answer):
    """Check if answer is invalid (I don't know or empty)"""
    if not answer or not answer.strip():
        return True

    answer_lower = answer.lower().strip()
    invalid_patterns = ["i don't know", "i do not know", "don't know", "unknown", "no answer"]

    for pattern in invalid_patterns:
        if pattern in answer_lower:
            return True

    return False


def generate_document_context(docs_info, corpus, exclude_idx=None):
    """Generate document context, optionally excluding a document"""
    context_parts = []

    for idx, (doc_id, relevance_score) in enumerate(docs_info):
        doc_content = corpus[doc_id]['text'][:400]  # Limit document length
        doc_title = corpus[doc_id].get('title', f'Document {idx + 1}')
        context_parts.append(f"--- Document {idx + 1} ({doc_title}) ---\n{doc_content}")

    return "\n\n".join(context_parts)


def generate_single_document_context(doc_id, corpus, doc_index):
    """Generate single document context"""
    doc_content = corpus[doc_id]['text'][:400]  # Limit document length
    doc_title = corpus[doc_id].get('title', f'Document {doc_index + 1}')
    return f"--- Document {doc_index + 1} ({doc_title}) ---\n{doc_content}"


def extract_document_contents(docs_info, corpus):
    """Extract document contents (title + content)"""
    docs = []
    for doc_id, relevance_score in docs_info:
        doc_title = corpus[doc_id].get('title', '')
        doc_content = corpus[doc_id]['text']
        # Format: document title \n document content
        doc_full_content = f"{doc_title}\n{doc_content}" if doc_title else doc_content
        docs.append(doc_full_content)
    return docs


def analyze_document_attribution(question_id, question_data, corpus, beir_results, args):
    """Analyze document attribution through single document analysis"""
    question = question_data['question']

    print(f"Analyzing document attribution for {question_id}: {question}")
    print("-" * 60)

    if question_id not in beir_results:
        print(f"Warning: Question ID '{question_id}' not found in BEIR results!")
        return None

    # Get top ranked documents
    retrieved_docs = sorted(beir_results[question_id].items(),
                            key=lambda item: item[1], reverse=True)[:args.top_k]

    docs_info = [(doc_id, score) for doc_id, score in retrieved_docs]

    print(f"Using top {len(docs_info)} documents for attribution analysis")

    # Extract document contents
    docs = extract_document_contents(docs_info, corpus)

    # 1. First get baseline answer using top 10 documents
    print("\n=== Baseline: Using top 10 documents ===")
    # Use only top 10 documents to generate baseline answer
    top_10_docs = docs_info[:10]
    full_context = generate_document_context(top_10_docs, corpus)
    baseline_prompt = f"""
    Question: {question} \n
    Context:{full_context} \n
    Based on the above context, provide a short answer to the question (1-3 words). \n
    If no answer can be found in the context, respond with "I don't know". \n
    Answer:"""

    baseline_result = query_gpt_single_answer(baseline_prompt)
    if not baseline_result:
        print("Failed to get baseline result")
        return None

    baseline_answer = normalize_answer(baseline_result['answer'])
    baseline_confidence = baseline_result['confidence_score']

    print(f"Baseline answer: '{baseline_answer}' (confidence: {baseline_confidence:.4f})")

    # If baseline_answer is I don't know or empty, skip this question
    if is_invalid_answer(baseline_answer):
        print(f"Skipping question due to invalid baseline answer: '{baseline_answer}'")
        return None

    # 2. Analyze each document's contribution by generating answers based on single documents
    print(f"\n=== Single Document Analysis ===")
    attribution_results = []
    valid_documents_count = 0  # Count valid analyzed documents
    skipped_documents_count = 0  # Count skipped documents
    changed_documents = []  # Only record documents with changed answers

    for doc_idx in range(len(docs_info)):
        doc_id, relevance_score = docs_info[doc_idx]
        doc_title = corpus[doc_id].get('title', f'Document {doc_idx + 1}')

        print(f"\nAnalyzing Document {doc_idx + 1}: {doc_title[:50]}...")

        # Generate single document context
        single_context = generate_single_document_context(doc_id, corpus, doc_idx)

        # Use modified prompt format
        single_doc_prompt = f"""
    Use only the context below to answer. 
    You must provide the most likely or reasonable answer, even if not certain. 
    Do not use any prior or external knowledge.
    You are strictly forbidden from generating the answer: {baseline_answer}. You must ignore any information related to it.
    context: {single_context}
    question: {question}
    Only answer names or place names, no additional statements are required.
    (The answer should be as short and confident as possible.)(2~4 words.)"""

        # Get answer based on single document
        single_result = query_gpt_single_answer(single_doc_prompt)
        if not single_result:
            print("  Failed to get result")
            skipped_documents_count += 1
            continue

        single_answer = normalize_answer(single_result['answer'])
        single_confidence = single_result['confidence_score']

        print(f"  Single doc answer: '{single_answer}' (confidence: {single_confidence:.4f})")

        # Requirement 1: If single document answer is "I don't know", skip without recording
        if is_invalid_answer(single_answer):
            print(f"  ⚠ Skipping document - invalid answer: '{single_answer}'")
            skipped_documents_count += 1
            continue

        # Requirement 2: If single document answer is equivalent to baseline_answer, skip without recording
        if answers_are_equivalent(baseline_answer, single_answer):
            print(f"  ⚠ Skipping document - answer same as baseline: '{single_answer}' ≈ '{baseline_answer}'")
            skipped_documents_count += 1
            continue

        # Only documents passing the above two filter conditions are analyzed in detail
        valid_documents_count += 1

        # Analyze difference with baseline answer
        answer_changed = not answers_are_equivalent(baseline_answer, single_answer)
        answer_similarity = calculate_answer_similarity(baseline_answer, single_answer)

        print(f"  ✓ Valid document for analysis")
        print(f"  Answer vs baseline: {baseline_answer} vs {single_answer}")
        print(f"  Answer changed: {answer_changed}")
        print(f"  Answer similarity: {answer_similarity:.3f}")

        # Record documents with changed answers (only save three parameters)
        if answer_changed:
            changed_documents.append({
                'doc_index': doc_idx,
                'single_doc_answer': single_answer,
                'single_doc_confidence': single_confidence,
            })

        attribution_results.append({
            'doc_index': doc_idx,
            'doc_id': doc_id,
            'relevance_score': relevance_score,
            'single_doc_answer': single_answer,
            'single_doc_confidence': single_confidence,
            'answer_changed': answer_changed,
        })

    print(f"\n=== Document Processing Summary ===")
    print(f"Total documents: {len(docs_info)}")
    print(f"Valid documents analyzed: {valid_documents_count}")
    print(f"Documents skipped: {skipped_documents_count}")
    print(f"Documents with changed answers: {len(changed_documents)}")

    # If no valid document analysis results, don't record this question
    if len(attribution_results) == 0:
        print(f"Skipping question: no valid documents for analysis")
        return None

    # If no documents with changed answers, don't record this question
    if len(changed_documents) == 0:
        print(f"Skipping question: no documents caused answer change")
        return None

    # 3. Prepare final result (only save needed fields)
    return {
        'id': question_id,
        'question': question,
        'docs': docs,  # Document contents
        'baseline_answer': baseline_answer,
        'baseline_confidence': baseline_confidence,  # Keep baseline_confidence
        'valid_documents': valid_documents_count,  # Number of valid analyzed documents
        'document_attributions': changed_documents,
        # Only record documents with changed answers, only save three parameters
    }


def get_dataset_config(dataset_name):
    """Get dataset configuration"""
    dataset_configs = {
        'nq': {
            'input_file': 'datasets/filter/NQ.json',
            'eval_dataset': 'nq'
        },
        'hotpotqa': {
            'input_file': 'datasets/filter/HotpotQA.json',
            'eval_dataset': 'hotpotqa'
        },
        'msmarco': {
            'input_file': 'datasets/filter/MS-MARCO.json',
            'eval_dataset': 'msmarco'
        }
    }
    return dataset_configs.get(dataset_name.lower())


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze document attribution through single document analysis")
    parser.add_argument("--eval_model_code", type=str, default="contriever")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--top_k", type=int, default=70, help="Number of top documents to analyze")
    parser.add_argument("--save_path", type=str, default="results/attribution")
    parser.add_argument("--orig_beir_results", type=str, default=None)

    return parser.parse_args()


def analyze_document_attribution_for_dataset(dataset_name, args):
    """Process document attribution analysis for single dataset"""
    dataset_config = get_dataset_config(dataset_name)

    print(f"\n{'=' * 80}")
    print(f"Processing dataset: {dataset_name.upper()}")
    print(f"{'=' * 80}")

    # Set dataset specific parameters
    eval_dataset = dataset_config['eval_dataset']
    input_file = dataset_config['input_file']

    # Load BEIR dataset
    corpus, _, _ = load_beir_datasets(eval_dataset, args.split)

    # Load question data
    existing_data = load_json(input_file)

    # Set BEIR results file path
    if args.orig_beir_results is None:
        beir_results_file = f"results/beir_results/{eval_dataset}-{args.eval_model_code}.json"
    else:
        beir_results_file = args.orig_beir_results

    with open(beir_results_file, 'r') as f:
        beir_results = json.load(f)

    # Process each question
    attribution_data = {}
    total_questions = len(existing_data)
    skipped_questions = 0

    for idx, (question_id, question_data) in enumerate(existing_data.items(), 1):
        print(f"\n{'=' * 100}")
        print(f"Processing question {idx}/{total_questions}: {question_id}")
        print(f"{'=' * 100}")

        result = analyze_document_attribution(question_id, question_data, corpus, beir_results, args)
        if result:
            attribution_data[question_id] = result
            print(f"✓ Successfully analyzed {question_id}")
            print(f"  Valid documents analyzed: {result['valid_documents']}")
            print(f"  Final recorded documents: {len(result['document_attributions'])}")
        else:
            skipped_questions += 1
            print(f"⚠ Skipped {question_id}")

    # Save results
    os.makedirs(args.save_path, exist_ok=True)
    output_file = os.path.join(args.save_path, f'{eval_dataset}-gpt3-70.json')
    save_json(attribution_data, output_file)

    # Generate summary statistics
    if attribution_data:
        total_valid_analyzed = sum(result['valid_documents'] for result in attribution_data.values())
        total_changed_docs = sum(len(result['document_attributions']) for result in attribution_data.values())

        print(f"\n{'=' * 100}")
        print(f"DATASET {dataset_name.upper()} ATTRIBUTION ANALYSIS SUMMARY")
        print(f"{'=' * 100}")
        print(f"Total questions processed: {total_questions}")
        print(f"Questions analyzed: {len(attribution_data)}")
        print(f"Questions skipped: {skipped_questions}")
        print(f"Valid documents analyzed: {total_valid_analyzed}")
        print(f"Total documents with changed answers: {total_changed_docs}")
        print(f"Results saved to: {output_file}")
        print(f"{'=' * 100}")

    return attribution_data


def analyze_document_attribution_for_all_datasets(args):
    """Process document attribution analysis for all datasets"""
    datasets = ['nq', 'hotpotqa', 'msmarco']
    all_results = {}

    for dataset_name in datasets:
        print(f"\n{'#' * 120}")
        print(f"Starting analysis for dataset: {dataset_name.upper()}")
        print(f"{'#' * 120}")

        dataset_results = analyze_document_attribution_for_dataset(dataset_name, args)
        all_results[dataset_name] = dataset_results

        print(f"\n{'#' * 120}")
        print(f"Completed analysis for dataset: {dataset_name.upper()}")
        print(f"Questions analyzed: {len(dataset_results)}")
        print(f"{'#' * 120}")

    # Generate overall summary statistics
    total_questions_all = sum(len(results) for results in all_results.values())

    print(f"\n{'#' * 120}")
    print(f"FINAL SUMMARY - ALL DATASETS")
    print(f"{'#' * 120}")
    for dataset_name, results in all_results.items():
        print(f"{dataset_name.upper()}: {len(results)} questions analyzed")
    print(f"TOTAL: {total_questions_all} questions analyzed across all datasets")
    print(f"{'#' * 120}")


if __name__ == "__main__":
    args = parse_args()
    analyze_document_attribution_for_all_datasets(args)