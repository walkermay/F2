import json
import os
import numpy as np
import re
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
from src.utils import load_beir_datasets, save_json
from preprocess.llm import LLMManager

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class AnswerAggregator:
    """Handle aggregation and comparison of multiple answers"""

    def __init__(self):
        # Use sentence-transformers model for vectorization
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def chunk_text(self, text):
        """Split text into sentences"""
        if not text or not text.strip():
            return []

        # Use NLTK for sentence splitting
        try:
            sentences = sent_tokenize(text.strip())
        except:
            # If NLTK fails, use simple splitting method
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        # Filter out sentences that are too short
        sentences = [s for s in sentences if len(s.split()) >= 2]
        return sentences

    def vectorize_sentences(self, sentences):
        """Convert sentence list to vectors"""
        if not sentences:
            return np.array([])

        embeddings = self.embedding_model.encode(sentences)
        return embeddings

    def calculate_sentence_scores(self, all_sentences, all_embeddings, answer_indices):
        """
        Calculate score for each sentence
        answer_indices: [(start_idx, end_idx), ...] representing sentence index ranges for each answer
        """
        sentence_scores = []

        for i, sentence in enumerate(all_sentences):
            # Determine which answer the current sentence belongs to
            current_answer_idx = None
            for ans_idx, (start, end) in enumerate(answer_indices):
                if start <= i < end:
                    current_answer_idx = ans_idx
                    break

            if current_answer_idx is None:
                sentence_scores.append(0.0)
                continue

            # Calculate maximum similarity with sentences from other answers
            max_similarities = []

            for ans_idx, (start, end) in enumerate(answer_indices):
                if ans_idx == current_answer_idx:
                    continue  # Skip the same answer

                # Calculate similarity with all sentences in this answer
                answer_embeddings = all_embeddings[start:end]
                if len(answer_embeddings) == 0:
                    max_similarities.append(0.0)
                    continue

                current_embedding = all_embeddings[i:i + 1]
                similarities = cosine_similarity(current_embedding, answer_embeddings)[0]
                max_similarities.append(np.max(similarities))

            # Sentence score is the sum of maximum similarities with other answers
            score = sum(max_similarities)
            sentence_scores.append(score)

        return sentence_scores

    def remove_duplicate_sentences(self, sentences, embeddings, scores, similarity_threshold=0.8):
        """Remove semantically duplicate sentences, keep the one with highest score"""
        if len(sentences) <= 1:
            return sentences, embeddings, scores

        # Calculate similarity matrix between all sentences
        similarity_matrix = cosine_similarity(embeddings)

        # Mark sentences to keep
        keep_indices = []
        removed_indices = set()

        # Sort by score in descending order
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        for i in sorted_indices:
            if i in removed_indices:
                continue

            keep_indices.append(i)

            # Mark other sentences similar to the current sentence as removed
            for j in range(len(sentences)):
                if j != i and j not in removed_indices:
                    if similarity_matrix[i][j] >= similarity_threshold:
                        removed_indices.add(j)

        # Return kept sentences
        keep_indices.sort()  # Maintain original order
        filtered_sentences = [sentences[i] for i in keep_indices]
        filtered_embeddings = embeddings[keep_indices]
        filtered_scores = [scores[i] for i in keep_indices]

        return filtered_sentences, filtered_embeddings, filtered_scores

    def select_top_sentences(self, sentences, scores, max_sentences=3):
        """Select top N sentences with highest scores"""
        if len(sentences) <= max_sentences:
            return sentences

        # Sort by score
        scored_sentences = list(zip(sentences, scores))
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        return [sent for sent, score in scored_sentences[:max_sentences]]

    def aggregate_answers(self, answer1, answer2, answer3):
        """
        Aggregate three answers and return the most likely answer
        """
        # Preprocess answers
        answers = [answer1, answer2, answer3]
        cleaned_answers = [self.clean_answer(ans) for ans in answers]

        # Check for empty answers
        valid_answers = [ans for ans in cleaned_answers if ans and ans.strip()]
        if len(valid_answers) == 0:
            return "I don't know"

        if len(valid_answers) == 1:
            return valid_answers[0]

        # Check for identical answers
        answer_counts = defaultdict(int)
        for ans in valid_answers:
            answer_counts[ans.lower().strip()] += 1

        # If any answer appears 2 or more times, return it directly
        for ans, count in answer_counts.items():
            if count >= 2:
                # Return original format answer
                for original_ans in valid_answers:
                    if original_ans.lower().strip() == ans:
                        return original_ans

        # Perform semantic aggregation
        return self.semantic_aggregation(valid_answers)

    def semantic_aggregation(self, answers):
        """Use semantic analysis for answer aggregation"""
        # 1. Text chunking
        all_sentences = []
        answer_indices = []
        current_start = 0

        for answer in answers:
            sentences = self.chunk_text(answer)
            all_sentences.extend(sentences)
            answer_indices.append((current_start, current_start + len(sentences)))
            current_start += len(sentences)

        if not all_sentences:
            return "I don't know"

        if len(all_sentences) == 1:
            return all_sentences[0]

        # 2. Vectorization
        all_embeddings = self.vectorize_sentences(all_sentences)

        # 3. Calculate sentence scores
        sentence_scores = self.calculate_sentence_scores(all_sentences, all_embeddings, answer_indices)

        # 4. Remove duplicates
        filtered_sentences, filtered_embeddings, filtered_scores = self.remove_duplicate_sentences(
            all_sentences, all_embeddings, sentence_scores
        )

        # 5. Select highest scoring sentences
        top_sentences = self.select_top_sentences(filtered_sentences, filtered_scores, max_sentences=2)

        # 6. Generate final answer
        if not top_sentences:
            return answers[0]  # If no high-scoring sentences, return first answer

        final_answer = ' '.join(top_sentences)
        return final_answer.strip()

    def clean_answer(self, answer):
        """Clean answer format"""
        if not answer:
            return ""

        answer = answer.strip()

        # Remove common invalid response patterns
        invalid_patterns = [
            r"i don't know",
            r"i do not know",
            r"don't know",
            r"cannot determine",
            r"unclear",
            r"uncertain",
            r"not mentioned",
            r"not specified"
        ]

        answer_lower = answer.lower()
        for pattern in invalid_patterns:
            if re.search(pattern, answer_lower):
                return ""

        # Remove quotes and extra punctuation
        answer = re.sub(r'^["\'"]*|["\'"]*$', '', answer)
        answer = re.sub(r'[.。!！?？]*$', '', answer)

        return answer.strip()


def is_invalid_answer(answer):
    """Check if answer is invalid (I don't know or empty)"""
    if not answer or not answer.strip():
        return True

    answer_lower = answer.lower().strip()
    invalid_patterns = ["i don't know", "i do not know", "don't know", "unknown", "no answer", "not specified",
                        "cannot determine", "unclear", "uncertain", "not mentioned"]

    for pattern in invalid_patterns:
        if pattern in answer_lower:
            return True

    return False


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


def save_results(results, dataset, topk):
    """Save test results"""
    save_path = "results/baseline"
    os.makedirs(save_path, exist_ok=True)
    # Format
    # gpt3 == gpt-3.5-turbo
    # gpt41 == gpt-4.1-mini
    # gpt4o == gpt-4o-mini
    output_file = os.path.join(save_path, f'{dataset}-gpt3-{topk}.json')
    save_json(results, output_file)
    print(f"Results saved to: {output_file}")
    return output_file


def create_normal_prompt(context, question):
    """Create normal prompt"""
    baseline_prompt = f"""
    Question: {question} \n
    Context:{context} \n
    Based on the above context, provide a short answer to the question (2-4 words). \n
    If no answer can be found in the context, respond with "I don't know". \n
    """
    return baseline_prompt


def determine_baseline_answer(context, question, llm_manager, aggregator):
    """
    Determine baseline_answer through three repeated queries
    """
    print(f"  Performing three queries to determine baseline answer...")

    # Create prompt
    prompt = create_normal_prompt(context, question)

    # Perform three queries
    answers = []
    for i in range(3):
        print(f"    Query {i + 1}...")
        answer = llm_manager.query(prompt)
        if answer is not None:
            answers.append(answer.strip())
            print(f"    Answer {i + 1}: '{answer.strip()}'")
        else:
            print(f"    Query {i + 1} failed")
            answers.append("")

    # Use aggregator to determine final answer
    if len(answers) == 3:
        final_answer = aggregator.aggregate_answers(answers[0], answers[1], answers[2])
        print(f"  Aggregated baseline answer: '{final_answer}'")
        return final_answer, answers
    else:
        print(f"  Warning: Query failed, using first valid answer")
        valid_answers = [ans for ans in answers if ans]
        return valid_answers[0] if valid_answers else "I don't know", answers


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


def process_single_dataset(dataset_name, topk, llm_manager, eval_model_code="contriever", split="test"):
    """Process single dataset"""
    print(f"\n{'=' * 60}")
    print(f"Starting to process dataset: {dataset_name}")
    print(f"{'=' * 60}")

    # Initialize answer aggregator
    aggregator = AnswerAggregator()

    # Get dataset configuration
    dataset_config = get_dataset_config(dataset_name)
    if not dataset_config:
        print(f"Error: Unknown dataset {dataset_name}")
        return None

    eval_dataset = dataset_config['eval_dataset']
    input_file = dataset_config['input_file']

    # Load BEIR dataset
    try:
        corpus, _, _ = load_beir_datasets(eval_dataset, split)
        print(f"Successfully loaded BEIR dataset: {eval_dataset}")
    except Exception as e:
        print(f"Error loading BEIR dataset: {e}")
        return None

    # Load question data
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        print(f"Successfully loaded question file: {input_file}")
    except Exception as e:
        print(f"Error loading question file: {e}")
        return None

    # Load BEIR results
    beir_results_file = f"results/beir_results/{eval_dataset}-{eval_model_code}.json"
    try:
        with open(beir_results_file, 'r') as f:
            beir_results = json.load(f)
        print(f"Successfully loaded BEIR results: {beir_results_file}")
    except Exception as e:
        print(f"Error loading BEIR results: {e}")
        return None

    print(f"Total {len(existing_data)} questions to process")

    # Statistics
    total_processed = 0
    skipped_questions = 0

    # Create new result dictionary
    result_data = {}

    for idx, (question_id, question_data) in enumerate(existing_data.items()):
        print(f"\n--- {question_id} (Question {idx + 1}) ---")

        # Extract basic information
        question = question_data.get('question', '')

        # Initialize result entry
        result_item = {
            "id": question_id,
            "question": question,
            "baseline_answer": "",
            "three_answers": [],
            "used_documents_count": 0,
            "docs": []
        }

        # Check if question exists in BEIR results
        if question_id not in beir_results:
            print(f"Warning: Question ID '{question_id}' not found in BEIR results")
            skipped_questions += 1
            continue

        # Get top ranked documents
        retrieved_docs = sorted(beir_results[question_id].items(),
                                key=lambda item: item[1], reverse=True)[:topk]

        docs_info = [(doc_id, score) for doc_id, score in retrieved_docs]

        if not docs_info:
            print(f"Warning: No documents available")
            skipped_questions += 1
            continue

        # Extract document contents
        docs = extract_document_contents(docs_info, corpus)
        result_item["docs"] = docs
        result_item["used_documents_count"] = len(docs)

        # Use first 5 documents to create context
        context_docs = docs[:5]
        context = '\n\n'.join(context_docs)

        # Determine baseline answer through three queries
        try:
            baseline_answer, three_answers = determine_baseline_answer(context, question, llm_manager, aggregator)

            # Check if baseline answer is invalid
            if is_invalid_answer(baseline_answer):
                print(f"  Skipping question due to invalid baseline answer: '{baseline_answer}'")
                skipped_questions += 1
                continue

            result_item["baseline_answer"] = baseline_answer
            result_item["three_answers"] = three_answers
            total_processed += 1

        except Exception as e:
            print(f"Error determining baseline answer: {e}")
            skipped_questions += 1
            continue

        # Add result to final data
        result_data[question_id] = result_item

    # Save results
    output_file = save_results(result_data, dataset_name, topk)

    # Print statistics
    print(f"\n{'=' * 50}")
    print(f"Dataset {dataset_name} processing completed! Statistics:")
    print(f"Total questions: {len(existing_data)}")
    print(f"Questions in result: {len(result_data)}")
    print(f"Successfully processed: {total_processed}")
    print(f"Skipped questions: {skipped_questions}")
    print(f"\nResults saved to: {output_file}")

    return {
        'dataset': dataset_name,
        'total_questions': len(existing_data),
        'processed_questions': len(result_data),
        'successful_processed': total_processed,
        'skipped_questions': skipped_questions,
        'output_file': output_file
    }


def main():
    """Main function - Focus on determining baseline answers"""
    # Set parameters directly in code
    datasets = ["hotpotqa", "msmarco", "nq"]  # List of datasets to process
    topk = 70  # Keep consistent with original (though now only use first 5 documents)
    test_config = "./model_configs/gpt3.5_config.json"  # Model configuration file path

    print(f"Starting batch processing of {len(datasets)} datasets")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Processing strategy: Determine baseline answer through three queries")
    print(f"Model configuration: {test_config}")

    try:
        # Initialize LLM manager
        llm_manager = LLMManager(test_config)
        model_info = llm_manager.get_model_info()
        print(f"Successfully loaded model configuration: {model_info['model_name']}")

    except FileNotFoundError:
        print(f"Error: Configuration file {test_config} does not exist")
        return
    except Exception as e:
        print(f"Error: Initialization failed: {e}")
        return

    # Store processing results for all datasets
    all_results = []

    # Process each dataset one by one
    for dataset_name in datasets:
        try:
            result = process_single_dataset(dataset_name, topk, llm_manager)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            continue

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"All datasets processing completed! Summary:")
    print(f"{'=' * 80}")

    total_questions = 0
    total_processed_questions = 0
    total_successful_processed = 0
    total_skipped = 0

    for result in all_results:
        print(f"\nDataset: {result['dataset']}")
        print(f"  Total questions: {result['total_questions']}")
        print(f"  Processed questions: {result['processed_questions']}")
        print(f"  Successfully processed: {result['successful_processed']}")
        print(f"  Skipped questions: {result['skipped_questions']}")
        print(f"  Output file: {result['output_file']}")

        total_questions += result['total_questions']
        total_processed_questions += result['processed_questions']
        total_successful_processed += result['successful_processed']
        total_skipped += result['skipped_questions']

    print(f"\nTotal:")
    print(f"  Total questions: {total_questions}")
    print(f"  Total processed questions: {total_processed_questions}")
    print(f"  Total successfully processed: {total_successful_processed}")
    print(f"  Total skipped questions: {total_skipped}")


if __name__ == '__main__':
    main()