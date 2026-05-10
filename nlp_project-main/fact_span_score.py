import argparse
import json
import os
import re
from typing import Dict, List, Set, Tuple


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASETS = ["nq", "sciq", "simple_questions_wiki", "truthfulQA"]
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "generally",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "kind",
    "last",
    "least",
    "less",
    "made",
    "make",
    "makes",
    "many",
    "may",
    "might",
    "more",
    "most",
    "much",
    "no",
    "not",
    "of",
    "on",
    "or",
    "she",
    "should",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "through",
    "to",
    "type",
    "under",
    "up",
    "used",
    "using",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "yes",
    "you",
    "your",
}
ALIASES = {
    "u.s": "united states",
    "us": "united states",
    "usa": "united states",
    "u.s.a": "united states",
    "uk": "united kingdom",
    "u.k": "united kingdom",
    "nyc": "new york city",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build an optimized span-aware factual score from BGE-M3 token metrics. "
            "This adds raw-answer term, entity/proper-noun, and number/date mismatch "
            "penalties on top of fact_sensitive_similarity."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument(
        "--results-root",
        default=os.path.join(PROJECT_ROOT, "updated_results"),
        help="Root folder containing per-dataset result folders.",
    )
    parser.add_argument(
        "--input-metrics-file",
        default="embedding_metrics_bge_m3_fact.json",
        help="Input metrics from bge_fact_token_score.py.",
    )
    parser.add_argument(
        "--bge-metrics-file",
        default="embedding_metrics_bge_m3.json",
        help="Original BGE metrics file, used for comparison fields.",
    )
    parser.add_argument(
        "--output-file",
        default="embedding_metrics_bge_m3_fact_span.json",
        help="Output metrics JSON filename inside each dataset folder.",
    )
    parser.add_argument(
        "--explain-file",
        default="bge_fact_span_explanations.json",
        help="Output qualitative examples filename inside each dataset folder.",
    )
    parser.add_argument("--explain-limit", type=int, default=30)
    parser.add_argument(
        "--term-penalty-weight",
        type=float,
        default=0.06,
        help="Penalty weight for answer/content terms absent from reference.",
    )
    parser.add_argument(
        "--entity-penalty-weight",
        type=float,
        default=0.12,
        help="Penalty weight for proper noun/entity-like terms absent from reference.",
    )
    parser.add_argument(
        "--number-penalty-weight",
        type=float,
        default=0.09,
        help="Penalty weight for numbers/dates absent from reference.",
    )
    return parser.parse_args()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


def flatten_prediction_records(prediction_json) -> List[Dict]:
    rows = []
    for item in prediction_json:
        rows.append(item[0] if isinstance(item, list) else item)
    return rows


def word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[,.]\d+)*(?:st|nd|rd|th)?", text or "")


def normalize_token(token: str) -> str:
    token = token.lower().strip("'.,;:!?()[]{}\"")
    token = ALIASES.get(token, token)
    if token.endswith("'s"):
        token = token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token


def number_set(text: str) -> Set[str]:
    return set(re.findall(r"\b\d{1,4}(?:[,.]\d+)*(?:st|nd|rd|th)?\b", text or "", re.I))


def content_terms(text: str, question: str = "") -> Set[str]:
    question_terms = {normalize_token(token) for token in word_tokens(question)}
    terms = set()
    for token in word_tokens(text):
        normalized = normalize_token(token)
        if not normalized:
            continue
        for part in normalized.split():
            if not part or part in STOPWORDS or part in question_terms:
                continue
            if len(part) <= 2 and not part.isdigit():
                continue
            terms.add(part)
    return terms


def proper_or_number_terms(text: str, question: str = "") -> Set[str]:
    question_terms = {normalize_token(token) for token in word_tokens(question)}
    terms = set()
    for token in word_tokens(text):
        normalized = normalize_token(token)
        if not normalized:
            continue
        is_entity_like = (
            bool(re.match(r"^[A-Z][A-Za-z]+", token))
            or bool(re.match(r"^[A-Z]{2,}$", token))
            or bool(re.search(r"\d", token))
            or normalized in ALIASES.values()
        )
        if not is_entity_like:
            continue
        for part in normalized.split():
            if part and part not in STOPWORDS and part not in question_terms:
                terms.add(part)
    return terms


def term_matches(term: str, candidates: Set[str]) -> bool:
    if term in candidates:
        return True
    if len(term) <= 3:
        return False
    for candidate in candidates:
        if len(candidate) > 3 and (term in candidate or candidate in term):
            return True
    return False


def missing_terms(pred_terms: Set[str], true_terms: Set[str]) -> List[str]:
    return sorted(term for term in pred_terms if not term_matches(term, true_terms))


def missing_ratio(pred_terms: Set[str], true_terms: Set[str]) -> Tuple[float, List[str]]:
    if not pred_terms:
        return 0.0, []
    missing = missing_terms(pred_terms, true_terms)
    return len(missing) / len(pred_terms), missing


def number_missing_ratio(pred_numbers: Set[str], true_numbers: Set[str], question: str) -> Tuple[float, List[str]]:
    question_numbers = number_set(question)
    pred_numbers = {number for number in pred_numbers if number not in question_numbers}
    if not pred_numbers:
        return 0.0, []
    missing = sorted(number for number in pred_numbers if number not in true_numbers)
    return len(missing) / len(pred_numbers), missing


def compute_span_features(row: Dict) -> Dict:
    question = row.get("question", "")
    prediction_text = f"{row.get('prediction', '')} {row.get('merged_prediction', '')}"
    reference_text = f"{row.get('true_answer', '')} {row.get('merged_true_answer', '')}"

    pred_terms = content_terms(prediction_text, question)
    true_terms = content_terms(reference_text, question)
    pred_entities = proper_or_number_terms(prediction_text, question)
    true_entities = proper_or_number_terms(reference_text, question) | true_terms

    term_ratio, missing_content = missing_ratio(pred_terms, true_terms)
    entity_ratio, missing_entities = missing_ratio(pred_entities, true_entities)
    number_ratio, missing_numbers = number_missing_ratio(
        number_set(prediction_text),
        number_set(reference_text),
        question,
    )

    return {
        "answer_term_mismatch": term_ratio,
        "answer_entity_mismatch": entity_ratio,
        "answer_number_mismatch": number_ratio,
        "missing_content_terms": missing_content,
        "missing_entity_terms": missing_entities,
        "missing_numbers": missing_numbers,
    }


def process_dataset(dataset: str, args):
    dataset_dir = os.path.join(args.results_root, dataset)
    prediction_path = os.path.join(dataset_dir, "prediction.json")
    input_metrics_path = os.path.join(dataset_dir, args.input_metrics_file)
    bge_metrics_path = os.path.join(dataset_dir, args.bge_metrics_file)
    output_path = os.path.join(dataset_dir, args.output_file)
    explain_path = os.path.join(dataset_dir, args.explain_file)

    rows = flatten_prediction_records(load_json(prediction_path))
    input_metrics = load_json(input_metrics_path)
    bge_metrics = load_json(bge_metrics_path)

    base_scores = input_metrics["fact_sensitive_similarity"]
    if len(rows) != len(base_scores):
        raise ValueError(
            f"Length mismatch for {dataset}: rows={len(rows)} base_scores={len(base_scores)}"
        )

    output = {
        "cosine_similarity": input_metrics["cosine_similarity"],
        "bge_m3_colbert": bge_metrics.get("bge_m3_colbert", []),
        "token_reweighting_similarity": input_metrics.get("fact_weighted_colbert", []),
        "v1_fact_sensitive_similarity": base_scores,
        "span_penalty_similarity": [],
        "answer_term_mismatch": [],
        "answer_entity_mismatch": [],
        "answer_number_mismatch": [],
        "fact_span_penalty": [],
        "fact_span_similarity": [],
    }
    explanations = []

    for index, (row, base_score) in enumerate(zip(rows, base_scores)):
        features = compute_span_features(row)
        penalty = (
            args.term_penalty_weight * features["answer_term_mismatch"]
            + args.entity_penalty_weight * features["answer_entity_mismatch"]
            + args.number_penalty_weight * features["answer_number_mismatch"]
        )
        bge_colbert_scores = output.get("bge_m3_colbert", [])
        span_only_score = (float(bge_colbert_scores[index]) - penalty) if index < len(bge_colbert_scores) else None
        span_score = base_score - penalty

        for key in [
            "answer_term_mismatch",
            "answer_entity_mismatch",
            "answer_number_mismatch",
        ]:
            output[key].append(features[key])
        output["fact_span_penalty"].append(penalty)
        output["span_penalty_similarity"].append(span_only_score)
        output["fact_span_similarity"].append(span_score)

        if len(explanations) < args.explain_limit and (
            features["missing_content_terms"]
            or features["missing_entity_terms"]
            or features["missing_numbers"]
        ):
            explanations.append(
                {
                    "index": index,
                    "question": row.get("question", ""),
                    "prediction": row.get("prediction", ""),
                    "true_answer": row.get("true_answer", ""),
                    "merged_prediction": row.get("merged_prediction", ""),
                    "merged_true_answer": row.get("merged_true_answer", ""),
                    "v1_fact_sensitive_similarity": base_score,
                    "fact_span_penalty": penalty,
                    "span_penalty_similarity": span_only_score,
                    "fact_span_similarity": span_score,
                    "missing_content_terms": features["missing_content_terms"],
                    "missing_entity_terms": features["missing_entity_terms"],
                    "missing_numbers": features["missing_numbers"],
                }
            )

    output["_metadata"] = {
        "method": "bge_m3_fact_span_v2",
        "input_metrics_file": args.input_metrics_file,
        "term_penalty_weight": args.term_penalty_weight,
        "entity_penalty_weight": args.entity_penalty_weight,
        "number_penalty_weight": args.number_penalty_weight,
    }
    save_json(output, output_path)
    save_json(explanations, explain_path)
    print(f"[{dataset}] saved metrics: {output_path}")
    print(f"[{dataset}] saved explanations: {explain_path}")


def main():
    args = parse_args()
    for dataset in args.datasets:
        process_dataset(dataset, args)


if __name__ == "__main__":
    main()
