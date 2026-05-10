import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


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
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute fact-aware BGE-M3 token similarity. Important factual tokens "
            "such as answer-slot words, proper nouns, dates, and numbers receive "
            "larger weights in the token-level average."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to process.",
    )
    parser.add_argument(
        "--results-root",
        default=os.path.join(PROJECT_ROOT, "updated_results"),
        help="Root folder containing per-dataset result folders.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Local BGE-M3 path. If omitted, use ckpt/bge-m3 when present, "
            "otherwise the cached Hugging Face snapshot, otherwise BAAI/bge-m3."
        ),
    )
    parser.add_argument("--device", default="cuda:0", help="Device for BGE-M3.")
    parser.add_argument("--batch-size", type=int, default=16, help="BGE-M3 batch size.")
    parser.add_argument("--max-length", type=int, default=256, help="Max token length.")
    parser.add_argument(
        "--output-file",
        default="embedding_metrics_bge_m3_fact.json",
        help="Output metrics JSON filename inside each dataset folder.",
    )
    parser.add_argument(
        "--explain-file",
        default="bge_fact_token_explanations.json",
        help="Output token explanation JSON filename inside each dataset folder.",
    )
    parser.add_argument(
        "--explain-limit",
        type=int,
        default=20,
        help="Number of examples saved per dataset for qualitative inspection.",
    )
    parser.add_argument(
        "--answer-weight",
        type=float,
        default=3.0,
        help="Extra multiplicative weight for tokens not appearing in the question.",
    )
    parser.add_argument(
        "--entity-weight",
        type=float,
        default=2.0,
        help="Extra multiplicative weight for proper nouns, numbers, and dates.",
    )
    parser.add_argument(
        "--stopword-weight",
        type=float,
        default=0.35,
        help="Weight assigned to stopwords.",
    )
    parser.add_argument(
        "--lexical-weight-scale",
        type=float,
        default=1.0,
        help="Scale for BGE lexical weights used in token importance.",
    )
    parser.add_argument(
        "--mismatch-penalty",
        type=float,
        default=0.08,
        help="Penalty strength for high-importance factual tokens missing from the reference.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only run the first few samples for quick validation.",
    )
    parser.add_argument(
        "--dry-run-limit",
        type=int,
        default=32,
        help="Number of samples per dataset in dry-run mode.",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable fp16 for BGE-M3 inference.",
    )
    return parser.parse_args()


def resolve_model_path(user_path: Optional[str]) -> str:
    if user_path:
        return user_path

    ckpt_path = Path(PROJECT_ROOT) / "ckpt" / "bge-m3"
    if ckpt_path.exists():
        return str(ckpt_path)

    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-m3" / "snapshots"
    if cache_root.exists():
        snapshots = sorted(
            [path for path in cache_root.iterdir() if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            return str(snapshots[0])

    return "BAAI/bge-m3"


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
        record = item[0] if isinstance(item, list) else item
        rows.append(record)
    return rows


def normalize_token(token: str) -> str:
    token = token.replace("▁", "").strip()
    token = token.lower()
    token = re.sub(r"^[^\w]+|[^\w]+$", "", token)
    return token


def normalize_word(word: str) -> str:
    return re.sub(r"^[^\w]+|[^\w]+$", "", word.lower())


def text_word_set(text: str) -> set:
    return {
        normalize_word(match.group(0))
        for match in re.finditer(r"\b[\w'-]+\b", text)
        if normalize_word(match.group(0))
    }


def is_number_or_date(token: str) -> bool:
    compact = token.replace(",", "")
    return bool(re.search(r"\d", compact))


def is_proper_like(token: str) -> bool:
    token = token.replace("▁", "").strip()
    if not token or is_number_or_date(token):
        return False
    return bool(re.match(r"[A-Z][A-Za-z]+(?:[-'][A-Za-z]+)?$", token))


def token_ids_and_texts(tokenizer, text: str, max_length: int) -> Tuple[List[int], List[str]]:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    return ids, tokens


def text_token_norm_set(model, text: str, max_length: int) -> set:
    ids, tokens = token_ids_and_texts(model.tokenizer, text, max_length)
    special_ids = set(model.tokenizer.all_special_ids)
    norms = set()
    for token_id, token in zip(ids, tokens):
        if token_id in special_ids:
            continue
        norm = normalize_token(token)
        if norm:
            norms.add(norm)
    return norms


def non_special_token_data(model, text: str, vecs: np.ndarray, max_length: int) -> List[Dict]:
    ids, tokens = token_ids_and_texts(model.tokenizer, text, max_length)
    usable = min(len(vecs), len(ids))
    special_ids = set(model.tokenizer.all_special_ids)
    items = []
    for index in range(usable):
        token_id = ids[index]
        if token_id in special_ids:
            continue
        raw = tokens[index]
        norm = normalize_token(raw)
        if not norm:
            continue
        items.append(
            {
                "index": index,
                "id": str(token_id),
                "token": raw,
                "clean": raw.replace("▁", ""),
                "norm": norm,
                "vector": vecs[index],
            }
        )
    return items


def token_importance(
    item: Dict,
    lexical_weights: Dict[str, float],
    question_terms: set,
    args,
) -> Tuple[float, List[str]]:
    norm = item["norm"]
    raw = item["clean"]
    is_short_fragment = len(norm) <= 2 and not is_number_or_date(raw)
    is_in_question = exact_or_subword_overlap(norm, question_terms)
    reasons = []

    if norm in STOPWORDS:
        weight = args.stopword_weight
        reasons.append("stopword")
    elif is_short_fragment:
        weight = args.stopword_weight
        reasons.append("short_subword")
    else:
        weight = 1.0
        reasons.append("content")

    is_answer_slot = norm and not is_in_question and norm not in STOPWORDS and not is_short_fragment
    if is_answer_slot:
        weight *= args.answer_weight
        reasons.append("answer_slot")

    is_word_start = item["token"].startswith("▁")
    is_entity_like = is_number_or_date(raw) or (is_word_start and is_proper_like(raw))
    if is_entity_like and (is_answer_slot or is_number_or_date(raw)):
        weight *= args.entity_weight
        reasons.append("entity_or_number")

    lexical_weight = float(lexical_weights.get(item["id"], 0.0))
    if lexical_weight > 0:
        weight *= 1.0 + args.lexical_weight_scale * lexical_weight
        reasons.append("bge_lexical")

    return float(weight), reasons


def exact_or_subword_overlap(token_norm: str, target_norms: set) -> bool:
    if token_norm in target_norms:
        return True
    if len(token_norm) <= 2:
        return False
    return any(token_norm in other or other in token_norm for other in target_norms if len(other) > 2)


def fact_token_penalty(
    pred_item: Dict,
    pred_weight: float,
    pred_reasons: List[str],
    true_norms: set,
    question_terms: set,
    penalty_strength: float,
) -> float:
    norm = pred_item["norm"]
    if exact_or_subword_overlap(norm, question_terms) or norm in STOPWORDS or len(norm) <= 2:
        return 0.0

    is_fact_sensitive = (
        "answer_slot" in pred_reasons
        or "entity_or_number" in pred_reasons
        or is_number_or_date(pred_item["clean"])
        or is_proper_like(pred_item["clean"])
    )
    if not is_fact_sensitive:
        return 0.0
    if exact_or_subword_overlap(norm, true_norms):
        return 0.0
    return penalty_strength * pred_weight


def compute_pair_fact_score(
    pred_text: str,
    true_text: str,
    question: str,
    pred_encoded: Dict,
    true_encoded: Dict,
    model,
    args,
) -> Tuple[Dict[str, float], Dict]:
    pred_items = non_special_token_data(
        model, pred_text, pred_encoded["colbert_vecs"], args.max_length
    )
    true_items = non_special_token_data(
        model, true_text, true_encoded["colbert_vecs"], args.max_length
    )

    if not pred_items or not true_items:
        return (
            {
                "fact_weighted_colbert": 0.0,
                "fact_mismatch_penalty": 0.0,
                "fact_sensitive_similarity": 0.0,
                "fact_weighted_sparse": 0.0,
            },
            {"tokens": []},
        )

    question_terms = text_word_set(question) | text_token_norm_set(model, question, args.max_length)
    true_norms = {item["norm"] for item in true_items}
    pred_vecs = torch.tensor(np.stack([item["vector"] for item in pred_items]), dtype=torch.float32)
    true_vecs = torch.tensor(np.stack([item["vector"] for item in true_items]), dtype=torch.float32)
    token_scores = pred_vecs @ true_vecs.T
    best_scores, best_indices = torch.max(token_scores, dim=1)

    weights = []
    penalties = []
    explanation_tokens = []
    weighted_sparse_num = 0.0
    weighted_sparse_den = 0.0
    true_lexical = true_encoded["lexical_weights"]

    for item, best_score, best_index in zip(pred_items, best_scores.tolist(), best_indices.tolist()):
        weight, reasons = token_importance(item, pred_encoded["lexical_weights"], question_terms, args)
        penalty = fact_token_penalty(
            item,
            weight,
            reasons,
            true_norms,
            question_terms,
            args.mismatch_penalty,
        )
        weights.append(weight)
        penalties.append(penalty)

        pred_lw = float(pred_encoded["lexical_weights"].get(item["id"], 0.0))
        matched = true_items[best_index]
        true_lw = float(true_lexical.get(matched["id"], 0.0))
        if item["id"] in true_lexical:
            weighted_sparse_num += weight * pred_lw * float(true_lexical[item["id"]])
        weighted_sparse_den += max(weight * pred_lw, 1e-12)

        explanation_tokens.append(
            {
                "token": item["clean"],
                "normalized": item["norm"],
                "weight": weight,
                "reasons": reasons,
                "best_reference_token": matched["clean"],
                "colbert_match": float(best_score),
                "missing_from_reference_penalty": float(penalty),
            }
        )

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    score_tensor = best_scores.detach().float()
    fact_weighted_colbert = float(torch.sum(weights_tensor * score_tensor) / torch.sum(weights_tensor))
    mismatch_penalty = float(sum(penalties) / max(sum(weights), 1e-12))
    fact_sensitive_similarity = fact_weighted_colbert - mismatch_penalty
    fact_weighted_sparse = weighted_sparse_num / max(weighted_sparse_den, 1e-12)

    dense_pred = torch.tensor(pred_encoded["dense_vecs"], dtype=torch.float32).reshape(1, -1)
    dense_true = torch.tensor(true_encoded["dense_vecs"], dtype=torch.float32).reshape(1, -1)
    dense_cosine = float(F.cosine_similarity(dense_pred, dense_true, dim=1).item())

    scores = {
        "cosine_similarity": dense_cosine,
        "bge_m3_dense": dense_cosine,
        "fact_weighted_colbert": fact_weighted_colbert,
        "fact_mismatch_penalty": mismatch_penalty,
        "fact_sensitive_similarity": fact_sensitive_similarity,
        "fact_weighted_sparse": float(fact_weighted_sparse),
    }
    explanation = {
        "question": question,
        "prediction": pred_text,
        "reference": true_text,
        "scores": scores,
        "tokens": sorted(
            explanation_tokens,
            key=lambda token: token["weight"] * token["colbert_match"],
            reverse=True,
        ),
    }
    return scores, explanation


def encode_text_batch(model, texts: List[str], args) -> Dict:
    return model.encode(
        texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )


def get_encoded_item(encoded: Dict, index: int) -> Dict:
    return {
        "dense_vecs": encoded["dense_vecs"][index],
        "lexical_weights": encoded["lexical_weights"][index],
        "colbert_vecs": encoded["colbert_vecs"][index],
    }


def process_dataset(dataset: str, model, args):
    dataset_dir = os.path.join(args.results_root, dataset)
    prediction_path = os.path.join(dataset_dir, "prediction.json")
    output_path = os.path.join(dataset_dir, args.output_file)
    explanation_path = os.path.join(dataset_dir, args.explain_file)

    if not os.path.exists(prediction_path):
        print(f"[{dataset}] missing prediction JSON: {prediction_path}")
        return

    rows = flatten_prediction_records(load_json(prediction_path))
    if args.dry_run:
        rows = rows[: args.dry_run_limit]

    metric_lists = {
        "cosine_similarity": [],
        "bge_m3_dense": [],
        "fact_weighted_colbert": [],
        "fact_mismatch_penalty": [],
        "fact_sensitive_similarity": [],
        "fact_weighted_sparse": [],
    }
    explanations = []

    for start in tqdm(range(0, len(rows), args.batch_size), desc=f"{dataset} fact scoring"):
        batch_rows = rows[start : start + args.batch_size]
        pred_texts = [row["merged_prediction"] for row in batch_rows]
        true_texts = [row["merged_true_answer"] for row in batch_rows]
        encoded = encode_text_batch(model, pred_texts + true_texts, args)

        for offset, row in enumerate(batch_rows):
            scores, explanation = compute_pair_fact_score(
                pred_text=row["merged_prediction"],
                true_text=row["merged_true_answer"],
                question=row.get("question", ""),
                pred_encoded=get_encoded_item(encoded, offset),
                true_encoded=get_encoded_item(encoded, len(batch_rows) + offset),
                model=model,
                args=args,
            )
            for name, value in scores.items():
                metric_lists[name].append(value)
            if len(explanations) < args.explain_limit:
                explanation["index"] = start + offset
                explanations.append(explanation)

    metric_lists["_metadata"] = {
        "embedding_method": "bge_m3_fact_token_weighted",
        "model_path": args.model_path,
        "max_length": args.max_length,
        "answer_weight": args.answer_weight,
        "entity_weight": args.entity_weight,
        "stopword_weight": args.stopword_weight,
        "lexical_weight_scale": args.lexical_weight_scale,
        "mismatch_penalty": args.mismatch_penalty,
        "dry_run": args.dry_run,
    }
    save_json(metric_lists, output_path)
    save_json(explanations, explanation_path)
    print(f"[{dataset}] saved metrics: {output_path}")
    print(f"[{dataset}] saved explanations: {explanation_path}")


def main():
    args = parse_args()
    args.model_path = resolve_model_path(args.model_path)
    if args.model_path != "BAAI/bge-m3":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from FlagEmbedding import BGEM3FlagModel

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested {args.device}, but CUDA is not available. Falling back to CPU.")
        args.device = "cpu"

    print(f"Loading BGE-M3 from: {args.model_path}")
    model = BGEM3FlagModel(args.model_path, use_fp16=not args.no_fp16, devices=args.device)

    for dataset in args.datasets:
        process_dataset(dataset, model, args)


if __name__ == "__main__":
    main()
