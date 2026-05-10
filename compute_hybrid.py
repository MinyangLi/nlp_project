import argparse
import json
import os

from tqdm import tqdm

from factual_ultils import compute_factual_scores_batch

DEFAULT_DATASETS = ["nq", "sciq", "simple_questions_wiki", "truthfulQA"]


def _flatten_prediction_items(node):
    if isinstance(node, dict):
        if "merged_prediction" in node and "merged_true_answer" in node:
            return [node]
        return []

    if not isinstance(node, list):
        return []

    flattened = []
    for item in node:
        flattened.extend(_flatten_prediction_items(item))
    return flattened


def load_prediction_pairs(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    items = _flatten_prediction_items(data)
    pairs = []
    for item in items:
        predicted_answer = item["merged_prediction"]
        true_answer = item["merged_true_answer"]
        pairs.append((predicted_answer, true_answer))
    return pairs


def load_similarity_scores(path):
    with open(path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    cosine_similarity = metrics.get("cosine_similarity")
    if not isinstance(cosine_similarity, list):
        raise ValueError(f"'cosine_similarity' not found or invalid in {path}")

    return [float(score) for score in cosine_similarity]


def min_max_normalize(scores):
    if not scores:
        return []

    score_min = min(scores)
    score_max = max(scores)
    if score_max == score_min:
        return [0.0 for _ in scores]

    return [(score - score_min) / (score_max - score_min) for score in scores]


def process_dataset(
    dataset,
    results_root,
    ner_batch_size,
    nli_batch_size,
    show_progress,
    normalize_scores,
    entailment_fbeta_beta,
    compute_entity,
):
    dataset_dir = os.path.join(results_root, dataset)
    prediction_path = os.path.join(dataset_dir, "prediction.json")
    embedding_path = os.path.join(dataset_dir, "embedding_metrics.json")
    output_path = os.path.join(dataset_dir, "hybrid_score.json")
    entity_path = os.path.join(dataset_dir, "entity_score.json")
    entailment_path = os.path.join(dataset_dir, "entailment_score.json")
    entailment_true_only_path = os.path.join(dataset_dir, "entailment_true_entail_pred_score.json")
    entailment_square_sqrt_path = os.path.join(dataset_dir, "entailment_square_sqrt_score.json")
    entailment_fbeta_path = os.path.join(dataset_dir, "entailment_fbeta_score.json")

    if not os.path.exists(prediction_path):
        return {
            "dataset": dataset,
            "status": "missing_prediction",
            "path": prediction_path,
        }

    if not os.path.exists(embedding_path):
        return {
            "dataset": dataset,
            "status": "missing_embedding_metrics",
            "path": embedding_path,
        }

    prediction_pairs = load_prediction_pairs(prediction_path)
    cosine_scores = load_similarity_scores(embedding_path)

    if len(prediction_pairs) != len(cosine_scores):
        return {
            "dataset": dataset,
            "status": "length_mismatch",
            "prediction_count": len(prediction_pairs),
            "similarity_count": len(cosine_scores),
        }

    predicted_answers = [pair[0] for pair in prediction_pairs]
    true_answers = [pair[1] for pair in prediction_pairs]
    factual_scores = compute_factual_scores_batch(
        predicted_answers,
        true_answers,
        ner_batch_size=ner_batch_size,
        nli_batch_size=nli_batch_size,
        show_progress=show_progress,
        progress_prefix=dataset,
        fbeta_beta=entailment_fbeta_beta,
        compute_entity=compute_entity,
    )

    hybrid_scores = []
    entity_scores = []
    entailment_scores = []
    entailment_true_only_scores = []
    entailment_square_sqrt_scores = []
    entailment_fbeta_scores = []
    iterator = zip(cosine_scores, factual_scores)
    if show_progress:
        iterator = tqdm(
            iterator,
            total=len(cosine_scores),
            desc=f"{dataset} combine",
        )

    for cosine_similarity, factual_score in iterator:
        entity_matching_score = factual_score["entity_matching_score"]
        entailment_score = factual_score["entailment_f1"]
        entailment_true_only_score = factual_score["true_entail_pred"]
        entailment_square_sqrt_score = factual_score["entailment_square_sqrt"]
        entailment_fbeta_score = factual_score["entailment_fbeta"]

        entity_scores.append(float(entity_matching_score))
        entailment_scores.append(float(entailment_score))
        entailment_true_only_scores.append(float(entailment_true_only_score))
        entailment_square_sqrt_scores.append(float(entailment_square_sqrt_score))
        entailment_fbeta_scores.append(float(entailment_fbeta_score))
        hybrid_score = 0.3 * float(cosine_similarity) + 0.7 * float(entailment_true_only_score)
        hybrid_scores.append(hybrid_score)

    final_scores = min_max_normalize(hybrid_scores) if normalize_scores else hybrid_scores
    payload = {"hybrid_score": final_scores}
    entity_payload = {"entity_score": entity_scores}
    entailment_payload = {"entailment_score": entailment_scores}
    entailment_true_only_payload = {"entailment_true_entail_pred_score": entailment_true_only_scores}
    entailment_square_sqrt_payload = {"entailment_square_sqrt_score": entailment_square_sqrt_scores}
    entailment_fbeta_payload = {"entailment_fbeta_score": entailment_fbeta_scores}

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file)
    if compute_entity:
        with open(entity_path, "w", encoding="utf-8") as file:
            json.dump(entity_payload, file)
    with open(entailment_path, "w", encoding="utf-8") as file:
        json.dump(entailment_payload, file)
    with open(entailment_true_only_path, "w", encoding="utf-8") as file:
        json.dump(entailment_true_only_payload, file)
    with open(entailment_square_sqrt_path, "w", encoding="utf-8") as file:
        json.dump(entailment_square_sqrt_payload, file)
    with open(entailment_fbeta_path, "w", encoding="utf-8") as file:
        json.dump(entailment_fbeta_payload, file)

    return {
        "dataset": dataset,
        "status": "ok",
        "count": len(final_scores),
        "output_path": output_path,
        "entity_path": entity_path if compute_entity else None,
        "entailment_path": entailment_path,
        "entailment_true_only_path": entailment_true_only_path,
        "entailment_square_sqrt_path": entailment_square_sqrt_path,
        "entailment_fbeta_path": entailment_fbeta_path,
        "normalized": normalize_scores,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute hybrid score from cosine similarity and true->pred entailment score "
            "for each dataset: hybrid = 0.5 * cosine + 0.5 * entailment_true_only."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to process, e.g. --datasets nq sciq",
    )
    parser.add_argument(
        "--results-root",
        default="updated_results",
        help="Root folder containing per-dataset result folders.",
    )
    parser.add_argument(
        "--ner-batch-size",
        type=int,
        default=64,
        help="Batch size for NER pipeline inference (default: 64).",
    )
    parser.add_argument(
        "--nli-batch-size",
        type=int,
        default=32,
        help="Batch size for NLI model inference (default: 32).",
    )
    parser.add_argument(
        "--disable-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not min-max normalize hybrid scores before saving.",
    )
    parser.add_argument(
        "--entailment-fbeta-beta",
        type=float,
        default=0.3,
        help="Beta value used in entailment F_beta score (default: 1.0).",
    )
    parser.add_argument(
        "--entity",
        action="store_true",
        help="Compute and save entity score. If not set, entity score is skipped.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    show_progress = not args.disable_progress
    normalize_scores = not args.no_normalize
    datasets = args.datasets
    dataset_iterator = datasets
    if show_progress:
        dataset_iterator = tqdm(datasets, desc="Datasets")

    for dataset in dataset_iterator:
        report = process_dataset(
            dataset,
            args.results_root,
            args.ner_batch_size,
            args.nli_batch_size,
            show_progress,
            normalize_scores,
            args.entailment_fbeta_beta,
            args.entity,
        )
        if report["status"] == "ok":
            mode = "normalized" if report["normalized"] else "raw"
            entity_text = (
                f"entity={report['entity_path']}" if report["entity_path"] else "entity=skipped"
            )
            print(
                f"[{report['dataset']}] count={report['count']} mode={mode} "
                f"saved={report['output_path']} "
                f"{entity_text} entailment={report['entailment_path']} "
                f"true_only={report['entailment_true_only_path']} "
                f"square_sqrt={report['entailment_square_sqrt_path']} "
                f"fbeta={report['entailment_fbeta_path']}"
            )
        else:
            details = ", ".join(f"{k}={v}" for k, v in report.items() if k != "dataset")
            print(f"[{report['dataset']}] {details}")


if __name__ == "__main__":
    main()
