import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


DATASET_ORDER = ["nq", "truthfulQA", "sciq", "simple_questions_wiki"]
DATASET_DISPLAY_NAMES = {
    "nq": "NQ",
    "truthfulQA": "TruthfulQA",
    "sciq": "SciQ",
    "simple_questions_wiki": "Simple Questions Wiki",
}

METRIC_CONFIG = {
    "cos": {
        "file_name": "embedding_metrics.json",
        "json_key": "cosine_similarity",
        "display_name": "Cosine similarity",
        "slug": "cos",
    },
    "entity": {
        "file_name": "entity_score.json",
        "json_key": "entity_score",
        "display_name": "Entity matching score",
        "slug": "entity",
    },
    "entailment": {
        "file_name": "entailment_score.json",
        "json_key": "entailment_score",
        "display_name": "Entailment score (F1)",
        "slug": "entailment",
    },
    "entailment_true_only": {
        "file_name": "entailment_true_entail_pred_score.json",
        "json_key": "entailment_true_entail_pred_score",
        "display_name": "Entailment score (true->pred)",
        "slug": "entailment_true_only",
    },
    "entailment_square_sqrt": {
        "file_name": "entailment_square_sqrt_score.json",
        "json_key": "entailment_square_sqrt_score",
        "display_name": "Entailment score (t2p^2 + sqrt(p2t))",
        "slug": "entailment_square_sqrt",
    },
    "entailment_fbeta": {
        "file_name": "entailment_fbeta_score.json",
        "json_key": "entailment_fbeta_score",
        "display_name": "Entailment score (F_beta)",
        "slug": "entailment_fbeta",
    },
    "hybrid": {
        "file_name": "hybrid_score.json",
        "json_key": "hybrid_score",
        "display_name": "Hybrid score",
        "slug": "hybrid",
    },
}


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_dataset_data(
    dataset: str, results_root: str, threshold: float, metric_name: str
) -> Dict[str, np.ndarray]:
    dataset_dir = os.path.join(results_root, dataset)
    correctness_path = os.path.join(dataset_dir, "correctness.json")
    metric_cfg = METRIC_CONFIG[metric_name]
    metric_path = os.path.join(dataset_dir, metric_cfg["file_name"])

    if not os.path.exists(correctness_path):
        raise FileNotFoundError(f"Missing correctness file: {correctness_path}")
    if not os.path.exists(metric_path):
        raise FileNotFoundError(f"Missing metric file: {metric_path}")

    correctness_scores = np.asarray(load_json(correctness_path), dtype=np.float64)
    metric_payload = load_json(metric_path)
    metric_key = metric_cfg["json_key"]
    if metric_key not in metric_payload:
        raise ValueError(f"Missing {metric_key} in: {metric_path}")

    metric_scores = np.asarray(metric_payload[metric_key], dtype=np.float64)
    if len(metric_scores) != len(correctness_scores):
        raise ValueError(
            f"Length mismatch for {dataset}: "
            f"correctness={len(correctness_scores)}, {metric_key}={len(metric_scores)}"
        )

    is_correct = correctness_scores > threshold
    return {
        "correctness_scores": correctness_scores,
        "metric_scores": metric_scores,
        "is_correct": is_correct,
    }


def _safe_bin_edges(values: np.ndarray, bins: int = 50) -> np.ndarray:
    min_v = float(np.min(values))
    max_v = float(np.max(values))
    if np.isclose(min_v, max_v):
        span = 1e-6
        return np.linspace(min_v - span, max_v + span, bins + 1)
    return np.linspace(min_v, max_v, bins + 1)


def plot_distributions(
    dataset_data: Dict[str, Dict[str, np.ndarray]],
    output_path: str,
    score_label: str,
    bins: int = 50,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.ravel()

    for idx, dataset in enumerate(DATASET_ORDER):
        ax = axes[idx]
        scores = dataset_data[dataset]["metric_scores"]
        is_correct = dataset_data[dataset]["is_correct"]
        correct_values = scores[is_correct]
        incorrect_values = scores[~is_correct]

        bin_edges = _safe_bin_edges(scores, bins=bins)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        correct_counts, _ = np.histogram(correct_values, bins=bin_edges)
        incorrect_counts, _ = np.histogram(incorrect_values, bins=bin_edges)

        ax.plot(centers, correct_counts, color="blue", linewidth=2, label="Correct answers")
        ax.plot(centers, incorrect_counts, color="red", linewidth=2, label="Incorrect answers")
        ax.set_title(DATASET_DISPLAY_NAMES[dataset])
        ax.set_xlabel(score_label)
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    fig.suptitle(f"{score_label} Distributions", fontsize=16)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def compute_roc_curve(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    labels = labels.astype(bool)
    if labels.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    pos_count = np.sum(labels)
    neg_count = labels.size - pos_count
    if pos_count == 0 or neg_count == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    tprs: List[float] = [0.0]
    fprs: List[float] = [0.0]
    tp = 0
    fp = 0

    for i in range(sorted_labels.size):
        if sorted_labels[i]:
            tp += 1
        else:
            fp += 1

        is_last = i == sorted_labels.size - 1
        score_changed = not is_last and sorted_scores[i + 1] != sorted_scores[i]
        if is_last or score_changed:
            tprs.append(tp / pos_count)
            fprs.append(fp / neg_count)

    fpr_arr = np.asarray(fprs, dtype=np.float64)
    tpr_arr = np.asarray(tprs, dtype=np.float64)
    auc = float(np.trapz(tpr_arr, fpr_arr))
    return fpr_arr, tpr_arr, auc


def plot_roc_curves(
    dataset_data: Dict[str, Dict[str, np.ndarray]],
    output_path: str,
    score_label: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.ravel()

    for idx, dataset in enumerate(DATASET_ORDER):
        ax = axes[idx]
        scores = dataset_data[dataset]["metric_scores"]
        labels = dataset_data[dataset]["is_correct"]
        fpr, tpr, auc = compute_roc_curve(labels, scores)

        auc_text = "nan" if np.isnan(auc) else f"{auc:.4f}"
        ax.plot(fpr, tpr, color="blue", linewidth=2, label=f"ROC (AUC={auc_text})")
        ax.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1.5, label="Random baseline")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(DATASET_DISPLAY_NAMES[dataset])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right")

    fig.suptitle(f"ROC Curves ({score_label} -> Correctness)", fontsize=16)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_correlations(
    dataset_data: Dict[str, Dict[str, np.ndarray]],
    output_path: str,
    score_label: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.ravel()

    for idx, dataset in enumerate(DATASET_ORDER):
        ax = axes[idx]
        scores = dataset_data[dataset]["metric_scores"]
        correctness_scores = dataset_data[dataset]["correctness_scores"]
        correctness_scores = np.clip(correctness_scores, 0.0, None)

        if scores.size > 1:
            pearson_r = float(np.corrcoef(scores, correctness_scores)[0, 1])
        else:
            pearson_r = float("nan")

        ax.scatter(scores, correctness_scores, s=8, alpha=0.3, color="blue", edgecolors="none")

        r_text = "nan" if np.isnan(pearson_r) else f"{pearson_r:.4f}"
        ax.set_title(f"{DATASET_DISPLAY_NAMES[dataset]} (r={r_text})")
        ax.set_xlabel(score_label)
        ax.set_ylabel("Correctness (hallucination score)")
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.25)

    fig.suptitle(f"Correlation: {score_label} vs Correctness", fontsize=16)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate 2x2 subplot figures for distributions, ROC curves, and "
            "correlation between a selected score and correctness."
        )
    )
    parser.add_argument(
        "--score",
        choices=list(METRIC_CONFIG.keys()),
        required=True,
        help=(
            "Which score to plot: cos/entity/entailment/"
            "entailment_true_only/entailment_square_sqrt/entailment_fbeta/hybrid"
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASET_ORDER,
        help="Datasets to plot. Default: nq truthfulQA sciq simple_questions_wiki",
    )
    parser.add_argument(
        "--results-root",
        default="updated_results",
        help="Root folder containing dataset result directories.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Correctness threshold: score > threshold is treated as correct.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save generated figures. Default: updated_results/plots_<score>",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of bins for distribution curves.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = args.datasets
    if datasets != DATASET_ORDER:
        raise ValueError(
            "This script expects 4 datasets in this order for 2x2 layout: "
            f"{DATASET_ORDER}. Received: {datasets}"
        )

    metric_cfg = METRIC_CONFIG[args.score]
    score_label = metric_cfg["display_name"]
    output_dir = args.output_dir or os.path.join("updated_results", f"plots_{metric_cfg['slug']}")

    dataset_data: Dict[str, Dict[str, np.ndarray]] = {}
    for dataset in DATASET_ORDER:
        dataset_data[dataset] = load_dataset_data(
            dataset=dataset,
            results_root=args.results_root,
            threshold=args.threshold,
            metric_name=args.score,
        )
        total = dataset_data[dataset]["is_correct"].size
        correct = int(np.sum(dataset_data[dataset]["is_correct"]))
        print(f"[{dataset}] total={total}, correct={correct}, incorrect={total - correct}")

    os.makedirs(output_dir, exist_ok=True)
    slug = metric_cfg["slug"]
    distribution_path = os.path.join(output_dir, f"{slug}_distributions.png")
    roc_path = os.path.join(output_dir, f"{slug}_roc_curves.png")
    correlation_path = os.path.join(output_dir, f"{slug}_correlations.png")

    plot_distributions(dataset_data, distribution_path, score_label, bins=args.bins)
    plot_roc_curves(dataset_data, roc_path, score_label)
    plot_correlations(dataset_data, correlation_path, score_label)

    print(f"Saved distributions: {distribution_path}")
    print(f"Saved ROC curves: {roc_path}")
    print(f"Saved correlations: {correlation_path}")


if __name__ == "__main__":
    main()
