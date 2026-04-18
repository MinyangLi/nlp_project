import argparse
import io
import json
import os
import pickle

import torch
import torch.nn.functional as F


DEFAULT_DATASETS = ["nq", "sciq", "simple_questions_wiki", "truthfulQA"]


def load_pickle_cpu_safe(path):
    try:
        with open(path, "rb") as file:
            return pickle.load(file)
    except RuntimeError as error:
        message = str(error)
        if "Attempting to deserialize object on a CUDA device" not in message:
            raise

    original_loader = torch.storage._load_from_bytes

    def cpu_loader(byte_data):
        return torch.load(io.BytesIO(byte_data), map_location=torch.device("cpu"))

    torch.storage._load_from_bytes = cpu_loader
    try:
        with open(path, "rb") as file:
            return pickle.load(file)
    finally:
        torch.storage._load_from_bytes = original_loader


def to_2d_tensor(embeddings):
    if torch.is_tensor(embeddings):
        if embeddings.ndim == 1:
            return embeddings.unsqueeze(0).detach().float().cpu()
        return embeddings.detach().float().cpu()

    if not isinstance(embeddings, list):
        raise TypeError(f"Unsupported embeddings type: {type(embeddings)}")

    if len(embeddings) == 0:
        return torch.empty((0, 0), dtype=torch.float32)

    rows = []
    for item in embeddings:
        if not torch.is_tensor(item):
            item = torch.tensor(item)
        rows.append(item.detach().float().cpu().reshape(-1))
    return torch.stack(rows, dim=0)


def compute_metrics(pred_embeddings, true_embeddings):
    pred_tensor = to_2d_tensor(pred_embeddings)
    true_tensor = to_2d_tensor(true_embeddings)

    if pred_tensor.shape != true_tensor.shape:
        raise ValueError(
            f"Shape mismatch: pred={tuple(pred_tensor.shape)} true={tuple(true_tensor.shape)}"
        )

    if pred_tensor.numel() == 0:
        return {
            "cosine_similarity": [],
            "l2_distance": [],
            "l1_distance": [],
        }

    diff = pred_tensor - true_tensor
    cosine_similarity = F.cosine_similarity(pred_tensor, true_tensor, dim=1)
    l2_distance = torch.norm(diff, p=2, dim=1)
    l1_distance = torch.norm(diff, p=1, dim=1)

    return {
        "cosine_similarity": cosine_similarity.tolist(),
        "l2_distance": l2_distance.tolist(),
        "l1_distance": l1_distance.tolist(),
    }


def process_dataset(dataset, results_root):
    dataset_dir = os.path.join(results_root, dataset)
    input_path = os.path.join(dataset_dir, "embeddings.pkl")
    metrics_path = os.path.join(dataset_dir, "embedding_metrics.json")

    if not os.path.exists(input_path):
        return {
            "dataset": dataset,
            "status": "missing_embeddings",
            "input_path": input_path,
        }

    embeddings = load_pickle_cpu_safe(input_path)
    pred_embeddings = embeddings.get("pred_embeddings")
    true_embeddings = embeddings.get("true_embeddings")

    if pred_embeddings is None or true_embeddings is None:
        return {
            "dataset": dataset,
            "status": "invalid_embeddings_file",
            "input_path": input_path,
        }

    metrics = compute_metrics(pred_embeddings, true_embeddings)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file)

    return {
        "dataset": dataset,
        "status": "ok",
        "count": len(metrics["cosine_similarity"]),
        "metrics_path": metrics_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute cosine similarity, L2 distance, and L1 distance from saved embeddings."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to process, e.g. --datasets nq sciq",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root folder containing per-dataset result folders.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for dataset in args.datasets:
        report = process_dataset(dataset, args.results_root)
        if report["status"] == "ok":
            print(
                f"[{report['dataset']}] count={report['count']} saved={report['metrics_path']}"
            )
        else:
            print(f"[{report['dataset']}] status={report['status']} path={report['input_path']}")


if __name__ == "__main__":
    main()
