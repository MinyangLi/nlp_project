import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASETS = ["nq", "truthfulQA", "sciq", "simple_questions_wiki"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute BGE-M3 dense/sparse/ColBERT/hybrid pair scores from "
            "updated_results/<dataset>/prediction.json."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument(
        "--results-root",
        default=os.path.join(PROJECT_ROOT, "updated_results"),
        help="Root folder containing per-dataset result folders.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local BGE-M3 path. If omitted, use ckpt/bge-m3, local HF cache, or BAAI/bge-m3.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--output-file", default="embedding_metrics_bge_m3.json")
    parser.add_argument("--no-fp16", action="store_true", help="Disable fp16 for BGE-M3 inference.")
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


def load_prediction_rows(path: str) -> List[Dict]:
    rows = _flatten_prediction_items(load_json(path))
    if not rows:
        raise ValueError(f"No prediction rows found in {path}")
    return rows


def _as_list(values):
    if values is None:
        return None
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (int, float)):
        return [float(values)]
    return [float(value) for value in values]


def _extract_bge_scores(score_dict):
    candidates = {
        "bge_m3_dense": ["dense", "dense_score", "dense_scores"],
        "bge_m3_sparse": ["sparse", "sparse_score", "sparse_scores", "lexical", "lexical_score"],
        "bge_m3_colbert": ["colbert", "colbert_score", "colbert_scores", "multi_vector"],
        "bge_m3_hybrid": [
            "colbert+sparse+dense",
            "sparse+dense",
            "hybrid",
            "hybrid_score",
            "hybrid_scores",
        ],
    }

    extracted = {}
    for output_name, keys in candidates.items():
        for key in keys:
            if key in score_dict:
                extracted[output_name] = _as_list(score_dict[key])
                break
    return extracted


def _extend_pair_scores(target, score_dict):
    extracted = _extract_bge_scores(score_dict)
    for key, values in extracted.items():
        target.setdefault(key, []).extend(values)


def process_dataset(dataset: str, model, args):
    dataset_dir = os.path.join(args.results_root, dataset)
    prediction_path = os.path.join(dataset_dir, "prediction.json")
    output_path = os.path.join(dataset_dir, args.output_file)

    if not os.path.exists(prediction_path):
        return {"dataset": dataset, "status": "missing_prediction", "path": prediction_path}

    rows = load_prediction_rows(prediction_path)
    pair_scores = {}

    for start in tqdm(range(0, len(rows), args.chunk_size), desc=f"{dataset} BGE-M3"):
        chunk = rows[start : start + args.chunk_size]
        pairs = [(row["merged_prediction"], row["merged_true_answer"]) for row in chunk]
        score_dict = model.compute_score(
            pairs,
            batch_size=args.batch_size,
            max_passage_length=args.max_length,
            weights_for_different_modes=[0.4, 0.2, 0.4],
        )
        _extend_pair_scores(pair_scores, score_dict)

    expected = len(rows)
    for metric_name, values in pair_scores.items():
        if len(values) != expected:
            raise ValueError(f"{dataset}: {metric_name} length={len(values)}, expected={expected}")

    pair_scores["_metadata"] = {
        "method": "bge_m3_pair_scores",
        "model_path": args.model_path,
        "weights_for_different_modes": [0.4, 0.2, 0.4],
        "prediction_file": "prediction.json",
    }
    save_json(pair_scores, output_path)
    return {"dataset": dataset, "status": "ok", "count": expected, "saved": output_path}


def main():
    args = parse_args()
    args.model_path = resolve_model_path(args.model_path)
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as error:
        raise ImportError("Install BGE-M3 dependencies with: python -m pip install FlagEmbedding") from error

    model = BGEM3FlagModel(args.model_path, use_fp16=not args.no_fp16, devices=args.device)
    for dataset in args.datasets:
        report = process_dataset(dataset, model, args)
        if report["status"] == "ok":
            print(f"[{dataset}] count={report['count']} saved={report['saved']}")
        else:
            print(f"[{dataset}] status={report['status']} path={report['path']}")


if __name__ == "__main__":
    main()
