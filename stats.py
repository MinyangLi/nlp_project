import argparse
import json
import os
from typing import Dict, List


DEFAULT_DATASETS = ["nq", "sciq", "simple_questions_wiki", "truthfulQA"]


def quantile(values: List[float], q: float) -> float:
	if not values:
		raise ValueError("Cannot compute quantile of empty list.")
	if q <= 0:
		return min(values)
	if q >= 1:
		return max(values)

	sorted_vals = sorted(values)
	pos = (len(sorted_vals) - 1) * q
	lower_idx = int(pos)
	upper_idx = min(lower_idx + 1, len(sorted_vals) - 1)
	weight = pos - lower_idx

	lower_val = sorted_vals[lower_idx]
	upper_val = sorted_vals[upper_idx]
	return lower_val * (1 - weight) + upper_val * weight


def summarize(values: List[float], lower_q: float, upper_q: float) -> Dict[str, float]:
	if not values:
		return {
			"count": 0,
			"mean": None,
			"median": None,
			"lower_quantile": None,
			"upper_quantile": None,
			"min": None,
			"max": None,
		}

	sorted_vals = sorted(values)
	count = len(sorted_vals)
	middle = count // 2
	if count % 2 == 1:
		median_val = sorted_vals[middle]
	else:
		median_val = (sorted_vals[middle - 1] + sorted_vals[middle]) / 2

	return {
		"count": count,
		"mean": sum(sorted_vals) / count,
		"median": median_val,
		"lower_quantile": quantile(sorted_vals, lower_q),
		"upper_quantile": quantile(sorted_vals, upper_q),
		"min": sorted_vals[0],
		"max": sorted_vals[-1],
	}


def load_json(path: str):
	with open(path, "r", encoding="utf-8") as file:
		return json.load(file)


def process_dataset(
	dataset: str,
	results_root: str,
	threshold: float,
	lower_q: float,
	upper_q: float,
) -> Dict:
	dataset_dir = os.path.join(results_root, dataset)
	correctness_path = os.path.join(dataset_dir, "correctness.json")
	metrics_path = os.path.join(dataset_dir, "embedding_metrics.json")
	output_path = os.path.join(dataset_dir, "correct_similarity_stats.json")

	if not os.path.exists(correctness_path):
		return {
			"dataset": dataset,
			"status": "missing_correctness",
			"path": correctness_path,
		}

	if not os.path.exists(metrics_path):
		return {
			"dataset": dataset,
			"status": "missing_similarity_metrics",
			"path": metrics_path,
		}

	correctness_scores = load_json(correctness_path)
	metrics = load_json(metrics_path)

	if "cosine_similarity" not in metrics:
		return {
			"dataset": dataset,
			"status": "invalid_metrics_format",
			"path": metrics_path,
		}

	sample_count = len(correctness_scores)
	metric_lengths = {metric_name: len(metric_values) for metric_name, metric_values in metrics.items()}
	for metric_name, metric_len in metric_lengths.items():
		if metric_len != sample_count:
			return {
				"dataset": dataset,
				"status": "length_mismatch",
				"correctness_count": sample_count,
				"metric": metric_name,
				"metric_count": metric_len,
			}

	is_correct = [score > threshold for score in correctness_scores]
	correct_count = sum(is_correct)
	wrong_count = sample_count - correct_count

	metric_stats_correct = {}
	metric_stats_incorrect = {}
	for metric_name, metric_values in metrics.items():
		correct_values = [value for value, flag in zip(metric_values, is_correct) if flag]
		incorrect_values = [value for value, flag in zip(metric_values, is_correct) if not flag]
		metric_stats_correct[metric_name] = summarize(correct_values, lower_q, upper_q)
		metric_stats_incorrect[metric_name] = summarize(incorrect_values, lower_q, upper_q)

	output_obj = {
		"dataset": dataset,
		"threshold": threshold,
		"lower_quantile_q": lower_q,
		"upper_quantile_q": upper_q,
		"total_samples": sample_count,
		"correct_samples": correct_count,
		"wrong_samples": wrong_count,
		"stats_for_correct_samples": metric_stats_correct,
		"stats_for_incorrect_samples": metric_stats_incorrect,
	}

	with open(output_path, "w", encoding="utf-8") as file:
		json.dump(output_obj, file, indent=2)

	return {
		"dataset": dataset,
		"status": "ok",
		"total": sample_count,
		"correct": correct_count,
		"incorrect": wrong_count,
		"saved": output_path,
	}


def parse_args():
	parser = argparse.ArgumentParser(
		description=(
			"For each dataset: threshold correctness scores and report stats "
			"of similarity metrics on correct and incorrect samples."
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
		default="results",
		help="Root directory containing dataset result folders.",
	)
	parser.add_argument(
		"--threshold",
		type=float,
		default=0.5,
		help="Correctness threshold: score > threshold means correct.",
	)
	parser.add_argument(
		"--lower-q",
		type=float,
		default=0.25,
		help="Lower quantile to report.",
	)
	parser.add_argument(
		"--upper-q",
		type=float,
		default=0.75,
		help="Upper quantile to report.",
	)
	return parser.parse_args()


def main():
	args = parse_args()

	for dataset in args.datasets:
		report = process_dataset(
			dataset=dataset,
			results_root=args.results_root,
			threshold=args.threshold,
			lower_q=args.lower_q,
			upper_q=args.upper_q,
		)
		if report["status"] == "ok":
			print(
				f"[{report['dataset']}] total={report['total']} correct={report['correct']} "
				f"incorrect={report['incorrect']} "
				f"saved={report['saved']}"
			)
		else:
			print(f"[{report['dataset']}] status={report['status']} details={report}")


if __name__ == "__main__":
	main()
